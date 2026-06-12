"""
Data catalog - inventory and quality validation of every OHLCV dataset.

Backtests are REFUSED (DataCatalogError) when data is missing, too short, or too
gappy. The catalog is persisted as JSON and a human-readable validation report.
"""

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.research.factory_config import load_factory_config, resolve_path, factory_root

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

TIMEFRAME_MINUTES = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "4h": 240, "1d": 1440,
}


class DataCatalogError(Exception):
    """Raised when a dataset is unusable. Never caught silently."""


@dataclass
class DatasetRecord:
    dataset_id: str
    path: str
    symbol: str
    timeframe: str
    rows: int
    start: str
    end: str
    missing_fraction: float
    max_gap_minutes: float
    gap_count: int
    source: str
    usable: bool
    issues: List[str]


def _infer_symbol_timeframe(path: Path) -> (str, str):
    """Infer symbol/timeframe from filenames like BTC-USD-15m.csv."""
    m = re.match(r"(.+?)[-_](1m|5m|15m|30m|1h|4h|1d)$", path.stem, re.IGNORECASE)
    if m:
        return m.group(1).upper(), m.group(2).lower()
    return path.stem.upper(), "unknown"


def load_ohlcv(path: Path) -> pd.DataFrame:
    """
    Load an OHLCV CSV into the canonical frame: DatetimeIndex + Open/High/Low/
    Close/Volume float columns. Extra numeric columns (signal features such as
    FundingRate) are preserved with their original names - backtesting.py
    exposes them to strategies via self.data.<Name>.
    Raises DataCatalogError on structural problems.
    """
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.lower().str.startswith("unnamed")]
    rename, time_col = {}, None
    for c in df.columns:
        lc = c.strip().lower()
        if time_col is None and lc in ("datetime", "date", "time", "timestamp"):
            time_col = c
        elif lc in ("open", "high", "low", "close", "volume"):
            rename[c] = lc.capitalize()
    if time_col is None:
        raise DataCatalogError(f"{path}: no datetime column found (columns={list(df.columns)})")
    missing = [c for c in ("Open", "High", "Low", "Close", "Volume")
               if c not in set(rename.values())]
    if missing:
        raise DataCatalogError(f"{path}: missing OHLCV columns {missing}")
    df[time_col] = pd.to_datetime(df[time_col].astype(str).str.strip())
    df = df.rename(columns=rename).set_index(time_col).sort_index()
    extras = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    df = df[REQUIRED_COLUMNS + extras].astype(float)
    if df.index.has_duplicates:
        dupes = int(df.index.duplicated().sum())
        raise DataCatalogError(f"{path}: {dupes} duplicate timestamps")
    return df


def _analyze(path: Path, config: dict) -> DatasetRecord:
    symbol, timeframe = _infer_symbol_timeframe(path)
    issues: List[str] = []
    try:
        df = load_ohlcv(path)
    except DataCatalogError as e:
        return DatasetRecord(
            dataset_id=path.stem, path=str(path), symbol=symbol, timeframe=timeframe,
            rows=0, start="", end="", missing_fraction=1.0, max_gap_minutes=0.0,
            gap_count=0, source="csv", usable=False, issues=[str(e)],
        )

    rows = len(df)
    tf_min = TIMEFRAME_MINUTES.get(timeframe)
    gaps = df.index.to_series().diff().dropna()
    max_gap_minutes = float(gaps.max().total_seconds() / 60) if len(gaps) else 0.0
    gap_count = 0
    missing_fraction = 0.0
    if tf_min:
        expected = pd.Timedelta(minutes=tf_min)
        gap_count = int((gaps > expected).sum())
        if timeframe == "1d":
            # market calendars: weekends/holidays are not missing data
            import numpy as np
            span_bars = max(1, int(np.busday_count(df.index[0].date(), df.index[-1].date())) + 1)
        else:
            span_bars = (df.index[-1] - df.index[0]) / expected + 1
        missing_fraction = max(0.0, 1.0 - rows / float(span_bars))
        if max_gap_minutes > tf_min * config["data"]["max_gap_multiple"]:
            issues.append(f"max gap {max_gap_minutes:.0f}min exceeds "
                          f"{config['data']['max_gap_multiple']}x timeframe")
        missing_limit = (config["data"].get("max_missing_fraction_1d",
                                            config["data"]["max_missing_fraction"])
                         if timeframe == "1d" else config["data"]["max_missing_fraction"])
        if missing_fraction > missing_limit:
            issues.append(f"missing fraction {missing_fraction:.4f} exceeds limit")
    else:
        issues.append(f"unknown timeframe '{timeframe}' - gap analysis skipped")
    min_bars = (config["data"].get("min_bars_1d", config["data"]["min_bars"])
                if timeframe == "1d" else config["data"]["min_bars"])
    if rows < min_bars:
        issues.append(f"only {rows} bars, minimum is {min_bars}")
    if (df["High"] < df["Low"]).any():
        issues.append("rows with High < Low")
        return DatasetRecord(path.stem, str(path), symbol, timeframe, rows,
                             str(df.index[0]), str(df.index[-1]), missing_fraction,
                             max_gap_minutes, gap_count, "csv", False, issues)

    usable = not any("exceeds" in i or "minimum" in i or "High < Low" in i for i in issues)
    return DatasetRecord(
        dataset_id=path.stem, path=str(path), symbol=symbol, timeframe=timeframe,
        rows=rows, start=str(df.index[0]), end=str(df.index[-1]),
        missing_fraction=round(missing_fraction, 5), max_gap_minutes=max_gap_minutes,
        gap_count=gap_count, source="csv", usable=usable, issues=issues,
    )


class DataCatalog:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_factory_config()
        self.root = factory_root(self.config)
        self.catalog_path = self.root / "data_catalog.json"
        self.records: Dict[str, DatasetRecord] = {}
        if self.catalog_path.exists():
            for item in json.loads(self.catalog_path.read_text()):
                self.records[item["dataset_id"]] = DatasetRecord(**item)

    def scan(self) -> List[DatasetRecord]:
        """Scan configured data dirs, validate every CSV, persist the catalog."""
        self.records = {}
        for d in self.config["data"]["dirs"]:
            directory = resolve_path(d)
            if not directory.exists():
                continue
            for csv in sorted(directory.glob("*.csv")):
                if csv.name in ("strategy_ideas.csv", "backtest_stats.csv"):
                    continue
                record = _analyze(csv, self.config)
                self.records[record.dataset_id] = record
        self._save()
        return list(self.records.values())

    def _save(self) -> None:
        self.catalog_path.write_text(
            json.dumps([asdict(r) for r in self.records.values()], indent=2)
        )

    def require(self, dataset_id: str, min_bars: Optional[int] = None) -> pd.DataFrame:
        """
        Return the canonical OHLCV frame for a dataset, refusing if it is absent
        or fails quality checks. This is the only sanctioned way for the backtest
        runner to obtain data.
        """
        record = self.records.get(dataset_id)
        if record is None:
            raise DataCatalogError(
                f"dataset '{dataset_id}' not in catalog. Known: {sorted(self.records)}. "
                f"Run --mode catalog after adding data."
            )
        if not record.usable:
            raise DataCatalogError(f"dataset '{dataset_id}' is flagged unusable: {record.issues}")
        if min_bars is not None:
            floor = min_bars
        elif record.timeframe == "1d":
            floor = self.config["data"].get("min_bars_1d", self.config["data"]["min_bars"])
        else:
            floor = self.config["data"]["min_bars"]
        if record.rows < floor:
            raise DataCatalogError(
                f"dataset '{dataset_id}' has {record.rows} bars, spec requires {floor}"
            )
        return load_ohlcv(Path(record.path))

    def write_report(self) -> Path:
        reports_dir = resolve_path(self.config["paths"]["reports_dir"])
        reports_dir.mkdir(parents=True, exist_ok=True)
        path = reports_dir / "data_validation_report.md"
        lines = [
            "# Data Validation Report",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "| dataset | symbol | tf | rows | start | end | missing | max gap (min) | usable | issues |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in sorted(self.records.values(), key=lambda x: x.dataset_id):
            lines.append(
                f"| {r.dataset_id} | {r.symbol} | {r.timeframe} | {r.rows} | {r.start} | {r.end} "
                f"| {r.missing_fraction:.4f} | {r.max_gap_minutes:.0f} | "
                f"{'YES' if r.usable else 'NO'} | {'; '.join(r.issues) or '-'} |"
            )
        if not self.records:
            lines.append("\n**No datasets found.** Add OHLCV CSVs to the configured data dirs.")
        path.write_text("\n".join(lines) + "\n")
        return path
