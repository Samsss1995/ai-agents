"""
Signal features - builds enriched OHLCV datasets by joining signal series
(currently: Hyperliquid hourly funding) onto catalog OHLCV bars.

Output: src/data/ohlcv/<COIN>-SIG-<tf>.csv with the standard OHLCV columns plus
signal columns (FundingRate, Premium). The join is merge_asof BACKWARD: each bar
carries the latest signal value stamped at or before the bar's open - no
lookahead by construction. Bars before the first signal observation are dropped,
never filled.

Usage:
  python -m src.research.signal_features            # BTC/ETH/SOL on 1h
  python -m src.research.signal_features --timeframe 4h
"""

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from src.research.data_catalog import DataCatalog, load_ohlcv
from src.research.factory_config import PROJECT_ROOT, load_factory_config

SIGNALS_DIR = PROJECT_ROOT / "src" / "data" / "signals"
OUT_DIR = PROJECT_ROOT / "src" / "data" / "ohlcv"

SIGNAL_COLUMNS = ["FundingRate", "Premium"]


def load_funding(coin: str) -> pd.DataFrame:
    path = SIGNALS_DIR / f"FUNDING-{coin}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing - run: python src/scripts/pull_hl_funding.py --coins {coin}")
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.rename(columns={"funding_rate": "FundingRate", "premium": "Premium"})
    return df.sort_values("datetime")


def build_enriched(coin: str, timeframe: str, catalog: DataCatalog) -> Path:
    base_id = f"{coin}-USD-{timeframe}"
    ohlcv = catalog.require(base_id).reset_index()
    time_col = ohlcv.columns[0]
    ohlcv = ohlcv.rename(columns={time_col: "datetime"})

    funding = load_funding(coin)
    merged = pd.merge_asof(
        ohlcv.sort_values("datetime"), funding,
        on="datetime", direction="backward",
        tolerance=pd.Timedelta(hours=4),
    )
    before = len(merged)
    merged = merged.dropna(subset=SIGNAL_COLUMNS)
    dropped = before - len(merged)
    if len(merged) == 0:
        raise RuntimeError(f"{base_id}: no overlap between OHLCV and funding history")

    out = OUT_DIR / f"{coin}-SIG-{timeframe}.csv"
    cols = ["datetime", "Open", "High", "Low", "Close", "Volume"] + SIGNAL_COLUMNS
    merged[cols].to_csv(out, index=False)
    print(f"  {out.name}: {len(merged)} bars "
          f"[{merged['datetime'].iloc[0]} -> {merged['datetime'].iloc[-1]}] "
          f"({dropped} pre-signal bars dropped)")
    return out


def signal_columns_of(dataset_path: Path) -> List[str]:
    """Extra (non-OHLCV, non-time) columns of an enriched dataset CSV."""
    header = pd.read_csv(dataset_path, nrows=0).columns
    base = {"datetime", "date", "time", "timestamp",
            "open", "high", "low", "close", "volume"}
    return [c for c in header if c.strip().lower() not in base]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build signal-enriched OHLCV datasets")
    parser.add_argument("--coins", nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--timeframe", default="1h")
    args = parser.parse_args()

    catalog = DataCatalog(load_factory_config())
    if not catalog.records:
        catalog.scan()
    for coin in args.coins:
        build_enriched(coin, args.timeframe, catalog)
    print("rescan the catalog to register the new datasets: --mode catalog")


if __name__ == "__main__":
    main()
