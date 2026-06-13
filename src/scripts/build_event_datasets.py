"""
Build derived datasets for the evidence-driven ideas batch (2026-06-12):

1. Ratio bars: ETHBTC-1h, SOLBTC-1h - market-neutral spread series.
   OHLC approximation documented: O=o1/o2, C=c1/c2, H=max(O,C,h1/h2),
   L=min(O,C,l1/l2). Volume = numerator volume (information proxy only).
2. Lead-lag enrichment: ETH-XL-1h, SOL-XL-1h - own OHLCV + BtcClose column
   (backward as-of join, no lookahead).
3. Idiosyncratic-gap enrichment: <SYM>-XG-1d for the wide universe - own OHLCV
   + SpxGapPct column (SPX same-day open gap, joined by calendar date; the SPX
   open is known at the stock's open, so same-bar join is information-legal).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.research.data_catalog import DataCatalog, load_ohlcv
from src.research.factory_config import load_factory_config
from src.scripts.pull_sp100 import UNIVERSE

OUT = Path(__file__).resolve().parent.parent / "data" / "ohlcv"


def build_ratio(num_id: str, den_id: str, out_name: str, catalog) -> None:
    a, b = catalog.require(num_id), catalog.require(den_id)
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    o, c = a["Open"] / b["Open"], a["Close"] / b["Close"]
    h = pd.concat([o, c, a["High"] / b["High"]], axis=1).max(axis=1)
    l = pd.concat([o, c, a["Low"] / b["Low"]], axis=1).min(axis=1)
    df = pd.DataFrame({"datetime": idx, "open": o.values, "high": h.values,
                       "low": l.values, "close": c.values,
                       "volume": a["Volume"].values})
    df.to_csv(OUT / f"{out_name}.csv", index=False)
    print(f"  {out_name}: {len(df)} bars [{idx[0]} -> {idx[-1]}]")


def build_leadlag(sym_id: str, btc_id: str, out_name: str, catalog) -> None:
    own, btc = catalog.require(sym_id), catalog.require(btc_id)
    df = own.reset_index().rename(columns={own.reset_index().columns[0]: "datetime"})
    btc_close = btc["Close"].rename("BtcClose").reset_index()
    btc_close.columns = ["datetime", "BtcClose"]
    merged = pd.merge_asof(df.sort_values("datetime"), btc_close.sort_values("datetime"),
                           on="datetime", direction="backward",
                           tolerance=pd.Timedelta(hours=2)).dropna(subset=["BtcClose"])
    merged.columns = [c if c == "BtcClose" else c.lower() for c in merged.columns]
    merged.to_csv(OUT / f"{out_name}.csv", index=False)
    print(f"  {out_name}: {len(merged)} bars")


def build_gap_enriched(catalog) -> None:
    spx = catalog.require("SPX-1d")
    spx_gap = (spx["Open"] / spx["Close"].shift(1) - 1).rename("SpxGapPct")
    spx_gap.index = spx_gap.index.date
    built = 0
    for sym in UNIVERSE:
        rec = catalog.records.get(f"{sym}-1d")
        if rec is None or not rec.usable:
            continue
        own = catalog.require(f"{sym}-1d").reset_index()
        own = own.rename(columns={own.columns[0]: "datetime"})
        own["SpxGapPct"] = [spx_gap.get(d.date(), float("nan"))
                            for d in own["datetime"]]
        own = own.dropna(subset=["SpxGapPct"])
        own.columns = [c if c in ("datetime", "SpxGapPct") else c.lower()
                       for c in own.columns]
        own.to_csv(OUT / f"{sym}-XG-1d.csv", index=False)
        built += 1
    print(f"  gap-enriched: {built} <SYM>-XG-1d datasets")


def main() -> None:
    catalog = DataCatalog(load_factory_config())
    if not catalog.records:
        catalog.scan()
    build_ratio("ETH-USD-1h", "BTC-USD-1h", "ETHBTC-1h", catalog)
    build_ratio("SOL-USD-1h", "BTC-USD-1h", "SOLBTC-1h", catalog)
    build_leadlag("ETH-USD-1h", "BTC-USD-1h", "ETH-XL-1h", catalog)
    build_leadlag("SOL-USD-1h", "BTC-USD-1h", "SOL-XL-1h", catalog)
    build_gap_enriched(catalog)
    print("rescan catalog: --mode catalog")


if __name__ == "__main__":
    main()
