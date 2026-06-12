"""
Pull daily OHLCV for stocks / indices / commodities / forex from Stooq's free
CSV endpoint into src/data/ohlcv/.

Stooq serves full daily history in one request per symbol. Forex and spot
metals have no volume; Volume is written as 0 and a warning is printed -
volume-dependent strategy logic is meaningless on those datasets.

Usage:
  python src/scripts/pull_stooq_ohlcv.py
  python src/scripts/pull_stooq_ohlcv.py --symbols eurusd:EURUSD ^spx:SPX --start 2000-01-01
"""

import argparse
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "src" / "data" / "ohlcv"

URL = "https://stooq.com/q/d/l/"

# stooq_symbol -> dataset name (catalog symbol)
DEFAULT_SYMBOLS = {
    # stocks
    "aapl.us": "AAPL", "msft.us": "MSFT", "jpm.us": "JPM",
    # indices
    "^spx": "SPX", "^ndq": "NDQ", "^dji": "DJI",
    # commodities
    "xauusd": "XAUUSD", "xagusd": "XAGUSD", "cl.f": "WTI",
    # forex
    "eurusd": "EURUSD", "usdjpy": "USDJPY", "gbpusd": "GBPUSD",
}


def pull(stooq_symbol: str, start: str) -> pd.DataFrame:
    resp = requests.get(URL, params={
        "s": stooq_symbol, "i": "d",
        "d1": start.replace("-", ""), "d2": pd.Timestamp.now().strftime("%Y%m%d"),
    }, timeout=30)
    resp.raise_for_status()
    if "No data" in resp.text[:100] or len(resp.text) < 50:
        raise RuntimeError(f"stooq returned no data for '{stooq_symbol}'")
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = [c.strip().lower() for c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 0.0
        print(f"  WARNING: {stooq_symbol} has no volume; Volume=0 "
              f"(volume-based logic meaningless)", flush=True)
    df = df.rename(columns={"date": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[["datetime", "open", "high", "low", "close", "volume"]].dropna()
    return df.sort_values("datetime")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull Stooq daily OHLCV")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="stooq_symbol:NAME pairs; default = built-in basket")
    parser.add_argument("--start", default="2000-01-01")
    args = parser.parse_args()

    symbols = (dict(s.split(":") for s in args.symbols)
               if args.symbols else DEFAULT_SYMBOLS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    for stooq_symbol, name in symbols.items():
        try:
            df = pull(stooq_symbol, args.start)
            out = OUT_DIR / f"{name}-1d.csv"
            df.to_csv(out, index=False)
            print(f"  {name}-1d: {len(df)} bars "
                  f"[{df['datetime'].iloc[0].date()} -> {df['datetime'].iloc[-1].date()}]",
                  flush=True)
        except Exception as e:
            failures.append(f"{stooq_symbol}: {type(e).__name__}: {e}")
            print(f"  FAILED {stooq_symbol}: {e}", flush=True)
        time.sleep(0.5)
    if failures:
        print(f"\n{len(failures)} symbols failed - fix or substitute before relying on coverage")


if __name__ == "__main__":
    sys.exit(main())
