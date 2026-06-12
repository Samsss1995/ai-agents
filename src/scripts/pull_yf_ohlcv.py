"""
Pull daily OHLCV for stocks / indices / commodities / forex from Yahoo Finance
(yfinance) into src/data/ohlcv/. Replaces pull_stooq_ohlcv.py (Stooq now fronts
a JS proof-of-work wall).

Forex and index volume from Yahoo is zero or unreliable - volume-based logic is
meaningless on those datasets and the review must say so.

Usage:
  python src/scripts/pull_yf_ohlcv.py
  python src/scripts/pull_yf_ohlcv.py --symbols "AAPL:AAPL" "^GSPC:SPX" --start 2000-01-01
"""

import argparse
import sys
import time
from pathlib import Path

import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "src" / "data" / "ohlcv"

# yahoo_symbol -> dataset name (catalog symbol)
DEFAULT_SYMBOLS = {
    # stocks
    "AAPL": "AAPL", "MSFT": "MSFT", "JPM": "JPM",
    # indices
    "^GSPC": "SPX", "^NDX": "NDX", "^DJI": "DJI",
    # commodities (front-month continuous futures)
    "GC=F": "GOLD", "CL=F": "WTI", "SI=F": "SILVER",
    # forex
    "EURUSD=X": "EURUSD", "USDJPY=X": "USDJPY", "GBPUSD=X": "GBPUSD",
}


def pull(yahoo_symbol: str, start: str):
    df = yf.download(yahoo_symbol, start=start, interval="1d",
                     auto_adjust=True, progress=False, multi_level_index=False)
    if df is None or len(df) == 0:
        raise RuntimeError(f"yahoo returned no data for '{yahoo_symbol}'")
    df = df.reset_index()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={"date": "datetime"})
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[["datetime", "open", "high", "low", "close", "volume"]].dropna(
        subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0.0)
    valid = ((df["high"] >= df["low"])
             & (df["high"] >= df[["open", "close"]].max(axis=1))
             & (df["low"] <= df[["open", "close"]].min(axis=1))
             & (df["close"] > 0))
    bad = int((~valid).sum())
    if bad:
        print(f"  WARNING: {yahoo_symbol}: dropped {bad} inconsistent OHLC rows", flush=True)
    return df[valid].sort_values("datetime")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull Yahoo Finance daily OHLCV")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="yahoo_symbol:NAME pairs; default = built-in basket")
    parser.add_argument("--start", default="2000-01-01")
    args = parser.parse_args()

    symbols = (dict(s.split(":") for s in args.symbols)
               if args.symbols else DEFAULT_SYMBOLS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    for yahoo_symbol, name in symbols.items():
        try:
            df = pull(yahoo_symbol, args.start)
            out = OUT_DIR / f"{name}-1d.csv"
            df.to_csv(out, index=False)
            zero_vol = (df["volume"] == 0).mean()
            note = "  [no real volume]" if zero_vol > 0.5 else ""
            print(f"  {name}-1d: {len(df)} bars "
                  f"[{df['datetime'].iloc[0].date()} -> {df['datetime'].iloc[-1].date()}]{note}",
                  flush=True)
        except Exception as e:
            failures.append(yahoo_symbol)
            print(f"  FAILED {yahoo_symbol}: {type(e).__name__}: {e}", flush=True)
        time.sleep(1.0)
    if failures:
        print(f"\nFAILED: {failures} - substitute before relying on coverage")


if __name__ == "__main__":
    sys.exit(main())
