"""
Pull daily OHLCV for a wide US large-cap universe (current S&P-100-type roster)
from Yahoo into src/data/ohlcv/ for the wide-universe trend-family test.

SURVIVORSHIP BIAS: this is the CURRENT roster - delisted/merged names are absent,
so any backtest on it is upper-biased. Documented in the registration; a gate
pass on this universe is necessary but not sufficient.

Usage: python src/scripts/pull_sp100.py [--start 2000-01-01]
"""

import argparse
import sys
import time
from pathlib import Path

import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "src" / "data" / "ohlcv"

UNIVERSE = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","BRK-B","LLY","AVGO","JPM",
    "TSLA","UNH","V","XOM","MA","PG","JNJ","HD","COST","ORCL",
    "ABBV","MRK","CVX","CRM","KO","BAC","AMD","PEP","WMT","NFLX",
    "TMO","ADBE","LIN","CSCO","ACN","MCD","ABT","WFC","IBM","GE",
    "QCOM","DHR","CAT","INTU","AXP","VZ","PFE","T","AMGN","ISRG",
    "MS","PM","NOW","GS","RTX","NEE","SPGI","LOW","HON","UNP",
    "COP","BKNG","ELV","TJX","BLK","SYK","VRTX","BA","LMT","C",
    "SCHW","MDT","ADP","BMY","DE","CB","MMC","PGR","SBUX","SO",
    "GILD","MO","BSX","ETN","ICE","DUK","CL","EMR","NKE","WM",
    "CME","KLAC","TXN","MU","GM","F","DIS","CMCSA","TGT","USB",
]


def pull(symbol: str, start: str):
    df = yf.download(symbol, start=start, interval="1d", auto_adjust=True,
                     progress=False, multi_level_index=False)
    if df is None or len(df) == 0:
        raise RuntimeError("no data")
    df = df.reset_index()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={"date": "datetime"})
    df = df[["datetime", "open", "high", "low", "close", "volume"]].dropna(
        subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0.0)
    valid = ((df["high"] >= df["low"])
             & (df["high"] >= df[["open", "close"]].max(axis=1))
             & (df["low"] <= df[["open", "close"]].min(axis=1))
             & (df["close"] > 0))
    return df[valid].sort_values("datetime")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2000-01-01")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok, failed = 0, []
    for sym in UNIVERSE:
        out = OUT_DIR / f"{sym}-1d.csv"
        if out.exists():
            ok += 1
            continue
        try:
            df = pull(sym, args.start)
            df.to_csv(out, index=False)
            ok += 1
            print(f"  {sym}: {len(df)} bars [{df['datetime'].iloc[0].date()} -> "
                  f"{df['datetime'].iloc[-1].date()}]", flush=True)
        except Exception as e:
            failed.append(sym)
            print(f"  FAILED {sym}: {e}", flush=True)
        time.sleep(0.4)
    print(f"\nUNIVERSE PULL DONE: {ok} ok, {len(failed)} failed: {failed}")


if __name__ == "__main__":
    sys.exit(main())
