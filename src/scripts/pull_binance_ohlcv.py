"""
Pull deep historical OHLCV from Binance's public market-data API into
src/data/ohlcv/. Complements pull_hl_ohlcv.py: Hyperliquid's candleSnapshot
only retains ~5000 recent candles per interval, so deep 15m/1h history for
backtesting comes from here. Execution venue remains Hyperliquid; only the
research data source differs.

Uses data-api.binance.vision (public mirror, no key). 1000 klines per request.

Usage:
  python src/scripts/pull_binance_ohlcv.py
  python src/scripts/pull_binance_ohlcv.py --symbols BTCUSDT --timeframes 15m --start 2022-01-01
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "src" / "data" / "ohlcv"

API_URL = "https://data-api.binance.vision/api/v3/klines"
INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000,
               "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
LIMIT = 1000


def pull(symbol: str, interval: str, start_ms: int) -> pd.DataFrame:
    now_ms = int(time.time() * 1000)
    rows = {}
    cursor = start_ms
    while cursor < now_ms:
        resp = requests.get(API_URL, params={
            "symbol": symbol, "interval": interval,
            "startTime": cursor, "limit": LIMIT,
        }, timeout=30)
        resp.raise_for_status()
        klines = resp.json()
        if not klines:
            break
        for k in klines:
            rows[int(k[0])] = (float(k[1]), float(k[2]), float(k[3]),
                               float(k[4]), float(k[5]))
        cursor = int(klines[-1][0]) + INTERVAL_MS[interval]
        time.sleep(0.15)
    if not rows:
        raise RuntimeError(f"no klines returned for {symbol} {interval}")
    df = pd.DataFrame(
        [(ts, *vals) for ts, vals in sorted(rows.items())],
        columns=["ts", "open", "high", "low", "close", "volume"],
    )
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
    return df[["datetime", "open", "high", "low", "close", "volume"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull Binance OHLCV to src/data/ohlcv/")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    parser.add_argument("--timeframes", nargs="+", default=["15m", "1h"],
                        choices=list(INTERVAL_MS))
    parser.add_argument("--start", default="2023-01-01")
    args = parser.parse_args()

    start_ms = int(datetime.strptime(args.start, "%Y-%m-%d")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for symbol in args.symbols:
        coin = symbol.replace("USDT", "")
        for tf in args.timeframes:
            out = OUT_DIR / f"{coin}-USD-{tf}.csv"
            print(f"pulling {symbol} {tf} since {args.start} ...", flush=True)
            df = pull(symbol, tf, start_ms)
            df.to_csv(out, index=False)
            print(f"  {len(df)} bars  {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}"
                  f"  -> {out.name}")


if __name__ == "__main__":
    sys.exit(main())
