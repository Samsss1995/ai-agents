"""
Pull historical OHLCV from Hyperliquid's public info API into src/data/ohlcv/.

Paginates candleSnapshot requests (max ~5000 candles each) from the earliest
available data to now, dedupes, validates monotonic timestamps, and writes
CSVs named <COIN>-USD-<tf>.csv in the canonical column format the data catalog
expects (datetime, open, high, low, close, volume).

Usage:
  python src/scripts/pull_hl_ohlcv.py                       # defaults below
  python src/scripts/pull_hl_ohlcv.py --coins BTC ETH --timeframes 15m 1h
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

API_URL = "https://api.hyperliquid.xyz/info"
INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000,
               "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
MAX_CANDLES_PER_REQ = 5000
EARLIEST_MS = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


def fetch_window(coin: str, interval: str, start_ms: int, end_ms: int) -> list:
    resp = requests.post(API_URL, json={
        "type": "candleSnapshot",
        "req": {"coin": coin, "interval": interval,
                "startTime": start_ms, "endTime": end_ms},
    }, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(f"unexpected response for {coin} {interval}: {str(data)[:200]}")
    return data


def pull(coin: str, interval: str) -> pd.DataFrame:
    step_ms = INTERVAL_MS[interval] * MAX_CANDLES_PER_REQ
    now_ms = int(time.time() * 1000)
    rows = {}
    start = EARLIEST_MS
    while start < now_ms:
        end = min(start + step_ms, now_ms)
        candles = fetch_window(coin, interval, start, end)
        for c in candles:
            rows[int(c["t"])] = (float(c["o"]), float(c["h"]), float(c["l"]),
                                 float(c["c"]), float(c["v"]))
        if candles:
            # advance past the last candle received to avoid re-fetching
            start = max(end, int(candles[-1]["t"]) + INTERVAL_MS[interval])
        else:
            start = end
        time.sleep(0.25)  # stay polite to the public endpoint
    if not rows:
        raise RuntimeError(f"no candles returned for {coin} {interval}")
    df = pd.DataFrame(
        [(ts, *vals) for ts, vals in sorted(rows.items())],
        columns=["ts", "open", "high", "low", "close", "volume"],
    )
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
    return df[["datetime", "open", "high", "low", "close", "volume"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull Hyperliquid OHLCV to src/data/ohlcv/")
    parser.add_argument("--coins", nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--timeframes", nargs="+", default=["15m", "1h", "4h"],
                        choices=list(INTERVAL_MS))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for coin in args.coins:
        for tf in args.timeframes:
            out = OUT_DIR / f"{coin}-USD-{tf}.csv"
            print(f"pulling {coin} {tf} ...", flush=True)
            df = pull(coin, tf)
            df.to_csv(out, index=False)
            print(f"  {len(df)} bars  {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}"
                  f"  -> {out.name}")


if __name__ == "__main__":
    sys.exit(main())
