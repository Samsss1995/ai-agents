"""
Backfill historical hourly funding rates from Hyperliquid into src/data/signals/.

fundingHistory returns up to ~500 hourly entries per request; paginates from
--start to now. Output: src/data/signals/FUNDING-<COIN>.csv with columns
datetime, funding_rate, premium.

Usage:
  python src/scripts/pull_hl_funding.py
  python src/scripts/pull_hl_funding.py --coins BTC --start 2023-01-01
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "src" / "data" / "signals"

API_URL = "https://api.hyperliquid.xyz/info"
HOUR_MS = 3_600_000


def pull(coin: str, start_ms: int) -> pd.DataFrame:
    now_ms = int(time.time() * 1000)
    rows = {}
    cursor = start_ms
    while cursor < now_ms:
        resp = requests.post(API_URL, json={
            "type": "fundingHistory", "coin": coin,
            "startTime": cursor, "endTime": now_ms,
        }, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        for e in batch:
            rows[int(e["time"])] = (float(e["fundingRate"]), float(e.get("premium") or 0))
        cursor = int(batch[-1]["time"]) + HOUR_MS
        time.sleep(0.25)
    if not rows:
        raise RuntimeError(f"no funding history returned for {coin}")
    df = pd.DataFrame(
        [(ts, *vals) for ts, vals in sorted(rows.items())],
        columns=["ts", "funding_rate", "premium"],
    )
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
    return df[["datetime", "funding_rate", "premium"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Hyperliquid funding history")
    parser.add_argument("--coins", nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--start", default="2023-01-01")
    args = parser.parse_args()

    start_ms = int(datetime.strptime(args.start, "%Y-%m-%d")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for coin in args.coins:
        print(f"pulling funding history {coin} since {args.start} ...", flush=True)
        df = pull(coin, start_ms)
        out = OUT_DIR / f"FUNDING-{coin}.csv"
        df.to_csv(out, index=False)
        print(f"  {len(df)} hourly rows  {df['datetime'].iloc[0]} -> "
              f"{df['datetime'].iloc[-1]}  -> {out.name}")


if __name__ == "__main__":
    sys.exit(main())
