"""
L2 Collector Agent - forward-collects Hyperliquid market microstructure that
cannot be backfilled later:

  - L2 order book (top N levels per side) with derived imbalance metrics
  - per-coin open interest, funding rate, premium, mark/oracle price

Every poll appends one row per coin to daily CSVs under src/data/l2_snapshots/.
Real data only: failed polls are logged and skipped, never interpolated.

Run continuously (survives transient API failures, reconnects forever):
  python src/agents/l2_collector_agent.py
  nohup .venv/bin/python src/agents/l2_collector_agent.py >> src/data/l2_snapshots/collector.log 2>&1 &

Test single poll:
  python src/agents/l2_collector_agent.py --once
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "src" / "data" / "l2_snapshots"

API_URL = "https://api.hyperliquid.xyz/info"
DEFAULT_COINS = ["BTC", "ETH", "SOL"]
DEFAULT_INTERVAL_S = 5
DEFAULT_LEVELS = 5

BOOK_FIELDS = ["ts_utc", "mid", "spread", "spread_bps",
               "imb_l1", "imb_l5", "bid_depth_usd", "ask_depth_usd"]
CTX_FIELDS = ["open_interest", "funding_rate", "premium", "mark_px", "oracle_px",
              "day_ntl_volume"]


def _post(payload: dict) -> dict:
    resp = requests.post(API_URL, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_l2(coin: str, levels: int) -> dict:
    book = _post({"type": "l2Book", "coin": coin})
    bids, asks = book["levels"][0][:levels], book["levels"][1][:levels]
    if not bids or not asks:
        raise RuntimeError(f"empty book for {coin}")
    best_bid, best_ask = float(bids[0]["px"]), float(asks[0]["px"])
    mid = (best_bid + best_ask) / 2
    spread = best_ask - best_bid
    bid1, ask1 = float(bids[0]["sz"]), float(asks[0]["sz"])
    bid5 = sum(float(b["sz"]) for b in bids)
    ask5 = sum(float(a["sz"]) for a in asks)
    row = {
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "mid": mid,
        "spread": spread,
        "spread_bps": spread / mid * 10_000,
        "imb_l1": (bid1 - ask1) / (bid1 + ask1) if (bid1 + ask1) else 0.0,
        "imb_l5": (bid5 - ask5) / (bid5 + ask5) if (bid5 + ask5) else 0.0,
        "bid_depth_usd": bid5 * mid,
        "ask_depth_usd": ask5 * mid,
    }
    for i, (b, a) in enumerate(zip(bids, asks), start=1):
        row[f"bid_px_{i}"], row[f"bid_sz_{i}"] = float(b["px"]), float(b["sz"])
        row[f"ask_px_{i}"], row[f"ask_sz_{i}"] = float(a["px"]), float(a["sz"])
    return row


def fetch_asset_ctxs(coins: list) -> dict:
    """openInterest / funding / premium per coin from metaAndAssetCtxs."""
    meta, ctxs = _post({"type": "metaAndAssetCtxs"})
    names = [u["name"] for u in meta["universe"]]
    out = {}
    for coin in coins:
        if coin not in names:
            continue
        c = ctxs[names.index(coin)]
        out[coin] = {
            "open_interest": float(c.get("openInterest") or 0),
            "funding_rate": float(c.get("funding") or 0),
            "premium": float(c.get("premium") or 0),
            "mark_px": float(c.get("markPx") or 0),
            "oracle_px": float(c.get("oraclePx") or 0),
            "day_ntl_volume": float(c.get("dayNtlVlm") or 0),
        }
    return out


def _csv_path(coin: str, levels: int) -> Path:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    d = OUT_DIR / coin
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{day}.csv"


def append_row(coin: str, row: dict, levels: int) -> None:
    path = _csv_path(coin, levels)
    level_fields = [f"{side}_{kind}_{i}" for i in range(1, levels + 1)
                    for side in ("bid", "ask") for kind in ("px", "sz")]
    fields = BOOK_FIELDS + CTX_FIELDS + level_fields
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def write_heartbeat(status: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "collector_status.json").write_text(json.dumps(status, indent=2))


def poll_once(coins: list, levels: int) -> dict:
    ctxs = fetch_asset_ctxs(coins)
    results = {}
    for coin in coins:
        row = fetch_l2(coin, levels)
        row.update(ctxs.get(coin, {}))
        append_row(coin, row, levels)
        results[coin] = {"mid": row["mid"], "imb_l5": round(row["imb_l5"], 4),
                         "oi": row.get("open_interest")}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperliquid L2 + OI/funding collector")
    parser.add_argument("--coins", nargs="+", default=DEFAULT_COINS)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S)
    parser.add_argument("--levels", type=int, default=DEFAULT_LEVELS)
    parser.add_argument("--once", action="store_true", help="single poll then exit")
    args = parser.parse_args()

    consecutive_failures = 0
    polls = 0
    while True:
        started = time.time()
        try:
            results = poll_once(args.coins, args.levels)
            polls += 1
            consecutive_failures = 0
            write_heartbeat({
                "last_poll_utc": datetime.now(timezone.utc).isoformat(),
                "polls_this_run": polls,
                "coins": results,
                "interval_s": args.interval,
            })
            if args.once or polls % 720 == 0:  # log roughly hourly at 5s cadence
                print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] "
                      f"poll {polls}: " + ", ".join(
                          f"{c} mid={r['mid']} imb={r['imb_l5']}" for c, r in results.items()),
                      flush=True)
        except Exception as e:
            consecutive_failures += 1
            print(f"poll failed ({consecutive_failures} consecutive): "
                  f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)
            if consecutive_failures >= 100:
                write_heartbeat({"status": "dead",
                                 "died_utc": datetime.now(timezone.utc).isoformat(),
                                 "error": str(e)})
                raise
        if args.once:
            break
        time.sleep(max(0.0, args.interval - (time.time() - started)))


if __name__ == "__main__":
    main()
