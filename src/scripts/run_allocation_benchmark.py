"""Allocation benchmark (list idea 'buy & hold with different risk allocations'),
2026-06-13. NOT an alpha test - measures whether risk-managed ways of HOLDING a
diversified asset set beat naive buy-and-hold on drawdown-adjusted metrics.
Reported, not gated as alpha (these are beta-harvesting, which the gates reject
by design). Common holdout window across the multi-asset set.

Variants: naive equal-weight, inverse-vol (risk parity), vol-targeted, and
200DMA-overlay (hold each sleeve only above its own 200-day MA).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.research.data_catalog import DataCatalog
from src.research.factory_config import load_factory_config
from src.research.xsectional import metrics_from_returns

ASSETS = ["SPX-1d", "NDX-1d", "DJI-1d", "GLD-1d", "SLV-1d", "WTI-1d",
          "EURUSD-1d", "BTC-USD-1h"]  # BTC resampled to daily below


def main() -> None:
    config = load_factory_config()
    catalog = DataCatalog(config)
    closes = {}
    for ds in ASSETS:
        df = catalog.require(ds)
        c = df["Close"]
        if "1h" in ds or "4h" in ds:
            c = c.resample("1D").last().dropna()
        closes[ds.split("-")[0]] = c
    wide = pd.concat(closes, axis=1).dropna()
    print(f"assets: {list(wide.columns)}  common window {wide.index[0].date()} -> "
          f"{wide.index[-1].date()}  ({len(wide)} days)")
    rets = wide.pct_change().dropna()

    def report(name, port_rets):
        m = metrics_from_returns(port_rets)
        print(f"\n{name}:")
        for k in ("return_pct", "annualized_return_pct", "sharpe", "sortino",
                  "calmar", "max_drawdown_pct"):
            v = m.get(k)
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # 1. naive equal weight
    report("Naive equal-weight B&H", rets.mean(axis=1))

    # 2. inverse-vol (risk parity), 63-day vol, monthly weights
    vol = rets.rolling(63).std()
    inv = (1.0 / vol).replace([np.inf, -np.inf], np.nan)
    w = inv.div(inv.sum(axis=1), axis=0).shift(1)
    report("Inverse-vol risk parity", (w * rets).sum(axis=1).dropna())

    # 3. vol-targeted equal weight (scale whole portfolio to 10% annual)
    base = rets.mean(axis=1)
    pvol = base.rolling(63).std() * np.sqrt(252)
    scale = (0.10 / pvol).clip(0, 3).shift(1)
    report("Vol-targeted (10% ann.)", (scale * base).dropna())

    # 4. 200DMA overlay: each sleeve in only when above its own 200DMA
    ma = wide.rolling(200).mean()
    inmkt = (wide > ma).shift(1).astype(float)
    active = inmkt.div(inmkt.sum(axis=1).replace(0, np.nan), axis=0)
    report("200DMA-overlay equal-weight", (active * rets).sum(axis=1).dropna())

    print("\nNote: these harvest the asset risk premium with managed drawdown - "
          "a way to HOLD, not alpha. The alpha gates reject beta by design.")
    print("ALLOCATION_BENCHMARK_DONE")


if __name__ == "__main__":
    main()
