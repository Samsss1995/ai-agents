"""Pre-registered VolRegime+VixRegime confirmation (2026-06-13). Criteria fixed
before computation - see chat registration. Three tests per candidate cell:
holdout (untouched final 20%), walk-forward retention, and beta decomposition
vs buy-and-hold. Verdict logic at the end is mechanical."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.research.backtest_runner import run_backtest, split_data
from src.research.data_catalog import DataCatalog
from src.research.factory_config import load_factory_config
from src.research.xsectional import metrics_from_returns

D = "src/data/research_factory/strategies_v2/"
CELLS = {
    "VolRegime_crypto": (D + "volregime.py", ["BTC-USD-4h", "ETH-USD-4h", "SOL-USD-4h"]),
    "VolRegime_stocks": (D + "volregime.py", ["AAPL-1d", "MSFT-1d", "JPM-1d"]),
    "VixRegime_stocks": (D + "vixregime.py", ["AAPL-VX-1d", "MSFT-VX-1d", "JPM-VX-1d"]),
}


def port_curve(stats_list, a, b):
    curves = []
    for s in stats_list:
        eq = s["_equity_curve"]["Equity"]
        seg = eq[(eq.index >= a) & (eq.index < b)]
        if len(seg) > 20:
            curves.append(seg / seg.iloc[0])
    if not curves:
        return None
    return pd.concat(curves, axis=1).ffill().dropna().mean(axis=1)


def bh_curve(dfs, a, b):
    curves = []
    for df in dfs:
        c = df["Close"]
        seg = c[(c.index >= a) & (c.index < b)]
        if len(seg) > 20:
            curves.append(seg / seg.iloc[0])
    return pd.concat(curves, axis=1).ffill().dropna().mean(axis=1)


def main():
    config = load_factory_config()
    catalog = DataCatalog(config)
    fr = config["splits"]
    results = {}

    for name, (module, ds_ids) in CELLS.items():
        stats_list, dfs = [], []
        for ds in ds_ids:
            df = catalog.require(ds)
            dfs.append(df)
            s, _ = run_backtest(Path(module), df, config=config)
            stats_list.append(s)
        starts = [s["_equity_curve"].index[0] for s in stats_list]
        ends = [s["_equity_curve"].index[-1] for s in stats_list]
        g0, g1 = min(starts), max(ends)
        span = g1 - g0
        t1, t2 = g0 + fr["train"] * span, g0 + (fr["train"] + fr["validation"]) * span

        # test (holdout) slice
        test_port = port_curve(stats_list, t2, g1)
        test_m = metrics_from_returns(test_port.pct_change().dropna())
        test_bh = bh_curve(dfs, t2, g1)
        bh_m = metrics_from_returns(test_bh.pct_change().dropna())

        # beta decomposition on the holdout
        sp = test_port.pct_change().dropna()
        bp = test_bh.pct_change().dropna()
        j = pd.concat([sp, bp], axis=1).dropna()
        beta = float(np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1] / np.var(j.iloc[:, 1])) if len(j) > 10 else None
        # regression alpha (annualized): strat = alpha + beta*bh
        alpha_ann = None
        if beta is not None:
            resid = j.iloc[:, 0] - beta * j.iloc[:, 1]
            ppy = len(j) / max((j.index[-1] - j.index[0]).total_seconds() / (365.25 * 86400), 1e-9)
            alpha_ann = float(resid.mean() * ppy * 100)

        # walk-forward on pre-test window
        pre = port_curve(stats_list, g0, t2).pct_change().dropna()
        n = len(pre)
        oos = []
        for k in range(4):
            a = int(n * (0.4 + k * 0.15))
            b = int(n * (0.4 + (k + 1) * 0.15))
            seg = pre.iloc[a:b]
            oos.append((1 + seg).prod() - 1)
        oos_mean = float(np.mean(oos) * 100)

        results[name] = {
            "holdout_return_pct": test_m.get("return_pct"),
            "holdout_sharpe": test_m.get("sharpe"),
            "holdout_maxdd_pct": test_m.get("max_drawdown_pct"),
            "bh_holdout_return_pct": bh_m.get("return_pct"),
            "bh_holdout_maxdd_pct": bh_m.get("max_drawdown_pct"),
            "beta_vs_bh": beta,
            "alpha_ann_pct": alpha_ann,
            "wf_oos_mean_pct": oos_mean,
        }

    print(f"{'cell':20}{'hold_ret%':>10}{'hold_shrp':>10}{'hold_DD%':>9}"
          f"{'bh_ret%':>9}{'bh_DD%':>8}{'beta':>6}{'alpha%':>8}{'wf_oos%':>8}")
    for name, r in results.items():
        def f(k, p=2):
            v = r.get(k)
            return "-" if v is None else f"{v:.{p}f}"
        print(f"{name:20}{f('holdout_return_pct',1):>10}{f('holdout_sharpe'):>10}"
              f"{f('holdout_maxdd_pct',1):>9}{f('bh_holdout_return_pct',1):>9}"
              f"{f('bh_holdout_maxdd_pct',1):>8}{f('beta_vs_bh'):>6}"
              f"{f('alpha_ann_pct',1):>8}{f('wf_oos_mean_pct',1):>8}")

    # mechanical verdict
    cells = list(results.values())
    holdout_ok = sum(1 for c in cells if (c["holdout_return_pct"] or -1) > 0
                     and (c["holdout_sharpe"] or -1) > 0.8)
    wf_ok = sum(1 for c in cells if c["wf_oos_mean_pct"] > 0)
    alpha_ok = sum(1 for c in cells if (c["alpha_ann_pct"] or -1) > 0)
    dd_better = sum(1 for c in cells
                    if abs(c["holdout_maxdd_pct"] or 99) < abs(c["bh_holdout_maxdd_pct"] or 0))
    print(f"\nholdout(ret>0 & sharpe>0.8): {holdout_ok}/3 | wf OOS positive: {wf_ok}/3 "
          f"| positive alpha vs B&H: {alpha_ok}/3 | lower DD than B&H: {dd_better}/3")
    if holdout_ok >= 2 and wf_ok >= 2 and alpha_ok >= 2:
        print("VERDICT: PASS - promote to paper_candidate (genuine alpha)")
    elif dd_better >= 2 and holdout_ok >= 2:
        print("VERDICT: RISK-OVERLAY ONLY - beats B&H on drawdown but not on alpha; "
              "a way to HOLD, not alpha. Not promoted as alpha.")
    else:
        print("VERDICT: FAIL - retire, like the prior six near-misses")
    print("VOLVIX_CONFIRMATION_DONE")


if __name__ == "__main__":
    main()
