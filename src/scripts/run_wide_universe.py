"""
Wide-universe trend-family test (pre-registered 2026-06-12).

Registration:
- Universe: current S&P-100-type roster, daily, names with >= 2500 bars in the
  catalog. SURVIVORSHIP-BIASED (current roster) - a pass here is necessary, not
  sufficient; promotion would additionally require a survivorship-clean check.
- Modules: the three retired stock-family modules, byte-unchanged. Breadth is
  the only variable under test.
- Method: ONE continuous full-history backtest per instrument; train (first 60%
  of the calendar span), validation (next 20%) and walk-forward folds are
  obtained by slicing the resulting equity curves and trade lists by date. Test
  (final 20%) untouched. Portfolio = equal-weight normalized curves.
- Robustness: portfolio-level walk-forward (4 anchored calendar folds inside
  train+validation), parameter neighborhood on the first instrument, cost
  stress at 1.5x re-run across the whole universe, Monte Carlo on pooled
  validation trades (seeded).
- Judgment: standard hard gates on portfolio validation metrics. One shot per
  module; no iteration.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.research.backtest_runner import run_backtest
from src.research.data_catalog import DataCatalog
from src.research.experiment_store import ExperimentStore
from src.research.factory_config import load_factory_config
from src.research.strategy_spec import StrategySpec
from src.research.validation_gates import evaluate_gates, hard_gates_passed
from src.research.xsectional import metrics_from_returns
from src.scripts.pull_sp100 import UNIVERSE

MODULES = {
    "WideTrendPyramid": ("src/data/research_factory/strategies/"
                         "spec_d0a4119045_PyramidTrendConvexity.py", False),
    "WideTrendPullback": ("src/data/research_factory/strategies/"
                          "spec_539152b420_TrendPullbackStochastic.py", True),
    "WideSqueezeTrend": ("src/data/research_factory/strategies/"
                         "spec_274f084b1d_SqueezeTrendGate.py", True),
}


def portfolio_segment(stats_list, a, b):
    """Equal-weight portfolio metrics for calendar window [a, b)."""
    curves, trades = [], []
    for s in stats_list:
        eq = s["_equity_curve"]["Equity"]
        seg = eq[(eq.index >= a) & (eq.index < b)]
        if len(seg) > 20:
            curves.append(seg / seg.iloc[0])
        t = s["_trades"]
        if len(t):
            et = pd.to_datetime(t["EntryTime"])
            trades.append(t[(et >= a) & (et < b)])
    if not curves:
        return {"error": "no curves in window"}
    port = pd.concat(curves, axis=1).ffill().dropna().mean(axis=1)
    m = metrics_from_returns(port.pct_change().dropna())
    tr = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    m["n_trades"] = int(len(tr))
    m["n_instruments"] = len(curves)
    if len(tr):
        pnl_pct = tr["ReturnPct"]
        wins, losses = pnl_pct[pnl_pct > 0], pnl_pct[pnl_pct <= 0]
        m["profit_factor"] = (float(wins.sum() / abs(losses.sum()))
                              if len(losses) and losses.sum() != 0 else None)
        m["win_rate_pct"] = float(len(wins) / len(tr) * 100)
        m["expectancy_pct"] = float(pnl_pct.mean() * 100)
        total = pnl_pct.sum()
        m["return_without_best_trade_pct"] = float((total - pnl_pct.max())
                                                   / max(m["n_instruments"], 1) * 100)
    return m


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None, help="run a single custom module")
    parser.add_argument("--module", default=None)
    parser.add_argument("--suffix", default="-1d",
                        help="dataset suffix, e.g. -1d or -XG-1d")
    args = parser.parse_args()

    config = load_factory_config()
    store = ExperimentStore()
    catalog = DataCatalog(config)
    seed = config["monte_carlo"]["seed"]

    datasets = []
    for sym in UNIVERSE:
        rec = catalog.records.get(f"{sym}{args.suffix}")
        if rec is not None and rec.usable:
            datasets.append(f"{sym}{args.suffix}")
    print(f"universe: {len(datasets)} usable instruments of {len(UNIVERSE)} roster names")

    modules = ({args.name: (args.module, True)} if args.name and args.module
               else MODULES)
    for name, (module, excl) in modules.items():
        module = Path(module)
        import copy
        cfg = copy.deepcopy(config)
        cfg["costs"]["exclusive_orders"] = excl

        stats_list, stress_list, failed = [], [], []
        for ds in datasets:
            df = catalog.require(ds)
            try:
                s, _ = run_backtest(module, df, config=cfg)
                stats_list.append(s)
                s2, _ = run_backtest(module, df, config=cfg, cost_multiplier=1.5)
                stress_list.append(s2)
            except Exception as e:
                failed.append(f"{ds}: {type(e).__name__}")
        print(f"\n{name}: {len(stats_list)} instruments backtested, {len(failed)} failed")

        starts = [s["_equity_curve"].index[0] for s in stats_list]
        ends = [s["_equity_curve"].index[-1] for s in stats_list]
        g0, g1 = min(starts), max(ends)
        span = g1 - g0
        t1, t2 = g0 + 0.6 * span, g0 + 0.8 * span

        train_m = portfolio_segment(stats_list, g0, t1)
        val_m = portfolio_segment(stats_list, t1, t2)
        stress_val = portfolio_segment(stress_list, t1, t2)

        # portfolio walk-forward: 4 anchored calendar folds inside [g0, t2)
        folds = []
        wspan = t2 - g0
        for k in range(4):
            fa = g0 + (0.4 + k * 0.15) * wspan
            fb = g0 + (0.4 + (k + 1) * 0.15) * wspan
            is_m = portfolio_segment(stats_list, g0, fa)
            oos_m = portfolio_segment(stats_list, fa, fb)
            folds.append({"fold": k, "in_sample_return_pct": is_m.get("return_pct"),
                          "oos_return_pct": oos_m.get("return_pct")})
        is_mean = float(np.mean([f["in_sample_return_pct"] for f in folds]))
        oos_mean = float(np.mean([f["oos_return_pct"] for f in folds]))

        # parameter neighborhood on the first instrument's train window
        from src.research.backtest_runner import load_strategy_module
        mod = load_strategy_module(module)
        df0 = catalog.require(datasets[0])
        df0_train = df0[df0.index < t1]
        base = run_backtest(module, df0_train, config=cfg)[1].get("return_pct")
        neigh = []
        for pname, default in mod.PARAMS.items():
            if not isinstance(default, (int, float)) or isinstance(default, bool):
                continue
            for mult in (0.8, 0.9, 1.1, 1.2):
                v = default * mult
                if isinstance(default, int):
                    v = max(1, int(round(v)))
                    if v == default:
                        continue
                try:
                    neigh.append(run_backtest(module, df0_train, params={pname: v},
                                              config=cfg)[1].get("return_pct"))
                except Exception:
                    neigh.append(None)
        clean = [x for x in neigh if x is not None]
        pn_ret = (float(np.mean(clean)) / base
                  if clean and base and base > 0 else None)

        # Monte Carlo on pooled validation trades
        val_trades = []
        for s in stats_list:
            t = s["_trades"]
            if len(t):
                et = pd.to_datetime(t["EntryTime"])
                val_trades += list(t[(et >= t1) & (et < t2)]["ReturnPct"])
        mc = {}
        if len(val_trades) >= 5:
            rng = np.random.default_rng(seed)
            arr = np.array(val_trades)
            dds = []
            for _ in range(1000):
                eq = np.cumprod(1 + rng.permutation(arr) / max(len(stats_list), 1))
                pk = np.maximum.accumulate(eq)
                dds.append(((eq - pk) / pk).min() * 100)
            mc = {"p05_max_drawdown_pct": float(np.percentile(dds, 5)),
                  "n_trades": len(arr), "seed": seed}

        robustness = {
            "folds": folds,
            "walk_forward_is_mean_return_pct": is_mean,
            "walk_forward_oos_mean_return_pct": oos_mean,
            "walk_forward_retention": (oos_mean / is_mean) if is_mean > 0 else None,
            "param_neighborhood_retention": pn_ret,
            "cost_stress": {"return_pct_at_1.5x": stress_val.get("return_pct")},
            "monte_carlo": mc,
            "failed_instruments": failed,
        }

        spec = StrategySpec(
            name=name, family="trend_following", asset_class="equity",
            instruments=[d.split("-1d")[0] for d in datasets], timeframe="1d",
            hypothesis=("Wide-universe deployment of the retired stocks trend family: "
                        "the 3-name basket's Sortino/Calmar/walk-forward failures were "
                        "diagnosed as intra-regime concentration; ~100-name breadth "
                        "diversifies the bleed between any one name's trends. Logic "
                        "byte-unchanged; breadth is the only variable. Survivorship-"
                        "biased roster: a pass is necessary, not sufficient."),
            regime_assumptions="needs some subset of the universe trending at all times",
            entry_logic=f"unchanged module {module.name}",
            exit_logic="unchanged", stop_logic="unchanged (ATR-based)",
            position_sizing="unchanged fractional sizing; equal-weight portfolio",
            risk_rules="portfolio equal weight across ~100 names",
            invalidation_rules="standard hard gates at portfolio level, one shot; "
                               "no iteration permitted",
            expected_trade_frequency="swing", low_frequency=False,
            costs_to_model={"exclusive_orders": excl},
            source="src/scripts/run_wide_universe.py (pre-registered 2026-06-12)",
        )
        try:
            sid = store.add_spec(spec)
        except ValueError:
            sid = store.conn.execute("SELECT spec_id FROM specs WHERE fingerprint=?",
                                     (spec.fingerprint(),)).fetchone()["spec_id"]
            spec.spec_id = sid
        store.record_experiment(sid, "train", True, dataset_id="WIDE", metrics=train_m)
        store.record_experiment(sid, "validation", True, dataset_id="PORTFOLIO",
                                metrics=val_m)
        store.record_experiment(sid, "robustness", True, dataset_id="WIDE",
                                metrics=robustness, seed=seed)
        gates = evaluate_gates(spec, val_m, train_m, robustness,
                               static_review_clean=True)
        store.record_gate_results(sid, None, [g.to_dict() for g in gates])
        passed = hard_gates_passed(gates)
        failed_g = [g.name for g in gates if g.hard and not g.passed]
        if not passed:
            store.record_rejection(sid, "gates", f"failed: {failed_g}")
        print(f"  val: ret={val_m.get('return_pct'):.2f}% sharpe={val_m.get('sharpe')} "
              f"sortino={val_m.get('sortino')} calmar={val_m.get('calmar')} "
              f"PF={val_m.get('profit_factor')} trades={val_m.get('n_trades')} "
              f"maxDD={val_m.get('max_drawdown_pct'):.1f}%")
        print(f"  WF retention={robustness['walk_forward_retention']} "
              f"oos_mean={oos_mean:.2f}% | param_ret={pn_ret} | "
              f"1.5x={stress_val.get('return_pct')}")
        print(f"  GATES: {'PASS ALL' if passed else 'failed ' + str(failed_g)}")
    print("\nWIDE_UNIVERSE_DONE")


if __name__ == "__main__":
    main()
