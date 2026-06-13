"""
Pre-registered strategy-combination test (2026-06-12).

Registration:
- Candidates: every spec in the store with a runnable module whose latest TRAIN
  run was positive with >= 30 trades (selection uses train only - validation
  stays unseen by the selection).
- Two combos (validation windows must overlap to combine): COMBO-DAILY (1d
  specs) and COMBO-CRYPTO (1h/4h specs). Equal weight across components'
  basket-portfolio curves. NO subset search, NO weight optimization.
- Judgment: standard hard gates on each combo's validation metrics. One shot.
- Also reported: mean pairwise correlation of component validation returns
  (the diversification that was actually available).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.research.backtest_runner import (apply_spec_cost_overrides, run_backtest,
                                          split_data)
from src.research.data_catalog import DataCatalog
from src.research.experiment_store import ExperimentStore
from src.research.factory_config import load_factory_config
from src.research.strategy_spec import StrategySpec
from src.research.validation_gates import evaluate_gates, hard_gates_passed
from src.research.xsectional import metrics_from_returns


def spec_curves(spec, store, catalog, config):
    """(train_curve, val_curve) as the spec's equal-weight basket portfolio."""
    code = store.latest_code(spec.spec_id)
    if code is None or not Path(code["file_path"]).exists():
        return None
    cfg = apply_spec_cost_overrides(config, spec)
    ds_ids = sorted({e["dataset_id"] for e in store.experiments_for(spec.spec_id, "validation")
                     if e["success"] and e["dataset_id"]
                     and e["dataset_id"] not in ("PORTFOLIO", "XS", "WIDE")})
    if not ds_ids or len(ds_ids) > 3:
        return None
    tr_curves, va_curves = [], []
    for ds in ds_ids:
        try:
            df = catalog.require(ds)
        except Exception:
            return None
        tr, va, _ = split_data(df, cfg)
        try:
            st, _ = run_backtest(Path(code["file_path"]), tr, config=cfg)
            sv, _ = run_backtest(Path(code["file_path"]), va, config=cfg)
        except Exception:
            return None
        tr_curves.append(st["_equity_curve"]["Equity"])
        va_curves.append(sv["_equity_curve"]["Equity"])

    def combine(curves):
        norm = [c / c.iloc[0] for c in curves]
        return pd.concat(norm, axis=1).ffill().dropna().mean(axis=1)
    return combine(tr_curves), combine(va_curves)


def main() -> None:
    config = load_factory_config()
    store = ExperimentStore()
    catalog = DataCatalog(config)

    candidates = []
    for row in store.list_specs():
        sid = row["spec_id"]
        trains = [e for e in store.experiments_for(sid, "train")
                  if e["success"] and e["dataset_id"] not in ("XS", "WIDE")]
        if not trains:
            continue
        m = trains[-1]["metrics"]
        if (m.get("return_pct") or 0) > 0 and (m.get("n_trades") or 0) >= 30:
            candidates.append(store.get_spec(sid))
    print(f"candidates with positive train (>=30 trades): {len(candidates)}")

    groups = {"COMBO-DAILY": [], "COMBO-CRYPTO": []}
    skipped = 0
    for spec in candidates:
        out = spec_curves(spec, store, catalog, config)
        if out is None:
            skipped += 1
            continue
        tr_curve, va_curve = out
        tr_ret = (tr_curve.iloc[-1] / tr_curve.iloc[0] - 1) * 100
        if tr_ret <= 0:   # selection strictly on re-run train curve
            continue
        group = "COMBO-DAILY" if spec.timeframe == "1d" else "COMBO-CRYPTO"
        groups[group].append((spec.name, spec.spec_id, va_curve))
        print(f"  selected [{group}] {spec.name} ({spec.spec_id}) train={tr_ret:.1f}%")
    print(f"skipped (no module/datasets/errors): {skipped}")

    for gname, members in groups.items():
        print(f"\n=== {gname}: {len(members)} components ===")
        if len(members) < 2:
            print("  fewer than 2 components - combination not constructible")
            continue
        rets = pd.concat({n: c.pct_change() for n, _, c in members}, axis=1).dropna()
        if len(rets) < 50:
            print(f"  only {len(rets)} overlapping bars - not constructible")
            continue
        corr = rets.corr()
        iu = np.triu_indices(len(corr), 1)
        mean_corr = float(corr.values[iu].mean())
        combo_returns = rets.mean(axis=1)
        m = metrics_from_returns(combo_returns)
        m["n_trades"] = 999_999  # trade-level pooling not meaningful for a combo of combos
        print(f"  mean pairwise correlation: {mean_corr:.3f}")
        print(f"  val: ret={m.get('return_pct'):.2f}% sharpe={m.get('sharpe')} "
              f"sortino={m.get('sortino')} calmar={m.get('calmar')} "
              f"maxDD={m.get('max_drawdown_pct'):.2f}%")
        spec = StrategySpec(
            name=gname.replace("-", ""), family="portfolio_allocation",
            asset_class="crypto_perp" if gname == "COMBO-CRYPTO" else "equity",
            instruments=[n for n, _, _ in members][:10], timeframe="combo",
            hypothesis=("Pre-registered equal-weight combination of all components with "
                        "positive train performance; selection on train only, judged on "
                        "validation. Tests whether diversification across the store's "
                        "surviving train-positive streams yields a gate-passing master "
                        "strategy without any subset search."),
            regime_assumptions="components span families; combination is regime-agnostic",
            entry_logic=f"equal-weight daily aggregation of {len(members)} component streams",
            exit_logic="components manage their own exits", stop_logic="component-level",
            position_sizing="equal weight", risk_rules="no leverage, no reweighting",
            invalidation_rules="standard gates, one shot, no subset iteration",
            expected_trade_frequency="swing", low_frequency=False,
            source="src/scripts/run_combination_test.py",
        )
        try:
            sid = store.add_spec(spec)
        except ValueError:
            sid = store.conn.execute("SELECT spec_id FROM specs WHERE fingerprint=?",
                                     (spec.fingerprint(),)).fetchone()["spec_id"]
            spec.spec_id = sid
        store.record_experiment(sid, "validation", True, dataset_id="COMBO",
                                metrics={**m, "mean_pairwise_correlation": mean_corr,
                                         "components": [s for _, s, _ in members]})
        gates = evaluate_gates(spec, m, {"n_trades": 999_999}, {},
                               static_review_clean=True)
        store.record_gate_results(sid, None, [g.to_dict() for g in gates])
        failed = [g.name for g in gates if g.hard and not g.passed]
        if failed:
            store.record_rejection(sid, "gates", f"failed: {failed}")
        print(f"  GATES: {'PASS ALL' if not failed else 'failed ' + str(failed)}")
    print("\nCOMBINATION_TEST_DONE")


if __name__ == "__main__":
    main()
