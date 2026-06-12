"""
Leaderboard - composite scoring of all researched strategies.

Score is built from OUT-OF-SAMPLE (validation) metrics with penalties for the
failure modes the legacy pipeline rewarded: drawdown, low trade count, OOS decay,
cost fragility, parameter fragility, concentration. Raw return never ranks alone.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.research.experiment_store import ExperimentStore
from src.research.factory_config import load_factory_config, load_gates_config, resolve_path


def _clamp(x: Optional[float], lo: float, hi: float) -> float:
    if x is None:
        return lo
    return max(lo, min(hi, float(x)))


def composite_score(validation: Dict[str, Any], robustness: Dict[str, Any],
                    spec_row: Dict[str, Any], config: Dict, gates_cfg: Dict) -> float:
    """
    Weighted sum of normalized OOS quality terms minus penalties. Each term is
    clamped to [0, 1] so no single metric can dominate.
    """
    w = config["leaderboard"]["weights"]
    p = config["leaderboard"]["penalties"]
    gates = gates_cfg["gates"]

    score = 0.0
    score += w["sharpe"] * _clamp((validation.get("sharpe") or 0) / 2.0, 0, 1)
    score += w["sortino"] * _clamp((validation.get("sortino") or 0) / 3.0, 0, 1)
    score += w["calmar"] * _clamp((validation.get("calmar") or 0) / 2.0, 0, 1)
    pf = validation.get("profit_factor")
    score += w["profit_factor"] * _clamp(((pf or 1.0) - 1.0) / 1.0, 0, 1)
    score += w["oos_retention"] * _clamp(robustness.get("walk_forward_retention"), 0, 1)

    n_trades = validation.get("n_trades") or 0
    min_trades = gates["min_trades"]["value"]
    if n_trades < min_trades:
        score -= p["low_trades"] * (1.0 - n_trades / max(min_trades, 1))
    dd = abs(validation.get("max_drawdown_pct") or 100.0)
    dd_limit = gates["max_drawdown_pct"]["value"]
    if dd > dd_limit:
        score -= p["drawdown_over_limit"]
    stress = robustness.get("cost_stress", {})
    stress_mult = gates["cost_stress_multiplier"]["value"]
    stress_ret = stress.get(f"return_pct_at_{stress_mult}x")
    if stress_ret is None or stress_ret <= 0:
        score -= p["cost_fragility"]
    pn = robustness.get("param_neighborhood_retention")
    if pn is None or pn < gates["param_neighborhood_retention"]["value"]:
        score -= p["param_fragility"]
    spec_json = json.loads(spec_row.get("spec_json", "{}")) if "spec_json" in spec_row else {}
    if len(spec_json.get("instruments", [])) <= 1 and not spec_json.get("intentionally_single_asset"):
        score -= p["single_asset"]
    return round(score, 4)


def _latest_metrics(store: ExperimentStore, spec_id: str, kind: str) -> Dict[str, Any]:
    runs = [e for e in store.experiments_for(spec_id, kind) if e["success"]]
    return runs[-1]["metrics"] if runs else {}


def build_leaderboard(store: Optional[ExperimentStore] = None,
                      config: Optional[Dict] = None) -> pd.DataFrame:
    config = config or load_factory_config()
    gates_cfg = load_gates_config()
    store = store or ExperimentStore()

    rows: List[Dict[str, Any]] = []
    for spec_row in store.list_specs():
        spec_id = spec_row["spec_id"]
        validation = _latest_metrics(store, spec_id, "validation")
        train = _latest_metrics(store, spec_id, "train")
        robustness = _latest_metrics(store, spec_id, "robustness")
        if not validation and not train:
            continue  # never executed - shows up in rejected/failures report instead
        full_spec = store.conn.execute(
            "SELECT spec_json FROM specs WHERE spec_id=?", (spec_id,)).fetchone()
        spec_dict = dict(spec_row)
        spec_dict["spec_json"] = full_spec["spec_json"] if full_spec else "{}"
        rows.append({
            "spec_id": spec_id,
            "name": spec_row["name"],
            "status": spec_row["status"],
            "score": composite_score(validation, robustness, spec_dict, config, gates_cfg),
            "val_return_pct": validation.get("return_pct"),
            "val_sharpe": validation.get("sharpe"),
            "val_sortino": validation.get("sortino"),
            "val_calmar": validation.get("calmar"),
            "val_profit_factor": validation.get("profit_factor"),
            "val_max_dd_pct": validation.get("max_drawdown_pct"),
            "val_trades": validation.get("n_trades"),
            "train_return_pct": train.get("return_pct"),
            "train_trades": train.get("n_trades"),
            "wf_retention": robustness.get("walk_forward_retention"),
            "param_retention": robustness.get("param_neighborhood_retention"),
            "buy_hold_val_pct": validation.get("buy_hold_return_pct"),
            "gates_passed": store.gates_passed(spec_id),
        })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df


def write_leaderboard_files(store: Optional[ExperimentStore] = None,
                            config: Optional[Dict] = None) -> Dict[str, Path]:
    config = config or load_factory_config()
    store = store or ExperimentStore()
    reports_dir = resolve_path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()

    out: Dict[str, Path] = {}
    df = build_leaderboard(store, config)
    lb_path = reports_dir / "strategy_leaderboard.csv"
    df.to_csv(lb_path, index=False)
    out["leaderboard"] = lb_path

    rejected = pd.DataFrame(store.list_rejections())
    rej_path = reports_dir / "rejected_strategies.csv"
    rejected.to_csv(rej_path, index=False)
    out["rejected"] = rej_path

    promoted_rows = [dict(r) for r in store.conn.execute(
        "SELECT p.*, s.name FROM promotions p JOIN specs s ON s.spec_id = p.spec_id "
        "ORDER BY p.created_at DESC")]
    prom_path = reports_dir / "promoted_strategies.csv"
    pd.DataFrame(promoted_rows).to_csv(prom_path, index=False)
    out["promoted"] = prom_path

    (reports_dir / "leaderboard_generated_at.txt").write_text(stamp + "\n")
    return out
