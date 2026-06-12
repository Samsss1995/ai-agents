"""
Report writer - markdown reports assembled from the experiment store.
LLM is never required here; prose is mechanical so numbers cannot be 'improved'.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.research.experiment_store import ExperimentStore
from src.research.factory_config import (
    load_factory_config, load_broker_profiles, resolve_path,
)
from src.research.leaderboard import build_leaderboard


def _reports_dir(config: Dict) -> Path:
    d = resolve_path(config["paths"]["reports_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_daily_research_report(store: Optional[ExperimentStore] = None,
                                config: Optional[Dict] = None) -> Path:
    config = config or load_factory_config()
    store = store or ExperimentStore()
    summary = store.summary()
    df = build_leaderboard(store, config)

    lines = [
        "# Daily Research Report",
        f"Generated: {_now()}",
        "",
        "## Store summary",
        f"- ideas: {summary['ideas']}",
        f"- specs: {summary['specs']} (by status: {summary['specs_by_status']})",
        f"- code versions: {summary['code_versions']}",
        f"- experiments: {summary['experiments']} "
        f"(failed: {summary['experiments_failed']})",
        f"- rejections: {summary['rejections']}",
        f"- promotions: {summary['promotions']}",
        "",
        "## Leaderboard (top 10 by composite score)",
    ]
    if len(df):
        lines += ["```", df.head(10).to_string(index=False), "```"]
    else:
        lines.append("No executed strategies yet.")

    lines += ["", "## Recent rejections (last 10)"]
    rejections = store.list_rejections()[:10]
    if rejections:
        for r in rejections:
            lines.append(f"- [{r['created_at']}] {r['stage']}: {r['reason']} "
                         f"(spec={r['spec_id']})")
    else:
        lines.append("None recorded.")

    path = _reports_dir(config) / "daily_research_report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_robustness_report(spec_id: str, robustness: Dict[str, Any],
                            store: Optional[ExperimentStore] = None,
                            config: Optional[Dict] = None) -> Path:
    config = config or load_factory_config()
    store = store or ExperimentStore()
    spec = store.get_spec(spec_id)

    lines = [
        f"# Robustness Report - {spec.name} ({spec_id})",
        f"Generated: {_now()}",
        "",
        "## Walk-forward",
        f"- in-sample mean return: {robustness.get('walk_forward_is_mean_return_pct')}%",
        f"- OOS mean return: {robustness.get('walk_forward_oos_mean_return_pct')}%",
        f"- retention (OOS/IS): {robustness.get('walk_forward_retention')}",
        "",
        "| fold | IS return % | OOS return % | OOS sharpe | OOS trades | OOS maxDD % |",
        "|---|---|---|---|---|---|",
    ]
    for f in robustness.get("folds", []):
        lines.append(
            f"| {f['fold']} | {f['in_sample_return_pct']} | {f['oos_return_pct']} "
            f"| {f['oos_sharpe']} | {f['oos_trades']} | {f['oos_max_drawdown_pct']} |"
        )
    lines += [
        "",
        "## Parameter neighborhood",
        f"- base return: {robustness.get('base_return_pct')}%",
        f"- neighborhood mean: {robustness.get('neighborhood_mean_return_pct')}%",
        f"- retention: {robustness.get('param_neighborhood_retention')}",
        "",
        "## Monte Carlo (trade reshuffle, seeded)",
        "```json",
        json.dumps(robustness.get("monte_carlo", {}), indent=2),
        "```",
        "",
        "## Cost stress",
        "```json",
        json.dumps(robustness.get("cost_stress", {}), indent=2),
        "```",
        "",
        "Limitation: Monte Carlo assumes trade independence; serial correlation "
        "understates tail risk.",
    ]
    path = _reports_dir(config) / f"robustness_{spec_id}.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_overfitting_report(store: Optional[ExperimentStore] = None,
                             config: Optional[Dict] = None) -> Path:
    """In-sample vs OOS decay across every executed strategy - researcher-bias meter."""
    config = config or load_factory_config()
    store = store or ExperimentStore()
    lines = [
        "# Overfitting Report",
        f"Generated: {_now()}",
        "",
        "In-sample (train) vs out-of-sample (validation) comparison. Large decay "
        "= curve fitting. All runs are recorded, including failures, so this table "
        "cannot be survivorship-biased by construction.",
        "",
        "| spec | name | train ret % | val ret % | decay | train trades | val trades |",
        "|---|---|---|---|---|---|---|",
    ]
    for spec_row in store.list_specs():
        spec_id = spec_row["spec_id"]
        train = [e for e in store.experiments_for(spec_id, "train") if e["success"]]
        val = [e for e in store.experiments_for(spec_id, "validation") if e["success"]]
        if not train or not val:
            continue
        t, v = train[-1]["metrics"], val[-1]["metrics"]
        tr, vr = t.get("return_pct"), v.get("return_pct")
        decay = (None if tr in (None, 0) or vr is None
                 else round(1 - vr / tr, 3) if tr > 0 else "n/a")
        lines.append(f"| {spec_id} | {spec_row['name']} | {tr} | {vr} | {decay} "
                     f"| {t.get('n_trades')} | {v.get('n_trades')} |")

    failed = store.conn.execute(
        "SELECT COUNT(*) AS c FROM experiments WHERE success=0").fetchone()["c"]
    total = store.conn.execute("SELECT COUNT(*) AS c FROM experiments").fetchone()["c"]
    lines += ["", f"Failure visibility: {failed}/{total} recorded experiments failed. "
                  "If this number looks too clean, suspect the pipeline, not the strategies."]
    path = _reports_dir(config) / "overfitting_report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_broker_readiness_report(config: Optional[Dict] = None) -> Path:
    config = config or load_factory_config()
    profiles = load_broker_profiles()
    lines = ["# Broker Readiness Report", f"Generated: {_now()}", ""]
    for name, p in profiles.items():
        lines += [
            f"## {name}",
            f"- type: {p.get('type')}",
            f"- live_enabled: {p.get('live_enabled', False)}",
            f"- limits: max_position_usd={p.get('max_position_usd')}, "
            f"max_total_exposure_usd={p.get('max_total_exposure_usd')}, "
            f"max_daily_loss_usd={p.get('max_daily_loss_usd')}",
            f"- note: {p.get('note', '-')}",
            "",
        ]
    lines += [
        "Live trading additionally requires env BROKER_LIVE_CONFIRM="
        "YES_I_APPROVE_LIVE_TRADING at runtime. No adapter ships enabled.",
    ]
    path = _reports_dir(config) / "broker_readiness_report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_paper_trading_report(fills: List[Dict[str, Any]],
                               account: Dict[str, Any],
                               config: Optional[Dict] = None) -> Path:
    """Populated by the paper-trading loop (phase 2); callable now for manual runs."""
    config = config or load_factory_config()
    lines = [
        "# Paper Trading Report",
        f"Generated: {_now()}",
        "",
        "## Account",
        "```json", json.dumps(account, indent=2, default=str), "```",
        "",
        f"## Fills ({len(fills)})",
    ]
    for f in fills[-100:]:
        lines.append(f"- {f}")
    path = _reports_dir(config) / "paper_trading_report.md"
    path.write_text("\n".join(lines) + "\n")
    return path
