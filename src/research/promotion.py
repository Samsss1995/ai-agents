"""
Promotion - the status state machine between research and live capital.

    research_only -> paper_candidate -> paper_active -> live_candidate
        -> live_approved -> retired (from any state)

Rules enforced here, not in prompts:
  - research_only -> paper_candidate: all hard validation gates passed AND the
    once-only holdout (test slice) run is positive.
  - paper_candidate -> paper_active: manual approval + broker profile + risk config.
  - any live_* transition: manual approval + paper report + named broker +
    emergency stop + rollback plan. The factory never performs these on its own.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.research.backtest_runner import run_holdout_test
from src.research.experiment_store import ExperimentStore
from src.research.factory_config import (
    load_factory_config, load_gates_config, load_broker_profiles, resolve_path,
)

STATUSES = ["research_only", "paper_candidate", "paper_active",
            "live_candidate", "live_approved", "retired"]

ALLOWED_TRANSITIONS = {
    "research_only": {"paper_candidate", "retired"},
    "paper_candidate": {"paper_active", "research_only", "retired"},
    "paper_active": {"live_candidate", "paper_candidate", "retired"},
    "live_candidate": {"live_approved", "paper_active", "retired"},
    "live_approved": {"retired", "paper_active"},
    "retired": set(),
}

MANUAL_APPROVAL_REQUIRED = {"paper_active", "live_candidate", "live_approved"}


class PromotionError(Exception):
    pass


def promote(spec_id: str, to_status: str, approved_by: Optional[str] = None,
            notes: str = "", store: Optional[ExperimentStore] = None,
            config: Optional[Dict] = None) -> Dict[str, Any]:
    config = config or load_factory_config()
    store = store or ExperimentStore()
    gates_cfg = load_gates_config()

    if to_status not in STATUSES:
        raise PromotionError(f"unknown status '{to_status}'. Valid: {STATUSES}")
    from_status = store.get_spec_status(spec_id)
    if to_status not in ALLOWED_TRANSITIONS[from_status]:
        raise PromotionError(
            f"illegal transition {from_status} -> {to_status}. "
            f"Allowed from {from_status}: {sorted(ALLOWED_TRANSITIONS[from_status])}"
        )
    if to_status in MANUAL_APPROVAL_REQUIRED and not approved_by:
        raise PromotionError(
            f"transition to '{to_status}' requires manual approval: pass --approved-by <name>"
        )

    spec = store.get_spec(spec_id)
    holdout_metrics: Dict[str, Any] = {}

    if to_status == "paper_candidate":
        if not store.gates_passed(spec_id):
            failed = [r["gate_name"] for r in store.latest_gate_results(spec_id)
                      if r["hard"] and not r["passed"]]
            raise PromotionError(
                f"hard gates not passed for {spec_id}: {failed or 'no gate results recorded'}"
            )
        code = store.latest_code(spec_id)
        if code is None:
            raise PromotionError(f"no code version recorded for {spec_id}")
        # the once-only holdout run
        existing_tests = store.experiments_for(spec_id, "test")
        if existing_tests:
            holdout_metrics = existing_tests[-1]["metrics"]
        else:
            results = run_holdout_test(spec, Path(code["file_path"]), store,
                                       code["code_id"], config)
            holdout_metrics = next(iter(results.values())) if results else {}
        ret = holdout_metrics.get("return_pct")
        if ret is None or ret <= 0:
            store.record_rejection(spec_id, "holdout",
                                   f"holdout test return {ret} not positive")
            raise PromotionError(f"holdout (test slice) return {ret} is not positive - rejected")

    if to_status == "paper_active":
        paper_cfg = gates_cfg["paper_promotion"]
        profiles = load_broker_profiles()
        if paper_cfg["require_broker_profile"] and "paper" not in profiles:
            raise PromotionError("no 'paper' broker profile in configs/broker_profiles.yaml")
        if paper_cfg["require_code_review"]:
            reviews = store.experiments_for(spec_id, "review")
            if not any(r["success"] for r in reviews):
                raise PromotionError("no successful code review recorded "
                                     "(run strategy_review_agent on the code first)")

    if to_status in ("live_candidate", "live_approved"):
        live_cfg = gates_cfg["live_promotion"]
        missing = []
        if live_cfg["require_paper_report"] and "paper_report=" not in notes:
            missing.append("paper_report=<path> in --notes")
        if live_cfg["require_named_broker"] and "broker=" not in notes:
            missing.append("broker=<profile> in --notes")
        if live_cfg["require_emergency_stop"] and "emergency_stop=" not in notes:
            missing.append("emergency_stop=<procedure> in --notes")
        if live_cfg["require_rollback_plan"] and "rollback=" not in notes:
            missing.append("rollback=<plan> in --notes")
        if missing:
            raise PromotionError(
                "live promotion blocked - missing required fields: " + ", ".join(missing)
            )

    store.set_spec_status(spec_id, to_status)
    store.record_promotion(spec_id, from_status, to_status, approved_by, notes)

    wrapper_path = None
    if to_status == "paper_candidate":
        wrapper_path = write_promoted_strategy(spec, store, holdout_metrics, config)

    return {
        "spec_id": spec_id,
        "from": from_status,
        "to": to_status,
        "approved_by": approved_by,
        "wrapper": str(wrapper_path) if wrapper_path else None,
        "holdout": holdout_metrics,
    }


def write_promoted_strategy(spec, store: ExperimentStore,
                            holdout_metrics: Dict[str, Any],
                            config: Dict) -> Path:
    """
    Write a project-convention strategy wrapper into src/strategies/custom/.
    The wrapper carries full metadata and delegates signal generation to the
    research module's generate_signal(df). It never trades by itself.
    """
    code = store.latest_code(spec.spec_id)
    gates = store.latest_gate_results(spec.spec_id)
    validation = store.experiments_for(spec.spec_id, "validation")
    val_metrics = validation[-1]["metrics"] if validation else {}

    metadata = {
        "spec_id": spec.spec_id,
        "name": spec.name,
        "version": code["version"] if code else None,
        "code_hash": code["code_hash"] if code else None,
        "date_created": datetime.now(timezone.utc).isoformat(),
        "status": "paper_candidate",
        "tested_instruments": spec.instruments,
        "tested_timeframe": spec.timeframe,
        "validation_summary": {
            "validation": val_metrics,
            "holdout": holdout_metrics,
            "gates": [{k: g[k] for k in ("gate_name", "passed", "value", "threshold")}
                      for g in gates],
        },
        "known_weaknesses": spec.invalidation_rules,
        "allowed_brokers": ["paper"],
        "risk_limits": load_gates_config()["paper_promotion"],
    }

    promoted_dir = resolve_path(config["paths"]["promoted_dir"])
    promoted_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c for c in spec.name if c.isalnum() or c == "_").lower()
    path = promoted_dir / f"rf_{safe_name}_v{metadata['version']}.py"

    module_file = code["file_path"] if code else ""
    content = f'''"""
Research-factory promoted strategy: {spec.name} (status: paper_candidate)
Generated by src/research/promotion.py - DO NOT enable live trading here.
Metadata below is the audit trail; edits invalidate the recorded code_hash.
"""

import importlib.util
import json
from pathlib import Path

from src.strategies.base_strategy import BaseStrategy

METADATA = json.loads(r\'\'\'{json.dumps(metadata, indent=2, default=str)}\'\'\')

RESEARCH_MODULE = Path(r"{module_file}")


class {spec.name.replace(" ", "")}Strategy(BaseStrategy):
    name = "{spec.name}"
    description = "{spec.family} | {spec.timeframe} | paper_candidate"

    def __init__(self):
        super().__init__(self.name)
        if not RESEARCH_MODULE.exists():
            raise FileNotFoundError(f"research module missing: {{RESEARCH_MODULE}}")
        mod_spec = importlib.util.spec_from_file_location(RESEARCH_MODULE.stem, RESEARCH_MODULE)
        self._module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(self._module)
        if not hasattr(self._module, "generate_signal"):
            raise AttributeError("research module lacks generate_signal(df)")

    def generate_signals(self, token=None, market_data=None):
        if market_data is None:
            raise ValueError("market_data (OHLCV DataFrame) is required")
        signal = self._module.generate_signal(market_data)
        signal.setdefault("token", token)
        signal["metadata"] = {{"spec_id": METADATA["spec_id"], "status": METADATA["status"]}}
        return signal
'''
    path.write_text(content)
    return path
