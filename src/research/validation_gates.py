"""
Validation gates - mechanical promotion criteria loaded from
configs/validation_gates.yaml. No LLM involvement; no override path in code.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from src.research.factory_config import load_gates_config


@dataclass
class GateResult:
    name: str
    passed: bool
    value: Any
    threshold: Any
    hard: bool
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _gate(gates: Dict, key: str) -> Dict[str, Any]:
    g = gates.get(key)
    if g is None:
        raise KeyError(f"gate '{key}' missing from validation_gates.yaml")
    return g


def evaluate_gates(
    spec,
    validation_metrics: Dict[str, Any],
    train_metrics: Dict[str, Any],
    robustness: Optional[Dict[str, Any]] = None,
    static_review_clean: bool = False,
    gates_config: Optional[Dict] = None,
) -> List[GateResult]:
    """
    Evaluate every gate against out-of-sample (validation) metrics plus robustness
    summary. Returns the full result list; caller decides what to do with it.
    """
    cfg = (gates_config or load_gates_config())["gates"]
    robustness = robustness or {}
    results: List[GateResult] = []

    def check(name: str, value: Optional[float], threshold: float, hard: bool,
              direction: str = ">=", reason: str = "") -> None:
        if value is None:
            results.append(GateResult(name, False, None, threshold, hard,
                                      reason or "metric missing/NaN (insufficient trades?)"))
            return
        passed = value >= threshold if direction == ">=" else value <= threshold
        results.append(GateResult(name, passed, round(float(value), 4), threshold, hard, reason))

    # trade count: train + validation combined
    g = _gate(cfg, "min_trades_low_frequency" if spec.low_frequency else "min_trades")
    total_trades = (train_metrics.get("n_trades") or 0) + (validation_metrics.get("n_trades") or 0)
    check("min_trades", total_trades, g["value"], g["hard"],
          reason="low_frequency spec" if spec.low_frequency else "")

    g = _gate(cfg, "positive_oos_return")
    check("positive_oos_return", validation_metrics.get("return_pct"), g["value"], g["hard"])

    g = _gate(cfg, "min_profit_factor")
    check("min_profit_factor", validation_metrics.get("profit_factor"), g["value"], g["hard"])

    g = _gate(cfg, "max_drawdown_pct")
    dd = validation_metrics.get("max_drawdown_pct")
    check("max_drawdown_pct", abs(dd) if dd is not None else None, g["value"], g["hard"],
          direction="<=")

    for gate_name, metric in (("min_sharpe", "sharpe"), ("min_sortino", "sortino"),
                              ("min_calmar", "calmar")):
        g = _gate(cfg, gate_name)
        check(gate_name, validation_metrics.get(metric), g["value"], g["hard"])

    # walk-forward
    g = _gate(cfg, "walk_forward_retention")
    check("walk_forward_retention", robustness.get("walk_forward_retention"),
          g["value"], g["hard"],
          reason="mean OOS fold return / in-sample return")
    g = _gate(cfg, "walk_forward_positive")
    check("walk_forward_positive", robustness.get("walk_forward_oos_mean_return_pct"),
          g["value"], g["hard"])

    # cost stress: return at the configured multiplier must stay positive
    g = _gate(cfg, "cost_stress_multiplier")
    stress_key = f"return_pct_at_{g['value']}x"
    check("cost_stress", robustness.get("cost_stress", {}).get(stress_key),
          0.0, g["hard"], reason=f"profitable at {g['value']}x costs")

    g = _gate(cfg, "param_neighborhood_retention")
    check("param_neighborhood_retention", robustness.get("param_neighborhood_retention"),
          g["value"], g["hard"], reason="neighborhood mean return / point estimate")

    g = _gate(cfg, "single_trade_dependence")
    check("single_trade_dependence", validation_metrics.get("return_without_best_trade_pct"),
          g["value"], g["hard"], reason="return with best trade removed")

    g = _gate(cfg, "monte_carlo_p05_drawdown_pct")
    mc_dd = robustness.get("monte_carlo", {}).get("p05_max_drawdown_pct")
    check("monte_carlo_p05_drawdown", abs(mc_dd) if mc_dd is not None else None,
          g["value"], g["hard"], direction="<=")

    g = _gate(cfg, "max_exposure_pct")
    check("max_exposure", validation_metrics.get("exposure_time_pct"), g["value"],
          g["hard"], direction="<=")

    g = _gate(cfg, "require_hypothesis")
    has_hypothesis = bool(spec.hypothesis and len(spec.hypothesis.strip()) >= 20)
    results.append(GateResult("require_hypothesis", has_hypothesis or not g["value"],
                              has_hypothesis, g["value"], g["hard"],
                              "spec must state a market mechanism"))

    g = _gate(cfg, "static_review_clean")
    results.append(GateResult("static_review_clean", static_review_clean or not g["value"],
                              static_review_clean, g["value"], g["hard"],
                              "no hard findings from static lookahead/bias checks"))

    return results


def hard_gates_passed(results: List[GateResult]) -> bool:
    return all(r.passed for r in results if r.hard)


def summarize(results: List[GateResult]) -> str:
    lines = []
    for r in results:
        status = "PASS" if r.passed else ("FAIL" if r.hard else "warn")
        lines.append(f"[{status}] {r.name}: value={r.value} threshold={r.threshold}"
                     + (f" ({r.reason})" if r.reason else ""))
    return "\n".join(lines)
