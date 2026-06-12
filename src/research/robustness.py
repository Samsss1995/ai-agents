"""
Robustness / anti-overfitting battery:
  - anchored walk-forward validation
  - parameter neighborhood testing
  - Monte Carlo trade reshuffling + bootstrap
  - cost stress testing

All randomness is seeded from config (monte_carlo.seed) for reproducibility.
Results feed validation_gates.evaluate_gates() and robustness_report.md.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.research.backtest_runner import run_backtest, load_strategy_module, split_data
from src.research.factory_config import load_factory_config
from src.research.metrics import trade_returns_pct


def walk_forward_folds(df: pd.DataFrame, n_folds: int, min_train_frac: float
                       ) -> List[Dict[str, pd.DataFrame]]:
    """
    Anchored walk-forward: train always starts at bar 0; each fold's test window
    is the next chronological slice. Example with 4 folds and min_train_frac 0.4:
    train [0..40%) test [40..55%), train [0..55%) test [55..70%), ...
    """
    n = len(df)
    test_frac = (1.0 - min_train_frac) / n_folds
    folds = []
    for k in range(n_folds):
        train_end = int(n * (min_train_frac + k * test_frac))
        test_end = int(n * (min_train_frac + (k + 1) * test_frac))
        train_df, test_df = df.iloc[:train_end], df.iloc[train_end:test_end]
        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError(f"walk-forward fold {k} produced an empty slice (n={n})")
        folds.append({"train": train_df, "test": test_df})
    return folds


def run_walk_forward(module_path: Path, df: pd.DataFrame,
                     config: Optional[Dict] = None) -> Dict[str, Any]:
    config = config or load_factory_config()
    wf = config["walk_forward"]
    min_fold_days = config["data"]["min_fold_days"]
    folds = walk_forward_folds(df, wf["n_folds"], wf["min_train_frac"])
    for k, fold in enumerate(folds):
        span_days = (fold["test"].index[-1] - fold["test"].index[0]).total_seconds() / 86_400
        if span_days < min_fold_days:
            raise ValueError(
                f"walk-forward fold {k} test slice spans {span_days:.1f} days "
                f"< min_fold_days {min_fold_days}. Dataset too short - extend data."
            )
    fold_results = []
    for k, fold in enumerate(folds):
        _, is_metrics = run_backtest(module_path, fold["train"], config=config)
        _, oos_metrics = run_backtest(module_path, fold["test"], config=config)
        fold_results.append({
            "fold": k,
            "in_sample_return_pct": is_metrics["return_pct"],
            "oos_return_pct": oos_metrics["return_pct"],
            "oos_sharpe": oos_metrics["sharpe"],
            "oos_trades": oos_metrics["n_trades"],
            "oos_max_drawdown_pct": oos_metrics["max_drawdown_pct"],
        })

    is_returns = [f["in_sample_return_pct"] for f in fold_results
                  if f["in_sample_return_pct"] is not None]
    oos_returns = [f["oos_return_pct"] for f in fold_results
                   if f["oos_return_pct"] is not None]
    is_mean = float(np.mean(is_returns)) if is_returns else None
    oos_mean = float(np.mean(oos_returns)) if oos_returns else None
    retention = None
    if is_mean is not None and oos_mean is not None and is_mean > 0:
        retention = oos_mean / is_mean
    return {
        "folds": fold_results,
        "walk_forward_is_mean_return_pct": is_mean,
        "walk_forward_oos_mean_return_pct": oos_mean,
        "walk_forward_retention": retention,
    }


def run_param_neighborhood(module_path: Path, df: pd.DataFrame,
                           config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Perturb each numeric parameter by the configured multipliers and re-run on the
    train slice. A robust strategy's neighborhood mean return should be close to
    the point estimate; a curve-fit one collapses.
    """
    config = config or load_factory_config()
    module = load_strategy_module(Path(module_path))
    multipliers = config["param_neighborhood"]["multipliers"]
    train_df, _, _ = split_data(df, config)

    _, base_metrics = run_backtest(module_path, train_df, config=config)
    base_return = base_metrics["return_pct"]

    runs: List[Dict[str, Any]] = []
    for name, default in module.PARAMS.items():
        if not isinstance(default, (int, float)) or isinstance(default, bool):
            continue
        for mult in multipliers:
            perturbed = default * mult
            if isinstance(default, int):
                perturbed = max(1, int(round(perturbed)))
                if perturbed == default:
                    continue
            try:
                _, m = run_backtest(module_path, train_df,
                                    params={name: perturbed}, config=config)
                runs.append({"param": name, "value": perturbed,
                             "return_pct": m["return_pct"], "n_trades": m["n_trades"]})
            except Exception as e:
                runs.append({"param": name, "value": perturbed,
                             "return_pct": None, "error": f"{type(e).__name__}: {e}"})

    neighborhood = [r["return_pct"] for r in runs if r.get("return_pct") is not None]
    retention = None
    if neighborhood and base_return is not None and base_return > 0:
        retention = float(np.mean(neighborhood)) / base_return
    return {
        "base_return_pct": base_return,
        "neighborhood_runs": runs,
        "neighborhood_mean_return_pct": float(np.mean(neighborhood)) if neighborhood else None,
        "param_neighborhood_retention": retention,
    }


def run_monte_carlo(stats: Any, config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Trade-sequence reshuffling (permutation) and bootstrap (resampling with
    replacement) on per-trade returns. Assumes trade independence - stated
    limitation, see RISK_AND_OVERFITTING_AUDIT.md section E.
    """
    config = config or load_factory_config()
    mc = config["monte_carlo"]
    rng = np.random.default_rng(mc["seed"])
    returns = trade_returns_pct(stats) / 100.0
    if len(returns) < 5:
        return {"error": f"only {len(returns)} trades - Monte Carlo not meaningful",
                "n_trades": int(len(returns))}

    def equity_stats(seq: np.ndarray) -> Dict[str, float]:
        equity = np.cumprod(1.0 + seq)
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        return {"total_return_pct": (equity[-1] - 1.0) * 100,
                "max_drawdown_pct": dd.min() * 100}

    shuffled, bootstrapped = [], []
    for _ in range(mc["n_iterations"]):
        shuffled.append(equity_stats(rng.permutation(returns)))
        bootstrapped.append(equity_stats(rng.choice(returns, size=len(returns), replace=True)))

    def pct(dist: List[Dict], key: str, q: float) -> float:
        return float(np.percentile([d[key] for d in dist], q))

    return {
        "n_iterations": mc["n_iterations"],
        "seed": mc["seed"],
        "n_trades": int(len(returns)),
        "median_total_return_pct": pct(shuffled, "total_return_pct", 50),
        "p05_total_return_pct": pct(shuffled, "total_return_pct", 5),
        "p05_max_drawdown_pct": pct(shuffled, "max_drawdown_pct", 5),
        "bootstrap_median_return_pct": pct(bootstrapped, "total_return_pct", 50),
        "bootstrap_p05_return_pct": pct(bootstrapped, "total_return_pct", 5),
        "bootstrap_p05_max_drawdown_pct": pct(bootstrapped, "max_drawdown_pct", 5),
    }


def run_cost_stress(module_path: Path, df: pd.DataFrame,
                    config: Optional[Dict] = None) -> Dict[str, Any]:
    """Re-run the validation slice at elevated cost multipliers."""
    config = config or load_factory_config()
    _, val_df, _ = split_data(df, config)
    out: Dict[str, Any] = {}
    for mult in config["cost_stress"]["multipliers"]:
        _, m = run_backtest(module_path, val_df, config=config, cost_multiplier=mult)
        out[f"return_pct_at_{mult}x"] = m["return_pct"]
        out[f"sharpe_at_{mult}x"] = m["sharpe"]
    return out


def full_robustness_battery(module_path: Path, df: pd.DataFrame, validation_stats: Any,
                            config: Optional[Dict] = None) -> Dict[str, Any]:
    """Run the whole battery. Caller records each block in the experiment store."""
    config = config or load_factory_config()
    result: Dict[str, Any] = {}
    result.update(run_walk_forward(module_path, df, config))
    result.update(run_param_neighborhood(module_path, df, config))
    result["monte_carlo"] = run_monte_carlo(validation_stats, config)
    result["cost_stress"] = run_cost_stress(module_path, df, config)
    return result
