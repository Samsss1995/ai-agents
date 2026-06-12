"""
Backtest runner - executes generated strategy modules with the standardized cost
model on chronological train/validation/test slices.

Generated module contract (enforced by codegen and checked here):
    STRATEGY_CLASS : a backtesting.py Strategy subclass; numeric tunables are
                     class attributes so bt.run(**params) can override them.
    PARAMS         : dict of those tunables and their default values.
    generate_signal(df) -> dict   (optional; reused by promotion wrappers)

Fails loudly if `backtesting` is not importable - the environment must be fixed,
not worked around.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.research.data_catalog import DataCatalog
from src.research.factory_config import load_factory_config
from src.research.metrics import extract_metrics


class BacktestEnvironmentError(Exception):
    pass


class StrategyContractError(Exception):
    pass


def _import_backtesting():
    """
    Prefer FractionalBacktest (backtesting >= 0.6): BTC trades above the default
    cash level, so whole-unit sizing floors every fractional position to 0 units
    ('size must be a positive fraction...' on every order). Fractional units
    (uBTC) make position sizing behave like real crypto trading.
    """
    try:
        import os
        os.environ.setdefault("TQDM_DISABLE", "1")  # no progress bars in logs
        try:
            from backtesting.lib import FractionalBacktest
            return FractionalBacktest
        except ImportError:
            from backtesting import Backtest
            return Backtest
    except ImportError as e:
        raise BacktestEnvironmentError(
            "The 'backtesting' package is not installed in this interpreter. "
            "Neither the conda env 'tflow' nor the repo .venv currently provides it "
            "(see docs/QUANT_RESEARCH_FACTORY_PLAN.md section 7). Install it, e.g.: "
            "pip install -r requirements.txt"
        ) from e


def load_strategy_module(module_path: Path):
    """Import a generated strategy module from a file path and check the contract."""
    module_path = Path(module_path)
    if not module_path.exists():
        raise FileNotFoundError(f"strategy module not found: {module_path}")
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    if not hasattr(module, "STRATEGY_CLASS"):
        raise StrategyContractError(f"{module_path}: missing STRATEGY_CLASS")
    if not hasattr(module, "PARAMS") or not isinstance(module.PARAMS, dict):
        raise StrategyContractError(f"{module_path}: missing PARAMS dict")
    for name in module.PARAMS:
        if not hasattr(module.STRATEGY_CLASS, name):
            raise StrategyContractError(
                f"{module_path}: PARAMS key '{name}' is not a class attribute of STRATEGY_CLASS"
            )
    return module


def split_data(df: pd.DataFrame, config: Optional[Dict] = None
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/validation/test split. Never shuffled."""
    config = config or load_factory_config()
    fr = config["splits"]
    total = fr["train"] + fr["validation"] + fr["test"]
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split fractions must sum to 1.0, got {total}")
    n = len(df)
    i1 = int(n * fr["train"])
    i2 = int(n * (fr["train"] + fr["validation"]))
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def effective_commission(config: Dict, cost_multiplier: float = 1.0) -> float:
    """Flat commission + slippage (bps) folded into backtesting.py's commission."""
    costs = config["costs"]
    return (costs["commission"] + costs["slippage_bps"] / 10_000.0) * cost_multiplier


def run_backtest(module_path: Path, df: pd.DataFrame,
                 params: Optional[Dict[str, Any]] = None,
                 config: Optional[Dict] = None,
                 cost_multiplier: float = 1.0) -> Tuple[Any, Dict[str, Any]]:
    """
    Run one backtest. Returns (raw stats, metrics dict). Raises on structural
    failure - the caller records failures in the experiment store.
    """
    config = config or load_factory_config()
    Backtest = _import_backtesting()
    module = load_strategy_module(Path(module_path))

    if len(df) == 0:
        raise ValueError("empty data slice passed to run_backtest")

    import warnings
    bt_kwargs = dict(
        cash=config["costs"]["cash"],
        commission=effective_commission(config, cost_multiplier),
        margin=config["costs"]["margin"],
        exclusive_orders=config["costs"]["exclusive_orders"],
    )
    if Backtest.__name__ == "FractionalBacktest":
        bt_kwargs["fractional_unit"] = 1e-06  # micro-units: BTC price >> cash otherwise
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="backtesting")
        try:  # backtesting >= 0.6: close dangling trades at slice end so metrics are complete
            bt = Backtest(df, module.STRATEGY_CLASS, finalize_trades=True, **bt_kwargs)
        except TypeError:  # backtesting 0.3.x has no finalize_trades
            bt = Backtest(df, module.STRATEGY_CLASS, **bt_kwargs)
        stats = bt.run(**(params or {}))
    metrics = extract_metrics(stats, benchmark_close=df["Close"])
    metrics["cost_multiplier"] = cost_multiplier
    metrics["commission_effective"] = effective_commission(config, cost_multiplier)
    return stats, metrics


def apply_spec_cost_overrides(config: Dict, spec) -> Dict:
    """
    Per-spec structural/cost overrides from spec.costs_to_model. Costs may only
    be overridden UPWARD (more conservative). exclusive_orders may be set false
    to allow pyramiding/grid strategies (multiple concurrent entries).
    """
    import copy
    cfg = copy.deepcopy(config)
    overrides = spec.costs_to_model or {}
    if "exclusive_orders" in overrides:
        cfg["costs"]["exclusive_orders"] = bool(overrides["exclusive_orders"])
    if "commission" in overrides:
        cfg["costs"]["commission"] = max(cfg["costs"]["commission"],
                                         float(overrides["commission"]))
    if "slippage_bps" in overrides:
        cfg["costs"]["slippage_bps"] = max(cfg["costs"]["slippage_bps"],
                                           float(overrides["slippage_bps"]))
    return cfg


def config_hash(config: Dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]


def run_spec_backtests(spec, module_path: Path, store, code_id: str,
                       catalog: Optional[DataCatalog] = None,
                       config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Standard research run for a spec: train + validation slices on every dataset
    the spec requires. The TEST slice is NOT touched here - it is reserved for
    promotion time (promotion.py runs it once).

    Returns {dataset_id: {"train": metrics, "validation": metrics}}.
    All outcomes (including failures) are recorded in the store.
    """
    config = config or load_factory_config()
    catalog = catalog or DataCatalog(config)
    chash = config_hash(config)
    requirements = spec.data_requirements or {}
    dataset_ids = requirements.get("datasets") or _match_datasets(spec, catalog)
    if not dataset_ids:
        raise ValueError(
            f"no catalog datasets match spec '{spec.name}' "
            f"(instruments={spec.instruments}, timeframe={spec.timeframe}). "
            f"Run --mode catalog or add data."
        )

    results: Dict[str, Any] = {}
    for dataset_id in dataset_ids:
        df = catalog.require(dataset_id, requirements.get("min_bars"))
        train_df, val_df, _test_df = split_data(df, config)
        results[dataset_id] = {}
        for kind, slice_df in (("train", train_df), ("validation", val_df)):
            try:
                _, metrics = run_backtest(module_path, slice_df, config=config)
                store.record_experiment(
                    spec.spec_id, kind, True, code_id=code_id, dataset_id=dataset_id,
                    metrics=metrics, config_hash=chash,
                )
                results[dataset_id][kind] = metrics
            except Exception as e:  # recorded, then re-raised: failures must be visible
                store.record_experiment(
                    spec.spec_id, kind, False, code_id=code_id, dataset_id=dataset_id,
                    error=f"{type(e).__name__}: {e}", config_hash=chash,
                )
                raise
    return results


def run_holdout_test(spec, module_path: Path, store, code_id: str,
                     config: Optional[Dict] = None) -> Dict[str, Any]:
    """The once-only holdout run, invoked by promotion.py."""
    config = config or load_factory_config()
    catalog = DataCatalog(config)
    requirements = spec.data_requirements or {}
    dataset_ids = requirements.get("datasets") or _match_datasets(spec, catalog)
    out: Dict[str, Any] = {}
    for dataset_id in dataset_ids:
        df = catalog.require(dataset_id, requirements.get("min_bars"))
        _, _, test_df = split_data(df, config)
        _, metrics = run_backtest(module_path, test_df, config=config)
        store.record_experiment(
            spec.spec_id, "test", True, code_id=code_id, dataset_id=dataset_id,
            metrics=metrics, config_hash=config_hash(config),
        )
        out[dataset_id] = metrics
    return out


def _match_datasets(spec, catalog: DataCatalog) -> list:
    """
    Match catalog datasets to spec instruments + timeframe. Exact symbol match
    only (BTC-USD or bare BTC) - enriched variants like BTC-SIG never match
    implicitly; signal specs must pin them via data_requirements.datasets.
    """
    wanted = set()
    for i in spec.instruments:
        sym = i.upper().replace("/", "-")
        wanted.add(sym)
        wanted.add(sym.split("-")[0])
    matches = []
    for record in catalog.records.values():
        if not record.usable or record.timeframe != spec.timeframe:
            continue
        if record.symbol.upper() in wanted:
            matches.append(record.dataset_id)
    return sorted(matches)
