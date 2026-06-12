"""
Quant Research Agent - CLI orchestrator for the research factory.

Modes:
  audit            summarize the experiment store + legacy RBI results
  catalog          scan/validate OHLCV data, write data_validation_report.md
  research-cycle   ideas file -> specs -> code -> costed backtests -> gates -> robustness
  backtest-spec    run the full pipeline for one StrategySpec JSON file
  backtest-folder  run backtest-spec for every *.json spec in a folder
  leaderboard      build composite-score leaderboard + rejected/promoted CSVs
  promote          move a strategy through the status state machine
  report           write daily research / overfitting / broker readiness reports

Examples:
  python src/agents/quant_research_agent.py --mode catalog
  python src/agents/quant_research_agent.py --mode research-cycle --ideas-file src/data/rbi_pp/ideas.txt --max-ideas 3
  python src/agents/quant_research_agent.py --mode promote --strategy-id spec_ab12cd34ef --to paper_candidate

This agent NEVER places live trades. Promotion beyond paper_candidate requires
--approved-by (manual human approval).
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

try:
    from termcolor import cprint
except ImportError:  # minimal envs: plain print, no color
    def cprint(text, *_args, **_kwargs):
        print(text)

from src.research.factory_config import (
    load_factory_config, resolve_path, factory_root,
)


# ---------------------------------------------------------------- pipeline core

def _backtest_with_debug(spec, module_path: Path, store, code_id: str,
                         config: Dict) -> Optional[Dict]:
    """Train+validation backtests with a bounded LLM debug loop. Returns results or None."""
    from src.research.backtest_runner import run_spec_backtests, BacktestEnvironmentError
    from src.research import codegen

    max_iters = config["llm"]["max_debug_iterations"]
    last_error = ""
    for attempt in range(max_iters + 1):
        try:
            results = run_spec_backtests(spec, module_path, store, code_id, config=config)
            total_trades = sum(
                (m.get("n_trades") or 0)
                for ds in results.values() for m in ds.values()
            )
            if total_trades > 0:
                return results
            # legacy pipeline's classic failure: code runs but never triggers.
            # Treat as a debuggable defect, not a result.
            last_error = (
                "ZERO TRADES: the strategy executed without errors but produced 0 trades "
                "across the train and validation slices (~11 months of 15m bars). The entry "
                "conditions never fire. Typical causes: mutually contradictory filters "
                "(e.g. requiring a breakout bar while also requiring volatility to stay low "
                "on that same bar), thresholds too strict for this timeframe, warmup longer "
                "than the data slice, or position size computed in wrong units "
                "(backtesting.py: size must be a fraction 0<size<1 of equity or a positive "
                "integer unit count). Relax/decouple the conditions so the strategy trades "
                "while preserving the spec's hypothesis."
            )
            cprint(f"  zero trades (attempt {attempt + 1}/{max_iters + 1}) - sending back for fix",
                   "yellow")
        except BacktestEnvironmentError:
            raise  # environment problems are not fixable by an LLM
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}"
            cprint(f"  debug attempt {attempt + 1}/{max_iters + 1}: {type(e).__name__}: {e}",
                   "yellow")
        if attempt >= max_iters:
            break
        codegen.debug_fix(module_path, last_error, config)
        code_id = store.add_code_version(spec.spec_id, module_path,
                                         f"debug_fix_{attempt + 1}")
    store.record_rejection(spec.spec_id, "backtest",
                           f"failed after {max_iters} debug iterations: {last_error[:1000]}",
                           code_id=code_id)
    cprint(f"  REJECTED after {max_iters} debug iterations", "red")
    return None


def run_pipeline_for_spec(spec, store, config: Dict,
                          module_path: Optional[Path] = None) -> Dict:
    """spec (already registered in store) -> code -> review -> backtests -> robustness -> gates."""
    from src.research import codegen
    from src.research.backtest_runner import (
        run_backtest, split_data, apply_spec_cost_overrides,
    )
    from src.research.data_catalog import DataCatalog
    from src.research.metrics import portfolio_metrics
    from src.research.robustness import full_robustness_battery
    from src.research.validation_gates import evaluate_gates, hard_gates_passed, summarize
    from src.research.report_writer import write_robustness_report
    from src.agents.strategy_review_agent import review_file

    strategies_dir = resolve_path(config["paths"]["strategies_dir"])

    # 1. code
    if module_path is None:
        existing = store.latest_code(spec.spec_id)
        if existing and Path(existing["file_path"]).exists():
            module_path = Path(existing["file_path"])
            code_id = existing["code_id"]
            model_used = existing["model_used"]
        else:
            module_path, model_used = codegen.spec_to_code(spec, strategies_dir, config)
            code_id = store.add_code_version(spec.spec_id, module_path, model_used)
    else:
        code_id = store.add_code_version(spec.spec_id, module_path, "manual")
    cprint(f"  code: {module_path}", "cyan")

    # 2. static review (hard gate input)
    review = review_file(module_path, use_llm=False, spec_id=spec.spec_id, store=store)
    if not review["static_clean"]:
        store.record_rejection(spec.spec_id, "static_review",
                               json.dumps(review["findings"])[:1000], code_id=code_id)
        cprint("  REJECTED: static review found hard issues "
               "(lookahead/nondeterminism/contract)", "red")
        return {"spec_id": spec.spec_id, "status": "rejected_static_review",
                "findings": review["findings"]}

    # 3. train + validation backtests (bounded debug loop), spec cost overrides applied
    spec_config = apply_spec_cost_overrides(config, spec)
    results = _backtest_with_debug(spec, module_path, store,
                                   code_id, spec_config)
    if results is None:
        return {"spec_id": spec.spec_id, "status": "rejected_backtest_failure"}
    code_id = store.latest_code(spec.spec_id)["code_id"]

    # 4. robustness battery on the first matched dataset
    catalog = DataCatalog(spec_config)
    dataset_ids = list(results)
    dataset_id = dataset_ids[0]
    df = catalog.require(dataset_id)
    _, val_df, _ = split_data(df, spec_config)
    val_stats, val_metrics = run_backtest(module_path, val_df, config=spec_config)
    try:
        robustness = full_robustness_battery(module_path, df, val_stats, spec_config)
        store.record_experiment(spec.spec_id, "robustness", True, code_id=code_id,
                                dataset_id=dataset_id, metrics=robustness,
                                seed=spec_config["monte_carlo"]["seed"])
    except Exception as e:
        store.record_experiment(spec.spec_id, "robustness", False, code_id=code_id,
                                dataset_id=dataset_id, error=f"{type(e).__name__}: {e}")
        robustness = {}
        cprint(f"  robustness battery failed: {e}", "yellow")

    # 5. portfolio aggregation: multi-instrument specs are judged on the
    # equal-weight portfolio of their per-instrument validation curves
    train_metrics = results[dataset_id].get("train", {})
    if len(dataset_ids) >= 2:
        stats_list = [val_stats]
        for ds in dataset_ids[1:]:
            ds_df = catalog.require(ds)
            _, ds_val, _ = split_data(ds_df, spec_config)
            s, _ = run_backtest(module_path, ds_val, config=spec_config)
            stats_list.append(s)
        port = portfolio_metrics(stats_list)
        if "error" not in port:
            store.record_experiment(spec.spec_id, "validation", True, code_id=code_id,
                                    dataset_id="PORTFOLIO", metrics=port)
            val_metrics = port
            train_metrics = {"n_trades": sum(
                (results[ds].get("train", {}).get("n_trades") or 0) for ds in dataset_ids)}
            cprint(f"  portfolio ({port['n_instruments']} instruments): "
                   f"ret={port['return_pct']}% sharpe={port['sharpe']} "
                   f"trades={port['n_trades']}", "cyan")

    # 6. gates
    gate_results = evaluate_gates(spec, val_metrics, train_metrics, robustness,
                                  static_review_clean=True)
    store.record_gate_results(spec.spec_id, code_id,
                              [g.to_dict() for g in gate_results])
    passed = hard_gates_passed(gate_results)
    cprint("\n" + summarize(gate_results), "white")
    if robustness:
        write_robustness_report(spec.spec_id, robustness, store, config)
    if passed:
        cprint(f"  ALL HARD GATES PASSED - eligible for: "
               f"--mode promote --strategy-id {spec.spec_id} --to paper_candidate", "green")
    else:
        failed = [g.name for g in gate_results if g.hard and not g.passed]
        store.record_rejection(spec.spec_id, "gates", f"failed: {failed}", code_id=code_id)
        cprint(f"  gates failed: {failed}", "red")
    return {"spec_id": spec.spec_id, "status": "gates_passed" if passed else "gates_failed",
            "validation": val_metrics}


# ---------------------------------------------------------------- modes

def mode_audit(config: Dict) -> None:
    from src.research.experiment_store import ExperimentStore
    store = ExperimentStore()
    cprint("\nExperiment store summary:", "cyan")
    print(json.dumps(store.summary(), indent=2))

    legacy = resolve_path("src/data/rbi_pp/backtest_stats.csv")
    if legacy.exists():
        cprint("\nLegacy rbi_pp/backtest_stats.csv:", "cyan")
        print(legacy.read_text())
    cprint("Full historical analysis: docs/RBI_RESULTS_AUDIT.md", "cyan")


def mode_catalog(config: Dict) -> None:
    from src.research.data_catalog import DataCatalog
    catalog = DataCatalog(config)
    records = catalog.scan()
    report = catalog.write_report()
    usable = [r for r in records if r.usable]
    cprint(f"\n{len(records)} datasets scanned, {len(usable)} usable", "cyan")
    for r in records:
        color = "green" if r.usable else "red"
        cprint(f"  {r.dataset_id}: {r.symbol} {r.timeframe} {r.rows} bars "
               f"[{r.start} -> {r.end}] {'OK' if r.usable else r.issues}", color)
    cprint(f"report: {report}", "cyan")


def mode_research_cycle(config: Dict, ideas_file: Optional[str], max_ideas: int,
                        timeframe: str = "15m",
                        instruments: Optional[list] = None,
                        reprocess_ideas: bool = False,
                        datasets: Optional[list] = None) -> None:
    from src.research import codegen
    from src.research.experiment_store import ExperimentStore

    ideas_path = resolve_path(ideas_file or "src/data/research_factory/ideas.txt")
    if not ideas_path.exists():
        raise FileNotFoundError(
            f"ideas file not found: {ideas_path}. Pass --ideas-file or create it "
            f"(one idea per line; lines starting with # ignored).")
    ideas = [ln.strip() for ln in ideas_path.read_text().splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    if max_ideas:
        ideas = ideas[:max_ideas]
    cprint(f"\nresearch cycle: {len(ideas)} ideas from {ideas_path}", "cyan")

    store = ExperimentStore()
    context = f"[{timeframe}|{','.join(instruments or ['default'])}]"
    for i, idea in enumerate(ideas):
        cprint(f"\n[{i + 1}/{len(ideas)}] {idea[:100]}", "cyan")
        tagged = f"{context} {idea}"  # same idea on a new timeframe/basket = new work
        seen = store.conn.execute(
            "SELECT idea_id, status FROM ideas WHERE text=?", (tagged,)).fetchone()
        if seen is not None and seen["status"] != "spec_failed" and not reprocess_ideas:
            cprint(f"  SKIPPED: idea already processed ({seen['idea_id']}, "
                   f"status={seen['status']})", "yellow")
            continue
        idea_id = seen["idea_id"] if seen is not None else store.add_idea(tagged, str(ideas_path))
        try:
            spec = codegen.idea_to_spec(idea, config, default_timeframe=timeframe,
                                        default_instruments=instruments)
            if datasets:
                from src.research.data_catalog import DataCatalog
                from src.research.signal_features import signal_columns_of
                catalog = DataCatalog(config)
                spec.data_requirements["datasets"] = datasets
                first = catalog.records.get(datasets[0])
                if first is None:
                    raise ValueError(f"--datasets entry '{datasets[0]}' not in catalog - "
                                     f"run --mode catalog first")
                spec.data_requirements["signal_columns"] = signal_columns_of(Path(first.path))
            store.add_spec(spec, idea_id)
            cprint(f"  spec: {spec.spec_id} '{spec.name}' ({spec.family})", "green")
        except ValueError as e:          # duplicate fingerprint
            store.record_rejection(None, "spec_dedup", str(e))
            store.set_idea_status(idea_id, "duplicate")
            cprint(f"  SKIPPED duplicate: {e}", "yellow")
            continue
        except Exception as e:
            store.record_rejection(None, "spec_generation", f"{type(e).__name__}: {e}")
            store.set_idea_status(idea_id, "spec_failed")
            cprint(f"  spec generation failed: {e}", "red")
            continue
        spec.save(factory_root(config) / "specs" / f"{spec.spec_id}.json")
        result = run_pipeline_for_spec(spec, store, config)
        store.set_idea_status(idea_id, result["status"])


def mode_backtest_spec(config: Dict, spec_path: str,
                       module: Optional[str] = None) -> None:
    from src.research.experiment_store import ExperimentStore
    from src.research.strategy_spec import StrategySpec

    spec = StrategySpec.load(Path(spec_path))
    errors = spec.validate()
    if errors:
        raise ValueError(f"spec invalid: {errors}")
    store = ExperimentStore()
    if spec.spec_id is None:
        try:
            store.add_spec(spec)
        except ValueError as e:
            existing = store.conn.execute(
                "SELECT spec_id FROM specs WHERE fingerprint=?",
                (spec.fingerprint(),)).fetchone()
            spec.spec_id = existing["spec_id"]
            cprint(f"  spec already registered as {spec.spec_id} ({e})", "yellow")
    run_pipeline_for_spec(spec, store, config,
                          module_path=Path(module) if module else None)


def mode_backtest_folder(config: Dict, folder: str) -> None:
    specs = sorted(resolve_path(folder).glob("*.json"))
    if not specs:
        raise FileNotFoundError(f"no *.json specs in {folder}")
    for p in specs:
        cprint(f"\n=== {p.name} ===", "cyan")
        try:
            mode_backtest_spec(config, str(p))
        except Exception as e:
            cprint(f"  failed: {type(e).__name__}: {e}", "red")


def mode_leaderboard(config: Dict) -> None:
    from src.research.leaderboard import build_leaderboard, write_leaderboard_files
    df = build_leaderboard(config=config)
    if len(df):
        print(df.to_string(index=False))
    else:
        cprint("no executed strategies yet", "yellow")
    paths = write_leaderboard_files(config=config)
    for k, v in paths.items():
        cprint(f"  {k}: {v}", "cyan")


def mode_promote(config: Dict, strategy_id: str, to_status: str,
                 approved_by: Optional[str], notes: str) -> None:
    from src.research.promotion import promote
    result = promote(strategy_id, to_status, approved_by=approved_by, notes=notes,
                     config=config)
    cprint(json.dumps(result, indent=2, default=str), "green")


def mode_report(config: Dict) -> None:
    from src.research.report_writer import (
        write_daily_research_report, write_overfitting_report,
        write_broker_readiness_report,
    )
    for fn in (write_daily_research_report, write_overfitting_report,
               write_broker_readiness_report):
        path = fn(config=config) if fn is not write_broker_readiness_report \
            else fn(config)
        cprint(f"  wrote {path}", "cyan")


# ---------------------------------------------------------------- entry

def main() -> None:
    parser = argparse.ArgumentParser(description="Quant Research Factory CLI")
    parser.add_argument("--mode", required=True,
                        choices=["audit", "catalog", "research-cycle", "backtest-spec",
                                 "backtest-folder", "leaderboard", "promote", "report"])
    parser.add_argument("--ideas-file", default=None)
    parser.add_argument("--max-ideas", type=int, default=0)
    parser.add_argument("--timeframe", default="15m",
                        help="target timeframe for generated specs (research-cycle)")
    parser.add_argument("--instruments", nargs="+", default=None,
                        help="target instruments for generated specs (research-cycle)")
    parser.add_argument("--reprocess-ideas", action="store_true",
                        help="re-run ideas already in the store (e.g. on a new timeframe); "
                             "spec fingerprint dedup still applies")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="pin specs to specific catalog datasets (e.g. BTC-SIG-1h); "
                             "their extra columns become signal_columns for codegen")
    parser.add_argument("--spec", default=None, help="StrategySpec JSON path")
    parser.add_argument("--module", default=None,
                        help="use this strategy module instead of LLM codegen (backtest-spec)")
    parser.add_argument("--folder", default=None, help="folder of StrategySpec JSONs")
    parser.add_argument("--strategy-id", default=None)
    parser.add_argument("--to", default=None, help="target status for promote")
    parser.add_argument("--approved-by", default=None,
                        help="manual approval (required beyond paper_candidate)")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    config = load_factory_config()

    if args.mode == "audit":
        mode_audit(config)
    elif args.mode == "catalog":
        mode_catalog(config)
    elif args.mode == "research-cycle":
        mode_research_cycle(config, args.ideas_file, args.max_ideas,
                            timeframe=args.timeframe, instruments=args.instruments,
                            reprocess_ideas=args.reprocess_ideas,
                            datasets=args.datasets)
    elif args.mode == "backtest-spec":
        if not args.spec:
            parser.error("--spec is required for backtest-spec")
        mode_backtest_spec(config, args.spec, module=args.module)
    elif args.mode == "backtest-folder":
        if not args.folder:
            parser.error("--folder is required for backtest-folder")
        mode_backtest_folder(config, args.folder)
    elif args.mode == "leaderboard":
        mode_leaderboard(config)
    elif args.mode == "promote":
        if not (args.strategy_id and args.to):
            parser.error("--strategy-id and --to are required for promote")
        mode_promote(config, args.strategy_id, args.to, args.approved_by, args.notes)
    elif args.mode == "report":
        mode_report(config)


if __name__ == "__main__":
    main()
