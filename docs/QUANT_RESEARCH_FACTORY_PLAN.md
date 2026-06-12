# QUANT RESEARCH FACTORY PLAN

Date: 2026-06-10. Implementation plan derived from the three audit documents. Phase-1 code ships with this commit.

---

## 1. Design goals

1. Replace "LLM optimizes return until 50%" with **hypothesis → spec → deterministic code → costed backtest → mechanical gates → robustness → human approval**.
2. Every artifact versioned, every run recorded (including failures and rejections).
3. No component can place a live trade. Paper is the ceiling until manual approval.
4. Reuse existing conventions: ModelFactory for LLM, backtesting.py + talib/pandas_ta for backtests, standalone argparse agents in `src/agents/`, outputs in `src/data/`, files <800 lines.

## 2. Architecture

```
ideas (text/PDF/results/hypotheses)
   │  research ingestion (quant_research_agent --mode research-cycle)
   ▼
StrategySpec (structured, validated, fingerprinted, deduped)      src/research/strategy_spec.py
   │  codegen (LLM via ModelFactory) + static review               src/agents/strategy_review_agent.py
   ▼
strategy module (deterministic backtesting.py Strategy)           src/data/research_factory/strategies/
   │  data catalog check (refuse if insufficient)                  src/research/data_catalog.py
   ▼
backtest runner (costs, train/val/test split)                     src/research/backtest_runner.py
   │  metrics (full institutional set)                             src/research/metrics.py
   ▼
validation gates (configs/validation_gates.yaml)                  src/research/validation_gates.py
   │  robustness (walk-forward, MC, param-neighborhood, cost stress) src/research/robustness.py
   ▼
experiment store (SQLite, all runs)                               src/research/experiment_store.py
   │  leaderboard (composite score)                                src/research/leaderboard.py
   ▼
promotion state machine (manual approval beyond paper_candidate)  src/research/promotion.py
   │  reports                                                      src/research/report_writer.py
   ▼
src/strategies/custom/ wrapper + metadata (status-gated)
   │
brokers: PaperBroker (default) / Hyperliquid adapter (disabled) / IBKR + Solana-DEX stubs   src/brokers/
```

## 3. Components shipped in phase 1

| File | Purpose | Notes |
|---|---|---|
| `configs/research_factory.yaml` | paths, data dirs, split fractions, cost model defaults, interpreter, LLM models per role | single source of truth |
| `configs/validation_gates.yaml` | every promotion gate, editable | defaults per RISK_AND_OVERFITTING_AUDIT.md §D |
| `configs/broker_profiles.yaml` | per-venue fees/limits/paper settings; live disabled | |
| `src/research/strategy_spec.py` | StrategySpec dataclass (all required fields incl. hypothesis, invalidation rules, forbidden assumptions), validation, semantic fingerprint, JSON I/O | |
| `src/research/data_catalog.py` | scan/validate OHLCV CSVs (columns, monotonic time, gaps, range), catalog JSON, `require()` refuses insufficient data, validation report | |
| `src/research/experiment_store.py` | SQLite: ideas, specs, code versions, experiments (all outcomes), gate results, promotions, rejections | |
| `src/research/backtest_runner.py` | load strategy module, run backtesting.py with standardized costs on train/val/test slices; subprocess optional; fails clearly if backtesting missing | |
| `src/research/metrics.py` | full metric extraction from backtesting.py stats/trades: Sortino, Calmar, PF, expectancy, worst day/week/month, DD duration, exposure, turnover, fees, benchmark corr | |
| `src/research/validation_gates.py` | mechanical gate evaluation → pass/fail + reasons | |
| `src/research/robustness.py` | walk-forward folds, parameter neighborhood, Monte Carlo reshuffle, bootstrap, cost stress; seeded | |
| `src/research/leaderboard.py` | composite score (penalizes DD, low trades, OOS decay, fragility, concentration, cost sensitivity); CSV outputs | |
| `src/research/promotion.py` | status machine: research_only → paper_candidate → paper_active → live_candidate → live_approved → retired; writes `src/strategies/custom/` wrapper + metadata; live transitions require explicit human token | |
| `src/research/report_writer.py` | daily_research_report.md, robustness_report.md, overfitting_report.md, broker_readiness_report.md, paper_trading_report.md | |
| `src/agents/quant_research_agent.py` | CLI orchestrator: audit / catalog / research-cycle / backtest-spec / backtest-folder / leaderboard / promote / report | uses ModelFactory |
| `src/agents/strategy_review_agent.py` | static lookahead/bias checks (hard) + LLM code review (advisory) | |
| `src/brokers/base.py` | BrokerAdapter ABC: get_account, get_positions, get_open_orders, get_market_data, place_order, cancel_order, close_position, get_fills, get_fees, get_margin, health_check | |
| `src/brokers/paper.py` | PaperBroker: simulated fills (slippage+fees from broker profile), persistent JSON state, kill switch, daily-loss/exposure limits | |
| `src/brokers/hyperliquid_adapter.py` | wraps nice_funcs_hyperliquid read paths; order methods hard-disabled unless profile `live_enabled` AND env confirmation | |
| `src/brokers/ibkr_adapter_stub.py`, `src/brokers/solana_dex_adapter_stub.py` | interface stubs, NotImplementedError with guidance | |

## 4. LLM role boundaries (enforced structurally)

LLM is invoked only in: idea→spec extraction, spec→code generation, debug-fix loop (bounded), advisory code review, report prose. LLM output never: sets gate thresholds, marks gates passed, changes status, touches broker code, or modifies an existing strategy version in place (new version + full re-validation required). Regime/allocation adaptivity is deferred to a later phase and will operate only within config bounds with versioned changes.

## 5. CLI

```
python src/agents/quant_research_agent.py --mode audit                 # summarize store + legacy results
python src/agents/quant_research_agent.py --mode catalog              # build/validate data catalog
python src/agents/quant_research_agent.py --mode research-cycle [--ideas-file F] [--max-ideas N]
python src/agents/quant_research_agent.py --mode backtest-spec --spec PATH
python src/agents/quant_research_agent.py --mode backtest-folder --folder PATH
python src/agents/quant_research_agent.py --mode leaderboard
python src/agents/quant_research_agent.py --mode promote --strategy-id ID --to paper_candidate
python src/agents/quant_research_agent.py --mode report
```

## 6. Phasing

- **Phase 1 (this commit)**: everything in §3. Research → paper_candidate fully functional once the Python environment is repaired and data is added.
- **Phase 2**: paper trading loop (PaperBroker driven by promoted strategies on live data feeds), paper_trading_report.md populated from fills, funding/liquidation/OI signal strategies using existing history CSVs.
- **Phase 3**: broker adapters hardened (Hyperliquid paper→live behind manual approval; IBKR via ib_insync when account exists; Solana DEX route when API chosen), regime classifier within bounds.
- **Phase 4**: portfolio layer (allocation across promoted strategies, correlation budget).

## 7. Blockers you must resolve (cannot be coded around)

1. **Python environment**: no interpreter on this machine has `backtesting` installed; conda env `tflow` does not exist (`~/miniforge3/envs` empty); repo `.venv` (Python 3.13.0) has zero packages. Fix: either recreate tflow (`conda create -n tflow python=3.11` + `pip install -r requirements.txt`) or install `requirements-py313.txt` into `.venv`. The factory reads `python_executable` from `configs/research_factory.yaml`.
2. **Data**: one stale 2023 BTC bull dataset. Need multi-asset, multi-timeframe, multi-regime OHLCV (Hyperliquid `_get_ohlcv` can fetch crypto; equities/futures need a source decision for IBKR phase).
3. **Manual approvals**: any transition beyond paper_candidate requires your sign-off (promotion.py enforces an `--approved-by` flag plus status preconditions).
