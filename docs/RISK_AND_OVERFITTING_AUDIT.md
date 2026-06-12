# RISK AND OVERFITTING AUDIT

Date: 2026-06-10. Companion to SYSTEM_ARCHITECTURE_AUDIT.md and RBI_RESULTS_AUDIT.md.

---

## A. Overfitting mechanisms present in the current RBI system

| # | Mechanism | Where | Severity |
|---|---|---|---|
| 1 | Metric-targeting optimization loop: LLM rewrites strategy up to 10× until `return >= TARGET_RETURN (50%)` on the SAME data | rbi_agent_v3.py:983-1043, pp, pp_multi | Critical |
| 2 | Single dataset, single regime: BTC 15m, Jan–Nov 2023 uninterrupted bull (+127.8%) | all generated backtests | Critical |
| 3 | Zero out-of-sample/walk-forward/holdout anywhere | whole pipeline | Critical |
| 4 | Selection bias in record-keeping: only runs >1% return logged (`SAVE_IF_OVER_RETURN`); losers/failures discarded | rbi_agent_pp.py:122 | High |
| 5 | No minimum-trade gate: a 1-trade strategy was promoted to "working" with Sharpe 1.48 | backtest_stats.csv row 2 | High |
| 6 | No vs-benchmark gate: 3% return accepted while B&H did 127.8% | pipeline thresholds | High |
| 7 | Non-comparable costs: commission varies 0 → 0.002 across runs; no slippage/spread/funding ever | 4,770 generated files | High |
| 8 | No lookahead guards in prompts or post-generation static checks | all BACKTEST_PROMPTs | High |
| 9 | Optimistic fill modeling: SL/TP emulated by closing at next bar close after level pierced (no intra-bar stop fills) | e.g. T08_MomentumSqueeze | Medium |
| 10 | Idea-generator mode collapse: hundreds of near-identical volatility-squeeze strategies under different names → multiple-comparisons problem (run enough seeds, something passes by luck) | strategy_ideas.csv | Medium |
| 11 | No parameter-sensitivity testing: winners are single points in parameter space | whole pipeline | Medium |
| 12 | LLM nondeterminism + cache-busting nonce: same idea yields different code each run; no seeds, no reproducibility | base_model.py:40-44 | Medium |

## B. Live-trading risk gaps (must be closed before any paper→live promotion)

1. **No pre-trade checks**: `ai_entry()` (Solana `nice_funcs.py:1064`, HL `nice_funcs_hyperliquid.py:844`) never verifies balance, available margin, or liquidation distance. Re-entries can be attempted without checking what is available.
2. **LLM can veto hard risk limits**: `risk_agent.handle_limit_breach()` (risk_agent.py:453-551) lets an LLM answer "HOLD_POSITIONS" to override MAX_LOSS_USD. Hard limits must be mechanical.
3. **skip_preflight=True** on Solana sends (nice_funcs.py:269,317).
4. **No atomicity/rollback**: `chunk_kill()` does 3 sequential market sells; partial failure leaves a half-closed position with no recovery.
5. **Leverage without liquidation math**: `set_leverage()` 1-50x with no liquidation-price computation.
6. **No paper mode**: every code path that can trade, trades live.
7. **No per-strategy loss limits, no correlation/concentration limits, no daily-loss kill switch independent of the LLM.**
8. **Fixed ~2% slippage tolerance** regardless of liquidity/volatility.

## C. Anti-overfitting controls implemented in the new factory

Configured in `configs/validation_gates.yaml` and `configs/research_factory.yaml`; enforced mechanically by `src/research/validation_gates.py` and `src/research/robustness.py`. None of these are LLM-overridable.

1. **Chronological train/validation/test split** (default 60/20/20). Gates are evaluated on validation; test (holdout) is touched once, at promotion time only. In-sample vs OOS metrics are labeled and stored separately.
2. **Walk-forward validation**: rolling anchored windows (default 4 folds); promotion requires OOS-fold mean not collapsing vs in-sample (default: OOS ≥ 50% of IS return, and positive).
3. **Parameter neighborhood testing**: each numeric parameter perturbed ±10%/±20%; strategy rejected if the neighborhood mean degrades beyond tolerance (fragile point-solutions die).
4. **Cost stress test**: re-run at 1.5× and 2× modeled costs; must remain profitable at 1.5×.
5. **Monte Carlo trade reshuffling + bootstrap**: resample trade sequence (default 1,000 draws, seeded); require 5th-percentile equity path drawdown within limits and positive median.
6. **Minimum trade count**: default 100 for intraday/short-term specs; spec may declare `low_frequency: true` with explicit lower bound — recorded, never silent.
7. **Single-trade / single-asset dependence checks**: remove best trade → must stay profitable; per-asset contribution recorded.
8. **Benchmark gates**: must be compared against buy-and-hold and (for long-only crypto in bull data) must add risk-adjusted value (Sharpe/Calmar gates), not raw return.
9. **All runs persisted** — including failures and rejections — in the experiment store (`src/data/research_factory/research.db`). Rejected runs feed `rejected_strategies.csv`; researcher bias becomes measurable.
10. **Determinism**: generated strategy code must be pure-function-of-data with fixed parameters; seeds recorded; every artifact content-hashed and versioned (spec_hash, code_hash, data_hash, config_hash stored per experiment).
11. **Static lookahead checks** before execution (`strategy_review_agent.py`): scans generated code for `.shift(-`, centered rolling windows, `df[...]` future indexing, use of full-dataset statistics inside signals (e.g., normalizing by global max), same-bar fill assumptions, and `random`/`datetime.now` in signal logic. LLM review is advisory on top; static check failures are hard rejects.
12. **Duplicate-spec rejection**: semantic fingerprint of StrategySpec (entry/exit/indicator canonicalized) checked against the store before any compute is spent.
13. **No return-targeting optimization**: parameter selection happens only inside walk-forward folds on train data; nothing iterates against validation/test metrics.

## D. Default promotion gates (configurable, in `configs/validation_gates.yaml`)

- ≥100 trades (unless spec declares low-frequency, then ≥30 and flagged)
- positive out-of-sample return (validation + walk-forward OOS folds)
- profit factor > 1.20 after costs
- max drawdown < 25%
- Sharpe > 1.0, Sortino > 1.5, Calmar > 0.75 after costs
- walk-forward OOS ≥ 50% of in-sample return
- survives 1.5× cost stress
- parameter neighborhood mean within 40% of point estimate
- not dependent on a single trade (top-trade removal still profitable)
- static lookahead checks pass
- spec contains an explicit market hypothesis (free text required, enforced at spec validation)

## E. Honest limitations

- Walk-forward on 323 days of 15m data gives short folds; conclusions remain weak until the data catalog is extended (more assets, more years, bear/chop regimes). The data catalog refuses runs below configurable minimum bars per fold rather than silently proceeding.
- Monte Carlo trade reshuffling assumes trade independence; serial correlation in trades understates tail risk. Reported as such in robustness_report.md.
- Costs are still models (flat commission + bps slippage + optional funding drag); real microstructure (queue position, partial fills, latency) is not simulated. Paper trading exists to close that gap before live.
