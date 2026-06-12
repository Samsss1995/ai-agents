# RBI RESULTS AUDIT

Date: 2026-06-10. Read-only audit of all historical RBI/backtest artifacts under `src/data/rbi*`.

---

## 1. Headline numbers

| Metric | Value |
|---|---|
| Generated backtest .py files (all variants, Mar 2025 – Oct 2025) | ~4,770 |
| Files with hardcoded dead `/Users/md/Dropbox/...` data paths | 3,699 (**cannot re-run on this machine**) |
| Strategies ever logged to a stats CSV as "working" | **2** |
| Strategies that beat buy-and-hold | **0** |
| Datasets ever used | 1 (`src/data/rbi/BTC-USD-15m.csv`) |
| Out-of-sample / walk-forward tests ever run | 0 |
| Costs modeled beyond flat commission | none (no spread, slippage, funding) |

>99.9% of generation compute produced no recorded result. The pipeline records winners only above a 1% return threshold (`SAVE_IF_OVER_RETURN`), so researcher bias is structurally built in: losing runs mostly vanish.

## 2. The only stats file: `src/data/rbi_pp/backtest_stats.csv` (2 rows)

| Strategy | Return % | B&H % | MaxDD % | Sharpe | Sortino | Verdict |
|---|---|---|---|---|---|---|
| DonchianAscent (T04, opt v10) | 3.04 | 127.77 | -9.53 | 0.28 | 0.64 | **REJECT** |
| MomentumSqueeze (T08, opt v7) | 1.06 | 127.77 | -1.13 | 1.48 | blank (=inf) | **REJECT** |

**DonchianAscent** — code is honest (correct `self.I()` usage, `[-1]/[-2]` indexing, next-bar fills, no lookahead found), but the result is bad: 3% in 323 days vs 127.8% B&H, exposure 2.5%, 73 trades, 31.5% win rate, SQN 0.26, profit factor 1.13, max DD 3× its return. Long-only with SMA200 bull filter on an uninterrupted bull dataset — a degraded long proxy. 10 optimizer iterations on the same in-sample data, comments literally say "pushing towards 50% target".

**MomentumSqueeze** — statistically meaningless: **1 trade** in 11 months (exposure 0.055%), win rate 100%, Sortino = inf, profit factor NaN. The Sharpe 1.48 is a near-zero-volatility artifact. Code flaw: emulates SL/TP by closing on the next bar's close after the level is pierced — intra-bar stop fills not modeled; live losses would exceed backtest losses. Promoted to "working" because no minimum-trade gate exists.

Also: the two rows used **different commissions** (0.002 vs 0.001) — the CSV is not even internally comparable.

## 3. Funnels per variant

- **rbi_pp** 10_23: 10 research → 50 optimized → 0 working (best embedded returns +0.71% … -0.72%). 10_24: 4 → 1 PKG → 0. 10_25: 14 research → 105 optimizer variants → 2 "working" (the table above).
- **rbi_pp_multi** 10_26: 20 ideas → 20 backtests → 3 PKG → **0 executed**. The run died: `EnvironmentLocationNotFound: Not a conda environment: /Users/sam/miniforge3/envs/tflow` (see `execution_results/T07_UnknownStrategy_225255.json`). 3 of 20 names are LLM template-following failures shipped as artifacts: `T01_Uniquetwo-wordname`, `T16_Uniquetwo-wordnamebasedonthestrategysspecificapproach`, `T07_UnknownStrategy`. Duplicates: StochasticReversal (T05+T06), OscillatorFilter (T04+T13).
- **rbi_v2** 07_24: 19 research → 87 "final", 109 execution results, no stats recorded. 10_20: 7 → 30 final, same.
- **rbi_v3** 10_20 BEST files: -2.0%, +0.69%, -1.19%, +2.25%, +0.71%, +0.37%. 10_23: **FractalCascade +7.70%** (best raw number in the whole archive, still 16× worse than B&H), GoldenCrossover -0.008%.
- **rbi legacy** (Mar–Aug 2025, ~3,900 files across 29 date folders): `FINAL_WINNING_STRATEGIES/FINAL_STATS_REPORT.md` records SimpleMomentumCross **-92.43%** (Sharpe -47.31) and RSIMeanReversion **-92.58%** (Sharpe -77.63), 8 other "strategies" produced 0 trades, "Successful Strategies: 0/3". The "WINNING" label is aspirational text, contradicted by the recorded numbers in the same folder.

## 4. Ideas inventory

- `src/data/rbi/strategy_ideas.csv`: 4,670 LLM-generated idea rows; many non-backtestable ("sentiment analysis of social media", "short gamma positions").
- Idea generation collapses onto ~6 volatility-squeeze archetypes with thesaurus names: 54× VolumetricBreakout, 46× VoltaicSqueeze, 36× VolatilityBreakout, 34× VolatilitySurge, 33× VoltaicBreakout, 28× VolatilityDivergence. Massive duplicate logic under different names.

## 5. Cost modeling reality

Across 4,770 files: commission `0.002` in 1,069, `0.001` in 352, zero/near-zero in 19, malformed (`commission=.`) in 5, unspecified in the rest. No slippage, spread, funding, borrow, or execution-delay modeling anywhere. `exclusive_orders=True` in 691. `cash=1_000_000` in ~1,580; pathological `cash=1` in 12 (guaranteed margin failure) and truncated `cash=1_000_` in 5.

## 6. Data and validation reality

- Single dataset: BTC-USD 15m, 2023-01-01 → 2023-11-20 (323 days, 31,066 rows, 2 minor gaps, one continuous bull regime +127.8%). Ends 19 months before today.
- 100% in-sample. The v3/pp/pp_multi optimization loop iterates up to 10× against the SAME data targeting `TARGET_RETURN=50%` — explicit metric-targeting on in-sample data.
- No train/test split, no walk-forward, no parameter sensitivity, no Monte Carlo, no multi-asset run ever completed (the multi_data_tester import path doesn't exist on this machine).
- Lookahead guards: none in prompts or code. Sampled winners happened to be clean, but nothing prevents leakage.
- Survivorship/selection bias: only runs >1% return logged to CSV; failures and losers mostly unrecorded.

## 7. Dispositions

**Permanently reject**
- T08_MomentumSqueeze (1 trade, meaningless stats, optimistic stop model).
- Everything in `src/data/rbi/FINAL_WINNING_STRATEGIES/` (-92% recorded results).
- rbi_pp_multi placeholder strategies (T01/T07/T16 names).
- All legacy rbi date-folder output (no recorded results, dead paths, unverifiable).

**Conditionally re-test under the new factory (low expectations, must beat long-only baseline and pass gates)**
- T04_DonchianAscent (`src/data/rbi_pp/10_25_2025/backtests_working/`) — clean code, real trade count; must be tested on bear/chop data and OOS.
- FractalCascade_BEST_7.70pct (`src/data/rbi_v3/10_23_2025/backtests_optimized/`) — trade count unverified; re-run with costs and OOS before any conclusion.

**Structural conclusions feeding the factory design**
1. Minimum-trade, vs-buy-and-hold, OOS, and cost-standardization gates must be mechanical, not optional.
2. All runs (including failures and rejects) must be stored — measure researcher bias.
3. Idea dedup must happen at spec level (semantic fingerprint), not name level.
4. Optimization toward a return target must be replaced by hypothesis-driven specs + robustness testing.
5. Data must be extended (more assets, timeframes, regimes, recency) and cataloged with quality checks before any new conclusions are drawn.
6. The execution environment must be fixed (no working interpreter with `backtesting` exists on this machine today).
