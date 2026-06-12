# SYSTEM ARCHITECTURE AUDIT

Date: 2026-06-10
Scope: read-only audit of /Users/sam/ai-agents (branch main, commit 07e1c17 + uncommitted changes to rbi_agent_pp_multi.py, model_factory.py, requirements.txt).
Auditor stance: institutional quant research engineering review.

---

## 1. Repository topline

| Item | Finding |
|---|---|
| Agents in src/agents/ | 48 Python files (excl. __init__, api.py, base_agent.py helpers) |
| Agents wired into main loop | 5 (risk, trading, strategy, copybot, sentiment) — **all disabled by default** in `src/main.py:29-38` |
| RBI pipeline variants | 6 (rbi_agent, v2, v2_simple, v3, pp, pp_multi) + batch_backtester |
| Strategy definitions | `src/strategies/` (BaseStrategy + 2 examples) |
| LLM abstraction | `src/models/model_factory.py` — only `claude` and `openai` registered; DeepSeek/Groq/Grok/Gemini hardcoded ad-hoc inside individual agents |
| Exchange abstraction | `src/exchange_manager.py` (~150 lines) wrapping Solana (nice_funcs) and Hyperliquid (nice_funcs_hyperliquid) |
| Paper/dry-run mode | **Does not exist anywhere** |
| Execution environment | **Broken**: conda env `tflow` does not exist on this machine; repo `.venv` is Python 3.13 with no packages installed (no backtesting, no talib, no pandas). No interpreter on this machine can currently run a backtest. |

---

## 2. Agent inventory (by category)

### Wired into the orchestrator (`src/main.py`, all `False` by default)
- `risk_agent.py` — portfolio circuit breaker (MAX_LOSS_USD / MAX_GAIN_USD / MINIMUM_BALANCE_USD), optional AI override of breaches.
- `trading_agent.py` — LLM (single model or 6-model swarm vote) → BUY/SELL/NOTHING → direct execution via `ai_entry()`/`chunk_kill()`. Supports Solana spot (Jupiter), Hyperliquid perps, Aster.
- `strategy_agent.py` — loads `src/strategies/custom/*`, calls `generate_signals()`, LLM evaluates EXECUTE/REJECT, executes immediately.
- `copybot_agent.py` — reads an external portfolio CSV (hardcoded `/Users/md/Dropbox/...` — broken on this machine).
- `sentiment_agent.py` — Twitter via twikit + HuggingFace sentiment.

### RBI / research (standalone)
`rbi_agent.py` (v1, codegen only, never executes), `rbi_agent_v2.py`, `rbi_agent_v2_simple.py` (POC), `rbi_agent_v3.py` (debug + optimize loops), `rbi_agent_pp.py` (5 threads + backtest_stats.csv), `rbi_agent_pp_multi.py` (18 threads + multi-dataset harness; most advanced; has uncommitted changes switching models from grok-4-fast-reasoning to claude-opus-4-1 and paths from /Users/md to /Users/sam), `rbi_batch_backtester.py`, `backtest_runner.py` (conda subprocess helper), `research_agent.py`.

### Market monitoring (standalone, log/announce + some can trade)
`whale_agent.py` (OI anomalies), `funding_agent.py`, `liquidation_agent.py`, `chartanalysis_agent.py` (vision), `fundingarb_agent.py`, `coingecko_agent.py`, `new_or_top_agent.py`, `listingarb_agent.py`.

### Content/social/other (irrelevant to quant factory; several broken by hardcoded paths)
chat_agent (+_og, +_ad), tweet_agent, tiktok_agent, clips_agent, realtime_clips_agent, shortvid_agent, phone_agent, stream_agent, video-related, focus_agent, compliance_agent, polymarket_agent, housecoin_agent, sniper_agent, solana_agent, tx_agent, million_agent, swarm_agent, code_runner_agent, demo_countdown, clean_ideas, example_unified_agent.

---

## 3. Main orchestration loop

`src/main.py:40-94`:
1. `ACTIVE_AGENTS` dict gates each agent (all False today → loop is a no-op).
2. Order: risk → trading → strategy (per token in `MONITORED_TOKENS`) → copybot → sentiment.
3. Sleep `SLEEP_BETWEEN_RUNS_MINUTES` (15 min default, `src/config.py:66`).
4. Exceptions logged, 60s sleep, continue. No state persistence between cycles beyond CSVs.

Risk agent runs first by design (risk-first philosophy) but is advisory in practice: with `USE_AI_CONFIRMATION=True` an LLM can override a breached hard limit (`risk_agent.py:453-551`, looks for "HOLD_POSITIONS" in the response). **A hard limit that an LLM can veto is not a hard limit.**

---

## 4. Data entry points

| Source | Used by | Path |
|---|---|---|
| BirdEye API | `nice_funcs.get_data()` (`nice_funcs.py:349`) | Solana OHLCV, cached to temp_data/ |
| Hyperliquid Info API | `nice_funcs_hyperliquid._get_ohlcv()` (line 466) | perp OHLCV 1m–1d |
| MoonDev API (`src/agents/api.py`) | funding/liquidation/whale agents | liquidations, funding, OI, copybot list |
| CoinGecko | coingecko/listingarb/new_or_top | token metadata |
| Static CSV | all RBI backtests | `src/data/rbi/BTC-USD-15m.csv` — the **only** backtest dataset that exists (2023-01-01 → 2023-11-20, 31,066 rows, one asset, one timeframe, one bull regime, 19 months stale) |
| Signal history CSVs | src/data/funding_history.csv, liquidation_history.csv, oi_history.csv, sentiment_history.csv | Oct-2025 onward only |

`src/data/ohlcv/` contains only `guidelines.txt` — the multi-data ambition has zero data behind it.

---

## 5. LLM call map

- `ModelFactory` (`src/models/model_factory.py`): registered providers `claude` (default claude-sonnet-4-5) and `openai` (default gpt-5) only. `get_model(type, name)` → model; `model.generate_response(system_prompt, user_content, temperature, max_tokens)` → `ModelResponse(.content)`.
- A cache-busting random nonce is appended to every request (`base_model.py:40-44`) — deliberately defeats prompt caching, wasting tokens on every call.
- Agents bypassing the factory: risk_agent (raw anthropic/deepseek clients), strategy_agent (raw client), RBI v2/v3/pp (xAI grok via raw OpenAI-compatible client), v2_simple (DeepSeek), batch_backtester (7-model fallback chain).
- Consequence: no single place to audit/limit LLM spend or models; the factory's own README overstates provider coverage.

---

## 6. Strategy definition & consumption

- `src/strategies/base_strategy.py`: `BaseStrategy.generate_signals() -> dict` (token, signal 0-1, direction, metadata).
- `strategy_agent.py:71-80` imports ExampleStrategy/MyStrategy from `custom/`, gets signals, has an LLM approve, then **executes immediately** (`strategy_agent.py:231-302`). No staging, no paper mode, no position-size validation beyond config percentages.
- The RBI pipelines never produce strategies in this format — generated backtests live as standalone backtesting.py scripts in `src/data/rbi*/<date>/`. **There is no bridge from research output to the live strategy format.** (The factory built in this work adds that bridge with gating.)

## 7. Execution paths

- Solana spot: Jupiter Lite API, `market_buy/market_sell` (`nice_funcs.py:228/276`), `TxOpts(skip_preflight=True)` (lines 269, 317) — transactions sent without preflight validation. Chunked exits via `chunk_kill()` (3 sequential market sells, no rollback on partial failure). Slippage hardcoded ~199 bps.
- Hyperliquid perps: `nice_funcs_hyperliquid.py` — IOC orders 0.1% through the book, `set_leverage()` with **no margin-availability or liquidation-distance check**, `ai_entry()` (line 844), `open_short()` (line 866), `kill_switch()` (line 233).
- ExchangeManager (`src/exchange_manager.py`) normalizes buy/sell/get_position across both. No account/margin/fees/health methods. This is the natural seed of the broker abstraction layer (see BROKER_ABSTRACTION_PLAN.md).

## 8. Risk control today

Config circuit breakers (`src/config.py`): MAX_LOSS_USD=25, MAX_GAIN_USD=25, MINIMUM_BALANCE_USD=50, CASH_PERCENTAGE=20, MAX_POSITION_PERCENTAGE, STOP_LOSS/TAKE_PROFIT 5%.

Missing entirely:
- pre-trade balance/margin checks (both venues)
- liquidation-distance check before leveraged entry
- per-strategy / per-position loss limits (portfolio-level only)
- correlation/concentration limits
- fill verification, partial-fill handling, slippage caps
- any paper/dry-run mode
- hard (non-LLM-overridable) kill switch

## 9. Experimental / duplicated / legacy / dangerous

- **Broken hardcoded paths**: ≥11 files reference `/Users/md/Dropbox/...` (tweet, copybot, sniper sounds, shortvid, code_runner, rbi prompts, batch_backtester default, nice_funcs_aster, multi_data_tester import). 3,699 of 4,770 generated backtest files embed the dead path — none can re-run here.
- **Version sprawl**: 6 RBI variants, 3 chat variants; no canonical designation.
- **Dead config**: config.py lines 50-62, 127-135 unused/“NOT USED YET”.
- **Dangerous**: skip_preflight=True; LLM-vetoable risk breaches; LLM-vote → immediate live order with no validation layer; cache-busting nonce on all LLM calls.
- **Environment drift**: CLAUDE.md mandates `conda activate tflow`, but tflow does not exist on this machine; `requirements-py313.txt` (uncommitted) targets the repo `.venv` (Python 3.13.0) which has **no packages installed**. Until one interpreter has `backtesting`, `pandas`, `numpy`, `TA-Lib`/`pandas_ta` installed, no backtest can execute.

## 10. Reusable for the institutional RBI factory

| Component | Verdict |
|---|---|
| `rbi_agent_pp_multi.py` pipeline shape (research → codegen → package → debug → execute → optimize, threaded, CSV stats) | Reuse the *shape*; replace return-targeting optimizer with validation gates |
| `ModelFactory` | Reuse as-is for all LLM access (claude/openai) |
| `exchange_manager.py` + `nice_funcs_hyperliquid.py` | Wrap into BrokerAdapter; do not call directly from research code |
| `backtest_runner.py` conda-subprocess pattern | Replace with configurable-interpreter runner (env broken) |
| BTC-USD-15m.csv | Usable as the seed dataset; must be extended and split train/val/test |
| backtest_stats.csv columns | Superseded by the experiment store schema |

## 11. Must be isolated before paper/live

1. All direct-execution paths (`ai_entry`, `chunk_kill`, `market_buy/sell`) behind a BrokerAdapter with a PaperBroker default.
2. `USE_AI_CONFIRMATION` override of hard limits — remove from any factory-managed flow.
3. skip_preflight on Solana.
4. Content/social agents (tweet/tiktok/clips/phone) — out of scope, never imported by factory code.
5. The v3/pp "optimize until TARGET_RETURN" loop — this is metric-targeting curve-fitting; replaced by gates + robustness (see RISK_AND_OVERFITTING_AUDIT.md).

## 12. Conventions observed (and followed by new code)

- Standalone agents under `src/agents/`, argparse CLI (`--ideas-file`, `--run-name` style), outputs under `src/data/<agent_name>/`, termcolor cprint logging, files <800 lines, conda env `tflow` nominal (configurable interpreter added because tflow is absent here), requirements.txt updated on new packages.
