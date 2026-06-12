# BROKER ABSTRACTION PLAN

Date: 2026-06-10.

---

## 1. What exists today

- `src/exchange_manager.py` (~150 lines): normalizes `market_buy / market_sell / get_position` across Solana (`src/nice_funcs.py`, Jupiter spot) and Hyperliquid (`src/nice_funcs_hyperliquid.py`, perps). No account/margin/fees/orders/health methods. Used optionally by strategy_agent and example_unified_agent.
- `src/nice_funcs_hyperliquid.py`: the most complete venue integration — `get_position` (L117), `set_leverage` (L154), `limit_order` (L210), `kill_switch` (L233), `market_buy/sell` (L319/359), `ai_entry` (L844), `open_short` (L866), `_get_ohlcv` (L466), key management `_get_account_from_env` (L455). Missing: available margin, liquidation price, cancel-by-id, fills, fees.
- Solana path: Jupiter Lite API market swaps with `skip_preflight=True` — unsafe, and spot-only.
- Aster: `nice_funcs_aster.py` references a dead `/Users/md/...` path; treat as nonexistent.
- **No paper/simulated execution anywhere.**

## 2. Target interface (`src/brokers/base.py`, shipped)

`BrokerAdapter` ABC with normalized dataclasses (`Account`, `Position`, `Order`, `Fill`, `FeeSchedule`):

```
get_account() -> Account            # equity, cash, margin_used, margin_available
get_positions() -> list[Position]
get_open_orders() -> list[Order]
get_market_data(symbol, timeframe, bars) -> DataFrame (OHLCV)
place_order(OrderRequest) -> Order  # idempotent via client_order_id
cancel_order(order_id) -> bool
close_position(symbol) -> Order
get_fills(since) -> list[Fill]
get_fees() -> FeeSchedule
get_margin(symbol) -> MarginInfo    # leverage, liq price, available
health_check() -> HealthStatus
```

Pre-trade checks live in the base class (`validate_order`): balance/margin sufficiency, min notional, max position size, max total exposure, daily-loss kill switch state. Adapters cannot skip them — `place_order` is final in the base and delegates to `_submit_order`.

## 3. Adapters

| Adapter | Status | Notes |
|---|---|---|
| `paper.py` PaperBroker | **Shipped, default** | Fills at reference price ± slippage bps + fees from `configs/broker_profiles.yaml`; persistent state in `src/data/research_factory/paper_broker/<profile>.json`; kill switch; daily loss / exposure limits; deterministic given a price feed. |
| `hyperliquid_adapter.py` | Shipped, **read-only by default** | Wraps nice_funcs_hyperliquid for market data/positions/account. Order endpoints raise unless `broker_profiles.yaml: hyperliquid.live_enabled: true` AND env `BROKER_LIVE_CONFIRM=YES_I_APPROVE_LIVE_TRADING`. Both default off. Must add margin/liquidation math before live (gap list in file docstring). |
| `ibkr_adapter_stub.py` | Stub | Designed for ib_insync (TWS/Gateway paper account first). Interface mapped; NotImplementedError with setup guidance. Do not implement until an IBKR account + TWS exist. |
| `solana_dex_adapter_stub.py` | Stub | Route undecided (Phantom/GMGN/Jupiter). Existing Jupiter code is spot-only with skip_preflight=True — must be fixed before wrapping. NotImplementedError with decision checklist. |

## 4. Profiles (`configs/broker_profiles.yaml`)

Per venue: maker/taker fees, slippage bps assumption, min notional, max leverage, max position USD, max total exposure USD, max daily loss USD, live_enabled (false), paper settings (starting cash). Strategies carry `allowed_brokers` in metadata; promotion checks the venue profile exists.

## 5. Promotion coupling

- paper_candidate → paper_active requires: gates passed, code reviewed, risk config present, PaperBroker profile resolves, logging on, kill switch armed, max daily loss / position size / exposure defined.
- live transitions additionally require: your manual approval (`--approved-by`), paper performance report, named venue profile, account size + risk limits, emergency stop procedure, rollback plan — enforced as recorded fields in promotion.py, absent any of them the transition raises.

## 6. Migration path

1. Keep `exchange_manager.py` untouched for the legacy agents.
2. New factory code talks only to `src/brokers/*`.
3. Phase 3: port trading_agent/strategy_agent execution onto BrokerAdapter; then fix Solana skip_preflight and add Hyperliquid margin math; then deprecate direct nice_funcs execution calls from agents.
