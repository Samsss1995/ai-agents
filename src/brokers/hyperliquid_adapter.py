"""
Hyperliquid adapter - READ-ONLY by default.

Market data, positions, and account state wrap src/nice_funcs_hyperliquid.py.
Order placement is hard-disabled unless BOTH:
  1. configs/broker_profiles.yaml: hyperliquid.live_enabled: true
  2. env BROKER_LIVE_CONFIRM=YES_I_APPROVE_LIVE_TRADING

Known gaps before live use (do NOT enable until closed - see
docs/BROKER_ABSTRACTION_PLAN.md):
  - no liquidation-distance computation before leveraged entries
  - no cancel-by-order-id wiring
  - fills/fees endpoints not wired (get_fills/get_fees raise)
"""

import os
from typing import Any, Dict, List, Optional

import pandas as pd

from src.brokers.base import (
    Account, BrokerAdapter, FeeSchedule, Fill, HealthStatus, MarginInfo, Order,
    OrderRequest, Position,
)
from src.research.factory_config import load_broker_profiles

LIVE_CONFIRM_ENV = "BROKER_LIVE_CONFIRM"
LIVE_CONFIRM_VALUE = "YES_I_APPROVE_LIVE_TRADING"


class LiveTradingDisabledError(Exception):
    pass


class HyperliquidAdapter(BrokerAdapter):
    def __init__(self, profile: Optional[Dict[str, Any]] = None):
        profile = profile or load_broker_profiles()["hyperliquid"]
        super().__init__(profile, "hyperliquid")
        import src.nice_funcs_hyperliquid as hl  # lazy: needs hyperliquid SDK installed
        self._hl = hl
        self._account = hl._get_account_from_env()

    def _assert_live_allowed(self) -> None:
        if not self.profile.get("live_enabled", False):
            raise LiveTradingDisabledError(
                "hyperliquid.live_enabled is false in configs/broker_profiles.yaml")
        if os.getenv(LIVE_CONFIRM_ENV) != LIVE_CONFIRM_VALUE:
            raise LiveTradingDisabledError(
                f"env {LIVE_CONFIRM_ENV} not set to the confirmation value")

    # ---------- read-only ----------
    def get_account(self) -> Account:
        state = self._hl.get_user_state(self._account)
        margin = state.get("marginSummary", {}) if isinstance(state, dict) else {}
        equity = float(margin.get("accountValue", 0) or 0)
        used = float(margin.get("totalMarginUsed", 0) or 0)
        return Account(equity_usd=equity, cash_usd=equity - used,
                       margin_used_usd=used, margin_available_usd=equity - used)

    def get_positions(self) -> List[Position]:
        out = []
        state = self._hl.get_user_state(self._account)
        for ap in (state.get("assetPositions", []) if isinstance(state, dict) else []):
            p = ap.get("position", {})
            size = float(p.get("szi", 0) or 0)
            if size == 0:
                continue
            entry = float(p.get("entryPx", 0) or 0)
            liq = p.get("liquidationPx")
            out.append(Position(
                symbol=p.get("coin", "?"), size=size, entry_price=entry,
                mark_price=float(p.get("positionValue", 0) or 0) / abs(size) if size else entry,
                unrealized_pnl_usd=float(p.get("unrealizedPnl", 0) or 0),
                leverage=float(p.get("leverage", {}).get("value", 1) or 1),
                liquidation_price=float(liq) if liq else None,
            ))
        return out

    def get_open_orders(self) -> List[Order]:
        return []  # not wired; gap documented in module docstring

    def get_market_data(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        df = self._hl.get_data(symbol=symbol, timeframe=timeframe, bars=bars,
                               add_indicators=False)
        if df is None or len(df) == 0:
            raise RuntimeError(f"hyperliquid returned no data for {symbol} {timeframe}")
        return df

    # ---------- order paths (disabled by default) ----------
    def _submit_order(self, request: OrderRequest) -> Order:
        self._assert_live_allowed()
        if request.side == "buy":
            result = self._hl.market_buy(request.symbol, request.notional_usd, self._account)
        else:
            result = self._hl.market_sell(request.symbol, request.notional_usd, self._account)
        return Order(
            order_id=str(result), client_order_id=request.client_order_id,
            symbol=request.symbol, side=request.side,
            notional_usd=request.notional_usd, status="open",
        )

    def cancel_order(self, order_id: str) -> bool:
        self._assert_live_allowed()
        raise NotImplementedError("cancel-by-order-id not wired - close this gap before live")

    def close_position(self, symbol: str) -> Optional[Order]:
        self._assert_live_allowed()
        result = self._hl.kill_switch(symbol, self._account)
        return Order(order_id=str(result), client_order_id="kill_switch", symbol=symbol,
                     side="close", notional_usd=0.0, status="open")

    def get_fills(self, since: Optional[str] = None) -> List[Fill]:
        raise NotImplementedError("fills endpoint not wired - close this gap before live")

    def get_fees(self) -> FeeSchedule:
        return FeeSchedule(maker=self.profile["maker_fee"], taker=self.profile["taker_fee"])

    def get_margin(self, symbol: str) -> MarginInfo:
        account = self.get_account()
        liq = next((p.liquidation_price for p in self.get_positions() if p.symbol == symbol), None)
        return MarginInfo(symbol=symbol, leverage=self.profile.get("max_leverage", 1),
                          margin_available_usd=account.margin_available_usd,
                          liquidation_price=liq)

    def health_check(self) -> HealthStatus:
        try:
            account = self.get_account()
            return HealthStatus(True, f"equity={account.equity_usd:.2f}, "
                                      f"live_enabled={self.profile.get('live_enabled', False)}")
        except Exception as e:
            return HealthStatus(False, f"{type(e).__name__}: {e}")
