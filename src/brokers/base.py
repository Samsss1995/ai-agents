"""
BrokerAdapter - the only interface factory code may use to touch any venue.

place_order() is final: it always runs validate_order() (balance, min notional,
position/exposure caps, daily-loss kill switch) before delegating to the
adapter's _submit_order(). Adapters cannot bypass the checks.
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Account:
    equity_usd: float
    cash_usd: float
    margin_used_usd: float = 0.0
    margin_available_usd: float = 0.0


@dataclass
class Position:
    symbol: str
    size: float                # signed: negative = short
    entry_price: float
    mark_price: float
    unrealized_pnl_usd: float
    leverage: float = 1.0
    liquidation_price: Optional[float] = None

    @property
    def notional_usd(self) -> float:
        return abs(self.size) * self.mark_price


@dataclass
class OrderRequest:
    symbol: str
    side: str                  # "buy" | "sell"
    notional_usd: float
    order_type: str = "market"  # "market" | "limit"
    limit_price: Optional[float] = None
    reduce_only: bool = False
    client_order_id: str = field(default_factory=lambda: f"rf_{uuid.uuid4().hex[:12]}")


@dataclass
class Order:
    order_id: str
    client_order_id: str
    symbol: str
    side: str
    notional_usd: float
    status: str                # "filled" | "open" | "rejected" | "cancelled"
    fill_price: Optional[float] = None
    fee_usd: float = 0.0
    created_at: str = field(default_factory=_now)
    reject_reason: Optional[str] = None


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: str
    size: float
    price: float
    fee_usd: float
    timestamp: str


@dataclass
class FeeSchedule:
    maker: float
    taker: float


@dataclass
class MarginInfo:
    symbol: str
    leverage: float
    margin_available_usd: float
    liquidation_price: Optional[float]


@dataclass
class HealthStatus:
    healthy: bool
    detail: str


class OrderValidationError(Exception):
    """Raised when a pre-trade check fails. Never caught inside broker code."""


class BrokerAdapter(ABC):
    """profile: one entry from configs/broker_profiles.yaml."""

    def __init__(self, profile: Dict[str, Any], profile_name: str):
        self.profile = profile
        self.profile_name = profile_name
        self._kill_switch_engaged = False
        self._daily_realized_loss_usd = 0.0

    # ---------- mandatory pre-trade checks (final) ----------
    def validate_order(self, request: OrderRequest) -> None:
        p = self.profile
        if self._kill_switch_engaged:
            raise OrderValidationError(f"[{self.profile_name}] kill switch engaged - no new orders")
        if request.side not in ("buy", "sell"):
            raise OrderValidationError(f"invalid side '{request.side}'")
        if request.notional_usd <= 0:
            raise OrderValidationError("notional must be positive")
        min_notional = p.get("min_notional_usd", 0)
        if request.notional_usd < min_notional:
            raise OrderValidationError(
                f"notional {request.notional_usd} below min_notional_usd {min_notional}")
        max_pos = p.get("max_position_usd")
        if max_pos is not None and not request.reduce_only:
            current = self._position_notional(request.symbol)
            if current + request.notional_usd > max_pos:
                raise OrderValidationError(
                    f"position cap: {current:.2f} + {request.notional_usd:.2f} "
                    f"> max_position_usd {max_pos}")
        max_exposure = p.get("max_total_exposure_usd")
        if max_exposure is not None and not request.reduce_only:
            total = sum(pos.notional_usd for pos in self.get_positions())
            if total + request.notional_usd > max_exposure:
                raise OrderValidationError(
                    f"exposure cap: {total:.2f} + {request.notional_usd:.2f} "
                    f"> max_total_exposure_usd {max_exposure}")
        max_daily_loss = p.get("max_daily_loss_usd")
        if max_daily_loss is not None and self._daily_realized_loss_usd >= max_daily_loss:
            self.engage_kill_switch(
                f"daily loss {self._daily_realized_loss_usd:.2f} >= {max_daily_loss}")
            raise OrderValidationError(f"[{self.profile_name}] daily loss limit reached")
        account = self.get_account()
        if not request.reduce_only and request.notional_usd > account.margin_available_usd \
                and request.notional_usd > account.cash_usd:
            raise OrderValidationError(
                f"insufficient funds: notional {request.notional_usd:.2f}, "
                f"cash {account.cash_usd:.2f}, margin available "
                f"{account.margin_available_usd:.2f}")

    def place_order(self, request: OrderRequest) -> Order:
        """Final. Validates, then delegates to the adapter."""
        self.validate_order(request)
        return self._submit_order(request)

    def engage_kill_switch(self, reason: str) -> None:
        self._kill_switch_engaged = True
        self._kill_switch_reason = reason

    def record_realized_pnl(self, pnl_usd: float) -> None:
        if pnl_usd < 0:
            self._daily_realized_loss_usd += -pnl_usd
            max_daily = self.profile.get("max_daily_loss_usd")
            if max_daily is not None and self._daily_realized_loss_usd >= max_daily:
                self.engage_kill_switch(
                    f"daily loss {self._daily_realized_loss_usd:.2f} >= {max_daily}")

    def reset_daily_counters(self) -> None:
        self._daily_realized_loss_usd = 0.0

    def _position_notional(self, symbol: str) -> float:
        for pos in self.get_positions():
            if pos.symbol == symbol:
                return pos.notional_usd
        return 0.0

    # ---------- adapter interface ----------
    @abstractmethod
    def get_account(self) -> Account: ...

    @abstractmethod
    def get_positions(self) -> List[Position]: ...

    @abstractmethod
    def get_open_orders(self) -> List[Order]: ...

    @abstractmethod
    def get_market_data(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame: ...

    @abstractmethod
    def _submit_order(self, request: OrderRequest) -> Order: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool: ...

    @abstractmethod
    def close_position(self, symbol: str) -> Optional[Order]: ...

    @abstractmethod
    def get_fills(self, since: Optional[str] = None) -> List[Fill]: ...

    @abstractmethod
    def get_fees(self) -> FeeSchedule: ...

    @abstractmethod
    def get_margin(self, symbol: str) -> MarginInfo: ...

    @abstractmethod
    def health_check(self) -> HealthStatus: ...
