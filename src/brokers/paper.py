"""
PaperBroker - simulated execution against a caller-supplied price feed.

Fills are immediate at reference price +/- slippage_bps, with taker fees from the
broker profile. State (cash, positions, fills) persists to JSON under
src/data/research_factory/paper_broker/<profile>.json so paper runs survive
restarts. Deterministic given the same price sequence.
"""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.brokers.base import (
    Account, BrokerAdapter, FeeSchedule, Fill, HealthStatus, MarginInfo, Order,
    OrderRequest, OrderValidationError, Position,
)
from src.research.factory_config import load_factory_config, load_broker_profiles, factory_root


class PaperBroker(BrokerAdapter):
    def __init__(self, profile_name: str = "paper",
                 profile: Optional[Dict[str, Any]] = None,
                 state_path: Optional[Path] = None):
        profile = profile or load_broker_profiles()[profile_name]
        super().__init__(profile, profile_name)
        root = factory_root(load_factory_config()) / "paper_broker"
        root.mkdir(parents=True, exist_ok=True)
        self.state_path = state_path or root / f"{profile_name}.json"
        self._prices: Dict[str, float] = {}
        self._market_data_fn = None
        self._load_state()

    # ---------- price feed ----------
    def set_price(self, symbol: str, price: float) -> None:
        if price <= 0:
            raise ValueError(f"non-positive price for {symbol}: {price}")
        self._prices[symbol] = float(price)

    def set_market_data_source(self, fn) -> None:
        """fn(symbol, timeframe, bars) -> OHLCV DataFrame. Real data only."""
        self._market_data_fn = fn

    def _ref_price(self, symbol: str) -> float:
        if symbol not in self._prices:
            raise OrderValidationError(
                f"no reference price for '{symbol}'. Call set_price() or "
                f"set_market_data_source() with a real feed - PaperBroker never invents prices."
            )
        return self._prices[symbol]

    # ---------- state ----------
    def _load_state(self) -> None:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            self._cash = state["cash"]
            self._positions = state["positions"]      # symbol -> {size, entry_price}
            self._fills = state["fills"]
            self._daily_realized_loss_usd = state.get("daily_realized_loss_usd", 0.0)
            self._kill_switch_engaged = state.get("kill_switch_engaged", False)
        else:
            self._cash = float(self.profile["starting_cash"])
            self._positions = {}
            self._fills = []
            self._save_state()

    def _save_state(self) -> None:
        self.state_path.write_text(json.dumps({
            "cash": self._cash,
            "positions": self._positions,
            "fills": self._fills,
            "daily_realized_loss_usd": self._daily_realized_loss_usd,
            "kill_switch_engaged": self._kill_switch_engaged,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))

    # ---------- adapter interface ----------
    def get_account(self) -> Account:
        unrealized = 0.0
        for symbol, pos in self._positions.items():
            if symbol in self._prices:
                unrealized += pos["size"] * (self._prices[symbol] - pos["entry_price"])
        equity = self._cash + unrealized
        return Account(equity_usd=equity, cash_usd=self._cash,
                       margin_used_usd=0.0, margin_available_usd=self._cash)

    def get_positions(self) -> List[Position]:
        out = []
        for symbol, pos in self._positions.items():
            if abs(pos["size"]) < 1e-12:
                continue
            mark = self._prices.get(symbol, pos["entry_price"])
            out.append(Position(
                symbol=symbol, size=pos["size"], entry_price=pos["entry_price"],
                mark_price=mark,
                unrealized_pnl_usd=pos["size"] * (mark - pos["entry_price"]),
            ))
        return out

    def get_open_orders(self) -> List[Order]:
        return []  # market-only simulation: orders fill immediately

    def get_market_data(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        if self._market_data_fn is None:
            raise RuntimeError("no market data source attached "
                               "(set_market_data_source); PaperBroker has no synthetic data")
        return self._market_data_fn(symbol, timeframe, bars)

    def _submit_order(self, request: OrderRequest) -> Order:
        ref = self._ref_price(request.symbol)
        slip = self.profile["slippage_bps"] / 10_000.0
        fill_price = ref * (1 + slip) if request.side == "buy" else ref * (1 - slip)
        size = request.notional_usd / fill_price * (1 if request.side == "buy" else -1)
        fee = request.notional_usd * self.profile["taker_fee"]

        pos = self._positions.setdefault(request.symbol, {"size": 0.0, "entry_price": 0.0})
        old_size = pos["size"]
        new_size = old_size + size

        realized = 0.0
        if old_size != 0 and size * old_size < 0:  # order reduces or flips the position
            closed_amount = min(abs(size), abs(old_size))
            direction = 1 if old_size > 0 else -1
            realized = closed_amount * direction * (fill_price - pos["entry_price"])

        if abs(new_size) < 1e-12:                       # fully closed
            pos["size"], pos["entry_price"] = 0.0, 0.0
        elif old_size == 0 or old_size * new_size < 0:  # opened or flipped
            pos["size"], pos["entry_price"] = new_size, fill_price
        elif abs(new_size) > abs(old_size):             # increased same-direction
            pos["entry_price"] = (
                (abs(old_size) * pos["entry_price"] + abs(size) * fill_price) / abs(new_size)
            )
            pos["size"] = new_size
        else:                                           # reduced, entry unchanged
            pos["size"] = new_size

        self._cash += realized - fee
        self.record_realized_pnl(realized - fee)

        order = Order(
            order_id=f"paper_{len(self._fills) + 1}",
            client_order_id=request.client_order_id,
            symbol=request.symbol, side=request.side,
            notional_usd=request.notional_usd, status="filled",
            fill_price=fill_price, fee_usd=fee,
        )
        self._fills.append({
            "order_id": order.order_id, "symbol": request.symbol, "side": request.side,
            "size": size, "price": fill_price, "fee_usd": fee,
            "realized_pnl_usd": realized, "timestamp": order.created_at,
        })
        self._save_state()
        return order

    def cancel_order(self, order_id: str) -> bool:
        return False  # nothing rests

    def close_position(self, symbol: str) -> Optional[Order]:
        pos = self._positions.get(symbol)
        if not pos or abs(pos["size"]) < 1e-12:
            return None
        ref = self._ref_price(symbol)
        # notional priced at the slipped fill so the closed size matches the
        # position exactly (notional/fill_price == |size|)
        slip = self.profile["slippage_bps"] / 10_000.0
        side = "sell" if pos["size"] > 0 else "buy"
        fill_price = ref * (1 - slip) if side == "sell" else ref * (1 + slip)
        request = OrderRequest(
            symbol=symbol,
            side=side,
            notional_usd=abs(pos["size"]) * fill_price,
            reduce_only=True,
        )
        return self.place_order(request)

    def get_fills(self, since: Optional[str] = None) -> List[Fill]:
        rows = [f for f in self._fills if since is None or f["timestamp"] >= since]
        return [Fill(order_id=f["order_id"], symbol=f["symbol"], side=f["side"],
                     size=f["size"], price=f["price"], fee_usd=f["fee_usd"],
                     timestamp=f["timestamp"]) for f in rows]

    def get_fees(self) -> FeeSchedule:
        return FeeSchedule(maker=self.profile["maker_fee"], taker=self.profile["taker_fee"])

    def get_margin(self, symbol: str) -> MarginInfo:
        return MarginInfo(symbol=symbol, leverage=1.0,
                          margin_available_usd=self._cash, liquidation_price=None)

    def health_check(self) -> HealthStatus:
        detail = f"cash={self._cash:.2f}, positions={len(self.get_positions())}, " \
                 f"kill_switch={self._kill_switch_engaged}"
        return HealthStatus(healthy=not self._kill_switch_engaged, detail=detail)

    def raw_fills(self) -> List[Dict[str, Any]]:
        return list(self._fills)
