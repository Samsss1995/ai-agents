"""
IBKR adapter STUB - future stocks / ETFs / indices / futures execution.

Not implemented because no IBKR account, TWS/Gateway, or ib_insync dependency
exists in this repo yet. The stub pins the interface so research code can target
it today; every method raises with setup guidance.

Implementation checklist (when approved):
  1. Open IBKR account; enable a PAPER account first.
  2. Install TWS or IB Gateway; enable API connections (socket port).
  3. pip install ib_insync (and update requirements.txt).
  4. Map: get_account -> accountSummary; get_positions -> positions();
     place_order -> MarketOrder/LimitOrder via ib.placeOrder with client_order_id
     as orderRef; get_fills -> fills(); get_market_data -> reqHistoricalData.
  5. Respect pacing limits (max ~50 msg/s, historical data pacing).
  6. Route through the paper account until promotion.py approves live.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from src.brokers.base import (
    Account, BrokerAdapter, FeeSchedule, Fill, HealthStatus, MarginInfo, Order,
    OrderRequest, Position,
)

_MSG = ("IBKRAdapter is a stub. See module docstring for the implementation "
        "checklist. No IBKR connectivity exists in this repo yet.")


class IBKRAdapter(BrokerAdapter):
    def __init__(self, profile: Optional[Dict[str, Any]] = None):
        super().__init__(profile or {"live_enabled": False}, "ibkr")

    def get_account(self) -> Account:
        raise NotImplementedError(_MSG)

    def get_positions(self) -> List[Position]:
        raise NotImplementedError(_MSG)

    def get_open_orders(self) -> List[Order]:
        raise NotImplementedError(_MSG)

    def get_market_data(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        raise NotImplementedError(_MSG)

    def _submit_order(self, request: OrderRequest) -> Order:
        raise NotImplementedError(_MSG)

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError(_MSG)

    def close_position(self, symbol: str) -> Optional[Order]:
        raise NotImplementedError(_MSG)

    def get_fills(self, since: Optional[str] = None) -> List[Fill]:
        raise NotImplementedError(_MSG)

    def get_fees(self) -> FeeSchedule:
        raise NotImplementedError(_MSG)

    def get_margin(self, symbol: str) -> MarginInfo:
        raise NotImplementedError(_MSG)

    def health_check(self) -> HealthStatus:
        return HealthStatus(False, _MSG)
