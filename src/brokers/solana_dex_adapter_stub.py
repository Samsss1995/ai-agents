"""
Solana / DEX adapter STUB - future Phantom / GMGN / Jupiter execution route.

Not implemented: the route is undecided and the existing Jupiter code in
src/nice_funcs.py is NOT safe to wrap as-is:
  - TxOpts(skip_preflight=True) at nice_funcs.py:269,317 (no preflight validation)
  - no balance check before swaps, no fill-price verification
  - spot-only; no margin/liquidation concepts

Decision checklist before implementing:
  1. Pick the route: Jupiter API direct (fix skip_preflight), GMGN API, or a
     Phantom-compatible signer service.
  2. Define fee model (DEX swap fees + priority fees + slippage) in broker_profiles.yaml.
  3. Implement quote -> simulate -> sign -> send -> confirm with fill verification.
  4. Paper-test through PaperBroker with real Solana prices first.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from src.brokers.base import (
    Account, BrokerAdapter, FeeSchedule, Fill, HealthStatus, MarginInfo, Order,
    OrderRequest, Position,
)

_MSG = ("SolanaDexAdapter is a stub. Route undecided (Phantom/GMGN/Jupiter) and the "
        "existing Jupiter code must be fixed first - see module docstring.")


class SolanaDexAdapter(BrokerAdapter):
    def __init__(self, profile: Optional[Dict[str, Any]] = None):
        super().__init__(profile or {"live_enabled": False}, "solana_dex")

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
