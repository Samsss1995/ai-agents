"""
Broker abstraction layer.

base                    - BrokerAdapter ABC + normalized dataclasses + mandatory pre-trade checks
paper                   - PaperBroker (default; the only broker that fills orders out of the box)
hyperliquid_adapter     - read-only by default; live requires profile flag + env confirmation
ibkr_adapter_stub       - interface stub for future IBKR (stocks/ETF/index/futures)
solana_dex_adapter_stub - interface stub for future Phantom/GMGN/Jupiter route

No live adapter is enabled by default.
"""

from src.brokers.base import BrokerAdapter, OrderRequest, Order, Position, Account, Fill
from src.brokers.paper import PaperBroker

__all__ = ["BrokerAdapter", "OrderRequest", "Order", "Position", "Account", "Fill",
           "PaperBroker"]
