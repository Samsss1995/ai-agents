"""200-day MA timing (Faber GTAA rule), hand-written 2026-06-13. Long-only:
hold while close > 200-day SMA, flat below. Purpose is drawdown reduction, not
alpha - judged on the drawdown-adjusted gates it targets. Adaptive lookback."""

import numpy as np
import pandas as pd
import talib
from backtesting import Strategy

PARAMS = {"ma_days": 200, "size_frac": 0.95}


def _bars_per_day(index):
    if len(index) < 3:
        return 1.0
    secs = pd.Series(index).diff().median().total_seconds()
    return max(1.0, 86400.0 / secs) if secs else 1.0


class DMA200(Strategy):
    ma_days = 200
    size_frac = 0.95

    def init(self):
        n = max(20, int(self.ma_days * _bars_per_day(self.data.index)))
        self.ma = self.I(talib.SMA, self.data.Close, n)

    def next(self):
        if np.isnan(self.ma[-1]):
            return
        price = self.data.Close[-1]
        if price > self.ma[-1] and not self.position:
            self.buy(size=self.size_frac)
        elif price < self.ma[-1] and self.position:
            self.position.close()


STRATEGY_CLASS = DMA200


def generate_signal(df):
    n = PARAMS["ma_days"]
    if len(df) < n + 1:
        return {"action": "NOTHING", "confidence": 0, "reasoning": "insufficient bars"}
    ma = df["Close"].rolling(n).mean().iloc[-1]
    price = df["Close"].iloc[-1]
    if price > ma:
        return {"action": "BUY", "confidence": 55, "reasoning": "above 200-day MA"}
    return {"action": "NOTHING", "confidence": 0, "reasoning": "below 200-day MA - flat"}
