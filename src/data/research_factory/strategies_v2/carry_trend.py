"""Adjusted carry+trend (crypto SIG datasets), hand-written 2026-06-13.
Covers list ideas 'carry + trend' and 'adjusted/accurate carry trade' for the
crypto-testable portion (real hourly funding in the FundingRate column).
Take carry ONLY in trend agreement: long an uptrend when funding is negative
(shorts pay longs - paid to hold the trend); short a downtrend when funding is
positive (longs overpay). The trend gate is what naked carry (dead, -34%) lacked."""

import numpy as np
import pandas as pd
import talib
from backtesting import Strategy

PARAMS = {"ma_days": 20, "funding_lookback": 72, "size_frac": 0.5}


def _bars_per_day(index):
    if len(index) < 3:
        return 1.0
    secs = pd.Series(index).diff().median().total_seconds()
    return max(1.0, 86400.0 / secs) if secs else 1.0


class CarryTrend(Strategy):
    ma_days = 20
    funding_lookback = 72
    size_frac = 0.5

    def init(self):
        n = max(10, int(self.ma_days * _bars_per_day(self.data.index)))
        self.ma = self.I(talib.SMA, self.data.Close, n)
        self.fund = self.I(
            lambda v: pd.Series(v).rolling(self.funding_lookback, min_periods=1).mean().values,
            np.asarray(self.data.FundingRate))

    def next(self):
        if np.isnan(self.ma[-1]) or np.isnan(self.fund[-1]):
            return
        price, up, f = self.data.Close[-1], self.data.Close[-1] > self.ma[-1], self.fund[-1]
        if self.position:
            if self.position.is_long and not up:
                self.position.close()
            elif self.position.is_short and up:
                self.position.close()
            return
        if up and f < 0:
            self.buy(size=self.size_frac)
        elif (not up) and f > 0:
            self.sell(size=self.size_frac)


STRATEGY_CLASS = CarryTrend


def generate_signal(df):
    p = PARAMS
    if len(df) < p["ma_days"] + 2 or "FundingRate" not in df.columns:
        return {"action": "NOTHING", "confidence": 0, "reasoning": "insufficient data"}
    ma = df["Close"].rolling(p["ma_days"]).mean().iloc[-1]
    f = df["FundingRate"].rolling(p["funding_lookback"], min_periods=1).mean().iloc[-1]
    up = df["Close"].iloc[-1] > ma
    if up and f < 0:
        return {"action": "BUY", "confidence": 60, "reasoning": "uptrend + favorable carry"}
    if (not up) and f > 0:
        return {"action": "SELL", "confidence": 60, "reasoning": "downtrend + longs overpaying"}
    return {"action": "NOTHING", "confidence": 0, "reasoning": "carry/trend disagree"}
