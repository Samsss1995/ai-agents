"""Canonical Turtle (System 2), hand-written 2026-06-13. Donchian 55-bar breakout
entry, 20-bar opposite-extreme exit, 2N (ATR) stop, ATR-risk position sizing.
Adaptive lookback: windows are specified in DAYS and scaled to the data's native
bar spacing, so one module is correct on daily and on crypto 4h. Long/short."""

import numpy as np
import pandas as pd
import talib
from backtesting import Strategy

PARAMS = {"entry_days": 55, "exit_days": 20, "atr_days": 20, "stop_atr": 2.0,
          "risk_frac": 0.02}


def _bars_per_day(index):
    if len(index) < 3:
        return 1.0
    secs = pd.Series(index).diff().median().total_seconds()
    return max(1.0, 86400.0 / secs) if secs else 1.0


class Turtle(Strategy):
    entry_days = 55
    exit_days = 20
    atr_days = 20
    stop_atr = 2.0
    risk_frac = 0.02

    def init(self):
        bpd = _bars_per_day(self.data.index)
        e = max(10, int(self.entry_days * bpd))
        x = max(5, int(self.exit_days * bpd))
        a = max(5, int(self.atr_days * bpd))
        h, l = np.asarray(self.data.High), np.asarray(self.data.Low)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, a)
        self.hh = self.I(lambda v: pd.Series(v).rolling(e).max().values, h)
        self.ll_entry = self.I(lambda v: pd.Series(v).rolling(e).min().values, l)
        self.hh_exit = self.I(lambda v: pd.Series(v).rolling(x).max().values, h)
        self.ll_exit = self.I(lambda v: pd.Series(v).rolling(x).min().values, l)
        self.stop = None

    def next(self):
        if np.isnan(self.atr[-1]) or np.isnan(self.hh[-2]) or np.isnan(self.ll_entry[-2]):
            return
        price, atr = self.data.Close[-1], self.atr[-1]
        if self.position:
            if self.position.is_long and (price < self.ll_exit[-2] or price < self.stop):
                self.position.close()
            elif self.position.is_short and (price > self.hh_exit[-2] or price > self.stop):
                self.position.close()
            return
        if atr <= 0:
            return
        size = min(0.95, max(0.01, self.risk_frac / (self.stop_atr * atr / price)))
        if price > self.hh[-2]:
            self.stop = price - self.stop_atr * atr
            self.buy(size=size, sl=self.stop)
        elif price < self.ll_entry[-2]:
            self.stop = price + self.stop_atr * atr
            self.sell(size=size, sl=self.stop)


STRATEGY_CLASS = Turtle


def generate_signal(df):
    p = PARAMS
    n = p["entry_days"]
    if len(df) < n + 2:
        return {"action": "NOTHING", "confidence": 0, "reasoning": "insufficient bars"}
    hh = df["High"].iloc[-(n + 1):-1].max()
    ll = df["Low"].iloc[-(n + 1):-1].min()
    price = df["Close"].iloc[-1]
    if price > hh:
        return {"action": "BUY", "confidence": 65, "reasoning": "55-bar Donchian breakout up"}
    if price < ll:
        return {"action": "SELL", "confidence": 65, "reasoning": "55-bar Donchian breakout down"}
    return {"action": "NOTHING", "confidence": 0, "reasoning": "inside channel"}
