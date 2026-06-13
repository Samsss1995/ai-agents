"""Evidence batch idea 5 (hand-written, 2026-06-12): idiosyncratic gap
overreaction reversion at wide-universe breadth. A stock gapping >= gap_atr_mult
ATRs while the index barely gaps (SpxGapPct column) is an idiosyncratic
overreaction; fade it for a few days. Detection at the gap bar's close, entry
next bar - captures the multi-day reversion component only (documented)."""

import numpy as np
import talib
from backtesting import Strategy

PARAMS = {
    "gap_atr_mult": 1.5,
    "idx_ratio": 0.5,
    "hold_bars": 3,
    "atr_period": 14,
    "size_frac": 0.05,
}


class IdioGapReversion(Strategy):
    gap_atr_mult = 1.5
    idx_ratio = 0.5
    hold_bars = 3
    atr_period = 14
    size_frac = 0.05

    def init(self):
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low,
                          self.data.Close, timeperiod=self.atr_period)
        self.entry_bar = None

    def next(self):
        if len(self.data) < self.atr_period + 2:
            return
        if self.position:
            if len(self.data) - (self.entry_bar or 0) >= self.hold_bars:
                self.position.close()
                self.entry_bar = None
            return
        prev_close = self.data.Close[-2]
        atr = self.atr[-2]
        if np.isnan(atr) or atr <= 0 or prev_close <= 0:
            return
        own_gap = self.data.Open[-1] / prev_close - 1.0
        spx_gap = self.data.SpxGapPct[-1]
        if np.isnan(spx_gap):
            return
        atr_pct = atr / prev_close
        if abs(own_gap) < self.gap_atr_mult * atr_pct:
            return
        if abs(spx_gap) > self.idx_ratio * abs(own_gap):
            return  # market-wide gap, not idiosyncratic
        if own_gap < 0:
            self.buy(size=self.size_frac)
        else:
            self.sell(size=self.size_frac)
        self.entry_bar = len(self.data)


STRATEGY_CLASS = IdioGapReversion


def generate_signal(df):
    p = PARAMS
    if len(df) < p["atr_period"] + 2 or "SpxGapPct" not in df.columns:
        return {"action": "NOTHING", "confidence": 0, "reasoning": "insufficient data"}
    high, low, close = df["High"].values, df["Low"].values, df["Close"].values
    atr = talib.ATR(high, low, close, p["atr_period"])[-2]
    prev_close = close[-2]
    own_gap = df["Open"].iloc[-1] / prev_close - 1.0
    spx_gap = df["SpxGapPct"].iloc[-1]
    if np.isnan(atr) or atr <= 0 or np.isnan(spx_gap):
        return {"action": "NOTHING", "confidence": 0, "reasoning": "invalid inputs"}
    if abs(own_gap) < p["gap_atr_mult"] * atr / prev_close:
        return {"action": "NOTHING", "confidence": 0, "reasoning": "no large gap"}
    if abs(spx_gap) > p["idx_ratio"] * abs(own_gap):
        return {"action": "NOTHING", "confidence": 0, "reasoning": "market-wide gap"}
    action = "BUY" if own_gap < 0 else "SELL"
    return {"action": action, "confidence": 65,
            "reasoning": f"idiosyncratic gap {own_gap*100:.1f}% vs index {spx_gap*100:.2f}%"}
