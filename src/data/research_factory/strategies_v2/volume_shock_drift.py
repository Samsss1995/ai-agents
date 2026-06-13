"""Evidence batch idea 3 (hand-written, 2026-06-12): volume-shock event drift,
long-short at wide-universe breadth. A daily volume spike >= vol_mult times its
average with a close in the outer quarter of the bar's range marks institutional
initiation; the displacement drifts for days. Long up-shocks, short down-shocks."""

import numpy as np
import talib
from backtesting import Strategy

PARAMS = {
    "vol_mult": 3.0,
    "vol_sma_period": 50,
    "close_loc_hi": 0.75,
    "close_loc_lo": 0.25,
    "hold_bars": 5,
    "size_frac": 0.05,
}


class VolumeShockDrift(Strategy):
    vol_mult = 3.0
    vol_sma_period = 50
    close_loc_hi = 0.75
    close_loc_lo = 0.25
    hold_bars = 5
    size_frac = 0.05

    def init(self):
        self.vol_sma = self.I(talib.SMA, self.data.Volume.astype(float),
                              timeperiod=self.vol_sma_period)
        self.entry_bar = None

    def next(self):
        if len(self.data) < self.vol_sma_period + 2:
            return
        if self.position:
            if len(self.data) - (self.entry_bar or 0) >= self.hold_bars:
                self.position.close()
                self.entry_bar = None
            return
        vol, sma = self.data.Volume[-1], self.vol_sma[-1]
        if np.isnan(sma) or sma <= 0 or vol < self.vol_mult * sma:
            return
        rng = self.data.High[-1] - self.data.Low[-1]
        if rng <= 0:
            return
        loc = (self.data.Close[-1] - self.data.Low[-1]) / rng
        if loc >= self.close_loc_hi:
            self.buy(size=self.size_frac)
            self.entry_bar = len(self.data)
        elif loc <= self.close_loc_lo:
            self.sell(size=self.size_frac)
            self.entry_bar = len(self.data)


STRATEGY_CLASS = VolumeShockDrift


def generate_signal(df):
    p = PARAMS
    if len(df) < p["vol_sma_period"] + 2:
        return {"action": "NOTHING", "confidence": 0, "reasoning": "insufficient history"}
    vol_sma = df["Volume"].rolling(p["vol_sma_period"]).mean().iloc[-1]
    vol = df["Volume"].iloc[-1]
    rng = df["High"].iloc[-1] - df["Low"].iloc[-1]
    if vol_sma <= 0 or rng <= 0 or vol < p["vol_mult"] * vol_sma:
        return {"action": "NOTHING", "confidence": 0, "reasoning": "no volume shock"}
    loc = (df["Close"].iloc[-1] - df["Low"].iloc[-1]) / rng
    if loc >= p["close_loc_hi"]:
        return {"action": "BUY", "confidence": 65,
                "reasoning": f"volume shock {vol/vol_sma:.1f}x with top-of-range close"}
    if loc <= p["close_loc_lo"]:
        return {"action": "SELL", "confidence": 65,
                "reasoning": f"volume shock {vol/vol_sma:.1f}x with bottom-of-range close"}
    return {"action": "NOTHING", "confidence": 0, "reasoning": "shock without directional close"}
