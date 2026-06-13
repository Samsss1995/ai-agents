"""Time-series momentum (Moskowitz-Ooi-Pedersen managed-futures), hand-written
2026-06-13. Each instrument independently: long if its own trailing-12-month
return > 0, short if < 0; position vol-targeted to a constant risk budget;
monthly rebalance. Distinct from the dead cross-sectional momentum - this is
each asset on its OWN trend. Adaptive lookback. Long/short (canonical)."""

import numpy as np
import pandas as pd
from backtesting import Strategy

PARAMS = {"lookback_days": 252, "rebal_days": 21, "vol_days": 63,
          "target_vol": 0.15, "max_frac": 0.95}


def _bars_per_day(index):
    if len(index) < 3:
        return 1.0
    secs = pd.Series(index).diff().median().total_seconds()
    return max(1.0, 86400.0 / secs) if secs else 1.0


class TSMOM(Strategy):
    lookback_days = 252
    rebal_days = 21
    vol_days = 63
    target_vol = 0.15
    max_frac = 0.95

    def init(self):
        bpd = _bars_per_day(self.data.index)
        self._lb = max(20, int(self.lookback_days * bpd))
        self._rebal = max(1, int(self.rebal_days * bpd))
        self._voln = max(10, int(self.vol_days * bpd))
        self._bpy = bpd * 252
        self.ret = self.I(lambda v: pd.Series(v).pct_change().values,
                          np.asarray(self.data.Close))
        self._last = -10 ** 9

    def next(self):
        i = len(self.data) - 1
        if i < self._lb + 1:
            return
        if self.position and (i - self._last) < self._rebal:
            return
        mom = self.data.Close[-1] / self.data.Close[-self._lb] - 1
        vol = np.nanstd(self.ret[-self._voln:]) * np.sqrt(self._bpy)
        if vol <= 0 or np.isnan(vol):
            return
        size = min(self.max_frac, max(0.01, self.target_vol / vol))
        self._last = i
        if mom > 0:
            if self.position.is_short:
                self.position.close()
            if not self.position:
                self.buy(size=size)
        else:
            if self.position.is_long:
                self.position.close()
            if not self.position:
                self.sell(size=size)


STRATEGY_CLASS = TSMOM


def generate_signal(df):
    n = PARAMS["lookback_days"]
    if len(df) < n + 1:
        return {"action": "NOTHING", "confidence": 0, "reasoning": "insufficient bars"}
    mom = df["Close"].iloc[-1] / df["Close"].iloc[-n] - 1
    if mom > 0:
        return {"action": "BUY", "confidence": 60, "reasoning": f"12mo momentum +{mom*100:.0f}%"}
    return {"action": "SELL", "confidence": 60, "reasoning": f"12mo momentum {mom*100:.0f}%"}
