"""Options-expiry flow strategy, hand-written 2026-06-12. Monthly index options
expire the third Friday; documented pattern: dealer hedging supports prices into
expiry, and the unwind weakens the post-expiry week. Long during expiry week
(Monday through the third Friday close), short the following week. Third Friday
computed from the bar timestamp alone - no external data."""

import datetime as dt

from backtesting import Strategy

PARAMS = {"size_frac": 0.95}


def third_friday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 1)
    fridays = 0
    while True:
        if d.weekday() == 4:
            fridays += 1
            if fridays == 3:
                return d
        d += dt.timedelta(days=1)


def _phase(date: dt.date) -> str:
    f = third_friday(date.year, date.month)
    if f - dt.timedelta(days=4) <= date <= f:
        return "long"           # expiry week: pinning/hedging support
    prev_month = date.month - 1 or 12
    prev_year = date.year if date.month > 1 else date.year - 1
    f_prev = third_friday(prev_year, prev_month)
    f_cur = third_friday(date.year, date.month)
    last_f = f_cur if f_cur < date else f_prev
    if last_f < date <= last_f + dt.timedelta(days=7):
        return "short"          # post-expiry week: hedge unwind
    return "flat"


class OpexFlows(Strategy):
    size_frac = 0.95

    def init(self):
        pass

    def next(self):
        phase = _phase(self.data.index[-1].date())
        if phase == "long":
            if not self.position.is_long:
                if self.position:
                    self.position.close()
                self.buy(size=self.size_frac)
        elif phase == "short":
            if not self.position.is_short:
                if self.position:
                    self.position.close()
                self.sell(size=self.size_frac)
        else:
            if self.position:
                self.position.close()


STRATEGY_CLASS = OpexFlows


def generate_signal(df):
    import pandas as pd
    d = (df.index[-1] if isinstance(df.index, pd.DatetimeIndex)
         else pd.Timestamp(df["datetime"].iloc[-1])).date()
    phase = _phase(d)
    if phase == "long":
        return {"action": "BUY", "confidence": 55, "reasoning": "options expiry week"}
    if phase == "short":
        return {"action": "SELL", "confidence": 55, "reasoning": "post-expiry unwind week"}
    return {"action": "NOTHING", "confidence": 0, "reasoning": "outside expiry windows"}
