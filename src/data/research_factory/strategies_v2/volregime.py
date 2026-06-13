"""Volatility-regime trend ('chill when orders are flying'), hand-written
2026-06-13. Take a simple 50-bar trend signal ONLY when realized vol is below its
median (calm); flatten when vol is in its top quartile (turbulent). Tests whether
sitting out high-vol regimes fixes the drawdown-adjusted gates. Adaptive."""
import numpy as np, pandas as pd, talib
from backtesting import Strategy
PARAMS={"ma_days":50,"vol_days":20,"vol_lookback":252,"size_frac":0.7}
def _bpd(idx):
    if len(idx)<3: return 1.0
    s=pd.Series(idx).diff().median().total_seconds(); return max(1.0,86400.0/s) if s else 1.0
class VolRegime(Strategy):
    ma_days=50;vol_days=20;vol_lookback=252;size_frac=0.7
    def init(self):
        b=_bpd(self.data.index)
        self.ma=self.I(talib.SMA,self.data.Close,max(10,int(self.ma_days*b)))
        self._vn=max(5,int(self.vol_days*b)); self._vl=max(50,int(self.vol_lookback*b))
        self.rv=self.I(lambda x: pd.Series(x).pct_change().rolling(self._vn).std().values, np.asarray(self.data.Close))
    def next(self):
        if np.isnan(self.ma[-1]) or np.isnan(self.rv[-1]): return
        hist=self.rv[-self._vl:]; hist=hist[~np.isnan(hist)]
        if len(hist)<self._vl//2: return
        med=np.percentile(hist,50); hi=np.percentile(hist,75); v=self.rv[-1]
        if v>hi:
            if self.position: self.position.close()
            return
        up=self.data.Close[-1]>self.ma[-1]
        if v<=med:
            if self.position and ((self.position.is_long and not up) or (self.position.is_short and up)):
                self.position.close()
            if not self.position:
                (self.buy if up else self.sell)(size=self.size_frac)
STRATEGY_CLASS=VolRegime
def generate_signal(df):
    p=PARAMS
    if len(df)<p["vol_lookback"]+2: return {"action":"NOTHING","confidence":0,"reasoning":"insufficient"}
    rv=df["Close"].pct_change().rolling(p["vol_days"]).std()
    hi=rv.iloc[-p["vol_lookback"]:].quantile(0.75); med=rv.iloc[-p["vol_lookback"]:].quantile(0.5)
    v=rv.iloc[-1]; ma=df["Close"].rolling(p["ma_days"]).mean().iloc[-1]; up=df["Close"].iloc[-1]>ma
    if v>hi: return {"action":"NOTHING","confidence":0,"reasoning":"high-vol: chill"}
    if v<=med: return {"action":"BUY" if up else "SELL","confidence":55,"reasoning":"calm regime trend"}
    return {"action":"NOTHING","confidence":0,"reasoning":"mid-vol: hold"}
