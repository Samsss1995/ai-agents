"""Quad moving-average alignment (Nick Kautz style), hand-written 2026-06-13.
Four EMAs; long only when fully stacked bullish (fast>...>slow), short when fully
bearish, flat otherwise. Stricter than the dead triple-MA (4-way alignment =
fewer, higher-conviction trades). Adaptive periods."""
import numpy as np, pandas as pd, talib
from backtesting import Strategy
PARAMS={"f1":5,"f2":13,"f3":34,"f4":89,"size_frac":0.6}
def _bpd(idx):
    if len(idx)<3: return 1.0
    s=pd.Series(idx).diff().median().total_seconds(); return max(1.0,86400.0/s) if s else 1.0
class QuadMA(Strategy):
    f1=5;f2=13;f3=34;f4=89;size_frac=0.6
    def init(self):
        b=_bpd(self.data.index)
        self.m1=self.I(talib.EMA,self.data.Close,max(2,int(self.f1*b)))
        self.m2=self.I(talib.EMA,self.data.Close,max(3,int(self.f2*b)))
        self.m3=self.I(talib.EMA,self.data.Close,max(5,int(self.f3*b)))
        self.m4=self.I(talib.EMA,self.data.Close,max(8,int(self.f4*b)))
    def next(self):
        a,b,c,d=self.m1[-1],self.m2[-1],self.m3[-1],self.m4[-1]
        if np.isnan(d): return
        bull=a>b>c>d; bear=a<b<c<d
        if self.position:
            if self.position.is_long and not bull: self.position.close()
            elif self.position.is_short and not bear: self.position.close()
            return
        if bull: self.buy(size=self.size_frac)
        elif bear: self.sell(size=self.size_frac)
STRATEGY_CLASS=QuadMA
def generate_signal(df):
    p=PARAMS
    if len(df)<p["f4"]+2: return {"action":"NOTHING","confidence":0,"reasoning":"insufficient"}
    e=lambda n: df["Close"].ewm(span=n).mean().iloc[-1]
    a,b,c,d=e(p["f1"]),e(p["f2"]),e(p["f3"]),e(p["f4"])
    if a>b>c>d: return {"action":"BUY","confidence":60,"reasoning":"quad MA bull stack"}
    if a<b<c<d: return {"action":"SELL","confidence":60,"reasoning":"quad MA bear stack"}
    return {"action":"NOTHING","confidence":0,"reasoning":"unaligned"}
