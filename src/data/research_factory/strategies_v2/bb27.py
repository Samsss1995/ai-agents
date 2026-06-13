"""Bollinger 2.7-SD breakout (Trent), hand-written 2026-06-13. Momentum breakout
at wide 2.7-SD bands (rarer, larger moves -> better cost ratio than the dead 2SD
version). Enter on close beyond the band in the breakout direction, exit at the
mid-band, ATR stop. Adaptive period."""
import numpy as np, pandas as pd, talib
from backtesting import Strategy
PARAMS={"bb_days":20,"sd":2.7,"atr_mult":2.0,"size_frac":0.6}
def _bpd(idx):
    if len(idx)<3: return 1.0
    s=pd.Series(idx).diff().median().total_seconds(); return max(1.0,86400.0/s) if s else 1.0
class BB27(Strategy):
    bb_days=20;sd=2.7;atr_mult=2.0;size_frac=0.6
    def init(self):
        n=max(10,int(self.bb_days*_bpd(self.data.index)))
        self.mid=self.I(talib.SMA,self.data.Close,n)
        self.sdv=self.I(lambda x: pd.Series(x).rolling(n).std().values, np.asarray(self.data.Close))
        self.atr=self.I(talib.ATR,self.data.High,self.data.Low,self.data.Close,14)
    def next(self):
        if np.isnan(self.sdv[-1]) or np.isnan(self.atr[-1]): return
        price=self.data.Close[-1]; up=self.mid[-1]+self.sd*self.sdv[-1]; lo=self.mid[-1]-self.sd*self.sdv[-1]
        if self.position:
            if self.position.is_long and price<self.mid[-1]: self.position.close()
            elif self.position.is_short and price>self.mid[-1]: self.position.close()
            return
        if price>up: self.buy(size=self.size_frac, sl=price-self.atr_mult*self.atr[-1])
        elif price<lo: self.sell(size=self.size_frac, sl=price+self.atr_mult*self.atr[-1])
STRATEGY_CLASS=BB27
def generate_signal(df):
    p=PARAMS
    if len(df)<p["bb_days"]+2: return {"action":"NOTHING","confidence":0,"reasoning":"insufficient"}
    mid=df["Close"].rolling(p["bb_days"]).mean().iloc[-1]; sd=df["Close"].rolling(p["bb_days"]).std().iloc[-1]
    price=df["Close"].iloc[-1]
    if price>mid+p["sd"]*sd: return {"action":"BUY","confidence":60,"reasoning":"2.7SD upside breakout"}
    if price<mid-p["sd"]*sd: return {"action":"SELL","confidence":60,"reasoning":"2.7SD downside breakout"}
    return {"action":"NOTHING","confidence":0,"reasoning":"inside bands"}
