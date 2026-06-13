"""Empirical Markov regime model, hand-written 2026-06-13. Each bar is assigned
one of 3 states by trailing-return tercile (down/flat/up). Using ONLY past bars
(expanding window), estimate each state's average forward 1-bar return; trade the
sign of the current state's historical forward return. Causal; coarse states (3)
to limit overfitting. The 'Markov/empirical-transition' approach (Simons-style
regime conditioning), judged by the gates like everything else."""
import numpy as np, pandas as pd
from backtesting import Strategy
PARAMS = {"ret_window": 20, "min_history": 250, "size_frac": 0.5}

class MarkovRegime(Strategy):
    ret_window=20; min_history=250; size_frac=0.5
    def init(self):
        c=np.asarray(self.data.Close)
        self.mom=self.I(lambda x: pd.Series(x).pct_change(self.ret_window).values, c)
        self.r1=self.I(lambda x: pd.Series(x).pct_change().values, c)
    def next(self):
        i=len(self.data)-1
        if i<self.min_history+1: return
        hist=self.mom[:i]; fwd=self.r1[1:i+1]  # forward returns aligned to past states
        valid=~np.isnan(hist) & ~np.isnan(fwd)
        h=hist[valid]; f=fwd[valid]
        if len(h)<self.min_history: return
        lo,hi=np.nanpercentile(h,33.3),np.nanpercentile(h,66.7)
        cur=self.mom[-1]
        if np.isnan(cur): return
        state = 0 if cur<lo else (2 if cur>hi else 1)
        mask = (f if state==1 else (f[h<lo] if state==0 else f[h>hi]))
        if state==1: mask=f[(h>=lo)&(h<=hi)]
        if len(mask)<30: return
        exp=np.mean(mask); up=exp>0
        if self.position:
            if self.position.is_long and not up: self.position.close()
            elif self.position.is_short and up: self.position.close()
            return
        if up: self.buy(size=self.size_frac)
        else: self.sell(size=self.size_frac)

STRATEGY_CLASS=MarkovRegime
def generate_signal(df):
    p=PARAMS
    if len(df)<p["min_history"]+p["ret_window"]+2:
        return {"action":"NOTHING","confidence":0,"reasoning":"insufficient"}
    mom=df["Close"].pct_change(p["ret_window"]).values; r1=df["Close"].pct_change().values
    # pair state[t] with realized forward return[t+1], using only bars before the last
    h=mom[:-2]; f=r1[1:-1]
    v=~np.isnan(h)&~np.isnan(f); h,f=h[v],f[v]
    lo,hi=np.nanpercentile(h,33.3),np.nanpercentile(h,66.7); cur=mom[-1]
    state=0 if cur<lo else (2 if cur>hi else 1)
    mask = f[h<lo] if state==0 else (f[h>hi] if state==2 else f[(h>=lo)&(h<=hi)])
    if len(mask)<30: return {"action":"NOTHING","confidence":0,"reasoning":"thin state"}
    return {"action":"BUY" if np.mean(mask)>0 else "SELL","confidence":55,"reasoning":f"state {state} exp {np.mean(mask):.4f}"}
