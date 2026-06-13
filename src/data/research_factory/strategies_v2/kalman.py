"""Kalman-filter trend (local-level + local-trend / integrated random walk),
hand-written 2026-06-13. Causal state-space estimate of price level and slope;
long when filtered slope > 0, short when < 0. One smoothing parameter (q_ratio =
process/measurement variance). Distinct from SMA/EMA: adaptive, no fixed window.
No lookahead - the filter only ingests past closes."""
import numpy as np, pandas as pd
from backtesting import Strategy
PARAMS = {"q_ratio": 1e-4, "min_slope_atr": 0.0, "size_frac": 0.5}

def _kalman_slope(close, q_ratio):
    n=len(close); slope=np.full(n,np.nan)
    x=np.array([close[0],0.0]); P=np.eye(2); R=1.0
    Q=np.array([[q_ratio,0],[0,q_ratio]]); F=np.array([[1.0,1.0],[0.0,1.0]]); H=np.array([1.0,0.0])
    for t in range(1,n):
        x=F@x; P=F@P@F.T+Q
        y=close[t]-H@x; S=H@P@H+R; K=P@H/S
        x=x+K*y; P=(np.eye(2)-np.outer(K,H))@P
        slope[t]=x[1]
    return slope

class KalmanTrend(Strategy):
    q_ratio=1e-4; min_slope_atr=0.0; size_frac=0.5
    def init(self):
        import talib
        self.slope=self.I(_kalman_slope, np.asarray(self.data.Close), self.q_ratio)
        self.atr=self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
    def next(self):
        s=self.slope[-1]
        if np.isnan(s): return
        up=s>0
        if self.position:
            if self.position.is_long and not up: self.position.close()
            elif self.position.is_short and up: self.position.close()
            return
        if up: self.buy(size=self.size_frac)
        else: self.sell(size=self.size_frac)

STRATEGY_CLASS=KalmanTrend
def generate_signal(df):
    if len(df)<30: return {"action":"NOTHING","confidence":0,"reasoning":"insufficient"}
    s=_kalman_slope(df["Close"].values, PARAMS["q_ratio"])[-1]
    if np.isnan(s): return {"action":"NOTHING","confidence":0,"reasoning":"warmup"}
    return {"action":"BUY" if s>0 else "SELL","confidence":60,"reasoning":f"kalman slope {s:.4f}"}
