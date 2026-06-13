"""VIX-regime equity timing, hand-written 2026-06-13. Uses the VixClose column
(equity vol index). Long equities only when VIX is below its 50-day average
(calm, risk-on); flat when VIX is elevated. Equity/index only - VIX is an equity
construct. Covers the 'VIX strategies' / vol-regime idea."""
import numpy as np, pandas as pd
from backtesting import Strategy
PARAMS={"vix_ma":50,"size_frac":0.95}
class VixRegime(Strategy):
    vix_ma=50; size_frac=0.95
    def init(self):
        self.vma=self.I(lambda x: pd.Series(x).rolling(self.vix_ma,min_periods=10).mean().values, np.asarray(self.data.VixClose))
    def next(self):
        v=self.data.VixClose[-1]; m=self.vma[-1]
        if np.isnan(v) or np.isnan(m): return
        calm=v<m
        if calm and not self.position: self.buy(size=self.size_frac)
        elif not calm and self.position: self.position.close()
STRATEGY_CLASS=VixRegime
def generate_signal(df):
    p=PARAMS
    if "VixClose" not in df.columns or len(df)<p["vix_ma"]+2:
        return {"action":"NOTHING","confidence":0,"reasoning":"no VIX/insufficient"}
    v=df["VixClose"].iloc[-1]; m=df["VixClose"].rolling(p["vix_ma"],min_periods=10).mean().iloc[-1]
    if v<m: return {"action":"BUY","confidence":55,"reasoning":f"VIX {v:.1f} < MA {m:.1f}: risk-on"}
    return {"action":"NOTHING","confidence":0,"reasoning":"VIX elevated: flat"}
