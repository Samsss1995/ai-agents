"""Regime-layer v2 of spec_539152b420_TrendPullbackStochastic.py (pre-registered 2026-06-12). Only change:
entry size scaled by _regime_factor; no other logic touched."""
import numpy as np
import pandas as pd
import talib
from backtesting import Strategy
from backtesting.lib import crossover

PARAMS = {
    "ema_fast": 50,
    "ema_slow": 200,
    "stochrsi_period": 14,
    "stochrsi_entry_long": 0.3,
    "stochrsi_entry_short": 0.7,
    "stochrsi_exit_long": 0.9,
    "stochrsi_exit_short": 0.1,
    "atr_period": 14,
    "atr_multiplier": 2.0,
    "risk_per_trade": 0.01,
    "max_positions": 3,
}


class TrendPullbackStochasticStrategy(Strategy):
    ema_fast = 50
    ema_slow = 200
    stochrsi_period = 14
    stochrsi_entry_long = 0.3
    stochrsi_entry_short = 0.7
    stochrsi_exit_long = 0.9
    stochrsi_exit_short = 0.1
    atr_period = 14
    atr_multiplier = 2.0
    risk_per_trade = 0.01
    max_positions = 3

    def init(self):
        self.regime_sma200 = self.I(talib.SMA, self.data.Close, timeperiod=200)
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        self.ema_fast_line = self.I(talib.EMA, close, timeperiod=self.ema_fast)
        self.ema_slow_line = self.I(talib.EMA, close, timeperiod=self.ema_slow)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        
        # StochRSI calculation
        rsi = talib.RSI(close, timeperiod=self.stochrsi_period)
        stoch_k = talib.STOCH(
            rsi, rsi, rsi,
            fastk_period=self.stochrsi_period,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0
        )[0]
        self.stochrsi = self.I(lambda x: x / 100.0, stoch_k)
        
        # ATR 60-day average for volatility filter
        self.atr_60ma = self.I(talib.SMA, self.atr, timeperiod=60)
        
        self.entry_price = None
        self.stop_price = None


    def _regime_factor(self, is_long):
        # Pre-registered regime layer (2026-06-12): 200-bar SMA side agreement.
        # factor 1.0 when entry direction agrees with the SMA200 side, else 0.25.
        # Constants fixed by registration - deliberately NOT in PARAMS.
        sma = self.regime_sma200[-1]
        if sma != sma:  # NaN warmup
            return 0.25
        above = self.data.Close[-1] > sma
        agrees = (is_long and above) or ((not is_long) and (not above))
        return 1.0 if agrees else 0.25

    def next(self):
        # Skip if insufficient data
        if len(self.data) < max(self.ema_slow, 60 + self.atr_period):
            return
        
        # Current values
        close = self.data.Close[-1]
        ema_fast_val = self.ema_fast_line[-1]
        ema_slow_val = self.ema_slow_line[-1]
        stochrsi_val = self.stochrsi[-1]
        stochrsi_prev = self.stochrsi[-2]
        atr_val = self.atr[-1]
        atr_60ma_val = self.atr_60ma[-1]
        
        # Check if we have valid indicator values
        if np.isnan(ema_fast_val) or np.isnan(ema_slow_val) or np.isnan(stochrsi_val) or np.isnan(atr_val):
            return
        
        # Volatility filter: skip if ATR below 50% of 60-day average
        if not np.isnan(atr_60ma_val) and atr_val < 0.5 * atr_60ma_val:
            return
        
        # Regime detection
        uptrend = ema_fast_val > ema_slow_val
        downtrend = ema_fast_val < ema_slow_val
        
        # Exit logic for existing position
        if self.position:
            if self.position.is_long:
                # Exit long: close < EMA50 OR stochRSI > 0.9
                if close < ema_fast_val or stochrsi_val > self.stochrsi_exit_long:
                    self.position.close()
                    self.entry_price = None
                    self.stop_price = None
            elif self.position.is_short:
                # Exit short: close > EMA50 OR stochRSI < 0.1
                if close > ema_fast_val or stochrsi_val < self.stochrsi_exit_short:
                    self.position.close()
                    self.entry_price = None
                    self.stop_price = None
        
        # Entry logic (only if no position)
        if not self.position:
            # Count current positions (always 0 or 1 in this single-instrument context)
            # In multi-instrument backtesting, this would need to track across instruments
            # For now, we proceed with entry logic
            
            # Long entry: uptrend + stochRSI crosses above 0.3 from below
            if uptrend and stochrsi_prev < self.stochrsi_entry_long and stochrsi_val >= self.stochrsi_entry_long:
                # Calculate stop loss
                stop_distance = self.atr_multiplier * atr_val
                stop_price = close - stop_distance
                
                # Position sizing: risk 1% of equity
                # size_fraction = (equity * risk_per_trade) / (entry_price - stop_price)
                # Capped at 20% of equity
                if stop_distance > 0:
                    risk_fraction = self.risk_per_trade / (stop_distance / close)
                    size_fraction = min(0.20, max(0.01, risk_fraction))
                    
                    self.entry_price = close
                    self.stop_price = stop_price
                    self.buy(size=min(0.99, size_fraction * self._regime_factor(True)), sl=stop_price)
            
            # Short entry: downtrend + stochRSI crosses below 0.7 from above
            elif downtrend and stochrsi_prev > self.stochrsi_entry_short and stochrsi_val <= self.stochrsi_entry_short:
                # Calculate stop loss
                stop_distance = self.atr_multiplier * atr_val
                stop_price = close + stop_distance
                
                # Position sizing: risk 1% of equity
                if stop_distance > 0:
                    risk_fraction = self.risk_per_trade / (stop_distance / close)
                    size_fraction = min(0.20, max(0.01, risk_fraction))
                    
                    self.entry_price = close
                    self.stop_price = stop_price
                    self.sell(size=min(0.99, size_fraction * self._regime_factor(False)), sl=stop_price)


STRATEGY_CLASS = TrendPullbackStochasticStrategy


def generate_signal(df):
    """
    Generate trading signal from OHLCV DataFrame.
    Returns dict with action, confidence, and reasoning.
    """
    if len(df) < 250:
        return {
            "action": "NOTHING",
            "confidence": 0,
            "reasoning": "Insufficient data for analysis (need 250+ bars)"
        }
    
    # Use parameters from PARAMS
    ema_fast = PARAMS["ema_fast"]
    ema_slow = PARAMS["ema_slow"]
    stochrsi_period = PARAMS["stochrsi_period"]
    stochrsi_entry_long = PARAMS["stochrsi_entry_long"]
    stochrsi_entry_short = PARAMS["stochrsi_entry_short"]
    atr_period = PARAMS["atr_period"]
    
    # Calculate indicators
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    
    ema_fast_line = talib.EMA(close, timeperiod=ema_fast)
    ema_slow_line = talib.EMA(close, timeperiod=ema_slow)
    atr = talib.ATR(high, low, close, timeperiod=atr_period)
    
    # StochRSI
    rsi = talib.RSI(close, timeperiod=stochrsi_period)
    stoch_k, _ = talib.STOCH(
        rsi, rsi, rsi,
        fastk_period=stochrsi_period,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=3,
        slowd_matype=0
    )
    stochrsi = stoch_k / 100.0
    
    # ATR 60-day average
    atr_60ma = talib.SMA(atr, timeperiod=60)
    
    # Current and previous values (last completed bar)
    close_curr = close[-1]
    ema_fast_curr = ema_fast_line[-1]
    ema_slow_curr = ema_slow_line[-1]
    stochrsi_curr = stochrsi[-1]
    stochrsi_prev = stochrsi[-2]
    atr_curr = atr[-1]
    atr_60ma_curr = atr_60ma[-1]
    
    # Check for NaN
    if np.isnan(ema_fast_curr) or np.isnan(ema_slow_curr) or np.isnan(stochrsi_curr) or np.isnan(atr_curr):
        return {
            "action": "NOTHING",
            "confidence": 0,
            "reasoning": "Indicators not yet calculated (NaN values)"
        }
    
    # Volatility filter
    if not np.isnan(atr_60ma_curr) and atr_curr < 0.5 * atr_60ma_curr:
        return {
            "action": "NOTHING",
            "confidence": 0,
            "reasoning": f"Insufficient volatility: ATR {atr_curr:.2f} below 50% of 60-day average {atr_60ma_curr:.2f}"
        }
    
    # Regime detection
    uptrend = ema_fast_curr > ema_slow_curr
    downtrend = ema_fast_curr < ema_slow_curr
    
    # Long signal
    if uptrend and stochrsi_prev < stochrsi_entry_long and stochrsi_curr >= stochrsi_entry_long:
        confidence = min(100, int(70 + (ema_fast_curr - ema_slow_curr) / ema_slow_curr * 1000))
        return {
            "action": "BUY",
            "confidence": confidence,
            "reasoning": f"Uptrend confirmed (EMA{ema_fast} {ema_fast_curr:.2f} > EMA{ema_slow} {ema_slow_curr:.2f}), "
                        f"stochRSI crossed above {stochrsi_entry_long} (from {stochrsi_prev:.3f} to {stochrsi_curr:.3f}), "
                        f"signaling pullback exhaustion in uptrend"
        }
    
    # Short signal
    if downtrend and stochrsi_prev > stochrsi_entry_short and stochrsi_curr <= stochrsi_entry_short:
        confidence = min(100, int(70 + (ema_slow_curr - ema_fast_curr) / ema_slow_curr * 1000))
        return {
            "action": "SELL",
            "confidence": confidence,
            "reasoning": f"Downtrend confirmed (EMA{ema_fast} {ema_fast_curr:.2f} < EMA{ema_slow} {ema_slow_curr:.2f}), "
                        f"stochRSI crossed below {stochrsi_entry_short} (from {stochrsi_prev:.3f} to {stochrsi_curr:.3f}), "
                        f"signaling rally exhaustion in downtrend"
        }
    
    # No signal
    regime_desc = "uptrend" if uptrend else "downtrend" if downtrend else "neutral"
    return {
        "action": "NOTHING",
        "confidence": 0,
        "reasoning": f"No entry signal: regime={regime_desc}, stochRSI={stochrsi_curr:.3f} (prev={stochrsi_prev:.3f}), "
                    f"waiting for pullback exhaustion cross"
    }