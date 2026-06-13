"""Regime-layer v2 of spec_274f084b1d_SqueezeTrendGate.py (pre-registered 2026-06-12). Only change:
entry size scaled by _regime_factor; no other logic touched."""
import numpy as np
import pandas as pd
import talib
from backtesting import Strategy
from backtesting.lib import crossover

PARAMS = {
    "bb_period": 20,
    "bb_std": 2.0,
    "ema_trend_period": 200,
    "ema_slope_lookback": 20,
    "band_width_lookback": 100,
    "band_width_percentile_threshold": 10,
    "account_risk_fraction": 0.02,
    "max_position_fraction": 0.1,
    "max_stop_distance_atr": 3.0,
    "atr_period": 14
}


class SqueezeTrendGate(Strategy):
    bb_period = 20
    bb_std = 2.0
    ema_trend_period = 200
    ema_slope_lookback = 20
    band_width_lookback = 100
    band_width_percentile_threshold = 10
    account_risk_fraction = 0.02
    max_position_fraction = 0.1
    max_stop_distance_atr = 3.0
    atr_period = 14
    
    def init(self):
        self.regime_sma200 = self.I(talib.SMA, self.data.Close, timeperiod=200)
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        
        # Bollinger Bands
        self.bb_upper = self.I(talib.BBANDS, close, self.bb_period, self.bb_std, self.bb_std, 0)[0]
        self.bb_middle = self.I(talib.BBANDS, close, self.bb_period, self.bb_std, self.bb_std, 0)[1]
        self.bb_lower = self.I(talib.BBANDS, close, self.bb_period, self.bb_std, self.bb_std, 0)[2]
        
        # Band width
        self.band_width = self.I(lambda: self.bb_upper - self.bb_lower)
        
        # 200 EMA
        self.ema_200 = self.I(talib.EMA, close, self.ema_trend_period)
        
        # ATR for stop distance validation
        self.atr = self.I(talib.ATR, high, low, close, self.atr_period)
        
        # Track concurrent positions
        self.position_count = 0
    

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
        # Need sufficient history
        if len(self.data) < max(self.ema_trend_period, self.band_width_lookback) + self.ema_slope_lookback:
            return
        
        # Current values
        close = self.data.Close[-1]
        prev_high = self.data.High[-2]
        prev_low = self.data.Low[-2]
        
        bb_middle = self.bb_middle[-1]
        band_width = self.band_width[-1]
        ema_200_current = self.ema_200[-1]
        ema_200_past = self.ema_200[-self.ema_slope_lookback]
        atr_current = self.atr[-1]
        
        # Calculate band width percentile (no lookahead)
        lookback_start = max(0, len(self.band_width) - self.band_width_lookback - 1)
        lookback_end = len(self.band_width) - 1  # Exclude current bar
        historical_band_widths = self.band_width[lookback_start:lookback_end]
        
        if len(historical_band_widths) < self.band_width_lookback:
            return
        
        percentile_threshold = np.percentile(historical_band_widths, self.band_width_percentile_threshold)
        in_squeeze = band_width <= percentile_threshold
        
        # EMA slope
        ema_slope_positive = ema_200_current > ema_200_past
        ema_slope_negative = ema_200_current < ema_200_past
        
        # Exit logic: close beyond middle band
        if self.position:
            if self.position.is_long and close < bb_middle:
                self.position.close()
                return
            elif self.position.is_short and close > bb_middle:
                self.position.close()
                return
        
        # Entry logic
        if not self.position:
            # Count concurrent positions (single instrument in backtesting.py)
            concurrent_positions = 1 if self.position else 0
            
            if concurrent_positions >= 3:
                return
            
            # Long entry conditions
            if in_squeeze and close > bb_middle and ema_slope_positive:
                stop_price = prev_low
                stop_distance = close - stop_price
                
                if stop_distance <= 0:
                    return
                
                # Validate stop distance
                if atr_current > 0 and stop_distance > self.max_stop_distance_atr * atr_current:
                    return
                
                # Position sizing: equal risk
                risk_amount = self.account_risk_fraction * self.equity
                position_value = risk_amount / (stop_distance / close)
                position_fraction = position_value / self.equity
                
                # Cap at max position fraction
                position_fraction = min(position_fraction, self.max_position_fraction)
                position_fraction = max(0.01, min(0.99, position_fraction))
                
                self.buy(size=min(0.99, position_fraction * self._regime_factor(True)), sl=stop_price)
            
            # Short entry conditions
            elif in_squeeze and close < bb_middle and ema_slope_negative:
                stop_price = prev_high
                stop_distance = stop_price - close
                
                if stop_distance <= 0:
                    return
                
                # Validate stop distance
                if atr_current > 0 and stop_distance > self.max_stop_distance_atr * atr_current:
                    return
                
                # Position sizing: equal risk
                risk_amount = self.account_risk_fraction * self.equity
                position_value = risk_amount / (stop_distance / close)
                position_fraction = position_value / self.equity
                
                # Cap at max position fraction
                position_fraction = min(position_fraction, self.max_position_fraction)
                position_fraction = max(0.01, min(0.99, position_fraction))
                
                self.sell(size=min(0.99, position_fraction * self._regime_factor(False)), sl=stop_price)


STRATEGY_CLASS = SqueezeTrendGate


def generate_signal(df):
    """
    Generate trading signal from OHLCV DataFrame.
    Returns dict with action, confidence, and reasoning.
    """
    if len(df) < max(PARAMS["ema_trend_period"], PARAMS["band_width_lookback"]) + PARAMS["ema_slope_lookback"]:
        return {
            "action": "NOTHING",
            "confidence": 0,
            "reasoning": "Insufficient data for signal generation"
        }
    
    # Calculate indicators
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = talib.BBANDS(
        close, PARAMS["bb_period"], PARAMS["bb_std"], PARAMS["bb_std"], 0
    )
    
    # Band width
    band_width = bb_upper - bb_lower
    
    # 200 EMA
    ema_200 = talib.EMA(close, PARAMS["ema_trend_period"])
    
    # ATR
    atr = talib.ATR(high, low, close, PARAMS["atr_period"])
    
    # Current values (last completed bar)
    close_current = close[-1]
    prev_high = high[-2]
    prev_low = low[-2]
    bb_middle_current = bb_middle[-1]
    band_width_current = band_width[-1]
    ema_200_current = ema_200[-1]
    ema_200_past = ema_200[-PARAMS["ema_slope_lookback"]]
    atr_current = atr[-1]
    
    # Band width percentile (no lookahead)
    lookback_start = max(0, len(band_width) - PARAMS["band_width_lookback"] - 1)
    lookback_end = len(band_width) - 1
    historical_band_widths = band_width[lookback_start:lookback_end]
    
    if len(historical_band_widths) < PARAMS["band_width_lookback"]:
        return {
            "action": "NOTHING",
            "confidence": 0,
            "reasoning": "Insufficient band width history"
        }
    
    percentile_threshold = np.percentile(historical_band_widths, PARAMS["band_width_percentile_threshold"])
    in_squeeze = band_width_current <= percentile_threshold
    
    # EMA slope
    ema_slope_positive = ema_200_current > ema_200_past
    ema_slope_negative = ema_200_current < ema_200_past
    
    # Check conditions
    if not in_squeeze:
        return {
            "action": "NOTHING",
            "confidence": 0,
            "reasoning": f"Band width {band_width_current:.4f} not in squeeze (threshold: {percentile_threshold:.4f})"
        }
    
    # Long signal
    if close_current > bb_middle_current and ema_slope_positive:
        stop_distance = close_current - prev_low
        if stop_distance <= 0:
            return {
                "action": "NOTHING",
                "confidence": 0,
                "reasoning": "Invalid stop distance for long"
            }
        
        if atr_current > 0 and stop_distance > PARAMS["max_stop_distance_atr"] * atr_current:
            return {
                "action": "NOTHING",
                "confidence": 0,
                "reasoning": f"Stop distance {stop_distance:.4f} exceeds max ATR multiple"
            }
        
        confidence = min(100, int(50 + (close_current - bb_middle_current) / (bb_upper[-1] - bb_middle_current) * 50))
        return {
            "action": "BUY",
            "confidence": confidence,
            "reasoning": f"Squeeze breakout long: band width in lowest decile, close above 20-SMA, 200-EMA slope positive"
        }
    
    # Short signal
    elif close_current < bb_middle_current and ema_slope_negative:
        stop_distance = prev_high - close_current
        if stop_distance <= 0:
            return {
                "action": "NOTHING",
                "confidence": 0,
                "reasoning": "Invalid stop distance for short"
            }
        
        if atr_current > 0 and stop_distance > PARAMS["max_stop_distance_atr"] * atr_current:
            return {
                "action": "NOTHING",
                "confidence": 0,
                "reasoning": f"Stop distance {stop_distance:.4f} exceeds max ATR multiple"
            }
        
        confidence = min(100, int(50 + (bb_middle_current - close_current) / (bb_middle_current - bb_lower[-1]) * 50))
        return {
            "action": "SELL",
            "confidence": confidence,
            "reasoning": f"Squeeze breakout short: band width in lowest decile, close below 20-SMA, 200-EMA slope negative"
        }
    
    return {
        "action": "NOTHING",
        "confidence": 0,
        "reasoning": "In squeeze but no directional breakout aligned with trend"
    }