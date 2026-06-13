"""Regime-layer v2 of spec_d0a4119045_PyramidTrendConvexity.py (pre-registered 2026-06-12). Only change:
entry size scaled by _regime_factor; no other logic touched."""
import numpy as np
import pandas as pd
import talib
from backtesting import Strategy
from backtesting.lib import crossover

PARAMS = {
    "ema_fast": 50,
    "ema_slow": 200,
    "reclaim_ma_period": 20,
    "atr_period": 14,
    "pyramid_atr_spacing": 1.0,
    "trailing_stop_atr_mult": 2.0,
    "max_pyramid_units": 4,
}


class PyramidTrendConvexityStrategy(Strategy):
    ema_fast = 50
    ema_slow = 200
    reclaim_ma_period = 20
    atr_period = 14
    pyramid_atr_spacing = 1.0
    trailing_stop_atr_mult = 2.0
    max_pyramid_units = 4

    def init(self):
        self.regime_sma200 = self.I(talib.SMA, self.data.Close, timeperiod=200)
        close = self.data.Close
        
        self.ema_fast_ind = self.I(talib.EMA, close, timeperiod=self.ema_fast)
        self.ema_slow_ind = self.I(talib.EMA, close, timeperiod=self.ema_slow)
        self.sma_reclaim = self.I(talib.SMA, close, timeperiod=self.reclaim_ma_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, close, timeperiod=self.atr_period)
        
        self.unit_count = 0
        self.last_entry_price = None
        self.highest_close_since_entry = None
        self.lowest_close_since_entry = None
        self.position_direction = None  # 'long' or 'short'
        self.prev_close_below_sma = None
        self.prev_close_above_sma = None


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
        close = self.data.Close[-1]
        ema_fast_val = self.ema_fast_ind[-1]
        ema_slow_val = self.ema_slow_ind[-1]
        sma_reclaim_val = self.sma_reclaim[-1]
        atr_val = self.atr[-1]
        
        if np.isnan(ema_fast_val) or np.isnan(ema_slow_val) or np.isnan(sma_reclaim_val) or np.isnan(atr_val):
            return
        
        long_regime = ema_fast_val > ema_slow_val
        short_regime = ema_fast_val < ema_slow_val
        
        # Track position state
        if self.position:
            if self.position.is_long:
                self.position_direction = 'long'
                if self.highest_close_since_entry is None or close > self.highest_close_since_entry:
                    self.highest_close_since_entry = close
            elif self.position.is_short:
                self.position_direction = 'short'
                if self.lowest_close_since_entry is None or close < self.lowest_close_since_entry:
                    self.lowest_close_since_entry = close
        else:
            self.unit_count = 0
            self.last_entry_price = None
            self.highest_close_since_entry = None
            self.lowest_close_since_entry = None
            self.position_direction = None
        
        # Regime invalidation check
        if self.position:
            if self.position.is_long and short_regime:
                self.position.close()
                return
            elif self.position.is_short and long_regime:
                self.position.close()
                return
        
        # Trailing stop check
        if self.position:
            if self.position.is_long and self.highest_close_since_entry is not None:
                trailing_stop = self.highest_close_since_entry - self.trailing_stop_atr_mult * atr_val
                if close <= trailing_stop:
                    self.position.close()
                    return
            elif self.position.is_short and self.lowest_close_since_entry is not None:
                trailing_stop = self.lowest_close_since_entry + self.trailing_stop_atr_mult * atr_val
                if close >= trailing_stop:
                    self.position.close()
                    return
        
        # Track SMA crossings
        prev_close = self.data.Close[-2] if len(self.data.Close) > 1 else None
        if prev_close is not None:
            self.prev_close_below_sma = prev_close < sma_reclaim_val
            self.prev_close_above_sma = prev_close > sma_reclaim_val
        
        # Entry and pyramiding logic
        if long_regime:
            if not self.position or (self.position and not self.position.is_long):
                # Initial long entry: reclaim 20-SMA
                if self.prev_close_below_sma and close > sma_reclaim_val:
                    # Check if trailing stop would trigger on same bar
                    hypothetical_highest = close
                    hypothetical_stop = hypothetical_highest - self.trailing_stop_atr_mult * atr_val
                    if close > hypothetical_stop:
                        size_fraction = min(0.25, max(0.01, 0.25))
                        self.buy(size=min(0.99, size_fraction * self._regime_factor(True)))
                        self.unit_count = 1
                        self.last_entry_price = close
                        self.highest_close_since_entry = close
            elif self.position and self.position.is_long:
                # Pyramiding: add unit if price is at least pyramid_atr_spacing ATR above last entry
                if self.unit_count < self.max_pyramid_units and self.last_entry_price is not None:
                    threshold = self.last_entry_price + self.pyramid_atr_spacing * atr_val
                    if close >= threshold:
                        # Check if trailing stop would trigger
                        hypothetical_highest = max(self.highest_close_since_entry, close)
                        hypothetical_stop = hypothetical_highest - self.trailing_stop_atr_mult * atr_val
                        if close > hypothetical_stop:
                            size_fraction = min(0.25, max(0.01, 0.25))
                            self.buy(size=min(0.99, size_fraction * self._regime_factor(True)))
                            self.unit_count += 1
                            self.last_entry_price = close
        
        elif short_regime:
            if not self.position or (self.position and not self.position.is_short):
                # Initial short entry: breakdown below 20-SMA
                if self.prev_close_above_sma and close < sma_reclaim_val:
                    # Check if trailing stop would trigger on same bar
                    hypothetical_lowest = close
                    hypothetical_stop = hypothetical_lowest + self.trailing_stop_atr_mult * atr_val
                    if close < hypothetical_stop:
                        size_fraction = min(0.25, max(0.01, 0.25))
                        self.sell(size=min(0.99, size_fraction * self._regime_factor(False)))
                        self.unit_count = 1
                        self.last_entry_price = close
                        self.lowest_close_since_entry = close
            elif self.position and self.position.is_short:
                # Pyramiding: add unit if price is at least pyramid_atr_spacing ATR below last entry
                if self.unit_count < self.max_pyramid_units and self.last_entry_price is not None:
                    threshold = self.last_entry_price - self.pyramid_atr_spacing * atr_val
                    if close <= threshold:
                        # Check if trailing stop would trigger
                        hypothetical_lowest = min(self.lowest_close_since_entry, close)
                        hypothetical_stop = hypothetical_lowest + self.trailing_stop_atr_mult * atr_val
                        if close < hypothetical_stop:
                            size_fraction = min(0.25, max(0.01, 0.25))
                            self.sell(size=min(0.99, size_fraction * self._regime_factor(False)))
                            self.unit_count += 1
                            self.last_entry_price = close


STRATEGY_CLASS = PyramidTrendConvexityStrategy


def generate_signal(df):
    """
    Generate trading signal from OHLCV DataFrame.
    Returns dict with action, confidence, and reasoning.
    """
    if len(df) < max(PARAMS["ema_slow"], PARAMS["atr_period"], PARAMS["reclaim_ma_period"]) + 1:
        return {
            "action": "NOTHING",
            "confidence": 0,
            "reasoning": "Insufficient data for indicator calculation"
        }
    
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    
    ema_fast = talib.EMA(close, timeperiod=PARAMS["ema_fast"])
    ema_slow = talib.EMA(close, timeperiod=PARAMS["ema_slow"])
    sma_reclaim = talib.SMA(close, timeperiod=PARAMS["reclaim_ma_period"])
    atr = talib.ATR(high, low, close, timeperiod=PARAMS["atr_period"])
    
    if np.isnan(ema_fast[-1]) or np.isnan(ema_slow[-1]) or np.isnan(sma_reclaim[-1]) or np.isnan(atr[-1]):
        return {
            "action": "NOTHING",
            "confidence": 0,
            "reasoning": "Indicators not yet valid"
        }
    
    current_close = close[-1]
    prev_close = close[-2]
    ema_fast_val = ema_fast[-1]
    ema_slow_val = ema_slow[-1]
    sma_val = sma_reclaim[-1]
    atr_val = atr[-1]
    
    long_regime = ema_fast_val > ema_slow_val
    short_regime = ema_fast_val < ema_slow_val
    
    # Initial entry signals
    if long_regime:
        if prev_close < sma_val and current_close > sma_val:
            confidence = min(100, int(50 + 50 * (ema_fast_val - ema_slow_val) / ema_slow_val * 100))
            return {
                "action": "BUY",
                "confidence": max(50, confidence),
                "reasoning": f"Long regime (EMA{PARAMS['ema_fast']} > EMA{PARAMS['ema_slow']}), price reclaimed {PARAMS['reclaim_ma_period']}-SMA from below"
            }
    
    elif short_regime:
        if prev_close > sma_val and current_close < sma_val:
            confidence = min(100, int(50 + 50 * (ema_slow_val - ema_fast_val) / ema_slow_val * 100))
            return {
                "action": "SELL",
                "confidence": max(50, confidence),
                "reasoning": f"Short regime (EMA{PARAMS['ema_fast']} < EMA{PARAMS['ema_slow']}), price broke down below {PARAMS['reclaim_ma_period']}-SMA"
            }
    
    return {
        "action": "NOTHING",
        "confidence": 0,
        "reasoning": "No entry signal: awaiting regime confirmation and SMA cross"
    }