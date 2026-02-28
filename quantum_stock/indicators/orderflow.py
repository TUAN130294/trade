# -*- coding: utf-8 -*-
"""
Order Flow Analysis Indicators
Advanced indicators for reading institutional order flow
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class OrderFlowSignal:
    """Order flow trading signal"""
    signal_type: str
    direction: str
    strength: float
    price: float
    description: str
    
    def to_dict(self) -> Dict:
        return {
            'type': self.signal_type,
            'direction': self.direction,
            'strength': self.strength,
            'price': self.price,
            'description': self.description
        }


class OrderFlowIndicators:
    """Advanced Order Flow Analysis"""
    
    @staticmethod
    def vwap_bands(high: pd.Series, low: pd.Series, close: pd.Series,
                   volume: pd.Series, num_std: float = 2.0) -> Dict:
        """VWAP with Standard Deviation Bands"""
        typical_price = (high + low + close) / 3
        cumulative_tp_vol = (typical_price * volume).cumsum()
        cumulative_vol = volume.cumsum()
        vwap = cumulative_tp_vol / cumulative_vol.replace(0, np.nan)
        
        squared_diff = (typical_price - vwap) ** 2
        variance = (squared_diff * volume).cumsum() / cumulative_vol.replace(0, np.nan)
        std = np.sqrt(variance)
        
        return {
            'vwap': vwap,
            'upper_1': vwap + std,
            'upper_2': vwap + 2 * std,
            'lower_1': vwap - std,
            'lower_2': vwap - 2 * std,
            'distance_from_vwap': (close - vwap) / vwap * 100
        }
    
    @staticmethod
    def cumulative_delta(open_: pd.Series, high: pd.Series, low: pd.Series,
                         close: pd.Series, volume: pd.Series) -> Dict:
        """Cumulative Volume Delta"""
        full_range = (high - low).replace(0, np.nan)
        buy_ratio = ((close - low) / full_range).clip(0.2, 0.8).fillna(0.5)
        
        buy_volume = volume * buy_ratio
        sell_volume = volume * (1 - buy_ratio)
        delta = buy_volume - sell_volume
        
        return {
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'delta': delta,
            'cumulative_delta': delta.cumsum(),
            'delta_ema': delta.ewm(span=14).mean()
        }
    
    @staticmethod
    def absorption_exhaustion(open_: pd.Series, high: pd.Series, low: pd.Series,
                              close: pd.Series, volume: pd.Series) -> Dict:
        """Detect Absorption and Exhaustion patterns"""
        price_range = high - low
        avg_volume = volume.rolling(20).mean()
        avg_range = price_range.rolling(20).mean()
        
        volume_ratio = volume / avg_volume.replace(0, np.nan)
        range_ratio = price_range / avg_range.replace(0, np.nan)
        
        absorption = (volume_ratio > 1.5) & (range_ratio < 0.5)
        bullish_absorption = absorption & (close > (high + low) / 2)
        bearish_absorption = absorption & (close <= (high + low) / 2)
        
        return {
            'absorption': absorption,
            'bullish_absorption': bullish_absorption,
            'bearish_absorption': bearish_absorption,
            'volume_ratio': volume_ratio
        }


class WyckoffPatternDetector:
    """Wyckoff Smart Money Pattern Detection"""

    @staticmethod
    def detect_initiative_buying(open_: pd.Series, high: pd.Series, low: pd.Series,
                                  close: pd.Series, volume: pd.Series) -> Dict:
        """Detect Initiative Buying pattern (gap up + volume + hold)"""
        prev_close = close.shift(1)
        gap_pct = (open_ - prev_close) / prev_close * 100

        avg_vol = volume.rolling(20).mean()
        vol_ratio = volume / avg_vol.replace(0, np.nan)

        hold_strength = (close - open_) / open_ * 100

        initiative_buying = (gap_pct > 1.5) & (vol_ratio > 1.5) & (hold_strength > -0.5)

        return {
            'signal': initiative_buying,
            'gap_pct': gap_pct,
            'vol_ratio': vol_ratio,
            'hold_strength': hold_strength,
            'confidence': np.where(initiative_buying, 0.85, 0)
        }

    @staticmethod
    def detect_initiative_selling(open_: pd.Series, high: pd.Series, low: pd.Series,
                                   close: pd.Series, volume: pd.Series) -> Dict:
        """Detect Initiative Selling pattern (gap down + volume + no bounce)"""
        prev_close = close.shift(1)
        gap_pct = (open_ - prev_close) / prev_close * 100

        avg_vol = volume.rolling(20).mean()
        vol_ratio = volume / avg_vol.replace(0, np.nan)

        price_range = high - low
        bounce_ratio = (close - low) / price_range.replace(0, np.nan)

        initiative_selling = (gap_pct < -1.5) & (vol_ratio > 1.5) & (bounce_ratio < 0.3)

        return {
            'signal': initiative_selling,
            'gap_pct': gap_pct,
            'vol_ratio': vol_ratio,
            'bounce_ratio': bounce_ratio,
            'confidence': np.where(initiative_selling, 0.85, 0)
        }

    @staticmethod
    def detect_spring(high: pd.Series, low: pd.Series, close: pd.Series) -> Dict:
        """Detect Wyckoff Spring pattern (breaks support then reclaims)"""
        prev_low = low.rolling(3).min().shift(1)
        close_loc = (close - low) / (high - low + 0.001)

        spring = (low < prev_low) & (close > prev_low) & (close_loc > 0.6)

        return {
            'signal': spring,
            'prev_low': prev_low,
            'close_location': close_loc,
            'confidence': np.where(spring, 0.7, 0)
        }

    @staticmethod
    def detect_upthrust(high: pd.Series, low: pd.Series, close: pd.Series) -> Dict:
        """Detect Upthrust pattern (breaks resistance then fails)"""
        prev_high = high.rolling(3).max().shift(1)
        close_loc = (close - low) / (high - low + 0.001)

        upthrust = (high > prev_high) & (close < prev_high) & (close_loc < 0.3)

        return {
            'signal': upthrust,
            'prev_high': prev_high,
            'close_location': close_loc,
            'confidence': np.where(upthrust, 0.7, 0)
        }

    @staticmethod
    def detect_stopping_volume(close: pd.Series, volume: pd.Series) -> Dict:
        """Detect Stopping Volume (volume spike halts trend)"""
        price_change_5d = close.pct_change(5) * 100

        avg_vol = volume.rolling(20).mean()
        vol_ratio = volume / avg_vol.replace(0, np.nan)

        close_loc = (close - close.rolling(5).min()) / (close.rolling(5).max() - close.rolling(5).min() + 0.001)

        stopping = (vol_ratio > 3.0) & (price_change_5d > 3) & (close_loc < 0.4)

        return {
            'signal': stopping,
            'vol_ratio': vol_ratio,
            'price_change_5d': price_change_5d,
            'confidence': np.where(stopping, 0.8, 0)
        }

    @staticmethod
    def detect_effort_vs_result(open_: pd.Series, close: pd.Series, volume: pd.Series) -> Dict:
        """Detect Effort vs Result divergence (high volume, small move)"""
        avg_vol = volume.rolling(20).mean()
        vol_ratio = volume / avg_vol.replace(0, np.nan)

        price_move = abs(close - open_) / open_ * 100

        divergence = (vol_ratio > 2.5) & (price_move < 0.5)

        return {
            'signal': divergence,
            'vol_ratio': vol_ratio,
            'price_move_pct': price_move,
            'confidence': np.where(divergence, 0.7, 0)
        }


class ForeignFlowIndicators:
    """Foreign & Proprietary Flow Analysis for Vietnamese market"""

    @staticmethod
    def foreign_flow_analysis(foreign_buy: pd.Series, foreign_sell: pd.Series,
                              total_volume: pd.Series, close: pd.Series) -> Dict:
        """Analyze foreign investor flow"""
        net_foreign = foreign_buy - foreign_sell
        cumulative = net_foreign.cumsum()
        participation = (foreign_buy + foreign_sell) / total_volume * 100
        
        return {
            'net_foreign': net_foreign,
            'cumulative_foreign': cumulative,
            'participation_ratio': participation,
            'is_accumulating': net_foreign.rolling(5).mean() > 0
        }
    
    @staticmethod
    def smart_money_index(close: pd.Series, volume: pd.Series,
                          foreign_net: pd.Series = None) -> pd.Series:
        """Smart Money Index combining institutional flows"""
        price_momentum = close.pct_change(5)
        volume_ratio = volume / volume.rolling(20).mean().replace(0, np.nan)
        
        if foreign_net is not None:
            foreign_component = foreign_net / volume * 100
        else:
            foreign_component = pd.Series(0, index=close.index)
        
        smi = price_momentum.fillna(0) * 50 + volume_ratio.fillna(1) * 30 + foreign_component.fillna(0) * 20
        
        return ((smi - smi.rolling(50).min()) / 
                (smi.rolling(50).max() - smi.rolling(50).min() + 0.0001) * 100).fillna(50)
