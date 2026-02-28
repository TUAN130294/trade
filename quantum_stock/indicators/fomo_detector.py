"""
FOMO (Fear of Missing Out) Detection Engine for Vietnamese Stock Market
Detects retail investor behavior patterns unique to VN market
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from enum import Enum


class FOMOSignal(Enum):
    """FOMO detection signal types"""
    NO_FOMO = "no_fomo"
    FOMO_BUILDING = "fomo_building"
    FOMO_PEAK = "fomo_peak"
    FOMO_EXHAUSTION = "fomo_exhaustion"
    FOMO_TRAP = "fomo_trap"


class FOMODetector:
    """
    FOMO Detection Engine for VN Market

    Detects retail investor panic buying patterns:
    - Ceiling chase velocity
    - Volume acceleration
    - RSI-volume divergence
    - Consecutive gap ups
    - Bid dominance ratio
    - Market breadth FOMO
    """

    def __init__(self):
        self.rsi_threshold_peak = 80
        self.rsi_threshold_building = 70
        self.volume_accel_threshold = 2.0
        self.gap_up_threshold = 0.02  # 2% gap

    def detect(
        self,
        df: pd.DataFrame,
        market_breadth: Dict[str, float] = None
    ) -> Tuple[FOMOSignal, float, Dict[str, any]]:
        """
        Main FOMO detection method

        Args:
            df: OHLCV DataFrame with columns [open, high, low, close, volume]
            market_breadth: Optional dict with keys:
                - advancing_pct: % of stocks advancing
                - turnover_ratio: current turnover / 20D avg
                - ceiling_hits: number of stocks hitting ceiling

        Returns:
            Tuple of (FOMOSignal, confidence 0-1, metrics dict)
        """
        if len(df) < 20:
            return FOMOSignal.NO_FOMO, 0.5, {'reason': 'insufficient_data'}

        metrics = {}

        # 1. Ceiling chase velocity
        ceiling_velocity = self._calculate_ceiling_chase_velocity(df)
        metrics['ceiling_chase_velocity'] = ceiling_velocity

        # 2. Volume acceleration
        vol_accel, consecutive_accel = self._calculate_volume_acceleration(df)
        metrics['volume_acceleration'] = vol_accel
        metrics['consecutive_volume_accel'] = consecutive_accel

        # 3. RSI-volume divergence
        rsi = self._calculate_rsi(df['close'], 14)
        rsi_vol_divergence = self._detect_rsi_volume_divergence(df, rsi)
        metrics['rsi'] = rsi.iloc[-1] if len(rsi) > 0 else 50
        metrics['rsi_volume_divergence'] = rsi_vol_divergence

        # 4. Consecutive gap ups
        gap_ups = self._count_consecutive_gap_ups(df)
        metrics['consecutive_gap_ups'] = gap_ups

        # 5. Bid dominance ratio (if bid/ask data available)
        bid_dominance = self._calculate_bid_dominance(df)
        metrics['bid_dominance_ratio'] = bid_dominance

        # 6. Breadth FOMO
        breadth_fomo = self._calculate_breadth_fomo(market_breadth)
        metrics['breadth_fomo'] = breadth_fomo

        # Determine FOMO signal
        signal, confidence = self._generate_signal(metrics)

        return signal, confidence, metrics

    def _calculate_ceiling_chase_velocity(self, df: pd.DataFrame) -> float:
        """
        Calculate velocity of price approaching ceiling (+7% limit)
        Higher velocity = more aggressive FOMO
        """
        if len(df) < 5:
            return 0.0

        close = df['close'].iloc[-5:]
        ref_price = df['close'].iloc[-6] if len(df) > 5 else df['close'].iloc[-5]
        ceiling = ref_price * 1.07

        # Distance to ceiling
        distance_to_ceiling = (ceiling - close) / close * 100

        # Velocity: rate of change in distance (negative = approaching ceiling)
        velocity = distance_to_ceiling.diff().mean()

        # Normalize: -1.0 (very fast) to 0 (not moving toward ceiling)
        normalized_velocity = max(-1.0, min(0.0, velocity / 2.0))

        return abs(normalized_velocity)

    def _calculate_volume_acceleration(self, df: pd.DataFrame) -> Tuple[float, int]:
        """
        Detect volume acceleration pattern
        Returns: (current acceleration ratio, consecutive accelerations count)
        """
        if len(df) < 10:
            return 1.0, 0

        volume = df['volume'].iloc[-10:]

        # Volume ratio day-over-day
        vol_ratios = volume / volume.shift(1)

        # Current acceleration
        current_accel = vol_ratios.iloc[-1] if not pd.isna(vol_ratios.iloc[-1]) else 1.0

        # Count consecutive days with vol_ratio > threshold
        consecutive = 0
        for ratio in vol_ratios.iloc[-5:]:
            if ratio > self.volume_accel_threshold:
                consecutive += 1
            else:
                consecutive = 0

        return current_accel, consecutive

    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _detect_rsi_volume_divergence(self, df: pd.DataFrame, rsi: pd.Series) -> bool:
        """
        Detect dangerous RSI-volume divergence:
        RSI > 80 + volume spike = peak FOMO
        """
        if len(df) < 20 or len(rsi) < 20:
            return False

        current_rsi = rsi.iloc[-1]
        avg_volume_20 = df['volume'].iloc[-20:].mean()
        current_volume = df['volume'].iloc[-1]

        vol_spike = current_volume > avg_volume_20 * 1.5
        high_rsi = current_rsi > self.rsi_threshold_peak

        return vol_spike and high_rsi

    def _count_consecutive_gap_ups(self, df: pd.DataFrame) -> int:
        """
        Count consecutive gap-up sessions
        Gap up = open > previous close by > 2%
        """
        if len(df) < 5:
            return 0

        gaps = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        gap_ups = gaps > self.gap_up_threshold

        # Count consecutive from most recent
        consecutive = 0
        for is_gap_up in gap_ups.iloc[-5:].values[::-1]:
            if is_gap_up:
                consecutive += 1
            else:
                break

        return consecutive

    def _calculate_bid_dominance(self, df: pd.DataFrame) -> float:
        """
        Calculate bid dominance ratio
        If bid/ask volume data not available, estimate from price action
        """
        if 'bid_volume' in df.columns and 'ask_volume' in df.columns:
            recent_bid = df['bid_volume'].iloc[-5:].sum()
            recent_ask = df['ask_volume'].iloc[-5:].sum()

            if recent_ask > 0:
                return recent_bid / recent_ask

        # Estimate from price action: strong closes = bid pressure
        recent_df = df.iloc[-5:]
        body_position = (recent_df['close'] - recent_df['low']) / (
            recent_df['high'] - recent_df['low']
        ).replace(0, np.nan)

        # Average close position (0.5 = neutral, 1.0 = all bid pressure)
        avg_position = body_position.mean()

        # Convert to ratio (0.5 -> 1.0, 1.0 -> 3.0)
        estimated_ratio = 1.0 + (avg_position - 0.5) * 4

        return max(0.5, min(3.0, estimated_ratio))

    def _calculate_breadth_fomo(self, market_breadth: Dict[str, float]) -> float:
        """
        Calculate market-wide FOMO score from breadth data

        FOMO conditions:
        - Advancing% > 80%
        - Turnover spike > 2x
        - Many ceiling hits
        """
        if not market_breadth:
            return 0.0

        score = 0.0

        advancing_pct = market_breadth.get('advancing_pct', 50)
        if advancing_pct > 80:
            score += 0.4
        elif advancing_pct > 70:
            score += 0.2

        turnover_ratio = market_breadth.get('turnover_ratio', 1.0)
        if turnover_ratio > 2.0:
            score += 0.3
        elif turnover_ratio > 1.5:
            score += 0.15

        ceiling_hits = market_breadth.get('ceiling_hits', 0)
        if ceiling_hits > 20:
            score += 0.3
        elif ceiling_hits > 10:
            score += 0.15

        return min(1.0, score)

    def _generate_signal(self, metrics: Dict) -> Tuple[FOMOSignal, float]:
        """
        Generate FOMO signal from metrics

        Decision logic:
        - FOMO_PEAK: RSI > 80, volume spike, high ceiling velocity
        - FOMO_BUILDING: RSI > 70, volume acceleration, gap ups
        - FOMO_EXHAUSTION: Peak conditions + bid dominance declining
        - FOMO_TRAP: Peak + breadth divergence (narrow rally)
        - NO_FOMO: Normal conditions
        """
        rsi = metrics.get('rsi', 50)
        vol_accel = metrics.get('consecutive_volume_accel', 0)
        gap_ups = metrics.get('consecutive_gap_ups', 0)
        ceiling_velocity = metrics.get('ceiling_chase_velocity', 0)
        rsi_vol_div = metrics.get('rsi_volume_divergence', False)
        bid_dominance = metrics.get('bid_dominance_ratio', 1.0)
        breadth_fomo = metrics.get('breadth_fomo', 0.0)

        # FOMO_PEAK: Extreme conditions
        if (rsi_vol_div or
            (rsi > self.rsi_threshold_peak and ceiling_velocity > 0.6)):

            # Check for TRAP condition
            if breadth_fomo < 0.3:  # Narrow rally without broad participation
                return FOMOSignal.FOMO_TRAP, 0.85

            return FOMOSignal.FOMO_PEAK, 0.90

        # FOMO_EXHAUSTION: Peak conditions + weakening bid pressure
        if rsi > 75 and bid_dominance < 1.2:
            return FOMOSignal.FOMO_EXHAUSTION, 0.80

        # FOMO_BUILDING: Early warning signs
        if ((rsi > self.rsi_threshold_building and vol_accel >= 2) or
            gap_ups >= 3 or
            (ceiling_velocity > 0.5 and breadth_fomo > 0.5)):

            return FOMOSignal.FOMO_BUILDING, 0.70

        # NO_FOMO: Normal market
        return FOMOSignal.NO_FOMO, 0.50


# Convenience function
def detect_fomo(
    df: pd.DataFrame,
    market_breadth: Dict[str, float] = None
) -> Dict[str, any]:
    """
    Convenience function for FOMO detection

    Returns dict with signal, confidence, and all metrics
    """
    detector = FOMODetector()
    signal, confidence, metrics = detector.detect(df, market_breadth)

    return {
        'signal': signal,
        'signal_name': signal.value,
        'confidence': confidence,
        'metrics': metrics
    }
