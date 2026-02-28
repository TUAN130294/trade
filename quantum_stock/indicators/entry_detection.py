# -*- coding: utf-8 -*-
"""
Entry Point Detection & Support/Resistance Algorithms
=====================================================
Comprehensive technical analysis for Vietnam stock market

Algorithms included:
1. Support/Resistance Detection (Multiple methods)
2. Fibonacci Retracement/Extension
3. Volume Profile (POC, VAH, VAL)
4. Entry Point Detection (Breakout, Pullback, Reversal)
5. VWAP Analysis
6. Smart Money Concepts (Order Blocks, FVG)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class EntryType(Enum):
    """Types of entry signals"""
    BREAKOUT = "breakout"
    PULLBACK = "pullback"
    REVERSAL = "reversal"
    SUPPORT_BOUNCE = "support_bounce"
    RESISTANCE_BREAK = "resistance_break"
    VWAP_RECLAIM = "vwap_reclaim"
    ORDER_BLOCK = "order_block"


@dataclass
class SupportResistanceLevel:
    """Support/Resistance level with metadata"""
    price: float
    level_type: str  # 'support' or 'resistance'
    strength: float  # 0-1, based on number of touches
    method: str  # 'pivot', 'fibonacci', 'volume_profile', 'swing'
    touches: int  # Number of times price touched this level
    last_touch: Optional[pd.Timestamp] = None

    def to_dict(self) -> Dict:
        return {
            'price': self.price,
            'type': self.level_type,
            'strength': self.strength,
            'method': self.method,
            'touches': self.touches,
            'last_touch': self.last_touch.isoformat() if self.last_touch else None
        }


@dataclass
class EntrySignal:
    """Entry point signal with full context"""
    entry_type: EntryType
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    risk_reward: float
    confidence: float  # 0-1
    reasoning: str
    timestamp: pd.Timestamp

    def to_dict(self) -> Dict:
        return {
            'entry_type': self.entry_type.value,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit_1': self.take_profit_1,
            'take_profit_2': self.take_profit_2,
            'take_profit_3': self.take_profit_3,
            'risk_reward': self.risk_reward,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


class SupportResistanceDetector:
    """
    Multi-method Support/Resistance Detection

    Methods:
    1. Swing High/Low (Local extrema)
    2. Pivot Points (Classic, Fibonacci, Camarilla)
    3. Volume Profile (POC, VAH, VAL)
    4. Fibonacci Levels
    5. Round Numbers (Psychological levels)
    """

    def __init__(self, lookback: int = 100):
        self.lookback = lookback

    def detect_all(self, df: pd.DataFrame) -> Dict[str, List[SupportResistanceLevel]]:
        """
        Detect S/R using all methods and combine

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dict with 'support' and 'resistance' lists
        """
        all_levels = []

        # Method 1: Swing High/Low
        swing_levels = self._swing_levels(df)
        all_levels.extend(swing_levels)

        # Method 2: Pivot Points
        pivot_levels = self._pivot_points(df)
        all_levels.extend(pivot_levels)

        # Method 3: Volume Profile
        volume_levels = self._volume_profile(df)
        all_levels.extend(volume_levels)

        # Method 4: Fibonacci
        fib_levels = self._fibonacci_levels(df)
        all_levels.extend(fib_levels)

        # Method 5: Round Numbers
        round_levels = self._round_numbers(df)
        all_levels.extend(round_levels)

        # Cluster and merge nearby levels
        merged = self._merge_levels(all_levels, df['close'].iloc[-1])

        return merged

    def _swing_levels(self, df: pd.DataFrame, window: int = 5) -> List[SupportResistanceLevel]:
        """Detect swing high/low levels"""
        levels = []
        high = df['high']
        low = df['low']
        close = df['close']

        # Find swing highs (local maxima)
        for i in range(window, len(high) - window):
            if high.iloc[i] == high.iloc[i-window:i+window+1].max():
                # Count touches
                touches = ((high >= high.iloc[i] * 0.99) &
                          (high <= high.iloc[i] * 1.01)).sum()

                levels.append(SupportResistanceLevel(
                    price=high.iloc[i],
                    level_type='resistance',
                    strength=min(1.0, touches / 5),
                    method='swing',
                    touches=touches,
                    last_touch=df.index[i] if isinstance(df.index[i], pd.Timestamp) else None
                ))

        # Find swing lows (local minima)
        for i in range(window, len(low) - window):
            if low.iloc[i] == low.iloc[i-window:i+window+1].min():
                touches = ((low >= low.iloc[i] * 0.99) &
                          (low <= low.iloc[i] * 1.01)).sum()

                levels.append(SupportResistanceLevel(
                    price=low.iloc[i],
                    level_type='support',
                    strength=min(1.0, touches / 5),
                    method='swing',
                    touches=touches,
                    last_touch=df.index[i] if isinstance(df.index[i], pd.Timestamp) else None
                ))

        return levels

    def _pivot_points(self, df: pd.DataFrame) -> List[SupportResistanceLevel]:
        """Calculate classic pivot points from recent data"""
        levels = []

        # Use last complete period
        h = df['high'].iloc[-20:].max()
        l = df['low'].iloc[-20:].min()
        c = df['close'].iloc[-1]

        # Classic Pivot Points
        pp = (h + l + c) / 3
        r1 = 2 * pp - l
        r2 = pp + (h - l)
        r3 = h + 2 * (pp - l)
        s1 = 2 * pp - h
        s2 = pp - (h - l)
        s3 = l - 2 * (h - pp)

        pivot_levels = [
            ('resistance', r3, 'R3'),
            ('resistance', r2, 'R2'),
            ('resistance', r1, 'R1'),
            ('support', s1, 'S1'),
            ('support', s2, 'S2'),
            ('support', s3, 'S3'),
        ]

        for level_type, price, name in pivot_levels:
            levels.append(SupportResistanceLevel(
                price=price,
                level_type=level_type,
                strength=0.7,  # Pivot points are generally reliable
                method=f'pivot_{name}',
                touches=0
            ))

        return levels

    def _volume_profile(self, df: pd.DataFrame, num_bins: int = 50) -> List[SupportResistanceLevel]:
        """
        Volume Profile analysis - POC, VAH, VAL

        POC = Point of Control (highest volume price)
        VAH = Value Area High (70% volume boundary)
        VAL = Value Area Low (70% volume boundary)
        """
        levels = []

        if 'volume' not in df.columns or df['volume'].sum() == 0:
            return levels

        # Create price bins
        price_range = df['high'].max() - df['low'].min()
        bin_size = price_range / num_bins

        # Calculate volume at each price level
        volume_profile = {}

        for i in range(len(df)):
            # Distribute volume across the candle's range
            candle_low = df['low'].iloc[i]
            candle_high = df['high'].iloc[i]
            candle_volume = df['volume'].iloc[i]

            # Number of bins this candle spans
            bins_in_candle = max(1, int((candle_high - candle_low) / bin_size))
            volume_per_bin = candle_volume / bins_in_candle

            for b in range(bins_in_candle):
                price_level = round(candle_low + b * bin_size, 2)
                volume_profile[price_level] = volume_profile.get(price_level, 0) + volume_per_bin

        if not volume_profile:
            return levels

        # Find POC (Point of Control)
        poc_price = max(volume_profile, key=volume_profile.get)

        # Calculate Value Area (70% of volume)
        total_volume = sum(volume_profile.values())
        target_volume = total_volume * 0.7

        # Start from POC and expand outward
        sorted_prices = sorted(volume_profile.keys())
        poc_idx = sorted_prices.index(poc_price) if poc_price in sorted_prices else len(sorted_prices) // 2

        accumulated = volume_profile.get(poc_price, 0)
        low_idx = poc_idx
        high_idx = poc_idx

        while accumulated < target_volume and (low_idx > 0 or high_idx < len(sorted_prices) - 1):
            # Expand to the side with more volume
            vol_below = volume_profile.get(sorted_prices[low_idx - 1], 0) if low_idx > 0 else 0
            vol_above = volume_profile.get(sorted_prices[high_idx + 1], 0) if high_idx < len(sorted_prices) - 1 else 0

            if vol_below >= vol_above and low_idx > 0:
                low_idx -= 1
                accumulated += vol_below
            elif high_idx < len(sorted_prices) - 1:
                high_idx += 1
                accumulated += vol_above
            else:
                break

        vah = sorted_prices[high_idx] if high_idx < len(sorted_prices) else poc_price
        val = sorted_prices[low_idx] if low_idx >= 0 else poc_price

        # Add levels
        current_price = df['close'].iloc[-1]

        levels.append(SupportResistanceLevel(
            price=poc_price,
            level_type='support' if poc_price < current_price else 'resistance',
            strength=0.9,  # POC is very strong
            method='volume_profile_POC',
            touches=0
        ))

        levels.append(SupportResistanceLevel(
            price=vah,
            level_type='resistance',
            strength=0.75,
            method='volume_profile_VAH',
            touches=0
        ))

        levels.append(SupportResistanceLevel(
            price=val,
            level_type='support',
            strength=0.75,
            method='volume_profile_VAL',
            touches=0
        ))

        return levels

    def _fibonacci_levels(self, df: pd.DataFrame) -> List[SupportResistanceLevel]:
        """
        Fibonacci retracement and extension levels

        Based on recent swing high/low
        """
        levels = []

        # Find recent swing high and low
        lookback = min(self.lookback, len(df))
        recent = df.iloc[-lookback:]

        swing_high = recent['high'].max()
        swing_low = recent['low'].min()
        swing_range = swing_high - swing_low

        if swing_range == 0:
            return levels

        # Determine trend direction
        first_half_avg = recent.iloc[:lookback//2]['close'].mean()
        second_half_avg = recent.iloc[lookback//2:]['close'].mean()
        uptrend = second_half_avg > first_half_avg

        # Fibonacci levels
        fib_ratios = {
            '0.236': 0.236,
            '0.382': 0.382,
            '0.5': 0.5,
            '0.618': 0.618,  # Golden ratio
            '0.786': 0.786,
            '1.0': 1.0,
            '1.272': 1.272,  # Extension
            '1.618': 1.618,  # Extension
        }

        for name, ratio in fib_ratios.items():
            if uptrend:
                # Retracement from high
                price = swing_high - swing_range * ratio
            else:
                # Retracement from low
                price = swing_low + swing_range * ratio

            current_price = df['close'].iloc[-1]
            level_type = 'support' if price < current_price else 'resistance'

            # Golden ratio levels are stronger
            strength = 0.85 if ratio in [0.618, 1.618] else 0.65

            levels.append(SupportResistanceLevel(
                price=price,
                level_type=level_type,
                strength=strength,
                method=f'fibonacci_{name}',
                touches=0
            ))

        return levels

    def _round_numbers(self, df: pd.DataFrame) -> List[SupportResistanceLevel]:
        """
        Psychological round number levels

        VN market: 10k, 20k, 50k, 100k, etc.
        """
        levels = []
        current_price = df['close'].iloc[-1]

        # Determine round number intervals based on price
        if current_price < 20:
            intervals = [1, 5, 10]
        elif current_price < 50:
            intervals = [5, 10, 20]
        elif current_price < 100:
            intervals = [10, 20, 50]
        else:
            intervals = [20, 50, 100]

        for interval in intervals:
            # Find nearest round numbers
            lower = (current_price // interval) * interval
            upper = lower + interval

            if lower > 0:
                levels.append(SupportResistanceLevel(
                    price=lower,
                    level_type='support',
                    strength=0.5,
                    method=f'round_{interval}k',
                    touches=0
                ))

            levels.append(SupportResistanceLevel(
                price=upper,
                level_type='resistance',
                strength=0.5,
                method=f'round_{interval}k',
                touches=0
            ))

        return levels

    def _merge_levels(self, levels: List[SupportResistanceLevel],
                      current_price: float,
                      tolerance: float = 0.02) -> Dict[str, List[SupportResistanceLevel]]:
        """
        Merge nearby levels and sort by relevance

        Confluence (multiple methods at same level) = higher strength
        """
        if not levels:
            return {'support': [], 'resistance': []}

        # Separate by type
        supports = [l for l in levels if l.level_type == 'support']
        resistances = [l for l in levels if l.level_type == 'resistance']

        def merge_group(group: List[SupportResistanceLevel]) -> List[SupportResistanceLevel]:
            if not group:
                return []

            # Sort by price
            sorted_levels = sorted(group, key=lambda x: x.price)
            merged = []

            i = 0
            while i < len(sorted_levels):
                # Find cluster of nearby levels
                cluster = [sorted_levels[i]]
                j = i + 1

                while j < len(sorted_levels):
                    if abs(sorted_levels[j].price - cluster[0].price) / cluster[0].price < tolerance:
                        cluster.append(sorted_levels[j])
                        j += 1
                    else:
                        break

                # Merge cluster
                avg_price = np.mean([l.price for l in cluster])
                max_strength = max(l.strength for l in cluster)
                total_touches = sum(l.touches for l in cluster)
                methods = list(set(l.method for l in cluster))

                # Confluence bonus
                confluence_bonus = min(0.2, len(cluster) * 0.05)

                merged.append(SupportResistanceLevel(
                    price=avg_price,
                    level_type=cluster[0].level_type,
                    strength=min(1.0, max_strength + confluence_bonus),
                    method=','.join(methods[:3]),
                    touches=total_touches
                ))

                i = j

            return merged

        merged_supports = merge_group(supports)
        merged_resistances = merge_group(resistances)

        # Sort by distance to current price
        merged_supports = sorted(merged_supports,
                                key=lambda x: -x.price)  # Closest first
        merged_resistances = sorted(merged_resistances,
                                   key=lambda x: x.price)  # Closest first

        # Keep top 5 each
        return {
            'support': merged_supports[:5],
            'resistance': merged_resistances[:5]
        }


class EntryPointDetector:
    """
    Entry Point Detection Algorithms

    Detects optimal entry points based on:
    1. Breakout with volume confirmation
    2. Pullback to support/moving average
    3. Reversal patterns at S/R
    4. VWAP reclaim
    5. Order block entries (Smart Money)
    """

    def __init__(self, sr_detector: SupportResistanceDetector = None):
        self.sr_detector = sr_detector or SupportResistanceDetector()

    def detect_entries(self, df: pd.DataFrame) -> List[EntrySignal]:
        """
        Detect all entry signals in current market condition

        Returns list of entry signals sorted by confidence
        """
        entries = []

        # Get S/R levels
        sr_levels = self.sr_detector.detect_all(df)

        # Check each entry type
        breakout = self._detect_breakout(df, sr_levels)
        if breakout:
            entries.append(breakout)

        pullback = self._detect_pullback(df, sr_levels)
        if pullback:
            entries.append(pullback)

        reversal = self._detect_reversal(df, sr_levels)
        if reversal:
            entries.append(reversal)

        vwap_entry = self._detect_vwap_entry(df)
        if vwap_entry:
            entries.append(vwap_entry)

        # Sort by confidence
        entries.sort(key=lambda x: x.confidence, reverse=True)

        return entries

    def _detect_breakout(self, df: pd.DataFrame,
                         sr_levels: Dict) -> Optional[EntrySignal]:
        """
        Detect breakout entry

        Criteria:
        - Close above resistance with volume spike
        - Previous candles tested but failed to break
        - Volume > 1.5x average
        """
        if not sr_levels['resistance']:
            return None

        current = df.iloc[-1]
        prev = df.iloc[-2]

        nearest_resistance = sr_levels['resistance'][0].price

        # Check for breakout
        broke_out = current['close'] > nearest_resistance
        was_below = prev['close'] <= nearest_resistance

        # Volume confirmation
        avg_volume = df['volume'].iloc[-20:].mean()
        volume_spike = current['volume'] > avg_volume * 1.5

        if broke_out and was_below and volume_spike:
            # Calculate entry levels
            entry = current['close']
            stop_loss = nearest_resistance * 0.98  # Below breakout level

            # Take profits using ATR
            atr = self._calculate_atr(df)
            tp1 = entry + atr * 1.5
            tp2 = entry + atr * 2.5
            tp3 = entry + atr * 4

            risk = entry - stop_loss
            reward = tp2 - entry
            rr = reward / risk if risk > 0 else 0

            # Confidence based on volume and strength of level
            confidence = min(0.9, 0.5 + (current['volume'] / avg_volume - 1) * 0.2 +
                           sr_levels['resistance'][0].strength * 0.2)

            return EntrySignal(
                entry_type=EntryType.BREAKOUT,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit_1=tp1,
                take_profit_2=tp2,
                take_profit_3=tp3,
                risk_reward=rr,
                confidence=confidence,
                reasoning=f"Breakout above {nearest_resistance:.2f} with {current['volume']/avg_volume:.1f}x volume",
                timestamp=df.index[-1] if isinstance(df.index[-1], pd.Timestamp) else pd.Timestamp.now()
            )

        return None

    def _detect_pullback(self, df: pd.DataFrame,
                         sr_levels: Dict) -> Optional[EntrySignal]:
        """
        Detect pullback entry

        Criteria:
        - In uptrend (price > EMA20)
        - Pulled back to support or EMA
        - Showing bounce (green candle)
        """
        if len(df) < 20:
            return None

        current = df.iloc[-1]
        prev = df.iloc[-2]

        # Calculate EMAs
        ema20 = df['close'].ewm(span=20).mean()
        ema50 = df['close'].ewm(span=50).mean()

        # Check uptrend
        in_uptrend = ema20.iloc[-1] > ema50.iloc[-1]

        if not in_uptrend:
            return None

        # Check pullback to EMA20
        touched_ema = prev['low'] <= ema20.iloc[-2] * 1.01
        bounced = current['close'] > current['open']  # Green candle
        above_ema = current['close'] > ema20.iloc[-1]

        if touched_ema and bounced and above_ema:
            entry = current['close']
            stop_loss = min(prev['low'], ema20.iloc[-1]) * 0.98

            atr = self._calculate_atr(df)
            tp1 = entry + atr * 1.5
            tp2 = entry + atr * 2.5
            tp3 = entry + atr * 4

            risk = entry - stop_loss
            reward = tp2 - entry
            rr = reward / risk if risk > 0 else 0

            confidence = 0.7 if current['volume'] > df['volume'].iloc[-20:].mean() else 0.6

            return EntrySignal(
                entry_type=EntryType.PULLBACK,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit_1=tp1,
                take_profit_2=tp2,
                take_profit_3=tp3,
                risk_reward=rr,
                confidence=confidence,
                reasoning=f"Pullback to EMA20 ({ema20.iloc[-1]:.2f}) in uptrend, showing bounce",
                timestamp=df.index[-1] if isinstance(df.index[-1], pd.Timestamp) else pd.Timestamp.now()
            )

        return None

    def _detect_reversal(self, df: pd.DataFrame,
                         sr_levels: Dict) -> Optional[EntrySignal]:
        """
        Detect reversal entry at support

        Criteria:
        - Price at strong support level
        - Reversal candlestick pattern (hammer, engulfing)
        - RSI oversold
        """
        if not sr_levels['support']:
            return None

        current = df.iloc[-1]

        nearest_support = sr_levels['support'][0].price
        support_strength = sr_levels['support'][0].strength

        # Check if at support
        at_support = abs(current['low'] - nearest_support) / nearest_support < 0.02

        if not at_support:
            return None

        # Check RSI
        rsi = self._calculate_rsi(df)
        oversold = rsi.iloc[-1] < 35

        # Check for hammer pattern
        body = abs(current['close'] - current['open'])
        lower_shadow = min(current['open'], current['close']) - current['low']
        upper_shadow = current['high'] - max(current['open'], current['close'])

        is_hammer = lower_shadow > 2 * body and upper_shadow < body * 0.5
        is_bullish = current['close'] > current['open']

        if at_support and (oversold or is_hammer) and is_bullish:
            entry = current['close']
            stop_loss = nearest_support * 0.97  # Below support

            atr = self._calculate_atr(df)
            tp1 = entry + atr * 1.5
            tp2 = entry + atr * 2.5
            tp3 = entry + atr * 4

            risk = entry - stop_loss
            reward = tp2 - entry
            rr = reward / risk if risk > 0 else 0

            confidence = 0.5 + support_strength * 0.2 + (0.15 if oversold else 0) + (0.15 if is_hammer else 0)

            return EntrySignal(
                entry_type=EntryType.REVERSAL,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit_1=tp1,
                take_profit_2=tp2,
                take_profit_3=tp3,
                risk_reward=rr,
                confidence=confidence,
                reasoning=f"Reversal at support {nearest_support:.2f} with RSI={rsi.iloc[-1]:.0f}",
                timestamp=df.index[-1] if isinstance(df.index[-1], pd.Timestamp) else pd.Timestamp.now()
            )

        return None

    def _detect_vwap_entry(self, df: pd.DataFrame) -> Optional[EntrySignal]:
        """
        Detect VWAP reclaim entry

        Criteria:
        - Price reclaimed VWAP from below
        - Volume confirmation
        """
        if len(df) < 20 or 'volume' not in df.columns:
            return None

        # Calculate VWAP
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

        current = df.iloc[-1]
        prev = df.iloc[-2]

        # Check VWAP reclaim
        was_below = prev['close'] < vwap.iloc[-2]
        reclaimed = current['close'] > vwap.iloc[-1]

        if was_below and reclaimed:
            entry = current['close']
            stop_loss = vwap.iloc[-1] * 0.98

            atr = self._calculate_atr(df)
            tp1 = entry + atr * 1.5
            tp2 = entry + atr * 2.5
            tp3 = entry + atr * 4

            risk = entry - stop_loss
            reward = tp2 - entry
            rr = reward / risk if risk > 0 else 0

            avg_volume = df['volume'].iloc[-20:].mean()
            volume_factor = min(0.2, (current['volume'] / avg_volume - 1) * 0.1)
            confidence = 0.6 + volume_factor

            return EntrySignal(
                entry_type=EntryType.VWAP_RECLAIM,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit_1=tp1,
                take_profit_2=tp2,
                take_profit_3=tp3,
                risk_reward=rr,
                confidence=confidence,
                reasoning=f"VWAP reclaim at {vwap.iloc[-1]:.2f}",
                timestamp=df.index[-1] if isinstance(df.index[-1], pd.Timestamp) else pd.Timestamp.now()
            )

        return None

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]

        return atr

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return rsi


class TechnicalAnalysisSuite:
    """
    Complete Technical Analysis Suite

    Combines all indicators and detection algorithms
    """

    def __init__(self):
        self.sr_detector = SupportResistanceDetector()
        self.entry_detector = EntryPointDetector(self.sr_detector)

    def full_analysis(self, df: pd.DataFrame, symbol: str = "") -> Dict:
        """
        Run complete technical analysis

        Returns:
            Dict with all analysis results
        """
        # Support/Resistance
        sr_levels = self.sr_detector.detect_all(df)

        # Entry signals
        entries = self.entry_detector.detect_entries(df)

        # Current price context
        current_price = df['close'].iloc[-1]

        # Nearest levels
        nearest_support = sr_levels['support'][0] if sr_levels['support'] else None
        nearest_resistance = sr_levels['resistance'][0] if sr_levels['resistance'] else None

        # Price position analysis
        if nearest_support and nearest_resistance:
            support_distance = (current_price - nearest_support.price) / current_price
            resistance_distance = (nearest_resistance.price - current_price) / current_price
            position_in_range = support_distance / (support_distance + resistance_distance)
        else:
            position_in_range = 0.5

        # Trend analysis
        ema20 = df['close'].ewm(span=20).mean().iloc[-1]
        ema50 = df['close'].ewm(span=50).mean().iloc[-1]
        trend = "UPTREND" if ema20 > ema50 else "DOWNTREND" if ema20 < ema50 else "SIDEWAYS"

        # Best entry
        best_entry = entries[0] if entries else None

        return {
            'symbol': symbol,
            'current_price': current_price,
            'trend': trend,
            'position_in_range': position_in_range,
            'support_resistance': {
                'support': [s.to_dict() for s in sr_levels['support']],
                'resistance': [r.to_dict() for r in sr_levels['resistance']]
            },
            'nearest_support': nearest_support.to_dict() if nearest_support else None,
            'nearest_resistance': nearest_resistance.to_dict() if nearest_resistance else None,
            'entry_signals': [e.to_dict() for e in entries],
            'best_entry': best_entry.to_dict() if best_entry else None,
            'recommendation': self._generate_recommendation(
                current_price, nearest_support, nearest_resistance, trend, best_entry
            )
        }

    def _generate_recommendation(self, price: float,
                                  support: Optional[SupportResistanceLevel],
                                  resistance: Optional[SupportResistanceLevel],
                                  trend: str,
                                  entry: Optional[EntrySignal]) -> str:
        """Generate actionable recommendation"""
        parts = []

        # Trend context
        if trend == "UPTREND":
            parts.append("Xu hướng TĂNG - ưu tiên tìm điểm MUA")
        elif trend == "DOWNTREND":
            parts.append("Xu hướng GIẢM - thận trọng, đợi đảo chiều")
        else:
            parts.append("Xu hướng SIDEWAY - giao dịch trong range")

        # S/R context
        if support:
            parts.append(f"Hỗ trợ gần nhất: {support.price:,.0f} (strength: {support.strength:.0%})")
        if resistance:
            parts.append(f"Kháng cự gần nhất: {resistance.price:,.0f} (strength: {resistance.strength:.0%})")

        # Entry signal
        if entry:
            parts.append(f"\n📍 TÍN HIỆU VÀO LỆNH: {entry.entry_type.value.upper()}")
            parts.append(f"   Entry: {entry.entry_price:,.0f}")
            parts.append(f"   Stop Loss: {entry.stop_loss:,.0f}")
            parts.append(f"   TP1/TP2/TP3: {entry.take_profit_1:,.0f} / {entry.take_profit_2:,.0f} / {entry.take_profit_3:,.0f}")
            parts.append(f"   R:R = 1:{entry.risk_reward:.1f}")
            parts.append(f"   Confidence: {entry.confidence:.0%}")
        else:
            parts.append("\n⏳ Chưa có tín hiệu entry rõ ràng - chờ setup")

        return "\n".join(parts)


# Convenience functions for direct use
def detect_support_resistance(df: pd.DataFrame) -> Dict:
    """Quick S/R detection"""
    detector = SupportResistanceDetector()
    levels = detector.detect_all(df)
    return {
        'support': [s.to_dict() for s in levels['support']],
        'resistance': [r.to_dict() for r in levels['resistance']]
    }


def detect_entry_points(df: pd.DataFrame) -> List[Dict]:
    """Quick entry detection"""
    detector = EntryPointDetector()
    entries = detector.detect_entries(df)
    return [e.to_dict() for e in entries]


def full_technical_analysis(df: pd.DataFrame, symbol: str = "") -> Dict:
    """Quick full analysis"""
    suite = TechnicalAnalysisSuite()
    return suite.full_analysis(df, symbol)
