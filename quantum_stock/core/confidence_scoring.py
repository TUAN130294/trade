# -*- coding: utf-8 -*-
"""
Multi-Factor Confidence Scoring System
======================================
Advanced confidence calculation for trading signals

OLD (naive):
    confidence = min(0.9, abs(expected_return) * 10 + 0.5)

NEW (multi-factor):
    - Model accuracy factor
    - Market volatility factor
    - Volume confirmation factor
    - Technical alignment factor
    - Market regime factor
    - Historical performance factor
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ConfidenceLevel(Enum):
    """Confidence level categories"""
    VERY_HIGH = "very_high"    # 0.85+
    HIGH = "high"              # 0.70-0.85
    MEDIUM = "medium"          # 0.55-0.70
    LOW = "low"                # 0.40-0.55
    VERY_LOW = "very_low"      # <0.40


@dataclass
class ConfidenceResult:
    """Confidence calculation result with breakdown"""
    total_confidence: float
    confidence_level: ConfidenceLevel

    # Factor breakdown (required fields)
    return_factor: float
    model_accuracy_factor: float
    volatility_factor: float
    volume_factor: float
    technical_factor: float
    market_regime_factor: float

    # Reasoning
    reasoning: str

    # New factors (optional with defaults)
    money_flow_factor: float = 0.5
    foreign_flow_factor: float = 0.5
    fomo_penalty_factor: float = 1.0
    warnings: list = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    def to_dict(self) -> Dict:
        return {
            'total_confidence': self.total_confidence,
            'confidence_level': self.confidence_level.value,
            'factors': {
                'return': self.return_factor,
                'model_accuracy': self.model_accuracy_factor,
                'volatility': self.volatility_factor,
                'volume': self.volume_factor,
                'technical': self.technical_factor,
                'market_regime': self.market_regime_factor,
                'money_flow': self.money_flow_factor,
                'foreign_flow': self.foreign_flow_factor,
                'fomo_penalty': self.fomo_penalty_factor
            },
            'reasoning': self.reasoning,
            'warnings': self.warnings
        }


class MultiFactorConfidence:
    """
    Multi-factor confidence scoring system

    Factors and weights (ENHANCED for VN market):
    1. Expected Return Magnitude (15%)
    2. Model Historical Accuracy (15%)
    3. Market Volatility (inverse) (10%)
    4. Volume Confirmation (10%)
    5. Technical Alignment (10%)
    6. Market Regime Alignment (10%)
    7. Money Flow Score (15%) - NEW: smart money + Wyckoff
    8. Foreign Flow Score (10%) - NEW: khối ngoại net
    9. FOMO Penalty (5%) - NEW: inverse FOMO scoring
    """

    # Factor weights (total = 100%)
    WEIGHTS = {
        'return': 0.15,
        'model_accuracy': 0.15,
        'volatility': 0.10,
        'volume': 0.10,
        'technical': 0.10,
        'market_regime': 0.10,
        'money_flow': 0.15,
        'foreign_flow': 0.10,
        'fomo_penalty': 0.05
    }

    def __init__(self):
        # Model historical accuracy cache (symbol -> accuracy)
        self.model_accuracy_cache: Dict[str, float] = {}

    def calculate_confidence(
        self,
        expected_return: float,
        df: pd.DataFrame,
        symbol: str,
        model_accuracy: float = None,
        market_regime: str = "NEUTRAL",
        technical_score: float = None,
        flow_data: Dict[str, any] = None,
        fomo_signal: str = None
    ) -> ConfidenceResult:
        """
        Calculate multi-factor confidence score (ENHANCED with 9 factors)

        Args:
            expected_return: Expected return (e.g., 0.05 for 5%)
            df: Historical OHLCV data
            symbol: Stock symbol
            model_accuracy: Historical model accuracy (0-1)
            market_regime: Current market regime (BULL/BEAR/NEUTRAL)
            technical_score: Technical analysis score (0-100)
            flow_data: Money flow data dict with keys:
                - smart_money_index: float
                - cumulative_delta: float
                - absorption_signal: str (BULLISH/BEARISH/NEUTRAL)
                - foreign_net_5d: float (5-day accumulated foreign net)
            fomo_signal: FOMO signal from FOMODetector (NO_FOMO, FOMO_BUILDING, FOMO_PEAK, etc.)

        Returns:
            ConfidenceResult with full breakdown
        """
        warnings = []

        # 1. Return Factor (0-1)
        return_factor = self._calculate_return_factor(expected_return)

        # 2. Model Accuracy Factor (0-1)
        if model_accuracy is None:
            model_accuracy = self.model_accuracy_cache.get(symbol, 0.55)
        model_accuracy_factor = self._calculate_model_accuracy_factor(model_accuracy)

        # 3. Volatility Factor (0-1, inverse - low volatility = higher confidence)
        volatility_factor, vol_warning = self._calculate_volatility_factor(df)
        if vol_warning:
            warnings.append(vol_warning)

        # 4. Volume Factor (0-1)
        volume_factor, vol_warn = self._calculate_volume_factor(df)
        if vol_warn:
            warnings.append(vol_warn)

        # 5. Technical Factor (0-1)
        if technical_score is None:
            technical_score = self._calculate_technical_score(df)
        technical_factor = technical_score / 100

        # 6. Market Regime Factor (0-1)
        regime_factor = self._calculate_regime_factor(market_regime, expected_return)

        # 7. Money Flow Factor (0-1) - NEW
        money_flow_factor, mf_warning = self._calculate_money_flow_factor(df, flow_data)
        if mf_warning:
            warnings.append(mf_warning)

        # 8. Foreign Flow Factor (0-1) - NEW
        foreign_flow_factor, ff_warning = self._calculate_foreign_flow_factor(flow_data)
        if ff_warning:
            warnings.append(ff_warning)

        # 9. FOMO Penalty Factor (0-1) - NEW
        fomo_penalty_factor, fomo_warning = self._calculate_fomo_penalty_factor(fomo_signal, expected_return)
        if fomo_warning:
            warnings.append(fomo_warning)

        # Weighted sum
        total = (
            return_factor * self.WEIGHTS['return'] +
            model_accuracy_factor * self.WEIGHTS['model_accuracy'] +
            volatility_factor * self.WEIGHTS['volatility'] +
            volume_factor * self.WEIGHTS['volume'] +
            technical_factor * self.WEIGHTS['technical'] +
            regime_factor * self.WEIGHTS['market_regime'] +
            money_flow_factor * self.WEIGHTS['money_flow'] +
            foreign_flow_factor * self.WEIGHTS['foreign_flow'] +
            fomo_penalty_factor * self.WEIGHTS['fomo_penalty']
        )

        # Cap at 0.95 (never 100% confident)
        total = min(0.95, max(0.1, total))

        # Determine confidence level
        if total >= 0.85:
            level = ConfidenceLevel.VERY_HIGH
        elif total >= 0.70:
            level = ConfidenceLevel.HIGH
        elif total >= 0.55:
            level = ConfidenceLevel.MEDIUM
        elif total >= 0.40:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.VERY_LOW

        # Generate reasoning
        reasoning = self._generate_reasoning(
            total, return_factor, model_accuracy_factor,
            volatility_factor, volume_factor, technical_factor, regime_factor,
            money_flow_factor, foreign_flow_factor, fomo_penalty_factor
        )

        return ConfidenceResult(
            total_confidence=total,
            confidence_level=level,
            return_factor=return_factor,
            model_accuracy_factor=model_accuracy_factor,
            volatility_factor=volatility_factor,
            volume_factor=volume_factor,
            technical_factor=technical_factor,
            market_regime_factor=regime_factor,
            money_flow_factor=money_flow_factor,
            foreign_flow_factor=foreign_flow_factor,
            fomo_penalty_factor=fomo_penalty_factor,
            reasoning=reasoning,
            warnings=warnings
        )

    def _calculate_return_factor(self, expected_return: float) -> float:
        """
        Convert expected return to confidence factor

        Scale:
        - < 1%: 0.2 (too small to be significant)
        - 1-3%: 0.4-0.6 (moderate)
        - 3-7%: 0.6-0.8 (good)
        - 7-15%: 0.8-0.9 (very good)
        - > 15%: 0.7 (suspicious, reduce confidence)
        """
        ret = abs(expected_return)

        if ret < 0.01:
            return 0.2
        elif ret < 0.03:
            return 0.4 + (ret - 0.01) * 10  # 0.4 to 0.6
        elif ret < 0.07:
            return 0.6 + (ret - 0.03) * 5  # 0.6 to 0.8
        elif ret < 0.15:
            return 0.8 + (ret - 0.07) * 1.25  # 0.8 to 0.9
        else:
            # Too good to be true - reduce confidence
            return max(0.5, 0.9 - (ret - 0.15) * 2)

    def _calculate_model_accuracy_factor(self, accuracy: float) -> float:
        """
        Convert model accuracy to confidence factor

        Scale:
        - < 50%: 0.2 (worse than random)
        - 50-55%: 0.4-0.6 (marginally better than random)
        - 55-65%: 0.6-0.8 (good for stock prediction)
        - > 65%: 0.8-1.0 (excellent)
        """
        if accuracy < 0.50:
            return 0.2
        elif accuracy < 0.55:
            return 0.4 + (accuracy - 0.50) * 4  # 0.4 to 0.6
        elif accuracy < 0.65:
            return 0.6 + (accuracy - 0.55) * 2  # 0.6 to 0.8
        else:
            return min(1.0, 0.8 + (accuracy - 0.65) * 2)

    def _calculate_volatility_factor(self, df: pd.DataFrame) -> Tuple[float, Optional[str]]:
        """
        Calculate volatility factor (inverse - lower vol = higher factor)

        High volatility = less predictable = lower confidence
        """
        warning = None

        if len(df) < 20:
            return 0.5, "Insufficient data for volatility calculation"

        # Calculate 20-day realized volatility
        returns = df['close'].pct_change().dropna()
        vol_20d = returns.iloc[-20:].std() * np.sqrt(252)  # Annualized

        # VN market typical: 20-40% annualized
        if vol_20d < 0.15:
            factor = 0.9  # Very low vol
        elif vol_20d < 0.25:
            factor = 0.8  # Low vol
        elif vol_20d < 0.35:
            factor = 0.6  # Normal vol
        elif vol_20d < 0.50:
            factor = 0.4  # High vol
            warning = f"High volatility ({vol_20d*100:.0f}% annualized)"
        else:
            factor = 0.2  # Extreme vol
            warning = f"Extreme volatility ({vol_20d*100:.0f}% annualized) - reduce position"

        return factor, warning

    def _calculate_volume_factor(self, df: pd.DataFrame) -> Tuple[float, Optional[str]]:
        """
        Calculate volume confirmation factor

        High volume on signal = higher confidence
        """
        warning = None

        if 'volume' not in df.columns or len(df) < 20:
            return 0.5, None

        recent_vol = df['volume'].iloc[-1]
        avg_vol_20 = df['volume'].iloc[-20:].mean()

        if avg_vol_20 == 0:
            return 0.5, None

        vol_ratio = recent_vol / avg_vol_20

        if vol_ratio >= 2.0:
            factor = 0.9  # Strong volume confirmation
        elif vol_ratio >= 1.5:
            factor = 0.8
        elif vol_ratio >= 1.0:
            factor = 0.6
        elif vol_ratio >= 0.5:
            factor = 0.4
            warning = "Below average volume - weak confirmation"
        else:
            factor = 0.2
            warning = "Very low volume - signal may be unreliable"

        return factor, warning

    def _calculate_technical_score(self, df: pd.DataFrame) -> float:
        """
        Calculate basic technical alignment score (0-100)
        """
        if len(df) < 50:
            return 50

        score = 50
        close = df['close']

        # Price vs EMAs
        ema20 = close.ewm(span=20).mean().iloc[-1]
        ema50 = close.ewm(span=50).mean().iloc[-1]
        current = close.iloc[-1]

        if current > ema20 > ema50:
            score += 20  # Strong uptrend
        elif current > ema20:
            score += 10
        elif current < ema20 < ema50:
            score -= 20  # Strong downtrend
        elif current < ema20:
            score -= 10

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = (100 - (100 / (1 + rs))).iloc[-1]

        if 30 <= rsi <= 70:
            score += 10  # Neutral zone
        elif rsi < 30:
            score += 15  # Oversold (bullish for buy signals)
        elif rsi > 70:
            score -= 10  # Overbought

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()

        if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
            score += 15  # Bullish crossover
        elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
            score -= 15  # Bearish crossover

        return max(0, min(100, score))

    def _calculate_regime_factor(self, regime: str, expected_return: float) -> float:
        """
        Calculate market regime alignment factor

        Buy signals in BULL market = higher confidence
        Buy signals in BEAR market = lower confidence
        """
        is_bullish_signal = expected_return > 0

        if regime == "BULL":
            return 0.9 if is_bullish_signal else 0.3
        elif regime == "BEAR":
            return 0.3 if is_bullish_signal else 0.7
        else:  # NEUTRAL
            return 0.6

    def _calculate_money_flow_factor(
        self,
        df: pd.DataFrame,
        flow_data: Dict[str, any] = None
    ) -> Tuple[float, Optional[str]]:
        """
        Calculate money flow factor (15% weight)

        Composite of:
        - Smart Money Index
        - Cumulative Delta
        - Absorption signals

        Returns: (factor 0-1, optional warning)
        """
        if not flow_data:
            # Fallback: calculate basic A/D from OHLCV
            try:
                from quantum_stock.indicators.volume import VolumeIndicators

                if len(df) < 20:
                    return 0.5, "Insufficient data for money flow"

                ad_line = VolumeIndicators.accumulation_distribution(
                    df['high'], df['low'], df['close'], df['volume']
                )
                ad_slope = ad_line.iloc[-5:].diff().mean()

                # Normalize slope to 0-1
                if ad_slope > 0:
                    factor = min(1.0, 0.6 + ad_slope * 100)
                else:
                    factor = max(0.0, 0.4 + ad_slope * 100)

                return factor, None

            except Exception:
                return 0.5, "Money flow data unavailable"

        # Use provided flow data
        score = 0.5
        warning = None

        # Smart Money Index contribution
        smi = flow_data.get('smart_money_index', 0)
        if smi > 0:
            score += 0.2
        elif smi < 0:
            score -= 0.2

        # Cumulative Delta contribution
        cum_delta = flow_data.get('cumulative_delta', 0)
        if cum_delta > 0:
            score += 0.2
        elif cum_delta < 0:
            score -= 0.2

        # Absorption signal contribution
        absorption = flow_data.get('absorption_signal', 'NEUTRAL')
        if absorption == 'BULLISH':
            score += 0.1
        elif absorption == 'BEARISH':
            score -= 0.1
            warning = "Bearish absorption detected - selling pressure absorbed"

        return max(0.0, min(1.0, score)), warning

    def _calculate_foreign_flow_factor(
        self,
        flow_data: Dict[str, any] = None
    ) -> Tuple[float, Optional[str]]:
        """
        Calculate foreign flow factor (10% weight)

        Based on 5-day accumulated foreign net buy/sell

        Returns: (factor 0-1, optional warning)
        """
        if not flow_data or 'foreign_net_5d' not in flow_data:
            return 0.5, None  # Neutral if no data

        foreign_net_5d = flow_data.get('foreign_net_5d', 0)
        warning = None

        # Normalize based on typical VN foreign flow
        # Assume typical 5D net ranges from -50B to +50B VND
        normalized = foreign_net_5d / 50_000_000_000  # 50 billion VND

        # Convert to 0-1 scale
        if normalized > 0:
            # Foreign buying
            factor = min(1.0, 0.5 + normalized * 0.5)
        else:
            # Foreign selling
            factor = max(0.0, 0.5 + normalized * 0.5)

            if normalized < -0.5:
                warning = "Heavy foreign selling - institutional exit warning"

        return factor, warning

    def _calculate_fomo_penalty_factor(
        self,
        fomo_signal: str = None,
        expected_return: float = 0
    ) -> Tuple[float, Optional[str]]:
        """
        Calculate FOMO penalty factor (5% weight)

        Applied inversely to BUY confidence:
        - FOMO_PEAK → 0.2 (strong penalty for buying)
        - FOMO_BUILDING → 0.7 (moderate penalty)
        - NO_FOMO → 1.0 (no penalty)

        For SELL signals, FOMO_PEAK actually increases confidence

        Returns: (factor 0-1, optional warning)
        """
        if not fomo_signal or fomo_signal == 'NO_FOMO':
            return 1.0, None

        is_buy_signal = expected_return > 0
        warning = None

        if fomo_signal == 'FOMO_PEAK':
            if is_buy_signal:
                factor = 0.2  # Strong penalty for buying at peak
                warning = "FOMO PEAK detected - buying at top is dangerous!"
            else:
                factor = 1.0  # Good time to sell

        elif fomo_signal == 'FOMO_BUILDING':
            if is_buy_signal:
                factor = 0.7  # Moderate penalty
                warning = "FOMO building - retail chase detected"
            else:
                factor = 0.9

        elif fomo_signal == 'FOMO_EXHAUSTION':
            if is_buy_signal:
                factor = 0.3  # Strong penalty, exhaustion imminent
                warning = "FOMO exhaustion - rally losing steam"
            else:
                factor = 0.95  # Good sell signal

        elif fomo_signal == 'FOMO_TRAP':
            if is_buy_signal:
                factor = 0.1  # Extreme penalty - trap!
                warning = "FOMO TRAP - narrow rally without breadth, extreme danger!"
            else:
                factor = 1.0

        else:
            factor = 1.0

        return factor, warning

    def _generate_reasoning(
        self, total: float, ret: float, model: float,
        vol: float, volume: float, tech: float, regime: float,
        money_flow: float = 0.5, foreign_flow: float = 0.5, fomo_penalty: float = 1.0
    ) -> str:
        """Generate human-readable reasoning"""
        factors = [
            ("Return magnitude", ret),
            ("Model accuracy", model),
            ("Low volatility", vol),
            ("Volume confirmation", volume),
            ("Technical alignment", tech),
            ("Market regime", regime),
            ("Money flow", money_flow),
            ("Foreign flow", foreign_flow),
            ("FOMO penalty", fomo_penalty)
        ]

        # Sort by contribution
        factors.sort(key=lambda x: x[1], reverse=True)

        strengths = [f[0] for f in factors[:3]]
        weaknesses = [f[0] for f in factors[-3:] if f[1] < 0.5]

        reasoning = f"Confidence {total*100:.0f}%: "
        reasoning += f"Strong factors: {', '.join(strengths)}. "

        if weaknesses:
            reasoning += f"Weak factors: {', '.join(weaknesses)}."

        return reasoning

    def update_model_accuracy(self, symbol: str, accuracy: float):
        """Update cached model accuracy for symbol"""
        self.model_accuracy_cache[symbol] = accuracy


# Singleton instance
_confidence_calculator = None


def get_confidence_calculator() -> MultiFactorConfidence:
    """Get singleton confidence calculator"""
    global _confidence_calculator
    if _confidence_calculator is None:
        _confidence_calculator = MultiFactorConfidence()
    return _confidence_calculator


def calculate_confidence(
    expected_return: float,
    df: pd.DataFrame,
    symbol: str = "",
    model_accuracy: float = None,
    market_regime: str = "NEUTRAL"
) -> float:
    """
    Quick confidence calculation (returns just the score)

    Replaces:
        confidence = min(0.9, abs(expected_return) * 10 + 0.5)

    With:
        confidence = calculate_confidence(expected_return, df, symbol)
    """
    calc = get_confidence_calculator()
    result = calc.calculate_confidence(
        expected_return, df, symbol, model_accuracy, market_regime
    )
    return result.total_confidence


def calculate_confidence_detailed(
    expected_return: float,
    df: pd.DataFrame,
    symbol: str = "",
    model_accuracy: float = None,
    market_regime: str = "NEUTRAL"
) -> ConfidenceResult:
    """Full confidence calculation with breakdown"""
    calc = get_confidence_calculator()
    return calc.calculate_confidence(
        expected_return, df, symbol, model_accuracy, market_regime
    )
