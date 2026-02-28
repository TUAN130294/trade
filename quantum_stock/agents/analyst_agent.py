"""
Analyst Agent (Alex) - The Technical Expert
Pure technical analysis without emotional bias
"""

import logging
from typing import Dict, Any, List, Tuple
from .base_agent import BaseAgent, AgentSignal, StockData, SignalType, MessageType

logger = logging.getLogger(__name__)

# Handle both package and direct run import styles
try:
    from quantum_stock.dataconnector.market_flow import MarketFlowConnector
except ImportError:
    try:
        from dataconnector.market_flow import MarketFlowConnector
    except ImportError:
        # Fallback: create dummy class if module not available
        class MarketFlowConnector:
            async def detect_smart_money_footprint(self, df):
                return {'detected': False}



class AnalystAgent(BaseAgent):
    """
    Analyst Agent (Alex) - Technical Analysis Expert
    """

    def __init__(self):
        super().__init__(
            name="Alex",
            emoji="📊",
            role="Technical Analyst",
            weight=1.0
        )
        self.market_flow = MarketFlowConnector()

    def get_perspective(self) -> str:
        return "Phân tích kỹ thuật thuần túy, dựa trên dữ liệu và chỉ báo"

    async def analyze(self, stock_data: StockData, context: Dict[str, Any] = None) -> AgentSignal:
        """
        Analyze stock using technical indicators AND smart money flow
        """
        context = context or {}
        df = stock_data.historical_data

        # 1. Standard Technical Score
        # Technical scores
        trend_score = self._analyze_trend(stock_data)
        momentum_score = self._analyze_momentum(stock_data)
        volume_score = self._analyze_volume(stock_data)
        pattern_score = self._analyze_patterns(stock_data)
        support_resistance_score = self._analyze_levels(stock_data)

        # Money flow analysis (NEW - using existing indicators)
        money_flow_score = await self._analyze_money_flow(stock_data)

        # Calculate weighted technical score (UPDATED WEIGHTS)
        weights = {
            'trend': 0.20,
            'momentum': 0.20,
            'volume': 0.15,
            'pattern': 0.10,
            'levels': 0.10,
            'money_flow': 0.25  # NEW - highest weight for money flow
        }

        total_score = (
            trend_score['score'] * weights['trend'] +
            momentum_score['score'] * weights['momentum'] +
            volume_score['score'] * weights['volume'] +
            pattern_score['score'] * weights['pattern'] +
            support_resistance_score['score'] * weights['levels'] +
            money_flow_score['score'] * weights['money_flow']
        )

        # Collect all signals
        all_signals = []
        all_signals.extend(trend_score['signals'])
        all_signals.extend(momentum_score['signals'])
        all_signals.extend(volume_score['signals'])
        all_signals.extend(pattern_score['signals'])
        all_signals.extend(support_resistance_score['signals'])
        all_signals.extend(money_flow_score['signals'])

        # Determine final signal
        signal_type = self._determine_signal(total_score)

        # Generate technical report
        direction = "UP" if total_score >= 50 else "DOWN"
        message = self._generate_technical_message(
            stock_data.symbol, signal_type, total_score, direction, all_signals
        )
        self.emit_message(message, MessageType.ANALYSIS, total_score)

        # Calculate technical targets
        price = stock_data.current_price
        atr = stock_data.indicators.get('atr', price * 0.02)

        if total_score >= 50:
            # Bullish setup
            stop_loss = price - (2 * atr)
            take_profit = price + (2.5 * atr)
        else:
            # Bearish setup
            stop_loss = price + (2 * atr)
            take_profit = price - (2.5 * atr)

        risk_reward = abs(take_profit - price) / abs(price - stop_loss) if abs(price - stop_loss) > 0 else 0

        self.last_signal = AgentSignal(
            signal_type=signal_type,
            confidence=round(total_score, 1),
            price_target=take_profit,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=round(risk_reward, 2),
            reasoning=" | ".join(all_signals[:5]),
            metadata={
                'trend_score': trend_score,
                'momentum_score': momentum_score,
                'volume_score': volume_score,
                'pattern_score': pattern_score,
                'levels_score': support_resistance_score,
                'money_flow_score': money_flow_score,
                'perspective': 'technical'
            }
        )

        return self.last_signal

    def _analyze_trend(self, stock_data: StockData) -> Dict[str, Any]:
        """Analyze trend indicators"""
        price = stock_data.current_price
        signals = []
        score = 50

        # EMA Analysis
        ema20 = stock_data.indicators.get('ema20', price)
        ema50 = stock_data.indicators.get('ema50', price)
        ema200 = stock_data.indicators.get('ema200', price)

        # Golden/Death Cross
        if ema20 > ema50:
            score += 10
            signals.append("🟢 EMA20 > EMA50: Tín hiệu Golden Cross (Xu hướng tăng)")
        else:
            score -= 10
            signals.append("🔴 EMA20 < EMA50: Tín hiệu Death Cross (Xu hướng giảm)")

        # Price vs EMAs
        if price > ema20:
            score += 10
            signals.append("🟢 Giá trên EMA20: Xu hướng ngắn hạn tích cực")
        else:
            score -= 10
            signals.append("🔴 Giá dưới EMA20: Xu hướng ngắn hạn tiêu cực")

        if price > ema50:
            score += 8
            signals.append("🟢 Giá trên EMA50: Xu hướng trung hạn tích cực")
        else:
            score -= 8
            signals.append("🔴 Giá dưới EMA50: Xu hướng trung hạn tiêu cực")

        if price > ema200:
            score += 7
            signals.append("🟢 Giá trên EMA200: Xu hướng dài hạn mạnh mẽ")
        else:
            score -= 7
            signals.append("🔴 Giá dưới EMA200: Xu hướng dài hạn suy yếu")

        # ADX Trend Strength
        adx = stock_data.indicators.get('adx', 25)
        if adx > 25:
            # Strong trend - amplify the direction
            if score > 50:
                score += 5
                signals.append(f"🟢 ADX={adx:.0f}: Xu hướng mạnh mẽ được xác nhận")
            else:
                score -= 5
                signals.append(f"🔴 ADX={adx:.0f}: Xu hướng giảm mạnh mẽ được xác nhận")
        elif adx < 20:
            signals.append(f"ADX={adx:.0f} (Weak/No trend)")
            score = 50 + (score - 50) * 0.5  # Reduce conviction

        return {'score': max(0, min(100, score)), 'signals': signals}

    def _analyze_momentum(self, stock_data: StockData) -> Dict[str, Any]:
        """Analyze momentum indicators"""
        signals = []
        score = 50

        # RSI
        rsi = stock_data.indicators.get('rsi', 50)
        if rsi < 30:
            score += 20
            signals.append(f"RSI={rsi:.1f} (Oversold - Buy signal)")
        elif rsi > 70:
            score -= 20
            signals.append(f"RSI={rsi:.1f} (Overbought - Sell signal)")
        elif 40 <= rsi <= 60:
            signals.append(f"RSI={rsi:.1f} (Neutral)")
        else:
            if rsi < 50:
                score -= 5
            else:
                score += 5
            signals.append(f"RSI={rsi:.1f}")

        # MACD
        macd = stock_data.indicators.get('macd', 0)
        macd_signal = stock_data.indicators.get('macd_signal', 0)
        macd_hist = stock_data.indicators.get('macd_hist', 0)

        if macd > macd_signal:
            score += 15
            if macd_hist > 0 and macd_hist > stock_data.indicators.get('macd_hist_prev', 0):
                signals.append("MACD Bullish + Increasing histogram")
                score += 5
            else:
                signals.append("MACD Bullish crossover")
        else:
            score -= 15
            if macd_hist < 0:
                signals.append("MACD Bearish")
                score -= 5

        # Stochastic
        stoch_k = stock_data.indicators.get('stoch_k', 50)
        stoch_d = stock_data.indicators.get('stoch_d', 50)

        if stoch_k < 20:
            score += 10
            signals.append(f"Stochastic oversold ({stoch_k:.0f})")
        elif stoch_k > 80:
            score -= 10
            signals.append(f"Stochastic overbought ({stoch_k:.0f})")

        if stoch_k > stoch_d and stoch_k < 50:
            score += 5
            signals.append("Stochastic bullish cross")

        # CCI
        cci = stock_data.indicators.get('cci', 0)
        if cci < -100:
            score += 10
            signals.append(f"CCI={cci:.0f} (Oversold)")
        elif cci > 100:
            score -= 10
            signals.append(f"CCI={cci:.0f} (Overbought)")

        return {'score': max(0, min(100, score)), 'signals': signals}

    def _analyze_volume(self, stock_data: StockData) -> Dict[str, Any]:
        """Analyze volume indicators"""
        signals = []
        score = 50

        volume = stock_data.volume
        avg_volume = stock_data.indicators.get('avg_volume', volume)
        change = stock_data.change_percent

        # Volume ratio
        vol_ratio = volume / avg_volume if avg_volume > 0 else 1

        if vol_ratio > 2:
            if change > 0:
                score += 25
                signals.append(f"Volume spike {vol_ratio:.1f}x + Price up (Accumulation)")
            else:
                score -= 25
                signals.append(f"Volume spike {vol_ratio:.1f}x + Price down (Distribution)")
        elif vol_ratio > 1.5:
            if change > 0:
                score += 15
                signals.append(f"High volume {vol_ratio:.1f}x (Bullish)")
            else:
                score -= 15
                signals.append(f"High volume {vol_ratio:.1f}x (Bearish)")
        elif vol_ratio < 0.5:
            signals.append("Low volume - Weak conviction")
            score = 50 + (score - 50) * 0.5

        # OBV Trend
        obv = stock_data.indicators.get('obv', 0)
        obv_ema = stock_data.indicators.get('obv_ema', obv)

        if obv > obv_ema:
            score += 10
            signals.append("OBV above EMA (Bullish volume)")
        else:
            score -= 10
            signals.append("OBV below EMA (Bearish volume)")

        # MFI
        mfi = stock_data.indicators.get('mfi', 50)
        if mfi < 20:
            score += 10
            signals.append(f"MFI={mfi:.0f} (Oversold)")
        elif mfi > 80:
            score -= 10
            signals.append(f"MFI={mfi:.0f} (Overbought)")

        return {'score': max(0, min(100, score)), 'signals': signals}

    def _analyze_patterns(self, stock_data: StockData) -> Dict[str, Any]:
        """Analyze chart patterns and candlestick formations"""
        signals = []
        score = 50

        # Bollinger Band position
        price = stock_data.current_price
        bb_upper = stock_data.indicators.get('bb_upper', price * 1.02)
        bb_lower = stock_data.indicators.get('bb_lower', price * 0.98)
        bb_mid = stock_data.indicators.get('bb_mid', price)

        if price < bb_lower:
            score += 15
            signals.append("Price below BB lower (Potential bounce)")
        elif price > bb_upper:
            score -= 15
            signals.append("Price above BB upper (Overextended)")
        elif price > bb_mid:
            score += 5
        else:
            score -= 5

        # BB Width (Squeeze detection)
        bb_width = stock_data.indicators.get('bb_width', 0.04)
        if bb_width < 0.03:
            signals.append("BB Squeeze - Breakout imminent")

        # Divergence patterns
        divergence = stock_data.indicators.get('divergence', None)
        if divergence == 'bullish':
            score += 20
            signals.append("Bullish divergence detected!")
        elif divergence == 'bearish':
            score -= 20
            signals.append("Bearish divergence detected!")

        # Candlestick patterns (if available)
        candle_pattern = stock_data.indicators.get('candle_pattern', None)
        if candle_pattern:
            if candle_pattern in ['hammer', 'morning_star', 'bullish_engulfing']:
                score += 15
                signals.append(f"Bullish pattern: {candle_pattern}")
            elif candle_pattern in ['shooting_star', 'evening_star', 'bearish_engulfing']:
                score -= 15
                signals.append(f"Bearish pattern: {candle_pattern}")

        return {'score': max(0, min(100, score)), 'signals': signals}

    def _analyze_levels(self, stock_data: StockData) -> Dict[str, Any]:
        """Analyze support/resistance levels"""
        signals = []
        score = 50
        price = stock_data.current_price

        support = stock_data.indicators.get('support', price * 0.95)
        resistance = stock_data.indicators.get('resistance', price * 1.05)

        dist_to_support = (price - support) / price * 100
        dist_to_resistance = (resistance - price) / price * 100

        # Near support (bullish)
        if dist_to_support < 1:
            score += 20
            signals.append(f"At support {support:.2f} - Strong buy zone")
        elif dist_to_support < 3:
            score += 10
            signals.append(f"Near support {support:.2f}")

        # Near resistance (bearish)
        if dist_to_resistance < 1:
            score -= 20
            signals.append(f"At resistance {resistance:.2f} - May face rejection")
        elif dist_to_resistance < 3:
            score -= 10
            signals.append(f"Approaching resistance {resistance:.2f}")

        # Risk/Reward based on levels
        if dist_to_support > 0 and dist_to_resistance > 0:
            rr_ratio = dist_to_resistance / dist_to_support
            if rr_ratio > 2:
                score += 10
                signals.append(f"Good R:R ratio {rr_ratio:.1f}")
            elif rr_ratio < 0.5:
                score -= 10
                signals.append(f"Poor R:R ratio {rr_ratio:.1f}")

        # VWAP
        vwap = stock_data.indicators.get('vwap', price)
        if price > vwap:
            score += 5
            signals.append("Above VWAP (Institutional bullish)")
        else:
            score -= 5
            signals.append("Below VWAP (Institutional bearish)")

        return {'score': max(0, min(100, score)), 'signals': signals}

    async def _analyze_money_flow(self, stock_data: StockData) -> Dict[str, Any]:
        """
        Analyze money flow using existing indicators from orderflow, custom, and volume modules
        NEW method wiring 6+ existing indicators
        """
        signals = []
        score = 50
        df = stock_data.historical_data

        try:
            # Import existing indicators
            from quantum_stock.indicators.orderflow import OrderFlowIndicators
            from quantum_stock.indicators.custom import CustomIndicators
            from quantum_stock.indicators.volume import VolumeIndicators

            # 1. Cumulative Delta from orderflow
            if len(df) >= 14:
                delta_data = OrderFlowIndicators.cumulative_delta(
                    df['open'], df['high'], df['low'], df['close'], df['volume']
                )
                cumulative_delta = delta_data['cumulative_delta'].iloc[-1]
                delta_ema = delta_data['delta_ema'].iloc[-1]

                if cumulative_delta > delta_ema:
                    score += 15
                    signals.append("Cumulative Delta tích cực - Buyers mạnh hơn Sellers")
                else:
                    score -= 15
                    signals.append("Cumulative Delta tiêu cực - Sellers mạnh hơn")

            # 2. Absorption/Exhaustion
            if len(df) >= 20:
                absorption = OrderFlowIndicators.absorption_exhaustion(
                    df['open'], df['high'], df['low'], df['close'], df['volume']
                )
                if absorption['bullish_absorption'].iloc[-1]:
                    score += 20
                    signals.append("🛡️ Bullish Absorption - Selling pressure bị hấp thụ")
                elif absorption['bearish_absorption'].iloc[-1]:
                    score -= 20
                    signals.append("🛡️ Bearish Absorption - Buying pressure bị hấp thụ")

            # 3. Smart Money Index from custom
            if len(df) >= 14:
                smi = CustomIndicators.smart_money_index(df['open'], df['close'], df['volume'], 14)
                if smi.iloc[-1] > smi.iloc[-5:].mean():
                    score += 15
                    signals.append("Smart Money Index tăng - Dòng tiền thông minh vào")
                else:
                    score -= 10
                    signals.append("Smart Money Index giảm - Dòng tiền rút khỏi")

            # 4. Twiggs Money Flow from volume
            if len(df) >= 21:
                tmf = VolumeIndicators.chaikin_money_flow(
                    df['high'], df['low'], df['close'], df['volume'], 21
                )
                if tmf.iloc[-1] > 0.1:
                    score += 15
                    signals.append(f"Twiggs MF={tmf.iloc[-1]:.3f} - Dòng tiền mạnh")
                elif tmf.iloc[-1] < -0.1:
                    score -= 15
                    signals.append(f"Twiggs MF={tmf.iloc[-1]:.3f} - Dòng tiền yếu")

            # 5. Accumulation/Distribution from volume
            ad = VolumeIndicators.accumulation_distribution(
                df['high'], df['low'], df['close'], df['volume']
            )
            ad_slope = ad.iloc[-1] - ad.iloc[-5] if len(df) >= 5 else 0
            if ad_slope > 0:
                score += 10
                signals.append("A/D Line tăng - Tích lũy")
            else:
                score -= 10
                signals.append("A/D Line giảm - Phân phối")

            # 6. VWAP Bands
            if len(df) >= 20:
                vwap_data = OrderFlowIndicators.vwap_bands(
                    df['high'], df['low'], df['close'], df['volume']
                )
                current_price = stock_data.current_price
                vwap = vwap_data['vwap'].iloc[-1]
                deviation = vwap_data['distance_from_vwap'].iloc[-1]

                if current_price > vwap and deviation < 2:
                    score += 10
                    signals.append(f"Giá trên VWAP +{deviation:.1f}% - Institutional support")
                elif current_price < vwap and deviation > -2:
                    score -= 10
                    signals.append(f"Giá dưới VWAP {deviation:.1f}% - Yếu thế")

        except Exception as e:
            logger.warning(f"Money flow analysis error: {e}")
            signals.append("Money flow data không đầy đủ")

        return {'score': max(0, min(100, score)), 'signals': signals}

    def _generate_technical_message(self, symbol: str, signal: SignalType,
                                    score: float, direction: str,
                                    all_signals: List[str]) -> str:
        """Generate technical analysis message"""
        # Count bullish vs bearish signals
        bullish = sum(1 for s in all_signals if any(w in s.lower() for w in ['bullish', 'buy', 'oversold', 'above', 'accumulation']))
        bearish = sum(1 for s in all_signals if any(w in s.lower() for w in ['bearish', 'sell', 'overbought', 'below', 'distribution']))

        if signal in [SignalType.STRONG_BUY, SignalType.BUY]:
            status = "đang tăng" if score > 60 else "có tín hiệu tăng"
        elif signal in [SignalType.STRONG_SELL, SignalType.SELL]:
            status = "đang giảm" if score < 40 else "có tín hiệu giảm"
        else:
            status = "đi ngang"

        change_pct = f"+{score-50:.1f}%" if score >= 50 else f"{score-50:.1f}%"

        msg = f"Báo cáo Chief! {symbol} {status}, giá hiện tại ({change_pct}). Confidence {score:.0f}%."
        return msg
