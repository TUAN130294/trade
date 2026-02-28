# -*- coding: utf-8 -*-
"""
Flow Agent - Agentic Level 3
Analyzes foreign and proprietary money flow for Vietnamese stocks

Features:
- Foreign investor flow tracking
- Proprietary trading analysis
- Institutional accumulation/distribution
- Block trade detection
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentSignal, AgentMessage


@dataclass
class FlowData:
    """Money flow data structure"""
    foreign_buy: float
    foreign_sell: float
    prop_buy: float
    prop_sell: float
    retail_buy: float
    retail_sell: float
    total_volume: float
    
    @property
    def foreign_net(self) -> float:
        return self.foreign_buy - self.foreign_sell
    
    @property
    def prop_net(self) -> float:
        return self.prop_buy - self.prop_sell
    
    @property
    def retail_net(self) -> float:
        return self.retail_buy - self.retail_sell
    
    @property
    def smart_money_net(self) -> float:
        return self.foreign_net + self.prop_net


class FlowAgent(BaseAgent):
    """
    Money Flow Analysis Agent - Level 3 Agentic
    
    Responsibilities:
    - Track foreign investor activity
    - Monitor proprietary trading
    - Identify accumulation/distribution
    - Detect unusual block trades
    """
    
    def __init__(self):
        super().__init__(
            name="FLOW_TRACKER",
            emoji="💰",
            role="Money Flow Analyst",
            weight=1.0
        )
        
        # Flow history for pattern detection
        self.flow_history: Dict[str, List[FlowData]] = {}

    def get_perspective(self) -> str:
        """Return the agent's analytical perspective"""
        return "Money flow analysis: foreign investors, proprietary trading, institutional accumulation/distribution patterns in VN market"
        
        # Thresholds for Vietnam market (in billion VND)
        self.large_foreign_threshold = 10  # 10 billion VND
        self.block_trade_threshold = 5
        
        # Historical accuracy
        self.accuracy_score = 0.75
    
    async def analyze(self, stock_data: Any, context: Dict[str, Any] = None) -> AgentSignal:
        """Analyze money flow for given stock using 10+ existing indicators"""
        from .base_agent import SignalType

        symbol = stock_data.symbol if hasattr(stock_data, 'symbol') else 'UNKNOWN'

        # Extract flow data from context
        flow_context = context.get('flow', {}) if context else {}

        flow_data = FlowData(
            foreign_buy=flow_context.get('foreign_buy', 0),
            foreign_sell=flow_context.get('foreign_sell', 0),
            prop_buy=flow_context.get('prop_buy', 0),
            prop_sell=flow_context.get('prop_sell', 0),
            retail_buy=flow_context.get('retail_buy', 0),
            retail_sell=flow_context.get('retail_sell', 0),
            total_volume=flow_context.get('total_volume', 1)
        )

        # Store in history
        if symbol not in self.flow_history:
            self.flow_history[symbol] = []
        self.flow_history[symbol].append(flow_data)

        # Keep last 50 data points
        if len(self.flow_history[symbol]) > 50:
            self.flow_history[symbol] = self.flow_history[symbol][-50:]

        # Analyze flow patterns
        analysis = self._analyze_flow_pattern(symbol, flow_data)

        # NEW: Use existing indicators from orderflow.py and custom.py
        indicator_signals = self._analyze_flow_indicators(stock_data)

        # NEW: FOMO detection
        fomo_signals = self._analyze_fomo(stock_data, context)

        # NEW: Session analysis
        session_signals = self._analyze_session(stock_data)

        # Combine traditional flow analysis with indicator signals + FOMO + session
        signal, confidence, reasoning = self._generate_signal(
            symbol, flow_data, analysis, indicator_signals, fomo_signals, session_signals
        )

        current_price = stock_data.current_price if hasattr(stock_data, 'current_price') else \
                       stock_data.close if hasattr(stock_data, 'close') else 0

        # Convert signal string to SignalType enum
        signal_type_map = {
            'LONG': SignalType.BUY,
            'SHORT': SignalType.SELL,
            'NEUTRAL': SignalType.HOLD
        }
        signal_type = signal_type_map.get(signal, SignalType.HOLD)

        return AgentSignal(
            signal_type=signal_type,
            confidence=confidence * 100,  # Convert to 0-100 scale
            price_target=current_price,
            stop_loss=None,
            take_profit=None,
            risk_reward_ratio=None,
            reasoning=reasoning,
            metadata={
                'flow_data': {
                    'foreign_net': flow_data.foreign_net,
                    'prop_net': flow_data.prop_net,
                    'smart_money': flow_data.smart_money_net
                },
                'flow_trend': analysis['trend'],
                'indicator_signals': indicator_signals,
                'fomo_signals': fomo_signals,
                'session_signals': session_signals,
                'data_quality': indicator_signals.get('data_quality', 'available')
            },
            timestamp=datetime.now()
        )
    
    def _analyze_flow_pattern(self, symbol: str, current: FlowData) -> Dict:
        """Analyze flow pattern from history"""
        history = self.flow_history.get(symbol, [])
        
        if len(history) < 5:
            return {
                'trend': 'INSUFFICIENT_DATA',
                'accumulation': False,
                'distribution': False,
                'foreign_trend': 'NEUTRAL',
                'strength': 0.5
            }
        
        # Calculate recent averages
        recent_5 = history[-5:]
        recent_20 = history[-20:] if len(history) >= 20 else history
        
        avg_foreign_5 = sum(f.foreign_net for f in recent_5) / 5
        avg_foreign_20 = sum(f.foreign_net for f in recent_20) / len(recent_20)
        
        avg_smart_5 = sum(f.smart_money_net for f in recent_5) / 5
        avg_smart_20 = sum(f.smart_money_net for f in recent_20) / len(recent_20)
        
        # Determine trend
        if avg_smart_5 > avg_smart_20 * 1.2:
            trend = 'STRONG_INFLOW'
        elif avg_smart_5 > avg_smart_20:
            trend = 'INFLOW'
        elif avg_smart_5 < avg_smart_20 * 0.8:
            trend = 'STRONG_OUTFLOW'
        elif avg_smart_5 < avg_smart_20:
            trend = 'OUTFLOW'
        else:
            trend = 'NEUTRAL'
        
        # Check accumulation/distribution
        consecutive_buy = sum(1 for f in recent_5 if f.smart_money_net > 0)
        consecutive_sell = sum(1 for f in recent_5 if f.smart_money_net < 0)
        
        accumulation = consecutive_buy >= 4
        distribution = consecutive_sell >= 4
        
        # Foreign flow trend
        if avg_foreign_5 > self.large_foreign_threshold:
            foreign_trend = 'HEAVY_BUY'
        elif avg_foreign_5 > 0:
            foreign_trend = 'NET_BUY'
        elif avg_foreign_5 < -self.large_foreign_threshold:
            foreign_trend = 'HEAVY_SELL'
        elif avg_foreign_5 < 0:
            foreign_trend = 'NET_SELL'
        else:
            foreign_trend = 'NEUTRAL'
        
        # Calculate strength (0-1)
        if current.total_volume > 0:
            strength = abs(current.smart_money_net) / current.total_volume
        else:
            strength = 0.5
        
        return {
            'trend': trend,
            'accumulation': accumulation,
            'distribution': distribution,
            'foreign_trend': foreign_trend,
            'strength': min(1.0, strength)
        }
    
    def _analyze_flow_indicators(self, stock_data: Any) -> Dict:
        """
        NEW: Analyze using existing indicators from orderflow.py, custom.py
        Wires 10+ indicators that were previously unused
        """
        signals = {}
        score_adjustments = 0
        data_quality = 'available'

        try:
            from quantum_stock.indicators.orderflow import OrderFlowIndicators
            from quantum_stock.indicators.custom import CustomIndicators
            from quantum_stock.indicators.volume import VolumeIndicators

            df = stock_data.historical_data

            # 1. Cumulative Delta (orderflow.py)
            if len(df) >= 14:
                delta_data = OrderFlowIndicators.cumulative_delta(
                    df['open'], df['high'], df['low'], df['close'], df['volume']
                )
                cum_delta = delta_data['cumulative_delta'].iloc[-1]
                delta_ema = delta_data['delta_ema'].iloc[-1]

                if cum_delta > delta_ema * 1.1:
                    score_adjustments += 15
                    signals['cumulative_delta'] = 'BULLISH'
                elif cum_delta < delta_ema * 0.9:
                    score_adjustments -= 15
                    signals['cumulative_delta'] = 'BEARISH'

            # 2. Absorption/Exhaustion (orderflow.py)
            if len(df) >= 20:
                absorption = OrderFlowIndicators.absorption_exhaustion(
                    df['open'], df['high'], df['low'], df['close'], df['volume']
                )
                if absorption['bullish_absorption'].iloc[-1]:
                    score_adjustments += 20
                    signals['absorption'] = 'BULLISH_ABSORPTION'
                elif absorption['bearish_absorption'].iloc[-1]:
                    score_adjustments -= 20
                    signals['absorption'] = 'BEARISH_ABSORPTION'

            # 3. Foreign Flow Analysis (orderflow.py - assuming exists)
            # Already handled in main flow_data, skip duplicate

            # 4. Smart Money Index (orderflow.py + custom.py)
            if len(df) >= 14:
                smi = CustomIndicators.smart_money_index(df['open'], df['close'], df['volume'], 14)
                smi_trend = smi.iloc[-1] - smi.iloc[-5:].mean()

                if smi_trend > 0:
                    score_adjustments += 12
                    signals['smart_money_index'] = 'POSITIVE'
                else:
                    score_adjustments -= 12
                    signals['smart_money_index'] = 'NEGATIVE'

            # 5. VWAP Bands (orderflow.py)
            if len(df) >= 20:
                vwap_data = OrderFlowIndicators.vwap_bands(
                    df['high'], df['low'], df['close'], df['volume']
                )
                deviation = vwap_data['distance_from_vwap'].iloc[-1]

                if abs(deviation) > 2:
                    signals['vwap_deviation'] = f'{deviation:.1f}%'

            # 6. Foreign Flow Indicator (custom.py)
            # Need foreign_buy/sell series - skip if not available

            # 7. Accumulation/Distribution Zone (custom.py via volume.py)
            ad_line = VolumeIndicators.accumulation_distribution(
                df['high'], df['low'], df['close'], df['volume']
            )
            ad_slope = ad_line.iloc[-1] - ad_line.iloc[-5] if len(df) >= 5 else 0

            if ad_slope > 0:
                score_adjustments += 10
                signals['accumulation_distribution'] = 'ACCUMULATION'
            else:
                score_adjustments -= 10
                signals['accumulation_distribution'] = 'DISTRIBUTION'

            # 8. Twiggs Money Flow (volume.py)
            if len(df) >= 21:
                tmf = VolumeIndicators.chaikin_money_flow(
                    df['high'], df['low'], df['close'], df['volume'], 21
                )
                if tmf.iloc[-1] > 0.1:
                    score_adjustments += 10
                    signals['twiggs_mf'] = 'POSITIVE'
                elif tmf.iloc[-1] < -0.1:
                    score_adjustments -= 10
                    signals['twiggs_mf'] = 'NEGATIVE'

            signals['score_adjustment'] = score_adjustments
            signals['data_quality'] = data_quality

        except Exception as e:
            signals['error'] = str(e)
            signals['data_quality'] = 'unavailable'
            signals['score_adjustment'] = 0

        return signals

    def _analyze_fomo(self, stock_data: Any, context: Dict[str, Any] = None) -> Dict:
        """
        NEW: Analyze FOMO patterns using FOMODetector
        """
        try:
            from quantum_stock.indicators.fomo_detector import FOMODetector

            df = stock_data.historical_data

            if len(df) < 20:
                return {'signal': 'NO_FOMO', 'confidence': 0.5, 'reason': 'insufficient_data'}

            # Extract market breadth from context if available
            market_breadth = None
            if context and 'market_breadth' in context:
                market_breadth = context['market_breadth']

            detector = FOMODetector()
            signal, confidence, metrics = detector.detect(df, market_breadth)

            return {
                'signal': signal.value if hasattr(signal, 'value') else str(signal),
                'confidence': confidence,
                'metrics': metrics
            }

        except Exception as e:
            return {'signal': 'NO_FOMO', 'error': str(e)}

    def _analyze_session(self, stock_data: Any) -> Dict:
        """
        NEW: Analyze VN trading session patterns
        """
        try:
            from quantum_stock.indicators.session_analyzer import SessionAnalyzer

            df = stock_data.historical_data

            if len(df) < 10:
                return {'signal': 'NORMAL_SESSION', 'confidence': 0.5, 'reason': 'insufficient_data'}

            # Check if intraday data available (has timestamp column)
            has_intraday = 'timestamp' in df.columns

            analyzer = SessionAnalyzer()
            signal, confidence, metrics = analyzer.analyze_session(df, has_intraday)

            return {
                'signal': signal.value if hasattr(signal, 'value') else str(signal),
                'confidence': confidence,
                'metrics': metrics
            }

        except Exception as e:
            return {'signal': 'NORMAL_SESSION', 'error': str(e)}

    def _generate_signal(self, symbol: str, flow: FlowData, analysis: Dict,
                        indicator_signals: Dict = None, fomo_signals: Dict = None,
                        session_signals: Dict = None) -> tuple:
        """Generate trading signal from flow analysis + indicators"""
        trend = analysis['trend']
        foreign_trend = analysis['foreign_trend']

        # Build reasoning
        reasoning_parts = [f"Phân tích dòng tiền {symbol}:"]

        if flow.foreign_net > 0:
            reasoning_parts.append(f"✅ Khối ngoại mua ròng {flow.foreign_net:,.0f}")
        else:
            reasoning_parts.append(f"❌ Khối ngoại bán ròng {abs(flow.foreign_net):,.0f}")

        if flow.prop_net > 0:
            reasoning_parts.append(f"✅ Tự doanh mua ròng {flow.prop_net:,.0f}")
        else:
            reasoning_parts.append(f"❌ Tự doanh bán ròng {abs(flow.prop_net):,.0f}")

        # NEW: Add indicator insights
        base_confidence = 0.5
        if indicator_signals:
            score_adj = indicator_signals.get('score_adjustment', 0)
            # Normalize score adjustment to confidence boost (-0.2 to +0.2)
            confidence_boost = max(-0.2, min(0.2, score_adj / 100))
            base_confidence += confidence_boost

            # Add key indicator signals to reasoning
            if indicator_signals.get('cumulative_delta') == 'BULLISH':
                reasoning_parts.append("📈 Cumulative Delta bullish")
            if indicator_signals.get('absorption') == 'BULLISH_ABSORPTION':
                reasoning_parts.append("🛡️ Bullish absorption detected")
            if indicator_signals.get('smart_money_index') == 'POSITIVE':
                reasoning_parts.append("💰 Smart Money Index positive")

        # NEW: Add FOMO signals
        if fomo_signals:
            fomo_signal = fomo_signals.get('signal', 'NO_FOMO')

            if fomo_signal == 'FOMO_PEAK':
                base_confidence -= 0.25  # Strong penalty for buying at peak
                reasoning_parts.append("⚠️ FOMO PEAK - retail chasing ceiling!")
            elif fomo_signal == 'FOMO_BUILDING':
                base_confidence -= 0.15  # Moderate penalty
                reasoning_parts.append("⚡ FOMO building - retail entry accelerating")
            elif fomo_signal == 'FOMO_TRAP':
                base_confidence -= 0.30  # Extreme penalty
                reasoning_parts.append("🚨 FOMO TRAP - narrow rally, extreme danger!")

        # NEW: Add session signals
        if session_signals:
            session_signal = session_signals.get('signal', 'NORMAL_SESSION')

            if session_signal == 'ATO_INSTITUTIONAL_BUY':
                base_confidence += 0.10
                reasoning_parts.append("🔵 ATO institutional buy positioning")
            elif session_signal == 'ATO_INSTITUTIONAL_SELL':
                base_confidence -= 0.10
                reasoning_parts.append("🔴 ATO institutional sell positioning")
            elif session_signal == 'MORNING_AFTERNOON_REVERSAL':
                reasoning_parts.append("🔄 Session reversal detected - caution!")
            elif session_signal == 'ATC_MANIPULATION_DOWN':
                reasoning_parts.append("📊 ATC kéo giá (price pull up) - smart money positioning")
            elif session_signal == 'ATC_MANIPULATION_UP':
                reasoning_parts.append("📉 ATC đập giá (price push down) - manipulation warning")

        # Determine signal with indicator adjustments
        if trend in ['STRONG_INFLOW', 'INFLOW'] and foreign_trend in ['HEAVY_BUY', 'NET_BUY']:
            signal = 'LONG'
            confidence = 0.7 + analysis['strength'] * 0.2 + (base_confidence - 0.5)
            reasoning_parts.append("🟢 Smart money tích lũy mạnh")

        elif trend in ['STRONG_OUTFLOW', 'OUTFLOW'] and foreign_trend in ['HEAVY_SELL', 'NET_SELL']:
            signal = 'SHORT'
            confidence = 0.7 + analysis['strength'] * 0.2 + (base_confidence - 0.5)
            reasoning_parts.append("🔴 Smart money phân phối")

        elif analysis['accumulation']:
            signal = 'LONG'
            confidence = 0.6 + analysis['strength'] * 0.15 + (base_confidence - 0.5)
            reasoning_parts.append("🟡 Có dấu hiệu tích lũy")

        elif analysis['distribution']:
            signal = 'SHORT'
            confidence = 0.6 + analysis['strength'] * 0.15 + (base_confidence - 0.5)
            reasoning_parts.append("🟡 Có dấu hiệu phân phối")

        else:
            signal = 'NEUTRAL'
            confidence = 0.4 + (base_confidence - 0.5)
            reasoning_parts.append("⚪ Chưa có xu hướng rõ ràng")

        # Clamp confidence to valid range
        confidence = max(0.3, min(0.95, confidence))

        reasoning = '\n'.join(reasoning_parts)

        return signal, confidence, reasoning
    
    async def respond_to_debate(self, topic: str, previous_rounds: List) -> str:
        """Participate in multi-agent debate"""
        response = f"[{self.name}] Từ góc độ dòng tiền: "
        
        if self.last_signal:
            if self.last_signal.signal == 'LONG':
                response += "Smart money đang tích lũy. "
            elif self.last_signal.signal == 'SHORT':
                response += "Cảnh báo dòng tiền rút mạnh. "
            else:
                response += "Dòng tiền chưa rõ xu hướng. "
        
        return response
