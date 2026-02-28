"""
Agent Coordinator - Multi-Agent Orchestration System
Manages communication between all agents and produces final verdicts
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

from .base_agent import BaseAgent, AgentSignal, AgentMessage, StockData, SignalType, MessageType
from .chief_agent import ChiefAgent
from .bull_agent import BullAgent
from .bear_agent import BearAgent
from .analyst_agent import AnalystAgent
from .risk_doctor import RiskDoctor
from .flow_agent import FlowAgent


@dataclass
class TeamDiscussion:
    """Container for multi-agent discussion"""
    symbol: str
    timestamp: datetime
    messages: List[AgentMessage]
    agent_signals: Dict[str, AgentSignal]
    final_verdict: Optional[AgentSignal]
    consensus_score: float
    has_conflict: bool
    market_context: Dict[str, Any] = field(default_factory=dict)
    is_mock: bool = False  # Flag to prevent mock discussions from triggering orders

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'messages': [m.to_dict() for m in self.messages],
            'agent_signals': {k: v.to_dict() for k, v in self.agent_signals.items()},
            'final_verdict': self.final_verdict.to_dict() if self.final_verdict else None,
            'consensus_score': self.consensus_score,
            'has_conflict': self.has_conflict,
            'market_context': self.market_context,
            'is_mock': self.is_mock
        }


class AgentCoordinator:
    """
    Orchestrates multiple AI agents for comprehensive stock analysis.
    Implements Agentic Architecture 4.0 with parallel analysis and consensus building.
    """

    def __init__(self, portfolio_value: float = 100000000):  # 100M VND default
        # Initialize all agents
        self.chief = ChiefAgent()
        self.bull = BullAgent()
        self.bear = BearAgent()
        self.analyst = AnalystAgent()
        self.risk_doctor = RiskDoctor(portfolio_value)
        self.flow_agent = FlowAgent()  # NEW: FlowAgent registration

        # Agent registry (UPDATED with FlowAgent)
        self.agents: Dict[str, BaseAgent] = {
            'Chief': self.chief,
            'Bull': self.bull,
            'Bear': self.bear,
            'Alex': self.analyst,
            'RiskDoctor': self.risk_doctor,
            'FlowAgent': self.flow_agent  # NEW
        }

        # Agent weights (UPDATED per Phase 06 spec)
        self.agent_weights = {
            'FlowAgent': 1.3,   # NEW - highest advisory weight
            'Alex': 1.0,        # Reduced from 1.2
            'Bull': 0.8,        # Reduced from 1.0
            'Bear': 0.8,        # Reduced from 1.0
            'RiskDoctor': 0.9,  # Increased from 0.8
            'Chief': 1.5        # Unchanged
        }

        # Advisory agents (provide perspective, don't vote) - UPDATED
        self.advisory_agents = ['Bull', 'Bear', 'Alex', 'FlowAgent']

        # Decision agents
        self.decision_agents = ['Chief']

        # Risk check agents
        self.risk_agents = ['RiskDoctor']

        # Discussion history
        self.discussions: List[TeamDiscussion] = []

        # Market context
        self.market_context: Dict[str, Any] = {}

        # Configuration
        self.parallel_analysis = True
        self.require_risk_check = True

    def set_portfolio_value(self, value: float):
        """Update portfolio value for risk calculations"""
        self.risk_doctor.set_portfolio_value(value)

    def set_market_context(self, context: Dict[str, Any]):
        """
        Set market-wide context for all agents

        Args:
            context: Dict containing:
                - vn_index: VN-Index value
                - vn_index_change: % change
                - vn30: VN30 value
                - vn30_change: % change
                - market_sentiment: BULLISH/BEARISH/NEUTRAL
                - sector_performance: Dict of sector -> performance
        """
        self.market_context = context

    async def analyze_stock(self, stock_data: StockData,
                           context: Dict[str, Any] = None) -> TeamDiscussion:
        """
        Run full multi-agent analysis on a stock

        Args:
            stock_data: Stock data to analyze
            context: Additional context (backtest results, news, etc.)

        Returns:
            TeamDiscussion with all agent inputs and final verdict
        """
        context = context or {}
        context['market'] = self.market_context

        # Clear previous messages
        for agent in self.agents.values():
            agent.clear_messages()

        # Phase 1: Parallel advisory analysis
        advisory_signals = await self._run_advisory_analysis(stock_data, context)

        # Phase 2: Risk assessment
        risk_signal = await self._run_risk_analysis(stock_data, context)
        advisory_signals['RiskDoctor'] = risk_signal

        # Phase 3: Data quality gating (NEW - per Phase 06 spec)
        # If FlowAgent reports bad data quality, Chief should default to HOLD
        data_quality_override = False
        if 'FlowAgent' in advisory_signals:
            flow_metadata = advisory_signals['FlowAgent'].metadata
            if flow_metadata and flow_metadata.get('data_quality') == 'unavailable':
                data_quality_override = True
                print(f"⚠️ Data quality unavailable - forcing HOLD signal")

        # Phase 4: Chief makes final decision
        chief_context = {
            'agent_signals': advisory_signals,
            'market': self.market_context,
            'data_quality_override': data_quality_override,  # NEW
            **context
        }
        final_verdict = await self.chief.analyze(stock_data, chief_context)

        # Override verdict if data quality is bad (NEW)
        if data_quality_override and final_verdict.signal_type not in [SignalType.HOLD, SignalType.WATCH]:
            final_verdict.signal_type = SignalType.HOLD
            final_verdict.confidence = 30.0
            final_verdict.reasoning = "Data quality unavailable - defaulting to HOLD for safety"

        # Collect all messages (UPDATED to include FlowAgent)
        all_messages = []
        for agent_name in ['Alex', 'Bull', 'Bear', 'FlowAgent', 'RiskDoctor', 'Chief']:
            agent = self.agents[agent_name]
            all_messages.extend(agent.messages)

        # Calculate consensus
        consensus_score = self._calculate_consensus(advisory_signals)
        has_conflict = self._detect_conflict(advisory_signals)

        # Create discussion record
        discussion = TeamDiscussion(
            symbol=stock_data.symbol,
            timestamp=datetime.now(),
            messages=all_messages,
            agent_signals=advisory_signals,
            final_verdict=final_verdict,
            consensus_score=consensus_score,
            has_conflict=has_conflict,
            market_context=self.market_context
        )

        # Store in history
        self.discussions.append(discussion)

        # 🔴 Register all signals with real-time cache for Radar display
        try:
            from quantum_stock.core.realtime_signals import (
                get_signal_cache, AgentSignal as CacheSignal
            )
            cache = get_signal_cache()

            # Convert and register each agent's signal
            for agent_name, signal in advisory_signals.items():
                cache_signal = CacheSignal(
                    agent_name=agent_name,
                    signal_type=signal.signal_type.value,  # Convert enum to string
                    symbol=stock_data.symbol,
                    message=signal.reasoning or f"{signal.signal_type.value} from {agent_name}",
                    confidence=signal.confidence / 100.0,  # Convert 0-100 to 0-1
                    timestamp=signal.timestamp,
                    metadata={
                        'price_target': signal.price_target,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                        'risk_reward_ratio': signal.risk_reward_ratio
                    }
                )
                cache.add_signal(agent_name, cache_signal)

            # Register Chief's final verdict
            if final_verdict:
                cache.add_chief_verdict(
                    symbol=stock_data.symbol,
                    verdict=final_verdict.signal_type.value,
                    confidence=final_verdict.confidence / 100.0
                )
        except Exception as e:
            # Don't let signal caching errors break the analysis
            import logging
            logging.warning(f"Could not cache signals for Radar: {e}")

        return discussion

    async def _run_advisory_analysis(self, stock_data: StockData,
                                     context: Dict[str, Any]) -> Dict[str, AgentSignal]:
        """Run all advisory agents in parallel (UPDATED with FlowAgent)"""
        signals = {}

        if self.parallel_analysis:
            # Run in parallel (UPDATED to include FlowAgent)
            tasks = [
                self.analyst.analyze(stock_data, context),
                self.bull.analyze(stock_data, context),
                self.bear.analyze(stock_data, context),
                self.flow_agent.analyze(stock_data, context)  # NEW
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            agent_names = ['Alex', 'Bull', 'Bear', 'FlowAgent']
            for name, result in zip(agent_names, results):
                if isinstance(result, Exception):
                    print(f"Agent {name} error: {result}")
                else:
                    signals[name] = result
        else:
            # Run sequentially (UPDATED)
            signals['Alex'] = await self.analyst.analyze(stock_data, context)
            signals['Bull'] = await self.bull.analyze(stock_data, context)
            signals['Bear'] = await self.bear.analyze(stock_data, context)
            signals['FlowAgent'] = await self.flow_agent.analyze(stock_data, context)  # NEW

        return signals

    async def _run_risk_analysis(self, stock_data: StockData,
                                 context: Dict[str, Any]) -> AgentSignal:
        """Run risk assessment"""
        return await self.risk_doctor.analyze(stock_data, context)

    def _calculate_consensus(self, signals: Dict[str, AgentSignal]) -> float:
        """Calculate consensus score among agents"""
        if not signals:
            return 0.0

        # Convert signals to numeric scores
        signal_scores = {
            SignalType.STRONG_BUY: 100,
            SignalType.BUY: 75,
            SignalType.HOLD: 50,
            SignalType.SELL: 25,
            SignalType.STRONG_SELL: 0,
            SignalType.WATCH: 50,
            SignalType.MIXED: 50
        }

        scores = []
        for signal in signals.values():
            score = signal_scores.get(signal.signal_type, 50)
            # Weight by confidence
            weighted_score = score * (signal.confidence / 100)
            scores.append(weighted_score)

        if not scores:
            return 0.0

        # Calculate standard deviation as measure of disagreement
        avg = sum(scores) / len(scores)
        variance = sum((s - avg) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5

        # Consensus = 100 - normalized std_dev
        # Max std_dev is ~50 (if scores are 0 and 100)
        consensus = 100 - min(100, std_dev * 2)

        return round(consensus, 1)

    def _detect_conflict(self, signals: Dict[str, AgentSignal]) -> bool:
        """Detect if there's significant conflict between agents"""
        bullish = 0
        bearish = 0

        for signal in signals.values():
            if signal.signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
                if signal.confidence > 60:
                    bullish += 1
            elif signal.signal_type in [SignalType.STRONG_SELL, SignalType.SELL]:
                if signal.confidence > 60:
                    bearish += 1

        return bullish >= 1 and bearish >= 1

    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            name: agent.get_status()
            for name, agent in self.agents.items()
        }

    def get_last_discussion(self) -> Optional[TeamDiscussion]:
        """Get the most recent discussion"""
        return self.discussions[-1] if self.discussions else None

    def format_discussion_for_display(self, discussion: TeamDiscussion) -> str:
        """Format discussion for terminal/chat display"""
        lines = []
        lines.append(f"{'='*50}")
        lines.append(f"📊 TEAM ANALYSIS: {discussion.symbol}")
        lines.append(f"⏰ {discussion.timestamp.strftime('%H:%M:%S')}")
        lines.append(f"{'='*50}")
        lines.append("")

        # Agent messages
        for msg in discussion.messages:
            status = "SUCCESS" if msg.message_type in [MessageType.ANALYSIS, MessageType.RECOMMENDATION] else "WARNING" if msg.message_type == MessageType.WARNING else "INFO"
            conf_str = f" Confidence {msg.confidence:.0f}%." if msg.confidence else ""
            lines.append(f"{msg.agent_emoji} {msg.agent_name}: {msg.content}{conf_str}")
            lines.append(f"   [{status}]")
            lines.append("")

        # Final verdict
        if discussion.final_verdict:
            lines.append(f"{'='*50}")
            lines.append(f"🎖 FINAL VERDICT: {discussion.final_verdict.signal_type.value}")
            lines.append(f"   Confidence: {discussion.final_verdict.confidence:.1f}%")
            lines.append(f"   Consensus: {discussion.consensus_score:.1f}%")
            if discussion.has_conflict:
                lines.append(f"   ⚠️ CONFLICT DETECTED between advisors")

            if discussion.final_verdict.stop_loss:
                lines.append(f"   SL: {discussion.final_verdict.stop_loss:.2f}")
            if discussion.final_verdict.take_profit:
                lines.append(f"   TP: {discussion.final_verdict.take_profit:.2f}")
            if discussion.final_verdict.risk_reward_ratio:
                lines.append(f"   R:R: 1:{discussion.final_verdict.risk_reward_ratio:.1f}")

        lines.append(f"{'='*50}")

        return "\n".join(lines)

    async def quick_scan(self, symbols: List[str],
                        data_provider: Any) -> List[Dict[str, Any]]:
        """
        Quick scan multiple stocks and rank by opportunity

        Args:
            symbols: List of stock symbols
            data_provider: Data provider instance

        Returns:
            List of stocks ranked by opportunity score
        """
        results = []

        for symbol in symbols:
            try:
                # Get stock data
                stock_data = await data_provider.get_stock_data(symbol)
                if not stock_data:
                    continue

                # Quick analysis (just analyst + risk)
                analyst_signal = await self.analyst.analyze(stock_data, {})
                risk_signal = await self.risk_doctor.analyze(stock_data, {})

                # Calculate opportunity score
                tech_score = analyst_signal.confidence
                risk_score = 100 - risk_signal.metadata.get('risk_assessment', {}).get('risk_score', 50)

                opportunity_score = (tech_score * 0.6 + risk_score * 0.4)

                results.append({
                    'symbol': symbol,
                    'price': stock_data.current_price,
                    'change': stock_data.change_percent,
                    'tech_signal': analyst_signal.signal_type.value,
                    'tech_confidence': tech_score,
                    'risk_level': risk_signal.metadata.get('risk_assessment', {}).get('risk_level', 'UNKNOWN'),
                    'opportunity_score': opportunity_score
                })
            except Exception as e:
                print(f"Error scanning {symbol}: {e}")

        # Sort by opportunity score
        results.sort(key=lambda x: x['opportunity_score'], reverse=True)

        return results

    def get_discussion_history(self, symbol: str = None,
                               limit: int = 10) -> List[TeamDiscussion]:
        """Get discussion history, optionally filtered by symbol"""
        discussions = self.discussions

        if symbol:
            discussions = [d for d in discussions if d.symbol == symbol]

        return discussions[-limit:]
