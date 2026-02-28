# -*- coding: utf-8 -*-
"""
Real-time Signal Cache System
=============================
Replaces hardcoded Radar signals with real-time data

This module provides:
1. Real-time signal caching from scanners
2. Agent activity tracking
3. Live signal retrieval for Radar display
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentSignal:
    """A signal from an agent"""
    agent_name: str
    signal_type: str  # BUY, SELL, HOLD, WARNING
    symbol: str
    message: str
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'agent_name': self.agent_name,
            'signal_type': self.signal_type,
            'symbol': self.symbol,
            'message': self.message,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class AgentActivity:
    """Tracks agent activity for status display"""
    name: str
    emoji: str
    role: str
    description: str
    status: str = "online"
    accuracy: float = 0.0
    signals_today: int = 0
    last_signal: str = ""
    last_signal_time: Optional[datetime] = None
    specialty: str = ""

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'emoji': self.emoji,
            'role': self.role,
            'description': self.description,
            'status': self.status,
            'accuracy': self.accuracy,
            'signals_today': self.signals_today,
            'last_signal': self.last_signal,
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
            'specialty': self.specialty
        }


class RealTimeSignalCache:
    """
    Central cache for all trading signals

    Provides:
    - Signal storage with TTL
    - Agent activity tracking
    - Real-time updates for Radar
    """

    # Maximum signals to keep per agent
    MAX_SIGNALS_PER_AGENT = 100

    # Signal TTL (24 hours)
    SIGNAL_TTL = timedelta(hours=24)

    def __init__(self):
        # Signals by agent
        self._signals: Dict[str, deque] = {}

        # Agent activity tracking
        self._agents: Dict[str, AgentActivity] = self._init_agents()

        # Daily counters (reset at midnight)
        self._daily_counts: Dict[str, int] = {}
        self._last_reset_date: Optional[datetime] = None

    def _init_agents(self) -> Dict[str, AgentActivity]:
        """Initialize agent configurations"""
        return {
            'Scout': AgentActivity(
                name='Scout',
                emoji='🔭',
                role='Market Scanner',
                description='Quét thị trường 24/7, phát hiện cơ hội đầu tư',
                specialty='Pattern Recognition'
            ),
            'Alex': AgentActivity(
                name='Alex',
                emoji='📊',
                role='Technical Analyst',
                description='Phân tích kỹ thuật: RSI, MACD, Bollinger, Support/Resistance',
                specialty='Technical Indicators'
            ),
            'Bull': AgentActivity(
                name='Bull',
                emoji='🐂',
                role='Growth Hunter',
                description='Tìm kiếm cổ phiếu tăng trưởng, breakout',
                specialty='Momentum Trading'
            ),
            'Bear': AgentActivity(
                name='Bear',
                emoji='🐻',
                role='Risk Sentinel',
                description='Phát hiện rủi ro, cảnh báo đảo chiều',
                specialty='Risk Detection'
            ),
            'RiskDoctor': AgentActivity(
                name='Risk Doctor',
                emoji='🏥',
                role='Position Sizer',
                description='Tính toán khối lượng giao dịch an toàn, quản lý vốn',
                specialty='Money Management'
            ),
            'Chief': AgentActivity(
                name='Chief',
                emoji='⚖️',
                role='Decision Maker',
                description='Tổng hợp ý kiến, ra quyết định cuối cùng',
                specialty='Consensus Building'
            )
        }

    def _reset_daily_if_needed(self):
        """Reset daily counters at midnight"""
        now = datetime.now()
        if self._last_reset_date is None or self._last_reset_date.date() < now.date():
            self._daily_counts = {name: 0 for name in self._agents}
            self._last_reset_date = now

    def add_signal(self, agent_name: str, signal: AgentSignal):
        """
        Add a new signal from an agent

        Args:
            agent_name: Name of the agent
            signal: The signal to add
        """
        self._reset_daily_if_needed()

        # Initialize deque if needed
        if agent_name not in self._signals:
            self._signals[agent_name] = deque(maxlen=self.MAX_SIGNALS_PER_AGENT)

        # Add signal
        self._signals[agent_name].append(signal)

        # Update agent activity
        if agent_name in self._agents:
            agent = self._agents[agent_name]
            agent.signals_today = self._daily_counts.get(agent_name, 0) + 1
            agent.last_signal = signal.message
            agent.last_signal_time = signal.timestamp

            self._daily_counts[agent_name] = agent.signals_today

        logger.info(f"Signal added: {agent_name} -> {signal.message}")

    def add_scout_signal(self, symbol: str, signal_type: str, confidence: float,
                         expected_return: float = None):
        """Convenience method for Scout signals"""
        if expected_return:
            message = f"Phát hiện cơ hội {symbol} (+{expected_return*100:.1f}%)"
        else:
            message = f"Phát hiện tín hiệu {signal_type} cho {symbol}"

        signal = AgentSignal(
            agent_name='Scout',
            signal_type=signal_type,
            symbol=symbol,
            message=message,
            confidence=confidence,
            timestamp=datetime.now()
        )
        self.add_signal('Scout', signal)

    def add_analyst_signal(self, symbol: str, analysis: str, score: float):
        """Convenience method for Alex (Analyst) signals"""
        signal = AgentSignal(
            agent_name='Alex',
            signal_type='ANALYSIS',
            symbol=symbol,
            message=analysis,
            confidence=score / 100,
            timestamp=datetime.now()
        )
        self.add_signal('Alex', signal)

    def add_chief_verdict(self, symbol: str, verdict: str, confidence: float):
        """Convenience method for Chief verdicts"""
        signal = AgentSignal(
            agent_name='Chief',
            signal_type=verdict,
            symbol=symbol,
            message=f"VERDICT: {symbol} → {verdict}",
            confidence=confidence,
            timestamp=datetime.now()
        )
        self.add_signal('Chief', signal)

    def add_risk_warning(self, message: str, severity: str = "WARNING"):
        """Convenience method for Bear risk warnings"""
        signal = AgentSignal(
            agent_name='Bear',
            signal_type=severity,
            symbol='MARKET',
            message=message,
            confidence=0.8,
            timestamp=datetime.now()
        )
        self.add_signal('Bear', signal)

    def get_agent_status(self) -> List[Dict]:
        """
        Get current status of all agents for Radar display

        Returns list of agent status dicts with real-time signals
        """
        self._reset_daily_if_needed()

        result = []
        for name, agent in self._agents.items():
            status = agent.to_dict()

            # Get last signal if available
            if name in self._signals and self._signals[name]:
                last = self._signals[name][-1]
                status['last_signal'] = last.message
                status['last_signal_time'] = last.timestamp.isoformat()

            # Calculate accuracy from recent signals (placeholder)
            # In production, this should come from actual trade results
            status['accuracy'] = agent.accuracy if agent.accuracy > 0 else 0.80

            result.append(status)

        return result

    def get_recent_signals(self, agent_name: str = None, limit: int = 10) -> List[Dict]:
        """Get recent signals, optionally filtered by agent"""
        signals = []

        if agent_name:
            if agent_name in self._signals:
                for s in list(self._signals[agent_name])[-limit:]:
                    signals.append(s.to_dict())
        else:
            # Get from all agents
            all_signals = []
            for agent_signals in self._signals.values():
                all_signals.extend(list(agent_signals))

            # Sort by timestamp and take recent
            all_signals.sort(key=lambda x: x.timestamp, reverse=True)
            signals = [s.to_dict() for s in all_signals[:limit]]

        return signals

    def get_opportunities(self, min_confidence: float = 0.7) -> List[Dict]:
        """Get current trading opportunities"""
        opportunities = []

        # Check Scout signals
        if 'Scout' in self._signals:
            for signal in self._signals['Scout']:
                if signal.confidence >= min_confidence and signal.signal_type in ['BUY', 'STRONG_BUY']:
                    # Check if not too old
                    if datetime.now() - signal.timestamp < timedelta(hours=4):
                        opportunities.append(signal.to_dict())

        return opportunities

    def update_agent_accuracy(self, agent_name: str, accuracy: float):
        """Update agent accuracy from trade results"""
        if agent_name in self._agents:
            self._agents[agent_name].accuracy = accuracy

    def clear_old_signals(self):
        """Remove signals older than TTL"""
        cutoff = datetime.now() - self.SIGNAL_TTL

        for agent_name in self._signals:
            # Filter out old signals
            self._signals[agent_name] = deque(
                [s for s in self._signals[agent_name] if s.timestamp > cutoff],
                maxlen=self.MAX_SIGNALS_PER_AGENT
            )


# Singleton instance
_signal_cache: Optional[RealTimeSignalCache] = None


def get_signal_cache() -> RealTimeSignalCache:
    """Get singleton signal cache instance"""
    global _signal_cache
    if _signal_cache is None:
        _signal_cache = RealTimeSignalCache()
    return _signal_cache


# Convenience functions
def add_trading_signal(agent_name: str, symbol: str, signal_type: str,
                       message: str, confidence: float):
    """Quick function to add a signal"""
    cache = get_signal_cache()
    signal = AgentSignal(
        agent_name=agent_name,
        signal_type=signal_type,
        symbol=symbol,
        message=message,
        confidence=confidence,
        timestamp=datetime.now()
    )
    cache.add_signal(agent_name, signal)


def get_radar_agent_status() -> List[Dict]:
    """Get agent status for Radar display"""
    return get_signal_cache().get_agent_status()
