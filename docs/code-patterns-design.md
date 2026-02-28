# Design Patterns & Architecture

**Version:** 1.0
**Updated:** 2026-02-27
**Scope:** VN-Quant design patterns and architectural patterns

---

## Agent Pattern (Multi-Agent System)

**Use Case:** Specialized AI agents analyzing same data differently

**Structure:**
```python
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Abstract base for all agents"""

    def __init__(self, name: str, role: str, weight: float = 1.0):
        self.name = name
        self.role = role
        self.weight = weight  # For consensus voting

    @abstractmethod
    async def analyze(self, data: StockData) -> AgentSignal:
        """Each agent implements own analysis"""
        pass

class BullAgent(BaseAgent):
    """Concrete agent implementation"""

    async def analyze(self, data: StockData) -> AgentSignal:
        # Specialized analysis logic
        confidence = self._calculate_confidence(data)
        signal_type = self._determine_signal(data, confidence)
        return AgentSignal(
            signal_type=signal_type,
            confidence=confidence,
            reasoning=self._build_reasoning(data)
        )
```

**Consensus Pattern:**
```python
class ChiefAgent(BaseAgent):
    """Aggregates signals from all agents"""

    async def analyze(self, data: StockData, context: Dict) -> AgentSignal:
        agent_signals = context.get('agent_signals', {})

        # Weighted aggregation
        total_weight = 0
        weighted_score = 0

        for agent_name, signal in agent_signals.items():
            weight = AGENT_WEIGHTS.get(agent_name, 1.0)
            score = self._signal_to_score(signal.signal_type)
            weighted_score += score * weight * (signal.confidence / 100)
            total_weight += weight

        final_score = weighted_score / total_weight if total_weight else 50
        return self._score_to_signal(final_score)
```

---

## Orchestrator Pattern (Central Coordinator)

**Use Case:** Manage complex workflows with multiple independent systems

**Structure:**
```python
class AutonomousOrchestrator:
    """Central coordinator for trading system"""

    def __init__(self):
        self.scanner_a = ModelPredictionScanner()
        self.scanner_b = NewsAlertScanner()
        self.agent_coordinator = AgentCoordinator()
        self.execution_engine = ExecutionEngine()

    async def start(self):
        """Run all components concurrently"""
        await asyncio.gather(
            self._run_model_scanner(),
            self._run_news_scanner(),
            self._process_opportunities(),
            self._monitor_positions()
        )

    async def _run_model_scanner(self):
        """Path A: Model scanning (3-minute intervals)"""
        while self.is_running:
            opportunities = self.scanner_a.scan()
            for opp in opportunities:
                await self.agent_message_queue.put(opp)
            await asyncio.sleep(180)

    async def _process_opportunities(self):
        """Consumer: Process queued opportunities"""
        while self.is_running:
            try:
                opp = await asyncio.wait_for(
                    self.agent_message_queue.get(),
                    timeout=60
                )
                await self._execute_opportunity(opp)
            except asyncio.TimeoutError:
                continue
```

---

## Factory Pattern (Broker Creation)

**Use Case:** Create different broker implementations

**Structure:**
```python
class BrokerFactory:
    """Factory for creating broker instances"""

    @staticmethod
    def create(broker_type: str, **kwargs) -> BaseBroker:
        """Create broker based on type"""
        brokers = {
            'paper': PaperBroker,
            'ssi': SSIBroker,
            'vndirect': VNDirectBroker,
        }

        broker_class = brokers.get(broker_type)
        if not broker_class:
            raise ValueError(f"Unknown broker type: {broker_type}")

        return broker_class(**kwargs)

# Usage
paper_broker = BrokerFactory.create('paper', initial_balance=100_000_000)
live_broker = BrokerFactory.create('ssi', api_key='xxx')
```

---

## Strategy Pattern (Multiple Signal Sources)

**Use Case:** Multiple scanning strategies with same interface

**Structure:**
```python
from abc import ABC, abstractmethod

class BaseScannerStrategy(ABC):
    @abstractmethod
    def scan(self) -> List[Opportunity]:
        """Each scanner has same interface"""
        pass

class ModelPredictionScanner(BaseScannerStrategy):
    def scan(self) -> List[Opportunity]:
        # ML-based scanning
        predictions = self.model.predict(self.data)
        return self._filter_opportunities(predictions)

class NewsAlertScanner(BaseScannerStrategy):
    def scan(self) -> List[Opportunity]:
        # News-based scanning
        alerts = self.news_fetcher.fetch()
        return self._filter_alerts(alerts)

# Usage (same interface for both)
for scanner in [ModelPredictionScanner(), NewsAlertScanner()]:
    opportunities = scanner.scan()  # Same call
```

---

## Dataclass Pattern (Data Transfer Objects)

**Use Case:** Transfer structured data between components

**Structure:**
```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any

@dataclass
class AgentSignal:
    """Standardized signal output from agents"""
    signal_type: SignalType
    confidence: float  # 0-100
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'signal_type': self.signal_type.value,
            'confidence': self.confidence,
            'price_target': self.price_target,
            # ... other fields
        }

@dataclass
class ModelPrediction:
    """Output from ML model predictions"""
    symbol: str
    expected_return_5d: float
    confidence: float
    has_opportunity: bool
    model_type: str = "Stockformer"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## Singleton Pattern (Single Instance)

**Use Case:** Shared resource like configuration or database connection

**Structure:**
```python
class ConfigManager:
    """Singleton configuration manager"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize configuration (called once)"""
        self.config = self._load_config()

    def get(self, key: str, default=None):
        return self.config.get(key, default)

# Usage
config = ConfigManager()
port = config.get('API_PORT', 8100)
# Every call to ConfigManager() returns same instance
```

---

## Cache Pattern (Performance Optimization)

**Use Case:** Avoid expensive computations

**Structure:**
```python
class CachedPredictor:
    """Predictor with caching"""

    def __init__(self, ttl_seconds: int = 300):
        self.cache: Dict[str, tuple] = {}
        self.ttl = ttl_seconds

    def predict(self, symbol: str) -> float:
        # Check cache
        if symbol in self.cache:
            result, timestamp = self.cache[symbol]
            if time.time() - timestamp < self.ttl:
                return result  # Return cached result

        # Compute if missing or expired
        result = self._expensive_prediction(symbol)
        self.cache[symbol] = (result, time.time())
        return result

    def _expensive_prediction(self, symbol: str) -> float:
        # Real prediction logic
        return self.model.predict(symbol)
```

---

## Builder Pattern (Complex Object Construction)

**Use Case:** Build complex objects with many options

**Structure:**
```python
class OrderBuilder:
    """Builder for creating Order objects"""

    def __init__(self):
        self.order = Order()

    def with_symbol(self, symbol: str) -> 'OrderBuilder':
        self.order.symbol = symbol
        return self

    def with_side(self, side: str) -> 'OrderBuilder':
        self.order.side = side
        return self

    def with_quantity(self, qty: int) -> 'OrderBuilder':
        self.order.quantity = qty
        return self

    def with_price(self, price: float) -> 'OrderBuilder':
        self.order.price = price
        return self

    def build(self) -> Order:
        # Validate before returning
        self.order.validate()
        return self.order

# Usage (fluent API)
order = (OrderBuilder()
    .with_symbol("ACB")
    .with_side("BUY")
    .with_quantity(100)
    .with_price(26500)
    .build())
```

---

## Observer Pattern (Event Publishing)

**Use Case:** Notify multiple subscribers of events

**Structure:**
```python
class EventPublisher:
    """Publish events to subscribers"""

    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)

    def publish(self, event_type: str, data: Any) -> None:
        """Publish event to all subscribers"""
        if event_type in self.subscribers:
            for handler in self.subscribers[event_type]:
                try:
                    handler(data)
                except Exception as e:
                    logger.error(f"Handler error: {e}")

# Usage
publisher = EventPublisher()
publisher.subscribe('order_executed', on_order_executed)
publisher.subscribe('order_executed', log_order_executed)

# Later...
publisher.publish('order_executed', order)
# Both handlers are called
```

---

*These design patterns provide proven solutions for common architectural challenges in VN-Quant.*
