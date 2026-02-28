# VN-Quant Code Standards & Patterns

**Version:** 1.0
**Updated:** 2026-01-12
**Scope:** All quantum_stock Python modules

---

## Table of Contents

1. [Naming Conventions](#naming-conventions)
2. [Code Organization](#code-organization)
3. [Design Patterns](#design-patterns)
4. [Error Handling](#error-handling)
5. [Logging Standards](#logging-standards)
6. [Type Hints](#type-hints)
7. [Documentation](#documentation)
8. [Testing Standards](#testing-standards)
9. [Performance Guidelines](#performance-guidelines)

---

## Naming Conventions

### Files & Modules

**Python Files:**
- Use `snake_case` for all filenames
- Descriptive names reflecting module purpose
- Examples: `model_prediction_scanner.py`, `confidence_scoring.py`, `vn_market_rules.py`

**Directories:**
- Use `snake_case` lowercase
- Organize by responsibility: `agents`, `scanners`, `core`, `models`
- No numbers or special characters (except underscores)

### Classes

**Class Names:**
- Use `PascalCase` (CapitalizedWords)
- Descriptive, noun-based names
- Examples:
  ```python
  class AutonomousOrchestrator:
  class ModelPredictionScanner:
  class ChiefAgent:
  class MultiFactorConfidence:
  ```

**Class Attributes (private/protected):**
- Prefix with underscore: `_private_attribute`
- Examples:
  ```python
  class Agent:
      _confidence_cache: Dict[str, float]
      _last_signal: Optional[AgentSignal]
  ```

### Functions & Methods

**Function Names:**
- Use `snake_case`
- Verb-based for actions
- Examples:
  ```python
  def calculate_confidence(...):
  def execute_order(...):
  def detect_entry_point(...):
  def _validate_vn_compliance(...):  # Private
  ```

**Parameters & Variables:**
- Use `snake_case`
- Clear, descriptive names
- Avoid single letters except loop counters
- Examples:
  ```python
  expected_return: float
  market_regime: str
  agent_signals: Dict[str, AgentSignal]
  ```

**Constants:**
- Use `UPPER_SNAKE_CASE`
- Group at module top level
- Examples:
  ```python
  MAX_POSITION_PCT = 0.125
  TAKE_PROFIT_THRESHOLD = 0.15
  MODEL_CONFIDENCE_MIN = 0.7
  VN_MARKET_OPEN = time(9, 15)
  VN_MARKET_CLOSE = time(14, 45)
  ```

### Enums

**Enum Names:**
- Use `PascalCase`
- Members in `UPPER_SNAKE_CASE`
- Examples:
  ```python
  class SignalType(Enum):
      STRONG_BUY = "STRONG_BUY"
      BUY = "BUY"
      HOLD = "HOLD"
      SELL = "SELL"
      STRONG_SELL = "STRONG_SELL"

  class OrderSide(Enum):
      BUY = "BUY"
      SELL = "SELL"

  class AlertLevel(Enum):
      CRITICAL = "CRITICAL"
      HIGH = "HIGH"
      MEDIUM = "MEDIUM"
      LOW = "LOW"
  ```

### Dataclasses

**Dataclass Names:**
- Use `PascalCase`
- Use `@dataclass` decorator (not inheritance)
- Examples:
  ```python
  @dataclass
  class AgentSignal:
      signal_type: SignalType
      confidence: float
      price_target: Optional[float] = None

  @dataclass
  class ModelPrediction:
      symbol: str
      expected_return_5d: float
      confidence: float
  ```

---

## Code Organization

### Module Structure

**Standard Module Layout:**
```python
# ============================================
# File docstring (purpose & role)
# ============================================
"""
ModuleName
==========
Brief description of what this module does

Responsibilities:
- First responsibility
- Second responsibility

Example:
    >>> result = main_function()
"""

# ============================================
# Imports
# ============================================
# 1. Standard library
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

# 2. Third-party
import pandas as pd
import numpy as np
import logging

# 3. Local
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from quantum_stock.core.base_models import BaseModel

# ============================================
# Constants
# ============================================
CONFIDENCE_THRESHOLD = 0.7
MAX_RETRIES = 3

# ============================================
# Logger
# ============================================
logger = logging.getLogger(__name__)

# ============================================
# Dataclasses / Enums
# ============================================
class SignalType(Enum):
    ...

@dataclass
class Signal:
    ...

# ============================================
# Main Classes
# ============================================
class MainClass:
    ...

# ============================================
# Helper Functions
# ============================================
def helper_function():
    ...

# ============================================
# Entry Point (if applicable)
# ============================================
if __name__ == "__main__":
    ...
```

### File Organization Rules

1. **Imports First:** All imports at top (stdlib → third-party → local)
2. **Constants Second:** All MODULE_CONSTANTS defined
3. **Logger Third:** Single logger per module
4. **Enums/Dataclasses:** Before main classes that use them
5. **Main Classes:** Core functionality
6. **Helpers:** Supporting functions (often private)
7. **Main Block:** Only if directly runnable

### Max File Sizes

- **Ideal:** < 150 lines (single responsibility, testable)
- **Small modules:** 150-250 lines (utilities, single class)
- **Medium modules:** 250-400 lines (agent class, analyzer)
- **Large modules:** 400-600 lines (orchestrator, engine)
- **Exceeded:** > 600 lines → must split into submodules
- **Monitor:** Files >800 LOC are candidates for refactoring

---

## Async Patterns

See `docs/code-patterns-async.md` for detailed AsyncIO patterns, concurrent execution, and async/await best practices.

---

## Design Patterns

See `docs/code-patterns-design.md` for comprehensive coverage of design patterns used in VN-Quant:
- Agent Pattern (multi-agent consensus)
- Orchestrator Pattern (central coordination)
- Factory Pattern (broker abstraction)
- Strategy Pattern (multiple scanners)
- Dataclass Pattern (data transfer objects)
- Singleton Pattern (shared resources)
- Cache Pattern (performance optimization)
- Builder Pattern (complex objects)
- Observer Pattern (event publishing)

---

## Error Handling

### Exception Hierarchy

**Create custom exceptions:**
```python
class VNQuantException(Exception):
    """Base exception for all VN-Quant errors"""
    pass

class MarketComplianceError(VNQuantException):
    """Raised when VN market rule is violated"""
    pass

class OrderExecutionError(VNQuantException):
    """Raised when order execution fails"""
    pass

class SignalProcessingError(VNQuantException):
    """Raised during signal analysis errors"""
    pass

class ModelPredictionError(VNQuantException):
    """Raised when model prediction fails"""
    pass
```

### Error Handling Patterns

**Pattern 1: Validate then Execute**
```python
def execute_trade(order: Order) -> bool:
    """Execute trade with validation"""
    try:
        # Pre-validation
        if not self._validate_order(order):
            raise ValueError("Order validation failed")

        # Compliance check
        if not self._check_vn_compliance(order):
            raise MarketComplianceError("VN market rule violated")

        # Execute
        result = self.broker.place_order(order)
        logger.info(f"Order executed: {order.order_id}")
        return True

    except MarketComplianceError as e:
        logger.warning(f"Compliance rejection: {e}")
        return False
    except OrderExecutionError as e:
        logger.error(f"Execution failed: {e}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise VNQuantException(f"Trade execution failed: {e}") from e
```

**Pattern 2: Graceful Degradation**
```python
def predict_returns(symbols: List[str]) -> Dict[str, float]:
    """Predict returns with graceful error handling"""
    predictions = {}

    for symbol in symbols:
        try:
            pred = self.model.predict(symbol)
            predictions[symbol] = pred
        except ModelPredictionError as e:
            logger.warning(f"Prediction failed for {symbol}: {e}")
            # Use fallback (e.g., 0 return)
            predictions[symbol] = 0.0
        except Exception as e:
            logger.exception(f"Unexpected error for {symbol}: {e}")
            predictions[symbol] = 0.0

    return predictions
```

**Pattern 3: Context Managers for Resource Management**
```python
class DatabaseConnection:
    def __enter__(self):
        self.conn = psycopg2.connect(...)
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
        return False

# Usage
with DatabaseConnection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT ...")
    # Auto-closed on exit, even if exception
```

---

## Logging Standards

### Logger Initialization

**Every module should have:**
```python
import logging

logger = logging.getLogger(__name__)

# In module usage:
logger.debug("Detailed diagnostic information")
logger.info("Confirmation that things are working")
logger.warning("Something unexpected, but continuing")
logger.error("Serious error, unable to continue function")
logger.critical("System cannot continue, immediate action needed")
```

### Logging Levels

| Level | Use Case | Example |
|-------|----------|---------|
| DEBUG | Development, detailed tracing | `logger.debug("Cache hit for ACB")` |
| INFO | Normal operation confirmations | `logger.info("Order executed: ORD123")` |
| WARNING | Suspicious but continuing | `logger.warning("Low confidence signal: 0.45")` |
| ERROR | Function failure, can recover | `logger.error("Model prediction failed, using fallback")` |
| CRITICAL | System failure, immediate action | `logger.critical("REAL TRADING enabled - manual review required")` |

### Logging Best Practices

```python
# Good: Include relevant context
logger.info(f"Order placed: symbol={order.symbol}, qty={order.quantity}, price={order.price}")

# Bad: Too vague
logger.info("Order placed")

# Good: Include values for debugging
logger.warning(f"Confidence below threshold: {confidence:.2f} < {threshold}")

# Bad: No specific info
logger.warning("Low confidence")

# Good: Exception logging with full context
try:
    result = risky_operation()
except Exception as e:
    logger.exception(f"Operation failed for symbol={symbol}, error={e}")
    # exception() includes full traceback

# Bad: Exception logging without traceback
except Exception as e:
    logger.error(f"Operation failed: {e}")  # Missing traceback
```

### Format & Structure

**Standard log format:**
```
2026-01-12 10:30:45.123 | INFO | quantum_stock.agents.chief_agent | Chief verdict: STRONG_BUY (confidence=0.87)
2026-01-12 10:30:46.456 | WARNING | quantum_stock.core.execution | Position limit reached: 12.5% >= 12.5%
2026-01-12 10:30:47.789 | ERROR | quantum_stock.scanners.model | Model prediction failed for ACB: OutOfMemory
```

---

## Type Hints

### Comprehensive Type Hints

**Required for all functions and methods:**
```python
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from datetime import datetime

# Good: Complete type hints
def calculate_confidence(
    expected_return: float,
    model_accuracy: float,
    volatility: float,
    volume_factor: float
) -> float:
    """Calculate confidence score with all factors."""
    return min(1.0, expected_return + model_accuracy - volatility * 0.5)

# Bad: Missing type hints
def calculate_confidence(expected_return, model_accuracy, volatility, volume_factor):
    pass
```

### Complex Type Hints

```python
# Dict with specific key/value types
agent_signals: Dict[str, AgentSignal] = {}

# Optional types
price_target: Optional[float] = None

# Union types (if multiple types possible)
data_source: Union[str, List[str]] = "cafef"

# Callable types (functions as parameters)
callback: Callable[[str, float], bool] = lambda symbol, return_pct: return_pct > 0.03

# Tuple with specific types
signal_components: Tuple[float, float, float] = (0.8, 0.9, 0.7)

# List of dataclasses
opportunities: List[ModelPrediction] = []

# Generic with multiple bounds
def process_data(data: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Process data and return symbol-value pairs."""
    pass
```

### Return Type Hints

```python
# Async functions
async def analyze(self, data: StockData) -> AgentSignal:
    pass

# Generator functions
def generate_opportunities(self) -> Iterator[Opportunity]:
    yield Opportunity(...)

# Functions that might raise
def validate_order(self, order: Order) -> bool:
    """Returns True if valid, raises exception if not."""
    pass
```

---

## Documentation

### Module Docstrings

```python
# Good: Comprehensive module docstring
"""
Multi-Factor Confidence Scoring System
======================================
Advanced confidence calculation for trading signals

Replaces naive formulas with 6-factor system:
1. Expected Return Magnitude (20% weight)
2. Model Historical Accuracy (20%)
3. Market Volatility (inverse) (15%)
4. Volume Confirmation (15%)
5. Technical Alignment (15%)
6. Market Regime Alignment (15%)

Example:
    >>> confidence = MultiFactorConfidence()
    >>> result = confidence.calculate_confidence(
    ...     expected_return=0.05,
    ...     df=price_data,
    ...     symbol="ACB"
    ... )
    >>> print(f"Confidence: {result.total_confidence:.2f}")
"""
```

### Function/Method Docstrings

```python
def calculate_confidence(
    self,
    expected_return: float,
    df: pd.DataFrame,
    symbol: str,
    market_regime: str = "BULL"
) -> ConfidenceResult:
    """
    Calculate multi-factor confidence score.

    Combines 6 factors with weighted averaging to produce
    a single confidence score (0-100) representing the
    strength of a trading signal.

    Args:
        expected_return: Predicted 5-day return (e.g., 0.05 for +5%)
        df: Historical price data (OHLCV columns required)
        symbol: Stock symbol (e.g., "ACB")
        market_regime: Current market regime ("BULL", "BEAR", "NEUTRAL")

    Returns:
        ConfidenceResult with:
            - total_confidence: 0-100 score
            - factor breakdown: individual factor scores
            - reasoning: explanation of factors
            - warnings: any concerns about signal

    Raises:
        ValueError: If expected_return not in (-1, 1) range
        KeyError: If required OHLCV columns missing from df

    Examples:
        >>> result = confidence.calculate_confidence(
        ...     expected_return=0.03,
        ...     df=df,
        ...     symbol="ACB",
        ...     market_regime="BULL"
        ... )
        >>> if result.total_confidence > 0.7:
        ...     print("Strong signal!")
    """
    # Implementation
    pass
```

### Class Docstrings

```python
class MultiFactorConfidence:
    """
    Multi-factor confidence scoring system for trading signals.

    Implements a weighted combination of 6 factors to produce
    a robust confidence score that accounts for market conditions,
    model accuracy, and technical alignment.

    Attributes:
        model_accuracy_cache: Historical accuracy per model (Dict[str, float])

    Methods:
        calculate_confidence: Main scoring method
        _factor_return_magnitude: Factor 1 implementation
        _factor_model_accuracy: Factor 2 implementation

    Examples:
        >>> scorer = MultiFactorConfidence()
        >>> result = scorer.calculate_confidence(...)
        >>> print(result.to_dict())
    """
    pass
```

### Inline Comments

```python
# Good: Explain WHY, not WHAT
# Use inverse volatility: lower volatility = higher confidence
# because stable stocks have more predictable patterns
volatility_factor = 1.0 - min(1.0, volatility / 0.15)

# Bad: Obvious from code
# Calculate volatility factor
volatility_factor = 1.0 - min(1.0, volatility / 0.15)

# Good: Explain business logic
# T+2.5 compliance: can't sell before 2 full trading days + 0.5
# This converts calendar days to trading days
can_exit = days_held >= 2.5

# Bad: Repeating code
# Check if days_held >= 2.5
can_exit = days_held >= 2.5
```

---

## Testing Standards

### Test File Organization

```python
"""
Test suite for confidence scoring system
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
from quantum_stock.core.confidence_scoring import MultiFactorConfidence

class TestMultiFactorConfidence:
    """Test suite for MultiFactorConfidence class"""

    @pytest.fixture
    def scorer(self):
        """Create scorer instance for tests"""
        return MultiFactorConfidence()

    @pytest.fixture
    def sample_data(self):
        """Create sample price data"""
        return pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [102, 103, 104, 105, 106],
            'low': [99, 100, 101, 102, 103],
            'close': [101, 102, 103, 104, 105],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })

    def test_high_return_high_confidence(self, scorer, sample_data):
        """Test that high expected return produces high confidence"""
        result = scorer.calculate_confidence(
            expected_return=0.10,  # 10%
            df=sample_data,
            symbol="ACB"
        )
        assert result.total_confidence > 0.7

    def test_low_return_low_confidence(self, scorer, sample_data):
        """Test that low expected return produces low confidence"""
        result = scorer.calculate_confidence(
            expected_return=0.01,  # 1%
            df=sample_data,
            symbol="ACB"
        )
        assert result.total_confidence < 0.5

    def test_invalid_return_raises_error(self, scorer, sample_data):
        """Test that out-of-range return raises ValueError"""
        with pytest.raises(ValueError):
            scorer.calculate_confidence(
                expected_return=2.0,  # Invalid: > 1.0
                df=sample_data,
                symbol="ACB"
            )

class TestIntegrationConfidence:
    """Integration tests with real data"""

    @pytest.mark.integration
    def test_confidence_with_market_data(self):
        """Test confidence scoring with real market data"""
        # Load real data, run full calculation
        pass
```

### Test Best Practices

- Use descriptive test names: `test_trailing_stop_triggers_at_five_percent_loss`
- Follow Arrange-Act-Assert pattern
- Use `@pytest.fixture` for reusable setup
- Mark integration tests: `@pytest.mark.integration`

---

## Performance Guidelines

1. **Correctness first** → readable code second → performance third
2. **Cache** frequently accessed data (dict cache, `@cached` decorator)
3. **Batch process** multiple symbols instead of individual calls
4. **Asyncio** for I/O-bound operations (`asyncio.gather`)
5. **Lazy load** expensive resources (property with `hasattr` check)

See `docs/code-patterns-async.md` for detailed async patterns.

---

## Code Review Checklist

### React/JSX Components

See `docs/code-patterns-websocket-react.md` for detailed React component patterns, WebSocket integration, and Tailwind CSS styling guidelines.

---

### Python & Async Code Review Checklist

Before submitting code for review, verify:

- [ ] Type hints on all function signatures
- [ ] Docstrings on all public functions/classes
- [ ] Error handling for all external calls
- [ ] Logging at appropriate levels
- [ ] No hardcoded values (use constants)
- [ ] Unit tests for complex logic
- [ ] Naming follows conventions
- [ ] No duplicate code (DRY principle)
- [ ] Performance acceptable (no obvious inefficiencies)
- [ ] Security reviewed (no credential leaks, input validation)

---

*VN-Quant Code Standards v1.0 | See pattern docs for detailed examples.*
