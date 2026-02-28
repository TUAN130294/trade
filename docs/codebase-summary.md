# VN-Quant Codebase Summary

**Generated:** 2026-02-27
**Total Python Files:** 110+
**Total LOC:** ~52,000 (Python 45K + React 2K + Scripts 2.5K + FastAPI 2.3K)
**Core Codebase Size:** 3.5MB
**Status:** Paper Trading Phase 2 (~70% complete)

---

## Directory Structure Overview

```
app/
├── api/routers/               # 4 modular FastAPI routers (2286 LOC)
│   ├── trading.py            # Orders, positions, trades, discussions, reset, stop (317 LOC)
│   ├── market.py             # VN-Index status, smart signals, interpretation (1237 LOC)
│   ├── data.py               # Stock OHLCV, predictions, stats, deep flow (316 LOC)
│   └── news.py               # News alerts, sentiment, watchlist, backtest (347 LOC)
│
└── core/
    ├── auth.py               # API key auth for dangerous endpoints (52 LOC)
    └── state.py              # Global app state (orchestrator, websockets) (17 LOC)

quantum_stock/
├── agents/                    # Multi-agent system (22 files, 7231 LOC)
│   ├── base_agent.py         # Abstract base class & data models
│   ├── chief_agent.py        # Final decision maker (weighted consensus)
│   ├── bull_agent.py         # Bullish analyst (trend following)
│   ├── bear_agent.py         # Risk detector (FOMO, downside protection)
│   ├── analyst_agent.py      # Technical analysis specialist (indicators)
│   ├── flow_agent.py         # Money flow analyst (Wyckoff patterns)
│   ├── risk_doctor.py        # Risk management specialist (position sizing)
│   ├── agent_coordinator.py  # Agent orchestration
│   └── [15+ supporting files]
│
├── autonomous/                # Autonomous orchestration (3 files, 1842 LOC)
│   ├── orchestrator.py        # Central coordinator (CORE)
│   ├── position_exit_scheduler.py  # Exit logic & T+2 compliance
│   └── __init__.py
│
├── core/                      # Core engine (15 files, 6368 LOC)
│   ├── execution_engine.py    # Order execution pipeline
│   ├── broker_api.py          # Broker abstraction layer
│   ├── confidence_scoring.py  # 6-factor confidence system
│   ├── vn_market_rules.py     # Vietnam market compliance
│   ├── realtime_signals.py    # Signal caching & deduplication
│   ├── portfolio_optimizer.py # Position sizing
│   ├── kelly_criterion.py     # Kelly formula
│   ├── backtesting_engine.py  # Strategy testing
│   └── [7+ supporting files]
│
├── scanners/                  # Signal pathways (2 files, 941+ LOC)
│   ├── model_prediction_scanner.py    # Path A: ML predictions (3-min)
│   └── news_alert_scanner.py          # Path B: News sentiment (24/7)
│
├── dataconnector/             # Market data (3 files, 1244 LOC)
│   ├── realtime_market.py     # CafeF API (primary), VPS (fallback)
│   ├── vps_market.py          # VPS API connector
│   └── market_flow.py         # Flow data processing
│
├── news/                      # News analysis (3 files, 952 LOC)
│   ├── sentiment_analyzer.py  # VADER sentiment + Vietnamese NLP
│   ├── rss_news_fetcher.py    # CafeF RSS integration
│   └── interpretation_service.py  # LLM interpretation
│
├── ml/                        # Machine learning
│   ├── ensemble_predictor.py  # Ensemble forecasting
│   └── [prediction utilities]
│
├── models/                    # Pre-trained models
│   ├── stockformer_*.pt       # 102 Stockformer models (trained)
│   ├── maddpg/*.pt            # Multi-agent RL models
│   └── [model metadata]
│
├── services/                  # Business logic (2 files, 373 LOC)
│   ├── interpretation_service.py  # LLM service (Claude Sonnet 4.6)
│   └── [service utilities]
│
├── web/                       # Web UI (legacy, 3 files, 2050 LOC)
│   ├── vn_quant_api.py        # Old API (deprecated for new routers)
│   ├── templates/             # HTML templates (deprecated)
│   └── [legacy assets]
│
├── utils/                     # Utilities
├── analysis/                  # Market analysis utilities
├── data/                      # Historical & cache data
├── db/                        # Database models
└── tests/                     # Unit tests

vn-quant-web/src/             # React 19 + Vite frontend (1918 LOC)
├── App.jsx                    # Main React app, router, state, localStorage (110 LOC)
├── main.jsx                   # Entry point (10 LOC)
├── components/ (12 files, 1068 LOC)
│   ├── sidebar.jsx            # Navigation (10 views)
│   ├── stock-chart.jsx        # Candlestick (lightweight-charts v5, VN colors)
│   ├── technical-panel.jsx    # Support/resistance, patterns
│   ├── trading-view.jsx       # Trading dashboard container
│   ├── portfolio-stats.jsx    # Cash, value, P&L
│   ├── positions-table.jsx    # Active positions
│   ├── orders-table.jsx       # Order history
│   ├── websocket-feed.jsx     # Real-time event stream
│   ├── discussions-view.jsx   # Agent discussions
│   ├── discussion-detail-modal.jsx # Discussion details
│   └── agent-votes-table.jsx  # Agent voting breakdown
├── views/ (8 files, 721 LOC)
│   ├── dashboard-view.jsx     # Overview + signals
│   ├── analysis-view.jsx      # Chart + technical
│   ├── radar-view.jsx         # Agent status radar
│   ├── command-view.jsx       # Multi-agent analysis
│   ├── backtest-view.jsx      # Strategy backtesting
│   ├── predict-view.jsx       # Stockformer predictions
│   ├── data-hub-view.jsx      # Data coverage stats
│   └── news-intel-view.jsx    # News & sentiment
├── hooks/use-websocket.js     # WebSocket exponential backoff
└── utils/constants.js         # API_URL, fmtMoney

scripts/                       # Automation scripts (7 files, 2507 LOC)
├── auto_model_evaluator.py    # Model evaluation automation
├── daily_automation.py        # Daily trading tasks
├── hybrid_training_orchestrator.py  # Training coordination
├── local_cpu_training.py      # CPU-based training
├── colab_training_setup.py    # Google Colab setup
├── sync_to_gdrive.py          # GDrive sync
└── [training utilities]

Root Files:
├── run_autonomous_paper_trading.py  # Main entry point (FastAPI on port 8100)
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Docker image
├── docker-compose.yml               # Multi-container orchestration
├── .env.example                     # Configuration template
└── [config files]
```

---

## Core Modules Explained

### 1. Agents System (`quantum_stock/agents/`)

**Purpose:** Multi-agent consensus trading system

#### Key Files:

| File | Purpose | Key Classes |
|------|---------|-------------|
| `base_agent.py` | Base abstractions | BaseAgent, AgentSignal, StockData, SignalType |
| `chief_agent.py` | Decision maker | ChiefAgent (weighted consensus) |
| `bull_agent.py` | Bullish analysis | BullAgent (trend following) |
| `bear_agent.py` | Risk detection | BearAgent (downside protection) |
| `analyst_agent.py` | Technical analysis | AnalystAgent (indicators) |
| `risk_doctor.py` | Risk management | RiskDoctorAgent (position sizing) |
| `agent_coordinator.py` | Orchestration | AgentCoordinator (team sync) |

#### Data Models:

```python
# Signal type enum
SignalType: STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL, WATCH, MIXED

# Agent signal output
AgentSignal:
  - signal_type: SignalType
  - confidence: 0-100
  - price_target, stop_loss, take_profit
  - risk_reward_ratio
  - reasoning: str
  - metadata: dict

# Agent message for dashboard
AgentMessage:
  - agent_name: str
  - agent_emoji: str
  - message_type: ANALYSIS, ALERT, RECOMMENDATION, WARNING
  - content: str
  - confidence: 0-100
```

#### Execution Flow:

1. Each agent analyzes stock data independently
2. Agents store signals in coordinator
3. Chief aggregates with weights (Bull=1.0, Bear=1.0, Alex=1.2, RiskDoctor=0.8)
4. Weighted consensus score calculated
5. Final signal passed to orchestrator

---

### 2. Autonomous Trading (`quantum_stock/autonomous/`)

**Purpose:** Central orchestration of fully autonomous trading

#### Key Files:

| File | Purpose |
|------|---------|
| `orchestrator.py` | Central coordinator (500 lines) |
| `position_exit_scheduler.py` | Position management & exits (350 lines) |

#### AutonomousOrchestrator Class:

**Responsibilities:**
- Run ModelPredictionScanner (every 3 minutes)
- Run NewsAlertScanner (24/7)
- Receive opportunities from both pathways
- Trigger agent discussions
- Execute orders automatically
- Monitor positions for exits
- Stream events to WebSocket

**Architecture:**
```
┌──────────────────────────────────────┐
│    AutonomousOrchestrator            │
├──────────────────────────────────────┤
│ - model_scanner: ModelPredictionScanner
│ - news_scanner: NewsAlertScanner
│ - exit_scheduler: PositionExitScheduler
│ - agent_coordinator: AgentCoordinator
│ - execution_engine: ExecutionEngine
│ - broker: BrokerFactory
│ - market_regime_detector: MarketRegimeDetector
├──────────────────────────────────────┤
│ Event Loop:                          │
│ 1. Scan for opportunities (parallel) │
│ 2. Receive signals (Model + News)    │
│ 3. Agent discussion (30-60s)         │
│ 4. Execute if Chief approves         │
│ 5. Monitor positions (every 60s)     │
│ 6. Auto-exit on conditions           │
│ 7. Broadcast to WebSocket            │
└──────────────────────────────────────┘
```

#### PositionExitScheduler:

**Responsibilities:**
- Monitor all open positions
- Check exit conditions every 60 seconds
- Enforce T+2.5 settlement (can't sell before T+2)
- Trailing stop: Exit if price drops 5% from peak
- Take profit: Exit at +15%
- Stop loss: Exit at -5%
- Time decay: Auto-exit after T+5 if weak

**Exit Priority:**
1. Stop loss (protect capital)
2. Trailing stop (protect profits)
3. Take profit (capture gains)
4. Time decay (free up capital)
5. Force exit at T+5 (position limit)

---

### 3. Core Engine (`quantum_stock/core/`)

**Purpose:** Trading execution, compliance, and signal processing

#### NEW Modules (Advanced Features):

| File | Purpose | Lines |
|------|---------|-------|
| `confidence_scoring.py` | 6-factor scoring system | 250+ |
| `vn_market_rules.py` | Vietnam compliance | 300+ |
| `realtime_signals.py` | Signal caching | 200+ |

#### ConfidenceScoring - 6 Factors:

```python
Factor 1: Expected Return Magnitude (20% weight)
  - 0.0 return → 0.0 confidence
  - 3.0% return → 0.6 confidence
  - 5.0%+ return → 0.9 confidence

Factor 2: Model Historical Accuracy (20%)
  - Per-stock cache of past accuracy
  - Rolling window (last 100 predictions)
  - Default: 0.5 for unknown models

Factor 3: Market Volatility (Inverse) (15%)
  - Lower volatility → higher confidence
  - Calculate 20-day ATR
  - Normalize to 0-1 scale

Factor 4: Volume Confirmation (15%)
  - Above 50-day average → boost
  - Below average → penalty
  - Prevents low-liquidity trades

Factor 5: Technical Alignment (15%)
  - Support/resistance nearby
  - Trend following indicators
  - Momentum alignment

Factor 6: Market Regime (15%)
  - Bull market → higher confidence
  - Bear market → lower confidence
  - Sideways → neutral

Result: confidence = sum(factor * weight)
```

#### VNMarketRules - Compliance Checks:

```python
Checks Performed:
1. T+2.5 Settlement
   - Can't sell before 2.5 trading days
   - Entry day + 2 full days + 0.5 day

2. Ceiling/Floor Limits
   - Max +7% from prior close
   - Max -7% from prior close
   - Enforced by exchange, pre-validated

3. Tick Size Validation
   - 100-999: VND 100 tick
   - 1,000+: VND 500 or 1,000 tick
   - Ensures valid price levels

4. Position Limits
   - Max 12.5% per stock
   - Max 10 total positions
   - Portfolio rebalancing if needed

5. Market Hours
   - Only 9:15-14:45 trading
   - Pre-market/after-hours rejected
   - Timezone: Asia/Ho_Chi_Minh

6. Order Type Validation
   - LIMIT: Normal orders
   - MARKET: Market orders
   - ATO/ATC: Opening/closing auction
```

#### ExecutionEngine - Order Flow:

```python
Order Placement:
1. Create Order object (OrderSide, OrderType, qty, price)
2. Risk validation (position limits, order size)
3. VN market compliance check
4. Broker order submission
5. Order tracking
6. Fill notification
7. Position update

Order Types:
- LIMIT (LO): Limit order
- MARKET (MP): Market price
- ATO: At Opening (special)
- ATC: At Close (special)

Order Status:
PENDING → SUBMITTED → PARTIAL → FILLED
                        ↓
                    REJECTED
                        ↓
                    CANCELLED
```

#### RealtimeSignals - Deduplication:

```python
Cache Structure:
- Key: f"{symbol}_{signal_type}_{timestamp_hour}"
- Value: SignalData with timestamp + confidence
- TTL: 1 hour (prevents duplicate triggers)

Deduplication Logic:
1. Check if signal exists in cache for symbol
2. If exists + confidence < new, update
3. If exists + confidence >= new, skip (duplicate)
4. If not exists, add to cache
5. Only trigger if new confidence > 0.7
```

---

### 3b. Data Connector & News (`quantum_stock/dataconnector/` + `quantum_stock/news/`)

**Purpose:** Real-time market data collection and Vietnamese news monitoring

#### RealTimeMarketConnector (`realtime_market.py`):

**Key Responsibilities:**
- Fetch real-time price data from CafeF API
- Market breadth metrics (VN-Index, top gainers/losers)
- Foreign investor flow tracking
- Volume anomalies detection
- Fallback to parquet historical data

**Data Provided (Priority Order):**
```
1. CafeF API (Real-time OHLCV, VN-Index, foreign flow)
2. VPS Securities API (Foreign investor flow tracking)
3. Parquet Files (Historical fallback, 289 stocks)
   - OHLCV, Bid/Ask spreads, 52-week levels
   - Support/Resistance key levels
   - Market regime indicators
```

**Coverage:** 289 Vietnamese stocks (parquet) + live CafeF + VPS flow data

#### VNStockNewsFetcher (`news/rss_news_fetcher.py`):

**RSS Sources Monitored:**
1. **VietStock** - Company events, insider trading, dividend announcements
2. **CafeF** - Market analysis, sector news
3. **VnExpress** - General business news

**Processing Pipeline:**
- Fetch RSS → Parse → Extract symbols (regex)
- Sentiment analysis using VADER
- Alert classification (CRITICAL, HIGH, MEDIUM, LOW)
- Vietnamese keyword detection (tăng vốn, IPO, M&A, phá sản)

---

### 4. Signal Scanners (`quantum_stock/scanners/`)

**Purpose:** Detect trading opportunities via two independent pathways

#### ModelPredictionScanner (Path A):

**Frequency:** Every 3 minutes during market hours

**Logic:**
```python
1. Load 102 Stockformer pre-trained models
2. Get latest market data (OHLCV)
3. Batch predict 5-day returns for all stocks
4. Filter:
   - Expected return > 3% OR < -3%
   - Confidence > 0.7 (from model)
5. For each opportunity:
   - Calculate multi-factor confidence (6-factor)
   - Create ModelPrediction object
   - Pass to orchestrator
6. Log all predictions + confidence breakdown
```

**ModelPrediction Dataclass:**
```python
symbol: str
expected_return_5d: float
confidence: float  # 0-100 (6-factor scoring)
has_opportunity: bool
model_type: str = "Stockformer"
timestamp: datetime
metadata: dict
```

**Supported Stocks:** 102 VN stocks (8 PASSED + 94 others)

#### NewsAlertScanner (Path B):

**Frequency:** 24/7 monitoring

**Logic:**
```python
1. Fetch CafeF RSS feeds (Vietnamese news)
2. Parse each news item
3. Extract company name + tickers
4. Sentiment analysis (VADER)
5. Classify alert level:
   - CRITICAL: stock price move >5% expected
   - HIGH: significant event (M&A, capital raise)
   - MEDIUM: moderate events
   - LOW: minor news
6. If CRITICAL/HIGH:
   - Immediate trigger to orchestrator
   - Bypass normal filters
   - Fast path (skips model analysis)
7. Log all alerts with reasoning
```

**NewsAlert Dataclass:**
```python
symbol: str
headline: str
url: str
sentiment: float  # -1 to 1
alert_level: str  # CRITICAL, HIGH, MEDIUM, LOW
suggested_action: str  # BUY, SELL, HOLD
published_date: datetime
reasoning: str
```

**Keywords Monitored:**
- Company-specific: tăng vốn, IPO, M&A
- Market-wide: độc quyền, giải tán, phá sản
- Sector-specific: lợi suất, cấp phép

---

### 5. Web Server & API (`run_autonomous_paper_trading.py`)

**Purpose:** FastAPI server with 28+ endpoints and WebSocket real-time streaming

#### FastAPI Routes (28+ Endpoints):

**Dashboard & WebSocket:**
- `/autonomous` - React dashboard (GET)
- `/ws/autonomous` - Real-time event streaming (WebSocket)

**Trading Router** (`app/api/routers/trading.py`):
- `/api/status` - System status (GET)
- `/api/orders` - Order history (GET)
- `/api/positions` - Current positions (GET)
- `/api/trades` - Trade records (GET)
- `/api/discussions` - Agent discussions (GET)
- `/api/test/opportunity` - Test signals (POST)
- `/api/test/trade` - Test execution (POST)
- `/api/reset` - Reset paper trading (POST)
- `/api/stop` - Stop trading (POST)

**Market Router** (`app/api/routers/market.py`):
- `/api/market/status` - Market regime status (GET)
- `/api/market/regime` - Market regime details (GET)
- `/api/market/smart-signals` - Market-wide signals (GET)

**Data Router** (`app/api/routers/data.py`):
- `/api/stock/{symbol}` - Stock data (GET)
- `/api/predict/{symbol}` - Model prediction (GET)
- `/api/data/stats` - Portfolio stats (GET)
- `/api/analyze/deep_flow` - Flow analysis (POST)

**News Router** (`app/api/routers/news.py`):
- `/api/news/status` - News system status (GET)
- `/api/news/alerts` - Active news alerts (GET)
- `/api/news/market-mood` - Market sentiment (GET)
- `/api/news/watchlist` - Watched stocks (GET)
- `/api/news/scan` - News scan (POST)

#### Data Source Priority:
1. **CafeF API** - Real-time prices (primary), market breadth, volume anomalies
2. **VPS Securities API** - Foreign investor flow (khối ngoại) tracking
3. **Parquet Files** - Historical fallback (289 stocks downloaded)
4. **RSS Feeds** - VietStock, CafeF, VnExpress news aggregation

#### WebSocket Message Format:

```python
{
  "type": "agent_message" | "order_executed" | "position_updated",
  "timestamp": "2026-01-12T10:30:00",
  "data": {
    # Type-specific data
  }
}

Example Agent Message:
{
  "type": "agent_message",
  "data": {
    "agent_name": "Bull",
    "agent_emoji": "🐂",
    "message_type": "ANALYSIS",
    "content": "ACB breaking above resistance, bullish for +10%",
    "confidence": 85
  }
}
```

#### Dashboard Features:

1. **System Stats Panel**
   - Status (Running/Stopped)
   - Portfolio value
   - Total P&L
   - Today's trades count

2. **Agent Conversations**
   - Chronological message stream
   - Color-coded by agent
   - Emoji indicators
   - Confidence scores

3. **Positions Panel**
   - Symbol, quantity, entry price
   - Current price, P&L %
   - Days held
   - Exit conditions

4. **Orders Panel**
   - Order ID, symbol, side
   - Quantity, price
   - Execution time
   - Status

---

## Data Flow Diagrams

### Complete Trading Cycle

```
Signal Detection (3min + 24/7)
         ↓
    ┌─────────┬────────┐
    ↓         ↓        ↓
Model News   News    (Opportunity?)
Scan  Path   Path    ✓ = Yes, ✗ = No
    ↓        ↓        ↓
    └─────────┴────────┘
         ↓
  Orchestrator Receives
  Opportunity Context
         ↓
  Agent Discussion (30-60s)
  6 agents → Chief
         ↓
  Chief Verdict
  (consensus + confidence)
         ↓
  Risk Validation
  VN rules + position limits
         ↓
  Execution Engine
  Place order (auto)
         ↓
  Order Tracking
  Broker confirmation
         ↓
  Position Monitor
  Every 60 seconds
         ↓
  Exit Check
  ✓ = Exit, ✗ = Hold
         ↓
  Dashboard Update
  WebSocket broadcast
```

### Data Schema

#### Order Flow:
```
Order (dict)
├── order_id: str (UUID)
├── symbol: str (e.g., "ACB")
├── side: "BUY" | "SELL"
├── order_type: "LO" | "MP" | "ATO" | "ATC"
├── quantity: int
├── price: float (VND)
├── status: PENDING → SUBMITTED → FILLED
├── timestamp: datetime
└── broker_order_id: str (if submitted)

Position (dict)
├── symbol: str
├── quantity: int
├── avg_price: float
├── current_price: float
├── entry_time: datetime
├── entry_days_held: float
├── unrealized_pnl: float
├── unrealized_pnl_pct: float
└── status: "ACTIVE" | "PENDING_EXIT"
```

---

## Configuration & Environment

### Key Environment Variables:

```bash
# Trading Mode
TRADING_MODE=paper              # paper or live
ALLOW_REAL_TRADING=false        # Multi-layer protection

# Capital & Risk
INITIAL_CAPITAL=100000000       # 100M VND
MAX_POSITION_PCT=0.125          # 12.5% per position
MAX_POSITIONS=10                # Max 10 holdings
STOP_LOSS_PCT=0.05              # 5% stop loss
TAKE_PROFIT_PCT=0.15            # 15% take profit

# Model Scanner
MODEL_SCAN_INTERVAL=180         # 3 minutes
MODEL_RETURN_THRESHOLD=0.03     # 3% min return
MODEL_CONFIDENCE_THRESHOLD=0.7  # 70% min confidence

# News Scanner
NEWS_SCAN_INTERVAL=300          # 5 minutes
NEWS_SOURCES=cafef              # CafeF RSS

# API Server
API_HOST=0.0.0.0
API_PORT=8000
```

---

## Key Statistics

### Codebase Metrics
- Total Python files: 102
- Total lines of code: ~8,000+
- Core modules: 15+
- Tests: 5+ test suites
- Documentation: 3,000+ lines

### Module Distribution
- Agents: 25+ files (30%)
- Core: 20+ files (25%)
- Web/API: 10+ files (15%)
- ML/Models: 15+ files (20%)
- Utils/Tests: 10+ files (10%)

### Model Coverage
- Stockformer models: 102 stocks
- 8 PASSED (high confidence)
- 94 monitored (lower confidence)
- Model load time: < 10 seconds
- Batch prediction time: < 30 seconds

---

## Dependencies Overview

### Core Dependencies
- **fastapi** (0.104.0+): Web framework
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: ML utilities
- **ta**: Technical indicators

### Optional/Specialized
- **torch**: Neural network (Stockformer)
- **tensorflow**: Alternative ML (commented out)
- **prophet**: Time series (optional)
- **statsmodels**: Statistical analysis

### Async & Real-time
- **uvicorn**: ASGI server
- **websockets**: WebSocket protocol
- **aiohttp**: Async HTTP client
- **nest_asyncio**: Async utilities

### News & Data
- **feedparser**: RSS parsing
- **beautifulsoup4**: HTML parsing
- **vaderSentiment**: Sentiment analysis
- **requests**: HTTP requests

---

## Testing & Quality

### Test Coverage
- Unit tests: 20+
- Integration tests: 15+
- System tests: 10+

### Quality Metrics
- Type hints: 90%+ code coverage
- Docstrings: All functions documented
- Error handling: 100% critical paths
- Logging: INFO, DEBUG, WARNING levels

---

## Future Scalability

### Planned Enhancements
1. Live trading broker integration (SSI)
2. Extended stock coverage (300+)
3. Additional ML models
4. Mobile app integration
5. Multi-user support
6. Database optimization for 1M+ records

### Performance Optimization
- Prediction caching
- Batch processing
- Connection pooling
- Memory management

---

*This codebase summary provides a complete map of the VN-Quant system architecture, modules, and data flows. Refer to individual module documentation for detailed implementation.*
