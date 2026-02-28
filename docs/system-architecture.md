# VN-Quant System Architecture

**Version:** 4.0.0
**Date:** 2026-01-12
**Scope:** Complete autonomous trading system for Vietnamese market

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [System Components](#system-components)
3. [Data Flow Architecture](#data-flow-architecture)
4. [Agent Communication](#agent-communication)
5. [Order Execution Pipeline](#order-execution-pipeline)
6. [Real-Time Infrastructure](#real-time-infrastructure)
7. [Database Schema](#database-schema)
8. [Integration Points](#integration-points)
9. [Deployment Topology](#deployment-topology)

---

## High-Level Overview

### System Vision

VN-Quant is designed as a fully autonomous trading system that:
1. Continuously monitors the Vietnamese stock market
2. Discovers trading opportunities via ML predictions and news sentiment
3. Engages 6 AI agents in consensus-based decision making
4. Executes orders automatically without user intervention
5. Manages positions intelligently with smart exits
6. Provides real-time visibility into all decisions

### Core Principle: Dual Pathway Execution

```
┌─────────────────────────────────────────────────────────────┐
│                  AUTONOMOUS ORCHESTRATOR                    │
├────────────────────┬────────────────────────────────────────┤
│   PATH A           │        PATH B                          │
│   (Technical)      │        (Fundamental)                   │
├────────────────────┼────────────────────────────────────────┤
│ ModelPrediction    │ NewsAlertScanner                       │
│ Scanner            │ - CafeF RSS feeds 24/7                │
│ - Stockformer      │ - Vietnamese NLP                       │
│ - 102 stocks       │ - Sentiment scoring                    │
│ - Every 3 min      │ - Alert classification                │
│                    │                                        │
│ Opportunity:       │ Opportunity:                           │
│ ✓ Return > 3%      │ ✓ CRITICAL/HIGH severity              │
│ ✓ Confidence > 70% │ ✓ Immediate path (no filters)        │
└────────────────────┴────────────────────────────────────────┘
        │                          │
        └──────────────┬───────────┘
                       ↓
        ┌──────────────────────────┐
        │  Opportunity Context     │
        │  - Source (Model/News)   │
        │  - Symbol, timestamp     │
        │  - Prediction/Alert data │
        └──────────────┬───────────┘
                       ↓
        ┌──────────────────────────┐
        │   Agent Discussion       │
        │   (30-60 seconds)        │
        │                          │
        │  Bull: 🐂 Bullish view   │
        │  Bear: 🐻 Risk analysis  │
        │  Alex: 📊 Technical      │
        │  Scout: 🔍 Opportunities│
        │  RiskDoc: 💊 Position    │
        │  Chief: 🎖 Final verdict │
        └──────────────┬───────────┘
                       ↓
        ┌──────────────────────────┐
        │  Chief Consensus Vote    │
        │  - Weighted aggregation  │
        │  - Confidence score      │
        │  - Risk assessment       │
        └──────────────┬───────────┘
                       ↓
        ┌──────────────────────────┐
        │  Compliance & Risk Checks│
        │  - VN market rules       │
        │  - Position limits       │
        │  - Order validation      │
        │  - Market hours check    │
        └──────────────┬───────────┘
                       ↓
        ┌──────────────────────────┐
        │  Order Execution         │
        │  (AUTOMATIC - No confirm)│
        │  - Create order          │
        │  - Submit to broker      │
        │  - Track execution       │
        │  - Log to database       │
        └──────────────┬───────────┘
                       ↓
        ┌──────────────────────────┐
        │  Position Monitoring     │
        │  (Check every 60 seconds)│
        │  - Trailing stop logic   │
        │  - T+2 exit eligibility  │
        │  - Profit/loss triggers  │
        │  - Exit or hold          │
        └──────────────┬───────────┘
                       ↓
        ┌──────────────────────────┐
        │  Real-Time Dashboard     │
        │  - WebSocket broadcast   │
        │  - Agent conversations   │
        │  - Portfolio stats       │
        │  - Order history         │
        └──────────────────────────┘
```

---

## System Components

### Data Flow & Integration Points

```
┌─────────────────────────────────────────────────────────────────┐
│              EXTERNAL DATA SOURCES                               │
├──────────────────────────┬──────────────────────────────────────┤
│ CafeF API                │ RSS News Feeds                        │
│ - Real-time OHLCV       │ - VietStock (events, insider)         │
│ - VN-Index breadth      │ - CafeF (analysis, market)            │
│ - Foreign flow          │ - VnExpress (business news)           │
│ - Volume anomalies      │ - Sentiment scoring (VADER)           │
│ - 52-week levels        │ - Alert classification                │
│ - Bid/Ask spreads       │                                       │
└──────────────────┬───────┴──────────────────┬────────────────────┘
                   │                          │
                   ↓                          ↓
        ┌──────────────────┐    ┌─────────────────────┐
        │ RealTimeMarket   │    │ VNStockNewsFetcher  │
        │ Connector        │    │ + NewsAnalyzer      │
        │                  │    │                     │
        │ - OHLCV data     │    │ - Parse RSS items   │
        │ - Market stats   │    │ - Extract symbols   │
        │ - Fallback cache │    │ - Sentiment→score   │
        │ (parquet)        │    │ - Create alerts     │
        └────────┬─────────┘    └──────────┬──────────┘
                 │                         │
                 └─────────────┬───────────┘
                               ↓
        ┌─────────────────────────────────────────┐
        │   FastAPI Server (Port 8100)            │
        │   (run_autonomous_paper_trading.py)     │
        │                                         │
        │   - 4 API routers:                      │
        │     * trading (orders, positions, reset)│
        │     * market (status, regime, signals)  │
        │     * data (stocks, predictions, stats) │
        │     * news (alerts, sentiment, scan)    │
        │   - WebSocket: /ws/autonomous           │
        │   - React 19 + Vite frontend (port 8100)│
        │   - LLM proxy (Claude via localhost:8317)
        └────────────────────┬────────────────────┘
                             ↓
           ┌─────────────────────────────────┐
           │  Autonomous Orchestrator        │
           │  (Central Trading Coordinator)  │
           │                                 │
           │  - Manages all components       │
           │  - Routes signals               │
           │  - Triggers decision makers     │
           │  - Monitors positions           │
           │  - Broadcasts events            │
           └────────────────────┬────────────┘
                                │
        ┌───────────────────────┼─────────────────────────┐
        ↓                       ↓                         ↓
┌────────────────┐   ┌──────────────────┐   ┌─────────────────────┐
│ ModelPrediction│   │ NewsAlertScanner │   │ PositionExitMonitor │
│ Scanner (Path A)   │ (Path B)        │   │                     │
│                    │                  │   │ Every 60 seconds:   │
│ Every 3 minutes:   │ 24/7 continuous: │   │ - Check exit rules  │
│ - Load 102 models  │ - Fetch RSS      │   │ - T+2 eligibility   │
│ - Predict returns  │ - Sentiment→alert│   │ - Trailing stop     │
│ - Filter > 3%      │ - Create objects │   │ - Take profit       │
│ - Confidence score │ - Immediate path │   │ - Stop loss         │
│ - Create opps      │                  │   │ - Exit order        │
└────────┬───────────┘ └────────┬────────┘   └──────────┬──────────┘
         │                      │                       │
         └──────────────────────┴───────────────────────┘
                                │
                                ↓
                  ┌─────────────────────────────┐
                  │  Agent Coordinator          │
                  │  (Discussion Orchestrator)  │
                  │                             │
                  │  Triggers: Bull, Bear,      │
                  │  Alex, Scout, RiskDoctor    │
                  │  → Chief consensus          │
                  │                             │
                  │  Duration: 30-60 seconds    │
                  │  Each agent provides:       │
                  │  - Signal (BUY/SELL/HOLD)  │
                  │  - Confidence (0-100)       │
                  │  - Reasoning                │
                  └──────────────┬──────────────┘
                                 ↓
                  ┌─────────────────────────────┐
                  │  Compliance & Risk Engine   │
                  │                             │
                  │  Validates:                 │
                  │  - VN market rules (T+2.5)  │
                  │  - Position limits (12.5%)  │
                  │  - Market hours (9:15-14:45)│
                  │  - Order structure          │
                  │  - Risk metrics             │
                  └──────────────┬──────────────┘
                                 ↓
                  ┌─────────────────────────────┐
                  │  Execution Engine           │
                  │                             │
                  │  Auto-execute if passed:    │
                  │  - Create order object      │
                  │  - Submit to broker         │
                  │  - Track fill               │
                  │  - Update position          │
                  │  - Log to database          │
                  └──────────────┬──────────────┘
                                 ↓
                  ┌─────────────────────────────┐
                  │  Message Queue & WebSocket  │
                  │                             │
                  │  Broadcasting:              │
                  │  - Agent messages           │
                  │  - Order fills              │
                  │  - Position updates         │
                  │  - Portfolio changes        │
                  │  → Real-time Dashboard      │
                  └─────────────────────────────┘
```

### LLM Interpretation Service

**Purpose:** AI-powered analysis for market insights

**Implementation:**
```
Claude Sonnet 4.6 (via localhost:8317 proxy)
    ↓
Endpoints:
- POST /api/agents/analyze - Multi-agent discussion interpretation
- GET /api/market/smart-signals - Market-wide signal interpretation
```

**Use Cases:**
1. **Agent Discussion Interpretation** - Summarize 6-agent reasoning
2. **Market Signal Interpretation** - Explain trading opportunity context
3. **News Sentiment Deep-Dive** - Detailed news impact analysis

---

### Component Responsibility Map

```
┌────────────────────────────────────────────────────────────────┐
│  FastAPI Web Server (run_autonomous_paper_trading.py)         │
│  - HTTP endpoints (/autonomous, /api/*)                        │
│  - WebSocket (/ws/autonomous)                                  │
│  - Session & auth (localhost only for now)                     │
└────────────────────┬───────────────────────────────────────────┘
                     ↑
       ┌─────────────┴──────────────┐
       ↓                            ↓
┌────────────────────┐  ┌───────────────────────┐
│ Orchestrator       │  │ Message Queue         │
│ (Central Hub)      │  │ (asyncio.Queue)       │
│                    │  │ - Agent messages      │
│ Responsibilities:  │  │ - Order updates       │
│ - Run scanners     │  │ - Position changes    │
│ - Trigger agents   │  │ - Maxsize: 1000       │
│ - Execute orders   │  │                       │
│ - Monitor exits    │  │ TLL: 60 sec events    │
│ - Broadcast events │  │                       │
│ - Error recovery   │  └───────────────────────┘
└────┬────────────┬──┴────────────────┬──────────┐
     ↓            ↓                   ↓          ↓
  ┌──────┐  ┌──────────┐  ┌──────────────┐  ┌──────────┐
  │Model │  │  News    │  │    Agent     │  │ Execution│
  │Scanner   │  Scanner   │  Coordinator │  │  Engine  │
  │          │            │              │  │          │
  │ - Stock- │ - CafeF   │ - 6 agents   │  │ - Orders │
  │   former │   RSS     │   discuss    │  │ - Fills  │
  │ - 102    │ - NLP     │ - Chief vote │  │ - Logs   │
  │   models │ - Senti-  │ - Consensus  │  │          │
  │ - 3 min  │   ment    │               │  │          │
  │   scan   │ - 24/7    │               │  │          │
  └──────┘  └──────────┘  └──────────────┘  └──────────┘
     ↓            ↓                   ↓
  ┌──────────────────────────────────────────┐
  │  Market Data Connectors                   │
  │  - Real-time prices (VCI/SSI)             │
  │  - News feeds (CafeF)                     │
  │  - Historical data (local cache)          │
  └──────────────────────────────────────────┘
     ↑            ↑
     └────┬───────┘
          ↓
  ┌──────────────────┐
  │ Position Monitor │
  │ - Check exits    │
  │ - Update P&L     │
  │ - Every 60s      │
  └──────────────────┘
```

### Core Components Detail

#### 1. ModelPredictionScanner

**Location:** `quantum_stock/scanners/model_prediction_scanner.py`

**Function:** Scan 102 stocks for ML-based trading opportunities

**Execution:**
- Schedule: Every 3 minutes during market hours (9:15-14:45)
- Batch prediction: Load all 102 Stockformer models
- Prediction target: 5-day returns
- Filter: Return > 3% AND Confidence > 0.7

**Output:**
```python
ModelPrediction(
    symbol='ACB',
    expected_return_5d=0.0566,  # +5.66%
    confidence=0.85,             # 85% multi-factor
    has_opportunity=True,
    timestamp=datetime.now()
)
```

**Error Handling:**
- Missing model: Use fallback prediction (0.0 return)
- Prediction error: Log and skip stock
- Data error: Skip timestamp, retry next scan

#### 2. NewsAlertScanner

**Location:** `quantum_stock/scanners/news_alert_scanner.py`

**Function:** Monitor Vietnamese news 24/7 for market alerts

**Execution:**
- Schedule: Every 5 minutes (24/7)
- Data source: CafeF RSS feeds
- Processing: Extract tickers → sentiment → classify
- Alert levels: CRITICAL, HIGH, MEDIUM, LOW

**Output:**
```python
NewsAlert(
    symbol='ACB',
    headline='ACB được chấp thuận tăng vốn 50,000 tỷ',
    sentiment=0.77,              # VADER sentiment
    alert_level='HIGH',
    suggested_action='BUY',
    url='https://cafef.vn/...',
    timestamp=datetime.now()
)
```

**Special Logic:**
- CRITICAL/HIGH news: Bypass normal filters, trigger immediately
- Skip low-conviction predictions mixed with news
- Prevent duplicate alerts (cache by symbol + headline hash)

#### 3. AgentCoordinator

**Location:** `quantum_stock/agents/agent_coordinator.py`

**Function:** Orchestrate 6 agents for consensus decision making

**Agent Roles:**

| Agent | Emoji | Role | Weight | Analysis |
|-------|-------|------|--------|----------|
| Bull | 🐂 | Bullish | 1.0 | Trend following, momentum |
| Bear | 🐻 | Risk detector | 1.0 | Downside protection, resistance |
| Alex | 📊 | Technical analyst | 1.2 | Indicators, support/resistance |
| Scout | 🔍 | Opportunity finder | 1.0 | Pattern recognition |
| RiskDoctor | 💊 | Risk manager | 0.8 | Position sizing, risk limits |
| Chief | 🎖 | Decision maker | — | Consensus aggregation |

**Consensus Algorithm:**
```python
1. Each agent analyzes independently
2. Convert signal → score:
   STRONG_BUY (90) > BUY (70) > HOLD (50) > SELL (30) > STRONG_SELL (10)
3. Weight by agent: score * weight * (confidence/100)
4. Aggregate: total = Σ(weighted_scores) / Σ(weights)
5. Chief verdict: score → signal_type

Thresholds:
- 80+: STRONG_BUY
- 65-79: BUY
- 50-64: HOLD
- 35-49: SELL
- <35: STRONG_SELL
```

#### 4. ExecutionEngine

**Location:** `quantum_stock/core/execution_engine.py`

**Function:** Bridge between trading strategy and broker execution

**Order Lifecycle:**
```
Order Create → Risk Check → VN Compliance → Broker Submit
    → Execution Track → Fill Notification → Position Update
    → Database Log
```

**Broker Abstraction:**
```python
class BaseBroker(ABC):
    def place_order(order: Order) -> OrderResult
    def cancel_order(order_id: str) -> bool
    def get_positions() -> List[Position]
    def get_account_balance() -> float

class PaperBroker(BaseBroker):
    # Paper trading simulation
    # Realistic slippage, delays, fills

class SSIBroker(BaseBroker):
    # Live SSI broker integration
    # Real order placement
```

#### 5. PositionExitScheduler

**Location:** `quantum_stock/autonomous/position_exit_scheduler.py`

**Function:** Monitor open positions and execute exits intelligently

**Check Interval:** Every 60 seconds

**Exit Conditions (Priority Order):**
1. **Stop Loss** (-5%)
   - Hard floor, protects capital
   - Executed immediately

2. **Trailing Stop** (5% below peak)
   - Protects profits dynamically
   - Peak price tracked since entry
   - Exit if price drops 5% from peak

3. **Take Profit** (+15%)
   - Lock in gains
   - Hard ceiling

4. **Time Decay** (T+5 or later)
   - After 5 full trading days
   - Auto-exit weak positions
   - Free up capital

5. **T+2 Compliance**
   - Can't exit before 2.5 days held
   - Enforced at order validation

---

## Data Flow Architecture

### Data Models Flow

```
StockData (Input)
├── symbol: str
├── prices: np.ndarray (OHLCV)
├── volumes: np.ndarray
├── dates: np.ndarray
└── technical_indicators: Dict

    ↓ Agent Analysis ↓

AgentSignal (Agent Output)
├── signal_type: SignalType
├── confidence: 0-100
├── price_target: Optional[float]
├── stop_loss: Optional[float]
├── take_profit: Optional[float]
├── reasoning: str
└── metadata: Dict

    ↓ Chief Aggregation ↓

ChiefSignal (Final Decision)
├── signal_type: SignalType
├── confidence: 0-100 (weighted consensus)
├── price_target: float
├── stop_loss: float
├── take_profit: float
└── reasoning: str (combined from agents)

    ↓ Order Creation ↓

Order (Trade Instruction)
├── order_id: str (UUID)
├── symbol: str
├── side: OrderSide (BUY/SELL)
├── order_type: OrderType (LO/MP/ATO/ATC)
├── quantity: int
├── price: float
├── status: OrderStatus
└── timestamp: datetime

    ↓ Execution ↓

OrderExecution (Confirmation)
├── order_id: str
├── broker_order_id: str
├── filled_quantity: int
├── filled_price: float
├── filled_time: datetime
└── commission: float

    ↓ Position Update ↓

Position (Holding)
├── symbol: str
├── quantity: int
├── avg_price: float
├── entry_time: datetime
├── current_price: float
├── unrealized_pnl: float
├── unrealized_pnl_pct: float
└── status: "ACTIVE"
```

---

## Agent Communication

### Agent Discussion Protocol

**Trigger:** Opportunity detected (Model or News pathway)

**Timeline:**
- T+0s: Orchestrator receives opportunity
- T+5-10s: Agents receive context and StockData
- T+10-50s: Each agent analyzes independently
- T+50-55s: Agents post messages to message queue
- T+55s: Chief aggregates all signals
- T+58s: Chief provides final verdict
- T+60s: Orchestrator makes trade decision

**Message Queue Structure:**
```python
asyncio.Queue(maxsize=1000)  # Bounded to prevent memory leak

Message format:
{
    'agent_name': str,
    'agent_emoji': str,
    'message_type': MessageType (ANALYSIS/ALERT/RECOMMENDATION),
    'content': str (agent's message),
    'confidence': float (0-100),
    'timestamp': datetime,
    'metadata': {
        'signal_type': SignalType,
        'price_target': float,
        'reasoning': str
    }
}
```

### Example Discussion Flow

```
Opportunity: ACB, Expected Return +5.66%, Confidence 0.85

T+5s: Bull Agent
"🐂 Bull: ACB breaking above MA200, strong uptrend.
 Target +8% in 5 days. Confidence 88%."

T+15s: Bear Agent
"🐻 Bear: RSI at 65, near overbought. Volume declining.
 Caution advised. Confidence 60%."

T+25s: Alex (Analyst)
"📊 Alex: Triple top forming at 27,500 resistance.
 Entry on breakout. Confidence 82%."

T+35s: Scout
"🔍 Scout: High volume at breakout, institutional buying.
 Pattern completion imminent. Confidence 75%."

T+45s: RiskDoctor
"💊 RiskDoctor: Risk/reward ratio: 2.1:1 favorable.
 Can position 100 shares (12% of capital). Confidence 80%."

T+50s: Chief Aggregation
"🎖 Chief: Consensus voting...
 Bull (1.0 × 88) + Bear (1.0 × 60) + Alex (1.2 × 82)
 + Scout (1.0 × 75) + RiskDoctor (0.8 × 80)
 = (88 + 60 + 98.4 + 75 + 64) / 4.0 = 71.35

 → STRONG BUY (confidence 71%)"

T+55s: Execution
"✓ Order created: BUY 100 ACB @ 26,500"
"✓ Submitted to broker"
"✓ Waiting for fill..."

T+60s: Fill Confirmation
"✓ Order filled: 100 ACB @ 26,520"
"→ Position opened, monitoring active"
"→ Trailing stop set: 25,194 (-4.9% from peak)"
```

---

## Order Execution Pipeline & Detailed Components

For complete implementation details, see `docs/system-architecture-detailed.md`:
- Order state machine and order flow
- Risk validation 7-point checklist
- Database schema (SQLite, PostgreSQL migration)
- Integration points (CafeF, VPS, RSS)
- Deployment topologies (local, Docker, cloud)
- System resilience and failure recovery
- Performance characteristics and latencies

---

## Real-Time Infrastructure

### WebSocket Architecture

**Connection:** Client connects to `/ws/autonomous`

**Server Stack:**
```
FastAPI app.WebSocket
    ↓
connected_connections: List[WebSocket]
    ↓
Message producers:
- orchestrator events
- agent messages
- order updates
- position changes
    ↓
broadcast_messages() coroutine
    ↓
JSON serialization
    ↓
Send to all active connections
```

**Message Types:**

| Type | Frequency | Content |
|------|-----------|---------|
| `agent_message` | Real-time | Agent name, emoji, message, confidence |
| `order_executed` | Per trade | Order ID, symbol, qty, price, status |
| `position_updated` | Every 60s | Holdings, avg_price, current_price, P&L% |
| `system_status` | Every 30s | Portfolio value, total P&L, trades today |
| `scan_result` | Every 3-5m | Opportunity symbol, return, confidence |
| `websocket_feed` | Real-time | Market data, agent discussions stream |

**Example WebSocket Message:**
```json
{
  "type": "agent_message",
  "timestamp": "2026-01-12T10:30:45.123",
  "data": {
    "agent_name": "Bull",
    "agent_emoji": "🐂",
    "message_type": "ANALYSIS",
    "content": "ACB breaking resistance at 27,500, strong uptrend signal",
    "confidence": 87,
    "metadata": {
      "signal_type": "STRONG_BUY",
      "price_target": 28000,
      "risk_reward_ratio": 2.1
    }
  }
}
```

### Dashboard Real-Time Updates

**React Frontend** (`vn-quant-web/src/`)

**Display Components:**
1. Sidebar - Navigation to 10 views (dashboard, analysis, radar, etc)
2. Agent Conversations - Chronological message stream with emojis
3. Portfolio Stats - Cash, portfolio value, total P&L, daily trades
4. Positions Table - Symbol, qty, entry price, current price, P&L%
5. Orders Table - Order ID, symbol, side, qty, execution time
6. Stock Chart - Candlestick with lightweight-charts v5, VN color scheme
7. Technical Panel - Support/resistance levels, chart patterns
8. WebSocket Feed - Real-time event notifications

**Real-Time Updates:** Auto-refresh via WebSocket, no manual action needed

---

## Additional Documentation

For implementation details, deployment, and configuration, see:
- **`docs/system-architecture-detailed.md`** - Database schema, order execution pipeline, integration points, deployment topologies, resilience, performance
- **`docs/code-patterns-design.md`** - Design patterns (Agent, Orchestrator, Factory, Strategy, etc)
- **`docs/code-patterns-async.md`** - AsyncIO patterns and concurrent execution
- **`docs/code-patterns-websocket-react.md`** - WebSocket and React component patterns

---

*VN-Quant System: 52K LOC • 6 AI agents • 102 ML models • 28+ API endpoints • React 19 frontend • Real-time WebSocket • Paper trading engine*

*See `docs/system-architecture-detailed.md` for integration points, configuration, and deployment details.*

*VN-Quant System Architecture: 52K LOC | 6 agents | 28+ endpoints | React 19 frontend | Paper trading engine*
