# VN-Quant Paper Trading Platform - Project Overview & PDR

**Project Version:** 4.0.0
**Last Updated:** 2026-02-25
**Status:** Paper Trading Phase In Progress
**Current Phase:** Validation & Optimization (Phase 2)

---

## Executive Summary

VN-Quant is a fully autonomous stock trading system optimized for the Vietnamese stock market. The platform orchestrates 6 specialized AI agents that provide real-time consensus-based trading decisions using dual signal pathways: technical ML predictions (102 Stockformer models) and fundamental news sentiment (24/7 CafeF RSS monitoring).

**Core Capability:** Fully autonomous trading orchestration executing orders without human intervention based on multi-agent consensus, with complete Vietnam market compliance.

**Technology Foundation:**
- **102+ Python modules** organized by responsibility (agents, scanners, core, ML, web)
- **102 Stockformer transformer models** for 5-day return forecasting
- **289 Vietnamese stocks** with historical data (parquet files)
- **Real-time CafeF API** integration for market data + news feeds
- **React 19 + Vite** responsive dashboard with WebSocket real-time updates
- **28 API endpoints** for analysis, signals, paper trading operations

**Key Achievement:** Transitioned from development to production-ready system with advanced confidence scoring, real-time signal deduplication, market regime detection, and Vietnam compliance enforcement.

---

## Product Development Requirements (PDR)

### 1. Functional Requirements

#### 1.1 Multi-Agent Architecture
- **Requirement:** Deploy 6 specialized agents for consensus trading decisions
- **Implementation:**
  - Bull Agent: Bullish technical analysis & trend following
  - Bear Agent: Risk detection & downside protection
  - Alex (Analyst): Advanced technical indicators & support/resistance
  - Scout: Market opportunity scanner & pattern detection
  - RiskDoctor: Risk management & position sizing
  - Chief: Final decision maker & consensus aggregation
- **Acceptance Criteria:**
  - All 6 agents can run in parallel
  - Agent discussions stream to WebSocket in real-time
  - Chief verdict based on weighted consensus (70%+ confidence threshold)

#### 1.2 Autonomous Order Execution
- **Requirement:** Auto-execute trades without user confirmation
- **Implementation:**
  - Parser receives opportunity from Model or News pathway
  - Agents discuss for 30-60 seconds max
  - Chief provides verdict
  - ExecutionEngine places order immediately
  - Order status streamed to dashboard
- **Acceptance Criteria:**
  - Order placed within 2 minutes of signal detection
  - Zero manual confirmations required
  - Full order trail logged to database

#### 1.3 Vietnam Market Compliance
- **Requirement:** Enforce Vietnam stock exchange rules
- **Implementation:**
  - T+2.5 settlement enforcement (can't sell before T+2)
  - Ceiling/floor price limits (7% per session)
  - Tick size validation (VND 100, 500, 1,000)
  - Position limits (max 12.5% per stock)
  - Market hours enforcement (9:15-14:45 trading)
- **Acceptance Criteria:**
  - All orders comply with VN market rules
  - Rejected orders logged with reason
  - Position limits enforced before order submission

#### 1.4 Signal Generation Pathways
- **Pathway A (Model):** Stockformer ML predictions on 102 stocks
  - Scan every 3 minutes during market hours
  - Filter: Expected return > 3% AND Confidence > 0.7
  - Trigger agent discussion if opportunity detected
- **Pathway B (News):** Real-time Vietnamese news sentiment
  - Monitor 24/7 from CafeF RSS feeds
  - Classify: CRITICAL, HIGH, MEDIUM, LOW
  - Immediate trigger for CRITICAL/HIGH news
- **Acceptance Criteria:**
  - Both pathways run independently
  - No signal lost due to buffering
  - News triggers execute within 60 seconds

#### 1.5 Position Management
- **Requirement:** Intelligent entry detection and exit automation
- **Implementation:**
  - Entry detection: 5 S/R methods + 4 entry types (breakout, bounce, reversion, trend-following)
  - Exit logic: Trailing stop (5% from peak) OR Take Profit (15%) OR Stop Loss (5%)
  - Time decay: Auto-exit weak positions after T+5
  - Monitor loop: Check positions every 60 seconds
- **Acceptance Criteria:**
  - Profitable positions held for max profit capture
  - Losing positions cut quickly
  - T+2 compliance enforced
  - No manual exit required

#### 1.6 Advanced Confidence Scoring
- **Requirement:** Replace naive formulas with 6-factor system
- **Implementation:**
  - Factor 1: Expected Return Magnitude (20% weight)
  - Factor 2: Model Historical Accuracy (20% weight)
  - Factor 3: Market Volatility (inverse) (15% weight)
  - Factor 4: Volume Confirmation (15% weight)
  - Factor 5: Technical Alignment (15% weight)
  - Factor 6: Market Regime Alignment (15% weight)
- **Acceptance Criteria:**
  - Confidence score 0-100 with clear reasoning
  - Breakdowns shown in dashboard
  - Warnings for low confidence trades

#### 1.7 Machine Learning Integration
- **Requirement:** Deploy Stockformer models for 102 VN stocks
- **Implementation:**
  - Pre-trained Stockformer models stored in `models/` directory
  - Scanner loads all 102 models at startup
  - Batch predictions every 3 minutes
  - 5-day return forecasting
- **Acceptance Criteria:**
  - Models load in < 10 seconds
  - Batch predictions complete in < 30 seconds
  - Error handling for missing models

#### 1.8 Real-Time Dashboard
- **Requirement:** Live visualization of autonomous trading
- **Implementation:**
  - WebSocket-based real-time updates
  - Agent conversations displayed chronologically
  - Portfolio stats (value, P&L, positions)
  - Order history with execution details
- **Acceptance Criteria:**
  - Dashboard accessible at `/autonomous`
  - Updates within 500ms of event
  - Auto-reconnect on disconnect

---

### 2. Non-Functional Requirements

#### 2.1 Performance
- Model predictions: < 30 seconds for 102 stocks
- Order execution: < 2 minutes from signal to placement
- Dashboard updates: < 500ms latency
- Memory usage: < 2GB for full system

#### 2.2 Reliability
- System uptime: 99%+ during trading hours
- Graceful shutdown without order loss
- Auto-restart on crash with position recovery
- Full audit trail of all decisions

#### 2.3 Security
- Paper trading by default (multi-layer protection)
- Environment variable guards for real trading
- No sensitive credentials in code
- API access restricted to localhost (development)

#### 2.4 Scalability
- Support 102+ stocks in scanner
- Handle 10+ concurrent positions
- WebSocket scale to 50+ clients
- Database optimized for 1M+ historical records

#### 2.5 Maintainability
- Code organized by responsibility (agents, core, scanners)
- Comprehensive logging at DEBUG/INFO levels
- Configuration externalized to `.env`
- Test suite for critical components

---

## Go-Live Checklist

### Pre-Launch (COMPLETE ✅)

#### Code Quality & Architecture
- [x] Security audit completed (multi-layer paper trading protection)
- [x] Algorithm review completed (consensus logic validated)
- [x] Major cleanup completed (86 unused files removed)
- [x] Code organized by responsibility (agents, core, scanners, models)
- [x] Comprehensive error handling implemented
- [x] Logging system integrated at all layers

#### Market Compliance
- [x] VN market rules enforcement verified
  - T+2.5 settlement validation
  - Ceiling/floor price checks
  - Tick size alignment
  - Position limits enforcement
- [x] Market hours enforcement (9:15-14:45 trading)
- [x] Order validation pipeline tested

#### Signal Generation
- [x] Model pathway working (Stockformer with 102 stocks)
- [x] News pathway working (CafeF RSS integration)
- [x] Real-time signal caching implemented
- [x] Both pathways tested independently

#### Agent System
- [x] All 6 agents functional and communicating
- [x] Chief verdict generation working
- [x] Consensus logic tested
- [x] Agent conversations logged

#### Data Integration
- [x] VN-Index & market breadth integration confirmed
- [x] Real-time market data feeds connected
- [x] Historical data properly stored
- [x] Data quality validation implemented

#### ML Models
- [x] Stockformer models loaded successfully
- [x] Model prediction pipeline working
- [x] Batch processing optimized
- [x] Fallback handling for errors

#### Web Interface
- [x] FastAPI server running on port 8100
- [x] WebSocket for real-time updates
- [x] Dashboard displaying agent conversations
- [x] Portfolio stats calculation working

### Launch Phase (IN PROGRESS ⏳ - 70% Complete)

#### Paper Trading Validation
- [x] Refactored monolithic run file into 4 modular routers (trading, market, data, news)
- [x] Removed inline HTML dashboard, replaced with React 19 + Vite frontend
- [x] Implemented 12-phase money flow behavioral improvements
- [x] Added VPS API as primary market data source (foreign flow tracking)
- [x] Integrated LLM interpretation service (Claude Sonnet 4.6 via localhost:8317)
- [x] Fixed 34 bugs (13 critical+high Phase 1 + 21 critical+high Phase 2)
- [x] Today's live candle appended to historical OHLCV data
- [x] VN-Index realtime status endpoint via CafeF banggia API
- [x] Chart candle colors VN market style (close vs previous close)
- [x] WebSocket with exponential backoff reconnection
- [ ] Run paper trading for 15-20 trading days (ongoing - ~10 days completed)
- [ ] Monitor signal quality and execution (ongoing)
- [ ] Analyze model prediction accuracy per stock
- [ ] Benchmark vs historical backtest results

#### Performance Baseline
- [x] Model prediction latency measured
- [x] Order execution latency verified (<2 min signal→execution)
- [x] Memory/CPU monitoring implemented
- [x] WebSocket latency < 500ms confirmed
- [ ] Dashboard performance under sustained load testing

#### Risk Management Testing
- [x] Trailing stop logic verified
- [x] Position limit enforcement tested
- [x] Stop loss triggers working
- [x] Take profit logic confirmed

---

## System Architecture Overview

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────┐
│         AUTONOMOUS ORCHESTRATOR                      │
├──────────────────┬──────────────────────────────────┤
│  PATH A          │         PATH B                   │
│  Model Scanner   │      News Scanner                │
│  (3min)          │      (24/7)                      │
│       ↓          │         ↓                        │
│  Stockformer     │   CafeF RSS                      │
│  102 stocks      │   Sentiment Analysis             │
│       ↓          │         ↓                        │
│  Opportunity?    │   CRITICAL/HIGH?                 │
│  (R>3%, C>0.7)   │                                  │
└──────┬───────────┴──────────────────┬───────────────┘
       │                              │
       └──────────────┬───────────────┘
                      ↓
        ┌─────────────────────────────┐
        │  Agent Discussion (30-60s)   │
        │  Bull, Bear, Alex, Scout     │
        │  RiskDoctor, Chief           │
        └──────────────┬────────────────┘
                       ↓
        ┌─────────────────────────────┐
        │ Chief Verdict Generated      │
        │ Consensus + Confidence       │
        └──────────────┬────────────────┘
                       ↓
        ┌─────────────────────────────┐
        │ Risk Checks + VN Compliance  │
        │ Position Limits + Market     │
        │ Hours + Order Validation     │
        └──────────────┬────────────────┘
                       ↓
        ┌─────────────────────────────┐
        │ Execution Engine             │
        │ Place Order (Auto)           │
        │ Log to Database              │
        └──────────────┬────────────────┘
                       ↓
        ┌─────────────────────────────┐
        │ Position Monitor             │
        │ Check every 60s              │
        │ Trailing Stop / Exit Logic   │
        └─────────────────────────────┘
                       ↓
        ┌─────────────────────────────┐
        │ Real-time Dashboard          │
        │ WebSocket to Frontend        │
        │ Agent Conversations          │
        └─────────────────────────────┘
```

### Key System Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **ModelPredictionScanner** | `quantum_stock/scanners/` | Scan 102 stocks for opportunities |
| **NewsAlertScanner** | `quantum_stock/scanners/` | Monitor Vietnamese news 24/7 |
| **AutonomousOrchestrator** | `quantum_stock/autonomous/` | Central coordinator |
| **PositionExitScheduler** | `quantum_stock/autonomous/` | Manage position exits |
| **AgentCoordinator** | `quantum_stock/agents/` | Agent discussion orchestration |
| **ExecutionEngine** | `quantum_stock/core/` | Order execution & trade management |
| **ConfidenceScoring** | `quantum_stock/core/` | Multi-factor confidence calculation |
| **VNMarketRules** | `quantum_stock/core/` | VN compliance validation |
| **RealtimeSignals** | `quantum_stock/core/` | Signal caching & deduplication |
| **MarketRegimeDetector** | `quantum_stock/utils/` | Bull trap & regime detection |
| **FastAPI Server** | `run_autonomous_paper_trading.py` | Web dashboard & WebSocket |

---

## Feature Highlights

### 1. Multi-Agent Consensus
- 6 specialized AI agents with distinct perspectives
- Each agent provides independent analysis
- Chief aggregates using weighted consensus
- Transparent voting visible in dashboard

### 2. Dual Pathway Intelligence
- **Model Path:** Fast, data-driven, technical
- **News Path:** Flexible, sentiment-aware, fundamental
- Both run independently, no blocking
- Combined opportunities for stronger signals

### 3. Vietnam-Optimized Trading
- T+2.5 settlement enforced at order placement
- Ceiling/floor limits checked
- VN market hours only (9:15-14:45)
- Sector-specific logic for banking/tech

### 4. Smart Position Management
- Entry detection with 5 support/resistance methods
- 4 entry type classifications (breakout, bounce, etc)
- Dynamic exit with trailing stop
- Time-decay auto-exit after T+5

### 5. Real-Time ML Integration
- Stockformer transformer models (102 stocks)
- 5-day return forecasting
- Batch prediction every 3 minutes
- Graceful degradation on model errors

### 6. Paper Trading with Realism
- Simulated slippage (0.1-0.3%)
- Realistic order fill delays
- Commission & tax simulation
- Full position tracking

---

## Success Metrics (Benchmarks)

### From Historical Backtest (8 PASSED Stocks)
```
Metric                Value
────────────────────────────
Average Sharpe Ratio  2.13
Average Return        +41.6%
Win Rate              53.2%
Best Performer        ACB (+56.6%)
Worst Performer       HPG (+29.5%)
```

### Paper Trading Targets
- Win rate: 50-55% (realistic expectation)
- Sharpe ratio: 1.8-2.2 (with drawdowns)
- Monthly return: 2.5-4% (conservative)
- Maximum drawdown: -10% to -15%

---

## Risk Mitigations

### Multi-Layer Paper Trading Protection
1. **Default Mode:** Paper trading enabled by default
2. **Environment Variable:** Requires `ALLOW_REAL_TRADING=true` to disable
3. **User Confirmation:** Interactive prompt for real trading
4. **Logging:** All real trading attempts logged as critical events

### Order Risk Controls
- Position size capped at 12.5% per stock
- Maximum 10 concurrent positions
- Stop loss enforced at -5%
- Take profit enforced at +15%

### Data Quality Checks
- Volume confirmation required
- Price outlier detection
- Missing data handling with fallback
- News sentiment validation

---

## Deployment Information

### System Requirements
- Python 3.10+
- 2GB RAM minimum
- Internet connection for market data
- Disk: 1GB for models + data cache

### Dependencies
- FastAPI + Uvicorn (web framework)
- Pandas + NumPy (data processing)
- Scikit-learn (ML preprocessing)
- TA (technical indicators)
- Feedparser (news RSS)

### Environment Variables (Key)
```bash
TRADING_MODE=paper              # paper or live
INITIAL_CAPITAL=100000000       # 100M VND
MAX_POSITION_PCT=0.125          # 12.5% per stock
MAX_POSITIONS=10                # 10 max holdings
ALLOW_REAL_TRADING=false        # Multi-layer protection
```

### Database
- SQLite for paper trading (default)
- PostgreSQL for production
- Tables: orders, positions, executions, conversations

---

## Known Limitations & Future Enhancements

### Current Limitations
1. **Paper Trading Only:** Real trading requires broker API integration
2. **Limited Stocks:** 102 stocks in Stockformer models
3. **News Sentiment:** Vietnamese-language bias
4. **Manual Configuration:** Thresholds require manual tuning

### Planned Enhancements
1. Live trading integration (SSI broker)
2. Extended model coverage (300+ stocks)
3. Multi-language news support
4. Automated hyperparameter optimization
5. Mobile app for alerts
6. Slippage machine learning models

---

## Contact & Support

**Project Lead:** VN-Quant Development Team
**Status Page:** Available via dashboard at `/status`
**Logs Location:** `logs/autonomous_trading.log`
**Configuration:** `.env` file in project root

**Quick Start:**
```bash
python run_autonomous_paper_trading.py
# Then open: http://localhost:8100/autonomous
# React frontend loads from port 8100 (integrated with FastAPI)
```

---

## Document Maintenance

| Section | Last Updated | Next Review |
|---------|--------------|-------------|
| Executive Summary | 2026-02-25 | 2026-03-15 |
| PDR - Functional | 2026-02-25 | 2026-03-15 |
| PDR - Non-Functional | 2026-02-25 | 2026-03-15 |
| Go-Live Checklist | 2026-02-25 | Weekly during validation |
| System Architecture | 2026-02-25 | 2026-03-15 |

---

*This document serves as the master reference for VN-Quant system capabilities, requirements, and deployment readiness. All team members should be familiar with this PDR before launch.*
