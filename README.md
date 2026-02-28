# VN-Quant: Autonomous Paper Trading Platform

![Status](https://img.shields.io/badge/status-production--ready-brightgreen)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-proprietary-red)
![Version](https://img.shields.io/badge/version-4.0.0-blue)

> AI-powered autonomous trading system for the Vietnamese stock market with 6-agent consensus, ML predictions, and real-time news sentiment analysis.

**Key Stats:** 102+ Python modules | 6 specialized AI agents | 102 Stockformer models | 289 Vietnamese stocks | Real-time CafeF data | 24/7 news monitoring

---

## Quick Start (2 Minutes)

### Installation

```bash
# 1. Clone/extract repository
cd D:\testpapertr  # Windows
# or: cd ~/testpapertr  # macOS/Linux

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure (optional - defaults work for paper trading)
cp .env.example .env
```

### Run System

```bash
python run_autonomous_paper_trading.py
```

**Dashboard opens at:** `http://localhost:8100/autonomous`

**Note:** React frontend now serves directly from FastAPI on port 8100

---

## Docker Quick Start

```bash
# Windows: scripts\docker-quickstart.bat
# Linux:   bash scripts/docker-quickstart.sh
```
See `docs/docker-deployment.md` for details.

---

## Key Features

- **6 AI Agents** for consensus trading: Bull, Bear, Alex, Scout, RiskDoctor, Chief
- **Dual Signal Pathways**: Path A (Stockformer ML, 102 stocks, 3-min scan) + Path B (CafeF news sentiment, 24/7)
- **6-Factor Confidence Scoring**: Return magnitude, model accuracy, volatility, volume, technical alignment, market regime
- **Smart Position Management**: Trailing stop (-5% from peak), take profit (+15%), stop loss (-5%), T+5 time decay
- **Real-Time Dashboard**: React 19 + WebSocket streaming, glass-morphism UI, live agent conversations
- **Vietnam Market Compliance**: T+2.5 settlement, ceiling/floor limits, tick validation, market hours (9:15-14:45)
- **Paper Trading Realism**: Simulated slippage, fill delays, commission & tax

---

## System Architecture

### Trading Pipeline
```
╔══════════════════════════════════════════════════════════════╗
║          AUTONOMOUS ORCHESTRATOR                             ║
╠════════════════════════╦════════════════════════════════════╣
║  PATH A (Technical)    ║  PATH B (Fundamental)              ║
║  ModelPredictionScan   ║  NewsAlertScanner                  ║
║  - Stockformer 102     ║  - CafeF RSS 24/7                  ║
║  - Every 3 minutes     ║  - Vietnamese NLP                  ║
║  - R>3%, C>70%         ║  - CRITICAL/HIGH alerts            ║
╚════════════╦═══════════╩════════════════╦═════════════════╝
             ║                            ║
             └──────────┬─────────────────┘
                        ↓
        ╔═══════════════════════════════╗
        ║  Agent Discussion (30-60s)    ║
        ║  Bull, Bear, Alex, Scout      ║
        ║  RiskDoctor, Chief            ║
        ╚═══════════════╦═══════════════╝
                        ↓
        ╔═══════════════════════════════╗
        ║  Chief Verdict Generated      ║
        ║  Weighted Consensus + Score   ║
        ╚═══════════════╦═══════════════╝
                        ↓
        ╔═══════════════════════════════╗
        ║  VN Compliance & Risk Checks  ║
        ║  T+2.5, Position Limits,      ║
        ║  Market Hours, Order Validate ║
        ╚═══════════════╦═══════════════╝
                        ↓
        ╔═══════════════════════════════╗
        ║  Execution Engine             ║
        ║  Place Order (AUTO)           ║
        ║  Broker Submission + Log      ║
        ╚═══════════════╦═══════════════╝
                        ↓
        ╔═══════════════════════════════╗
        ║  Position Monitor             ║
        ║  Check Every 60s              ║
        ║  Exit Conditions Check        ║
        ╚═══════════════╦═══════════════╝
                        ↓
        ╔═══════════════════════════════╗
        ║  Real-Time Dashboard          ║
        ║  WebSocket Broadcast          ║
        ║  Agent Conversations          ║
        ║  Portfolio Stats              ║
        ╚═══════════════════════════════╝
```

### Technology Stack
- **Backend:** FastAPI + Uvicorn (async Python web framework, port 8100)
- **Frontend:** React 19 + Vite 7.3 + Tailwind CSS 3.4 (glass-morphism UI)
- **Charts:** lightweight-charts v5 (VN-style candle colors)
- **ML:** Stockformer transformers (102 pre-trained models)
- **Data:** CafeF API (primary), VPS API (foreign flow), Parquet (historical fallback)
- **LLM:** Claude Sonnet 4.6 (interpretation service via localhost:8317)
- **WebSocket:** Real-time event streaming (agent messages, orders, positions)
- **NLP:** VADER sentiment + Vietnamese NLP for news classification

---

## System Requirements

**Minimum:**
- Python 3.10+
- 2GB RAM
- 5GB disk space
- Internet connection

**Recommended:**
- 4-core CPU, 4GB RAM
- 10GB SSD disk
- 10+ Mbps internet

---

## Documentation

| Document | Purpose |
|----------|---------|
| [project-overview-pdr.md](docs/project-overview-pdr.md) | Executive overview & go-live checklist |
| [system-architecture.md](docs/system-architecture.md) | Complete system design & data flows |
| [codebase-summary.md](docs/codebase-summary.md) | Code structure & module guide |
| [code-standards.md](docs/code-standards.md) | Coding standards & patterns |
| [deployment-guide.md](docs/deployment-guide.md) | Setup & deployment instructions |

---

## Configuration

**Key environment variables** (in `.env`):

```bash
# Trading mode (CRITICAL: keep as "paper")
TRADING_MODE=paper

# Capital & risk
INITIAL_CAPITAL=100000000       # 100M VND
MAX_POSITION_PCT=0.125          # 12.5% per stock
STOP_LOSS_PCT=0.05              # 5% stop loss
TAKE_PROFIT_PCT=0.15            # 15% take profit

# Scanning intervals
MODEL_SCAN_INTERVAL=180         # 3 minutes
MODEL_CONFIDENCE_THRESHOLD=0.7  # 70% minimum

# Security
ALLOW_REAL_TRADING=false        # Multi-layer protection
```

---

## Usage Examples

### Start System

```bash
# Run autonomous trading
python run_autonomous_paper_trading.py

# System will:
# 1. Load 102 ML models
# 2. Start scanning for opportunities
# 3. Listen for news alerts
# 4. Open WebSocket for dashboard
# 5. Begin trading automatically
```

### Access Dashboard

```
http://localhost:8100/autonomous
```

**Dashboard shows (React + Vite Frontend):**
- Real-time agent conversations (WebSocket streaming)
- Current positions & P&L with live candles
- Order history with execution details
- Portfolio statistics (cash, value, daily trades)
- Technical analysis panel (support/resistance, patterns)
- Agent voting breakdown

### View Logs

```bash
# Real-time logs
tail -f logs/autonomous_trading.log

# Search for specific trades
grep "Order executed" logs/autonomous_trading.log

# Check errors
grep "ERROR" logs/autonomous_trading.log
```

### API Endpoints

```bash
# System status
curl http://localhost:8100/api/status

# Portfolio info
curl http://localhost:8100/api/portfolio

# Current positions
curl http://localhost:8100/api/positions

# Order history
curl http://localhost:8100/api/orders
```

---

## Safety & Risk Controls

- Multi-layer paper trading protection (env var + confirmation + logging)
- Position cap: 12.5% per stock, max 10 concurrent
- Stop loss: -5%, Take profit: +15%
- T+2.5, ceiling/floor, tick size, market hours enforced

See [deployment-guide.md](docs/deployment-guide.md) for troubleshooting.

---

## Project Status

**Phase 2 (Validation & Optimization): ~70% complete**

| Status | Items |
|--------|-------|
| Done | 6-agent consensus, 102 ML models, news sentiment, VN compliance, auto-execution, smart exits, WebSocket dashboard, React frontend, LLM service |
| In Progress | Paper trading validation, performance baseline |
| Planned | Live broker (SSI), 300+ stocks, mobile alerts, auto hyperparameter tuning |

**Stats:** ~52K LOC | 110+ modules | 28+ endpoints | 102 Stockformer models | 8 RL agents

---

*Proprietary - Paper trading only | Version 4.0.0 | Last Updated: 2026-02-27*
