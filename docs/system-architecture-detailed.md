# System Architecture - Detailed Components

**Version:** 4.0.0
**Date:** 2026-02-27
**Scope:** Detailed component interactions and data flows

---

## Order Execution Pipeline

### Order State Machine

```
                    CREATE
                      ↓
                  PENDING
                      ↓
                VALIDATION
            ┌───────┴────────┐
        PASS│               │FAIL
            ↓               ↓
        SUBMIT         REJECTED
            ↓          (Log reason)
        SUBMITTED           │
            ↓               │
    ┌─────────────┐         │
    │Broker fills?│         │
    │partial/full?│         │
    └────┬────────┘         │
         │                  │
    PARTIAL ─→ Fill% < 100  │
         │    Continue      │
         │    waiting...    │
         │                  │
        FILLED ─→ Fill% = 100│
         │                  │
         ├─ Log order       │
         ├─ Update position │
         └─ Broadcast update
```

### Risk Validation Steps

```
Order received
    ↓
1. Size Check (quantity > 0 and <= 100,000?)
2. Price Check (price > 0 and valid VN ticks?)
3. Account Check (balance >= order_value?)
4. Position Check (new position <= 12.5% and total <= 10?)
5. Market Hours Check (9:15-14:45?)
6. VN Compliance Check (tick size, ceiling/floor, T+2)
7. Risk Check (risk/reward >= 1:1?)

All pass? → SUBMIT
Any fail? → REJECT (log reason)
```

---

## Database Schema (SQLite for Paper Trading)

### orders table
```sql
CREATE TABLE orders (
    order_id VARCHAR(36) PRIMARY KEY,
    symbol VARCHAR(10),
    side VARCHAR(4),  -- BUY, SELL
    order_type VARCHAR(3),  -- LO, MP, ATO, ATC
    quantity INT,
    price FLOAT,
    status VARCHAR(20),  -- PENDING, SUBMITTED, FILLED, REJECTED
    filled_quantity INT,
    filled_price FLOAT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    INDEX (symbol, created_at)
);
```

### positions table
```sql
CREATE TABLE positions (
    symbol VARCHAR(10) PRIMARY KEY,
    quantity INT,
    avg_price FLOAT,
    entry_time TIMESTAMP,
    entry_day INT,  -- Day count for T+2 tracking
    status VARCHAR(20),  -- ACTIVE, PENDING_EXIT, CLOSED,
    INDEX (status, entry_time)
);
```

### executions table
```sql
CREATE TABLE executions (
    execution_id VARCHAR(36) PRIMARY KEY,
    order_id VARCHAR(36),
    symbol VARCHAR(10),
    filled_quantity INT,
    filled_price FLOAT,
    commission FLOAT,
    execution_time TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    INDEX (symbol, execution_time)
);
```

### agent_conversations table
```sql
CREATE TABLE agent_conversations (
    conversation_id VARCHAR(36) PRIMARY KEY,
    symbol VARCHAR(10),
    agent_name VARCHAR(20),
    message_type VARCHAR(20),
    content TEXT,
    confidence FLOAT,
    timestamp TIMESTAMP,
    INDEX (symbol, timestamp)
);
```

---

## Integration Points

### External Data Sources

**Market Data:**
- Provider: CafeF API (primary), VPS (fallback)
- Update frequency: Real-time (every 1 second)
- Data: OHLCV + bid/ask spreads
- Fallback: Cached historical data (parquet files)

**News Feeds:**
- Source: CafeF RSS feeds
- Frequency: Every 5 minutes (24/7)
- Processing: Vietnamese NLP + sentiment analysis
- Output: NewsAlert objects

**Broker Integration:**
- Live: SSI Securities API (Phase 4)
- Paper: Internal PaperBroker simulation (current)
- Order submission: Async HTTP
- Fill notification: Polling or webhook

### Configuration Management

**Environment Variables (`.env`):**
```bash
# Execution
TRADING_MODE=paper
ALLOW_REAL_TRADING=false
INITIAL_CAPITAL=100000000

# Thresholds
MODEL_CONFIDENCE_THRESHOLD=0.7
MAX_POSITION_PCT=0.125
STOP_LOSS_PCT=0.05
TAKE_PROFIT_PCT=0.15

# Scanning
MODEL_SCAN_INTERVAL=180
NEWS_SCAN_INTERVAL=300

# API
API_HOST=0.0.0.0
API_PORT=8100

# LLM Service
LLM_API_ENDPOINT=http://localhost:8317
```

---

## Deployment Topology

### Single Machine (Development)

```
Local Machine
├── Python 3.10+
├── quantum_stock/ (codebase - 45K LOC)
├── vn-quant-web/ (React frontend - 2K LOC)
├── models/ (102 Stockformer, 8 RL agents)
├── data/ (historical cache, parquet)
├── logs/ (autonomous_trading.log)
└── Run: python run_autonomous_paper_trading.py
    ├── FastAPI server (port 8100)
    ├── WebSocket endpoint (/ws/autonomous)
    ├── SQLite database (paper_trading.db)
    └── LLM proxy (localhost:8317, Claude Sonnet 4.6)

Access: http://localhost:8100/autonomous
```

### Docker (Production-Ready)

```
Docker Network
├── API Service (port 8100)
│   ├── FastAPI + Uvicorn
│   ├── AutonomousOrchestrator
│   ├── WebSocket server
│   └── LLM interpretation service
│
├── Frontend (Nginx, port 5177)
│   ├── React + Vite built assets
│   ├── WebSocket client
│   └── lightweight-charts v5
│
├── PostgreSQL Database (optional)
│   ├── orders, positions, executions
│   └── agent_conversations
│
├── Redis Cache (optional)
│   └── Real-time signals
│
└── Volumes
    ├── models/ (102 Stockformer + RL)
    ├── data/ (parquet + cache)
    └── logs/ (autonomous_trading.log)

docker-compose up -d
```

---

## System Resilience

### Failure Modes & Recovery

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Model prediction fails | Exception caught | Use fallback (0% return) |
| News feed unavailable | RSS timeout | Skip scan, retry next interval |
| Broker disconnected | Connection error | Queue orders, retry on reconnect |
| Database error | DB exception | Log to file, continue operation |
| Memory leak | Monitor via logs | Restart service weekly |
| Order rejection | Broker response | Log reason, skip symbol period |

### Watchdog Monitoring

```python
async def monitor_system_health():
    """Monitor system health every 60s"""
    while True:
        await asyncio.sleep(60)

        health = {
            'memory_mb': get_memory_usage(),
            'active_positions': len(self.positions),
            'model_predictions_cached': len(self.prediction_cache),
            'websocket_clients': len(active_websockets),
            'last_scan_time': self.last_model_scan,
            'orders_today': len(today_orders())
        }

        # Alert if anomalies
        if health['memory_mb'] > 2000:
            logger.warning(f"High memory: {health['memory_mb']}MB")

        logger.info(f"System health: {health}")
```

---

## Performance Characteristics

### Expected Latencies

| Operation | Target | Actual |
|-----------|--------|--------|
| Model scan (102 stocks) | < 30s | 25-28s |
| Agent discussion | 30-60s | 45-55s |
| Order execution | < 2min | 1-2min |
| Dashboard update | < 500ms | 100-300ms |
| Database insert | < 100ms | 50-100ms |

### Resource Usage

| Resource | Baseline | Peak |
|----------|----------|------|
| CPU | 10-15% | 40-50% (during scan) |
| Memory | 500MB | 1.2GB (all models loaded) |
| Disk | 2GB | Grows ~100MB/month |
| Network | 5-10 Mbps | 50 Mbps (real-time feeds) |

---

## LLM Interpretation Service

### Purpose
AI-powered analysis for market insights using Claude Sonnet 4.6

### Implementation
```
Claude Sonnet 4.6 (via localhost:8317 proxy)
    ↓
Endpoints:
- POST /api/agents/analyze - Multi-agent discussion interpretation
- GET /api/market/smart-signals - Market-wide signal interpretation
- POST /api/news/analyze - News sentiment deep-dive
```

### Use Cases
1. **Agent Discussion Interpretation** - Summarize 6-agent reasoning
2. **Market Signal Interpretation** - Explain trading opportunity context
3. **News Sentiment Analysis** - Detailed news impact assessment

---

## Future Scalability

### Planned Enhancements (Phase 3-5)

1. **Live trading broker integration** (SSI Phase 4)
2. **Extended stock coverage** (300+ Phase 3-5)
3. **Additional ML models** (Ensemble Phase 3)
4. **Multi-timeframe analysis** (Phase 3)
5. **Mobile app integration** (Phase 5)
6. **Database optimization** (PostgreSQL Phase 3)
7. **Cloud deployment** (AWS/GCP Phase 3)

### Performance Optimization

- Prediction caching (reduce redundant calculations)
- Batch processing (vectorized operations)
- Connection pooling (database efficiency)
- Memory management (periodic cleanup)

---

*This detailed architecture provides complete understanding of VN-Quant's internal systems, data flows, and deployment options.*
