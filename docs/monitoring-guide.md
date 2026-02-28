# VN-QUANT Live Monitoring Guide

**Real-time Agent Signals & Trade Monitoring**

---

## 1. Watch Agent Discussions in Real-Time

### Option A: Docker Logs (Recommended)

```bash
# Watch autonomous trading logs (agent discussions + trades)
docker-compose logs -f autonomous
```

**What to look for:**
```
✅ Agent Analysis Starting
   Scout  🔭 Market Scanner: [signal type]
   Alex   📊 Technical Analysis: [signal]
   Bull   🐂 Growth Hunter: [signal]
   Bear   🐻 Risk Sentinel: [signal]
   Chief  ⚖️  Final Decision: [verdict]

✅ Order Execution
   [ORDER EXECUTED] Symbol | Price | Size | P&L

✅ Position Management
   [POSITION EXIT] T+2 compliance | Trailing stop | Take profit
```

### Option B: Real-time Monitoring Dashboard

```bash
# Terminal-based monitoring (watch every 2 seconds)
watch -n 2 'docker-compose logs autonomous | tail -50'
```

---

## 2. Monitor Trading Statistics

### API Endpoint - Live Status

```bash
# Get current trading status
curl http://localhost:5176/api/status
```

**Response includes:**
```json
{
  "is_running": true,
  "paper_trading": true,
  "active_positions": 0,
  "balance": 500000000,
  "statistics": {
    "opportunities_detected": 84,
    "agent_discussions": 0,
    "orders_executed": 0,
    "positions_exited": 0
  }
}
```

### Monitor Portfolio

```bash
# Check current positions
curl http://localhost:5176/api/positions

# Check order history
curl http://localhost:5176/api/orders

# Check P&L
curl http://localhost:5176/api/portfolio
```

---

## 3. Real-Time Agent Signal Flow

### Agent Signal Sources

**🔭 Scout (Market Scanner)**
- Detects opportunities from models
- Real-time news alerts
- Technical opportunities

**📊 Alex (Technical Analyst)**
- Support/Resistance analysis
- Entry point detection
- Technical indicators (RSI, MACD, etc.)

**🐂 Bull (Growth Hunter)**
- Identifies bullish setups
- Breakout detection
- Momentum analysis

**🐻 Bear (Risk Sentinel)**
- Detects risk signals
- Bull trap warnings
- Reversal patterns

**🏥 Risk Doctor (Position Sizer)**
- Calculates position size
- Risk/reward validation
- Portfolio limits check

**⚖️ Chief (Final Decision)**
- Aggregates all signals
- Makes final verdict
- Weighted consensus

---

## 4. Monitor Agent Signals via Dashboard

### Radar Display
```
http://localhost:5176
→ Look for "Radar" or "Agents" section
→ Shows each agent's status
→ Displays latest signals
→ Accuracy metrics
```

### Agent Status Endpoint
```bash
# Get all agents' current status
curl http://localhost:5176/api/agents/status | head -100
```

---

## 5. Track Model Predictions

### Monitor Model Scanning

```bash
# Watch model prediction scanner logs
docker-compose logs -f autonomous | grep -i "model\|prediction\|scan"
```

**What to watch for:**
```
INFO: Model prediction scan starting
INFO: Found 102 models
INFO: Scanning PASSED stocks...
INFO: 🎯 OPPORTUNITY: MWG | Return: +5.2% | Confidence: 0.82
```

---

## 6. Monitor News Analysis

### Watch News Sentiment Analysis

```bash
# Real-time news alerts
docker-compose logs -f autonomous | grep -i "news\|alert\|sentiment"
```

**Example output:**
```
INFO: 📰 NEWS ALERT: MWG [HIGH]
   Headline: Cổ phiếu tăng giá...
   Sentiment: 0.75 (HIGH)
   Confidence: 0.82
   Action: BUY
```

---

## 7. Monitor Trade Execution

### Watch Live Order Execution

```bash
# Watch for order execution
docker-compose logs -f autonomous | grep -i "order\|executed\|position"
```

**Example output:**
```
INFO: [ORDER EXECUTED]
  Symbol: MWG
  Type: BUY
  Price: 86,000 VND
  Volume: 100 shares
  Value: 8,600,000 VND
  Slippage: 0.15%

INFO: [POSITION CREATED]
  Entry: 86,000
  Stop Loss: 81,700
  Take Profit: 98,900
  Risk/Reward: 1:2.5
```

---

## 8. Full Monitoring Stack

### Terminal 1: Watch Trading Logs
```bash
docker-compose logs -f autonomous | grep -E "Scout|Alex|Bull|Bear|Chief|ORDER|POSITION"
```

### Terminal 2: Watch Database Activity
```bash
docker-compose logs -f postgres | grep -i "query\|transaction"
```

### Terminal 3: Monitor Service Health
```bash
while true; do
  echo "=== Service Status ==="
  docker-compose ps --format "table {{.Names}}\t{{.Status}}"
  echo ""
  sleep 5
done
```

### Terminal 4: Watch API Metrics
```bash
while true; do
  echo "=== API Status ==="
  curl -s http://localhost:5176/api/status | head -c 200
  echo ""
  sleep 5
done
```

---

## 9. Key Log Patterns to Watch

### Agent Signals
```
🔭 Scout:      "Phát hiện cơ hội"
📊 Alex:       "Technical analysis"
🐂 Bull:       "Bullish setup"
🐻 Bear:       "Risk detected"
🏥 RiskDoc:    "Position size"
⚖️  Chief:      "VERDICT:"
```

### Trade Events
```
[OPPORTUNITY]       → New signal detected
[DISCUSSION]        → Agent analysis happening
[ORDER EXECUTED]    → Trade placed
[POSITION CREATED]  → Position opened
[POSITION EXIT]     → Position closed
[P&L]               → Profit/Loss calculated
```

### Risk Events
```
[RISK ALERT]        → Risk threshold exceeded
[BULL TRAP]         → False signal detected
[LIMIT CHECK]       → Position limit reached
[COMPLIANCE]        → T+2 settlement issue
```

---

## 10. Quick Monitoring Commands

### All-in-One Monitoring
```bash
# Watch autonomous trading with agent signals
watch -n 2 'docker-compose logs --tail=50 autonomous'
```

### Filter by Agent
```bash
# Watch only Scout signals
docker-compose logs -f autonomous | grep -i scout

# Watch only Chief decisions
docker-compose logs -f autonomous | grep -i chief

# Watch only orders
docker-compose logs -f autonomous | grep -i order
```

### Monitor Performance
```bash
# Track opportunities found
docker-compose logs -f autonomous | grep "OPPORTUNITY" | wc -l

# Track trades executed
docker-compose logs -f autonomous | grep "ORDER EXECUTED" | wc -l

# Track wins
docker-compose logs -f autonomous | grep "POSITION EXIT.*+" | wc -l

# Track losses
docker-compose logs -f autonomous | grep "POSITION EXIT.*-" | wc -l
```

---

## 11. Expected Signal Flow

```
Market Open (9:15 AM)
    ↓
Model Scan (every 3 minutes)
    ↓
Scout detects opportunities
    ↓
Alex performs technical analysis
    ↓
Bull/Bear assess sentiment
    ↓
Risk Doctor calculates position size
    ↓
Chief makes final verdict
    ↓
Order executed (if consensus high enough)
    ↓
Position monitored
    ↓
Exit triggered (profit/stop-loss/T+2)
    ↓
P&L recorded
```

---

## 12. Dashboard Access

### Main Dashboard
```
http://localhost:5176

Should show:
- Portfolio stats
- Active positions
- Order history
- Agent status
- Real-time chart
```

### Direct Trading System
```
http://localhost:8001/autonomous

Shows:
- Live agent discussions
- Current positions
- Order history
- Agent conversations in real-time
```

---

## 13. Common Signals to Monitor

### Bullish Signals 🟢
```
Scout:      "Phát hiện cơ hội +5.2%"
Alex:       "Support bounce confirmed"
Bull:       "Breakout with volume"
Chief:      "VERDICT: BUY"
```

### Bearish Signals 🔴
```
Scout:      "Risk detected -3%"
Bear:       "Reversal pattern formed"
RiskDoctor: "Risk limit exceeded"
Chief:      "VERDICT: SELL"
```

### Hold Signals 🟡
```
Chief:      "VERDICT: HOLD"
Alex:       "No clear signal"
Bear:       "Risk/Reward unfavorable"
```

---

## 14. Troubleshooting Monitoring

### No Signals Appearing?
```bash
# Check if autonomous service is running
docker-compose ps autonomous

# Check for errors
docker-compose logs autonomous --tail=100
```

### No Opportunities Detected?
```bash
# Model prediction might need market movement
# Check if models are loaded
docker-compose logs autonomous | grep -i "model"

# Check news scanner
docker-compose logs autonomous | grep -i "news"
```

### Dashboard Not Updating?
```bash
# Check WebSocket connection
docker-compose logs autonomous | grep -i "websocket"

# Check frontend logs
docker-compose logs frontend
```

---

## 15. Set Up Alerts (Optional)

### Email Alerts on Trade
```bash
# Add to monitoring script
if docker-compose logs autonomous --tail=10 | grep -q "ORDER EXECUTED"; then
  mail -s "VN-QUANT Trade Executed" your-email@example.com
fi
```

### Telegram Alerts (if configured in .env)
```
Already configured in system
Requires: TELEGRAM_BOT_TOKEN + TELEGRAM_ADMIN_IDS in .env
```

---

## Quick Start - Choose Your Monitoring Method

### 🚀 Option 1: Python Dashboard (Recommended - Best UI)
```bash
python monitor_live.py
```
✅ Cleanest interface | Real-time updates | Color-coded output

### 🪟 Option 2: Windows Batch Script
```bash
monitor_live.bat
```
✅ Native Windows | No dependencies | Auto-refresh

### 🐧 Option 3: Linux/macOS Shell Script
```bash
bash monitor_live.sh
```
✅ Full colors | Terminal-optimized | Advanced filtering

### 📺 Option 4: Simple Docker Logs (Quick & Dirty)
```bash
docker-compose logs -f autonomous
```
✅ Direct output | No filtering | Real-time streaming

---

## Real-Time Monitoring Examples

### Watch Everything
```bash
python monitor_live.py
```

### Filter by Agent
```bash
docker-compose logs -f autonomous | grep -i "scout"
docker-compose logs -f autonomous | grep -i "chief"
docker-compose logs -f autonomous | grep -i "bear"
```

### Watch Only Trades
```bash
docker-compose logs -f autonomous | grep -i "order\|executed\|position"
```

### Count Opportunities
```bash
docker-compose logs autonomous | grep "OPPORTUNITY" | wc -l
```

### Count Trades
```bash
docker-compose logs autonomous | grep "ORDER EXECUTED" | wc -l
```

---

## Dashboard Quick Reference

### In Your Browser
```
http://localhost:5176
```
- Main dashboard
- Real-time charts
- Portfolio stats
- Agent status

### Via API
```bash
# Check status
curl http://localhost:5176/api/status

# Get positions
curl http://localhost:5176/api/positions

# Get orders
curl http://localhost:5176/api/orders
```

---

## What to Watch For

### 🟢 Good Signs
- Scout finding opportunities
- Alex confirming with technical analysis
- Bull/Bear agreeing on direction
- Chief making final verdict BUY/SELL
- Orders executing successfully
- Positions exiting with profit (+)

### 🔴 Warning Signs
- No signals for 30+ minutes
- Agent disagreement (conflicting signals)
- Bear constantly warning of risk
- Orders failing to execute
- Positions exiting with loss (-)
- Database/Redis disconnection errors

### 🟡 Normal Operations
- Model predictions every 3 minutes
- News alerts flowing through
- Agent discussions starting/ending
- Position monitoring active

---

## System Status: ✅ Ready for monitoring

**Start with:** `python monitor_live.py`

**Or quick check:** `docker-compose logs -f autonomous`
