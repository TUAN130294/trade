# 🔴 LIVE MONITORING - START HERE

**Real-Time Agent Signals & Trade Tracking**

---

## ⚡ Quick Start (30 seconds)

### Watch Live Trading System
```bash
docker-compose logs -f autonomous
```

**What you'll see:**
- 📰 News alerts being detected
- 🎯 Model opportunities found
- 🗣️ Agent discussions happening
- 📊 Trades being executed
- 📍 Positions being managed

---

## 📊 Current System Status

```
Status: ✅ RUNNING
Mode: 📄 PAPER TRADING (safe)
Capital: 500,000,000 VND
Opportunities: 42 detected
Discussions: 0 active
Trades: 0 executed (waiting for signals)
```

---

## 🎯 What to Watch For

### 🟢 Active Signals
```
📰 NEWS ALERT: Stock [HIGH]
   Sentiment: 0.75 (High confidence)
   → Agent discussion starting...
```

### 🎭 Agent Discussion Sequence
```
🔭 Scout:   "Phát hiện cơ hội MWG (+5%)"
📊 Alex:    "Technical analysis: Confirm BUY"
🐂 Bull:    "Momentum support - YES"
🐻 Bear:    "Risk level acceptable"
🏥 Risk:    "Position size: 1,000 shares"
⚖️  Chief:   "VERDICT: BUY ✅"
```

### ✅ Order Execution
```
[ORDER EXECUTED]
Symbol: MWG
Price: 86,000 VND
Volume: 1,000 shares
Value: 86M VND
Entry: 86,000 | Stop: 81,700 | Target: 98,900
```

---

## 📺 Monitoring Methods

### Method 1: Simple Docker Logs (Recommended for beginners)
```bash
docker-compose logs -f autonomous
```
✅ Shows everything in real-time
✅ Easy to understand
✅ No dependencies

---

### Method 2: Filter by Agent (Watch specific agents)
```bash
# Watch Scout (Market Scanner)
docker-compose logs -f autonomous | grep -i scout

# Watch Alex (Technical Analyst)
docker-compose logs -f autonomous | grep -i alex

# Watch Chief (Final Decision)
docker-compose logs -f autonomous | grep -i chief

# Watch Orders
docker-compose logs -f autonomous | grep -i "order\|executed"
```

---

### Method 3: Windows Batch Script
```bash
monitor_live.bat
```
✅ Auto-refresh dashboard
✅ Color-coded output
✅ Windows-native

---

### Method 4: Python Dashboard (Best UI)
```bash
python monitor_live.py
```
✅ Cleanest interface
✅ Real-time updates
✅ Color-coded signals
⚠️ Requires Python (not available in WSL)

---

### Method 5: Shell Script (Linux/macOS)
```bash
bash monitor_live.sh
```
✅ Full automation
✅ Advanced filtering
✅ Unix-optimized

---

## 📱 Web Dashboard

### Check Status in Browser
```
http://localhost:5176
```
Shows:
- Portfolio overview
- Active positions
- Order history
- Agent status
- Real-time updates

---

## 📊 API Monitoring

### Check Trading Status
```bash
curl http://localhost:5176/api/status
```

**Response:**
```json
{
  "is_running": true,
  "paper_trading": true,
  "balance": 500000000,
  "active_positions": 0,
  "statistics": {
    "opportunities_detected": 42,
    "agent_discussions": 0,
    "orders_executed": 0
  }
}
```

### Get Positions
```bash
curl http://localhost:5176/api/positions
```

### Get Order History
```bash
curl http://localhost:5176/api/orders
```

---

## 🔍 Real-Time Signal Flow

### What Happens Every 3 Minutes

```
1️⃣  MODEL SCAN
    Stockformer checks 102 stocks
    ↓
2️⃣  OPPORTUNITY DETECTION
    Scout finds candidates: MWG, HPG, ACB...
    ↓
3️⃣  AGENT ANALYSIS
    Alex → Bull → Bear → Risk Doctor
    ↓
4️⃣  CONSENSUS
    Chief aggregates votes
    ↓
5️⃣  DECISION
    If confidence > 70%: EXECUTE
    ↓
6️⃣  ORDER PLACEMENT
    [ORDER EXECUTED]
    ↓
7️⃣  POSITION MONITORING
    Real-time P&L tracking
    ↓
8️⃣  EXIT TRIGGER
    Profit Target | Stop Loss | T+2
```

### What Happens 24/7 (News Path)

```
1️⃣  NEWS SCANNING
    RSS feeds from VietStock, CafeF, etc.
    ↓
2️⃣  SENTIMENT ANALYSIS
    Calculate positive/negative score
    ↓
3️⃣  CRITICAL NEWS DETECTION
    High sentiment → trigger analysis
    ↓
4️⃣  SAME AGENT DISCUSSION
    Full multi-agent consensus
    ↓
5️⃣  TRADE IF CONSENSUS HIGH
```

---

## 🚨 Key Log Patterns

### Scout (Market Scanner)
```
🔭 Scout: "Phát hiện cơ hội MWG (+5%)"
```

### Alex (Technical Analyst)
```
📊 Alex: "Support/Resistance: Price at critical level"
```

### Bull (Growth Hunter)
```
🐂 Bull: "Bullish breakout with volume confirmation"
```

### Bear (Risk Sentinel)
```
🐻 Bear: "Risk alert: Bull trap pattern detected"
```

### Chief (Final Decision)
```
⚖️  Chief: "VERDICT: BUY (Confidence: 85%)"
```

### Orders
```
✅ [ORDER EXECUTED] MWG | 86,000 VND | 1,000 shares
```

### Positions
```
📍 [POSITION] Entry: 86,000 | Stop: 81,700 | Target: 98,900
```

### Exits
```
📈 [POSITION EXIT] PROFIT +2,000 VND (P&L: +2M)
📉 [POSITION EXIT] LOSS -5,000 VND (P&L: -5M)
```

---

## ⚙️ Common Commands

### Monitor Everything
```bash
docker-compose logs -f autonomous
```

### Count Opportunities Found
```bash
docker-compose logs autonomous | grep "OPPORTUNITY" | wc -l
```

### Count Trades Executed
```bash
docker-compose logs autonomous | grep "ORDER EXECUTED" | wc -l
```

### Watch Only Trades
```bash
docker-compose logs -f autonomous | grep -i "order\|position\|verdict"
```

### Watch Only News
```bash
docker-compose logs -f autonomous | grep -i "news\|alert\|sentiment"
```

### See Last 100 Lines
```bash
docker-compose logs --tail=100 autonomous
```

### Follow New Logs (Live)
```bash
docker-compose logs -f autonomous
```

---

## 📈 Healthy System Indicators

✅ **Good Signs:**
- Scout finding opportunities
- Model scans running every 3 minutes
- News alerts being detected
- Agent discussions happening
- Orders executing successfully
- Positions exiting with profit

⚠️ **Warning Signs:**
- No signals for 30+ minutes
- Repeated error messages
- Same news alert repeating
- Database connection errors
- Zero opportunities detected

---

## 🎯 Example Monitoring Session

### Terminal 1: Watch All Logs
```bash
$ docker-compose logs -f autonomous
```

**Output:**
```
INFO: Model prediction scan starting...
INFO: Found 102 models
INFO: Scanning PASSED stocks...
INFO: 🎯 OPPORTUNITY: MWG | Return: +5.2% | Confidence: 0.82
INFO: Agent analysis starting for MWG...
🔭 Scout: "Phát hiện cơ hội MWG (+5.2%)"
📊 Alex: "Technical analysis: Support bounce confirmed"
🐂 Bull: "Breakout pattern with volume"
🐻 Bear: "Risk assessment: Low"
🏥 Risk Doctor: "Position size: 1,000 shares"
⚖️  Chief: "VERDICT: BUY (Confidence: 85%)"
✅ [ORDER EXECUTED]
   Symbol: MWG
   Price: 86,000 VND
   Volume: 1,000 shares
   Total: 86,000,000 VND
```

### Terminal 2: Watch API Status
```bash
$ while true; do curl -s http://localhost:5176/api/status | head -c 150; echo ""; sleep 5; done
```

**Output:**
```
{"is_running":true,"paper_trading":true,...,"orders_executed":1}
{"is_running":true,"paper_trading":true,...,"orders_executed":1}
{"is_running":true,"paper_trading":true,...,"orders_executed":2}
...
```

### Terminal 3: Watch Service Health
```bash
$ watch -n 5 'docker-compose ps'
```

**Output:**
```
CONTAINER             STATUS
vnquant-autonomous    Up 5 minutes
vnquant-frontend      Up 5 minutes (healthy)
vnquant-postgres      Up 5 minutes (healthy)
vnquant-redis         Up 5 minutes (healthy)
vnquant-trainer       Up 5 minutes (healthy)
```

---

## 🎬 Start Monitoring Now

### Quick Start (Copy & Paste)
```bash
# Watch the autonomous trading system
docker-compose logs -f autonomous

# In another terminal, check API status
curl http://localhost:5176/api/status

# In browser, visit dashboard
http://localhost:5176
```

---

## 📚 More Information

For detailed monitoring guide:
```
MONITORING_GUIDE.md
```

For system architecture:
```
FINAL_STATUS.md
```

For deployment details:
```
DEPLOYMENT_SUMMARY.md
```

---

## ✅ System Ready

**Everything is running and ready to monitor.**

Start with:
```bash
docker-compose logs -f autonomous
```

Then open dashboard:
```
http://localhost:5176
```

**Happy Trading! 🚀**

---

*Last Updated: 2026-01-12 14:23 UTC+7*
*System Status: 🟢 OPERATING NORMALLY*
