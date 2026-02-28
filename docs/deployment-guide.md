# VN-Quant Deployment Guide

**Version:** 1.0
**Last Updated:** 2026-01-12
**Environments:** Development, Docker, Production

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Development Setup](#development-setup)
3. [Docker Deployment](#docker-deployment)
4. [Configuration](#configuration)
5. [Troubleshooting](#troubleshooting)
6. [Monitoring](#monitoring)
7. [Scaling](#scaling)

---

## System Requirements

### Hardware Requirements

**Minimum:**
- CPU: 2-core processor (2 GHz+)
- RAM: 2GB
- Disk: 5GB (includes models)
- Network: 5 Mbps internet connection

**Recommended:**
- CPU: 4-core processor (2.5 GHz+)
- RAM: 4GB
- Disk: 10GB SSD
- Network: 10+ Mbps dedicated connection

**For High-Performance:**
- CPU: 8-core processor (3 GHz+)
- RAM: 8GB
- Disk: 20GB SSD
- Network: 50+ Mbps dedicated

### Software Requirements

**Python Environment:**
- Python 3.10+ (tested on 3.10, 3.11, 3.12)
- pip package manager
- Virtual environment support

**Operating System:**
- Windows 10+ (tested on Windows 10, 11)
- macOS 10.14+ (not extensively tested)
- Linux Ubuntu 20.04+ (recommended for production)

**Dependencies (auto-installed):**
- FastAPI 0.104.0+ (web framework)
- Pandas 2.0+ (data processing)
- NumPy 1.20+ (numerical computing)
- scikit-learn 1.3+ (ML utilities)
- TA 0.11+ (technical indicators)
- Uvicorn 0.24+ (ASGI server)

---

## Development Setup

### Quick Start (5 minutes)

**Step 1: Clone or extract repository**
```bash
cd D:\testpapertr  # Windows
# or
cd ~/testpapertr   # macOS/Linux
```

**Step 2: Create virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**Step 3: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Configure environment**
```bash
# Copy example config
cp .env.example .env

# Edit .env with your settings
# Minimum required:
TRADING_MODE=paper
INITIAL_CAPITAL=100000000
```

**Step 5: Run system**
```bash
python run_autonomous_paper_trading.py
```

**Step 6: Access dashboard**
Open browser: `http://localhost:8000/autonomous`

### Detailed Setup Guide

#### Step 1: Environment Preparation

**Windows:**
```bash
# Create dedicated directory
mkdir D:\vn-quant
cd D:\vn-quant

# Clone repository (or extract)
git clone <repo-url>
cd testpapertr

# Verify Python installation
python --version  # Should be 3.10+
pip --version
```

**macOS/Linux:**
```bash
# Create directory
mkdir ~/vn-quant
cd ~/vn-quant

# Clone repository
git clone <repo-url>
cd testpapertr

# Verify Python
python3 --version  # Should be 3.10+
```

#### Step 2: Virtual Environment

**Why:** Isolate project dependencies from system Python

**Windows:**
```bash
# Create venv
python -m venv venv

# Activate
venv\Scripts\activate

# Verify (should show (venv) prefix)
python --version
```

**macOS/Linux:**
```bash
# Create venv
python3 -m venv venv

# Activate
source venv/bin/activate

# Verify
python --version
```

#### Step 3: Dependency Installation

**Standard installation:**
```bash
pip install --upgrade pip

# Install from requirements.txt
pip install -r requirements.txt

# Verify installation
python -c "import pandas; import numpy; print('✓ Dependencies OK')"
```

**For CPU-only systems (recommended):**
```bash
# Torch with CPU support (faster install)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**Installation troubleshooting:**
```bash
# If installation fails, try:
pip install --upgrade pip setuptools wheel

# Install core first
pip install numpy pandas scikit-learn

# Then rest
pip install -r requirements.txt
```

#### Step 4: Environment Configuration

**Option A: Quick setup (use defaults)**
```bash
# Just run with defaults
python run_autonomous_paper_trading.py
```

**Option B: Custom configuration**
```bash
# Copy example
cp .env.example .env

# Edit with your editor
# Windows: notepad .env
# macOS/Linux: nano .env

# Key settings:
TRADING_MODE=paper              # NEVER paper for live
ALLOW_REAL_TRADING=false        # Multi-layer protection
INITIAL_CAPITAL=100000000       # 100M VND
MAX_POSITION_PCT=0.125          # 12.5% per stock
MODEL_SCAN_INTERVAL=180         # 3 minutes
API_PORT=8000                   # HTTP port
```

#### Step 5: Verify Installation

```bash
# Run quick test
python -c "
from quantum_stock.scanners.model_prediction_scanner import ModelPredictionScanner
from quantum_stock.autonomous.orchestrator import AutonomousOrchestrator
print('✓ Imports OK')
"

# Check models directory
python -c "
import os
model_count = len([f for f in os.listdir('models') if f.endswith('.pkl')])
print(f'✓ Found {model_count} models')
"

# Check data directory
python -c "
import os
if os.path.exists('data'):
    print('✓ Data directory OK')
else:
    print('✗ Data directory missing')
"
```

#### Step 6: Start System

```bash
# Run autonomous trading
python run_autonomous_paper_trading.py

# Expected output:
# 2026-01-12 10:30:00 | INFO | Starting AUTONOMOUS PAPER TRADING SYSTEM
# 2026-01-12 10:30:05 | INFO | ✅ System ready!
# 2026-01-12 10:30:05 | INFO | 📊 Open http://localhost:8000/autonomous

# Keep terminal running (don't close window)
```

### Post-Installation Checks

```bash
# 1. Database initialization
ls -la data/  # Should have SQLite db

# 2. Model availability
ls -la models/ | grep stockformer | wc -l  # Should show 102+

# 3. Log file
ls -la logs/autonomous_trading.log  # Should be created

# 4. Configuration
grep TRADING_MODE .env  # Should show paper
```

---

## Docker Deployment

### Overview

VN-Quant runs in Docker containers with:
- **PostgreSQL** for data storage (optional)
- **Redis** for caching (optional)
- **Autonomous Trading Service** (main trading system)
- **Backend API** (REST endpoints)
- **Model Training Service** (weekly ML training)

### Prerequisites

- **Docker Desktop** (Windows/macOS) or **Docker Engine** (Linux)
- **Docker Compose** 1.29+
- **4 GB RAM minimum** (8 GB recommended)
- **2 GB free disk space** (models + data)
- **Internet connection** (for data feeds)

### Installation

**Windows/macOS:**
```bash
# Download Docker Desktop from https://www.docker.com/products/docker-desktop
# Follow installation wizard
docker --version  # Verify
```

**Linux (Ubuntu/Debian):**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
docker --version
```

### Docker Quick Start (5 Minutes)

**Step 1: Navigate to project**
```bash
cd D:\testpapertr  # Windows
cd ~/testpapertr   # macOS/Linux
```

**Step 2: Configure environment**
```bash
# Copy example configuration
cp .env.example .env

# Edit .env with your settings
# Most defaults work fine for testing
```

**Step 3: Start all services**
```bash
# Build and start containers
docker-compose up -d

# Check status
docker-compose ps

# Expected output:
# NAME                    STATUS              PORTS
# vnquant-postgres        Up 30s (healthy)
# vnquant-redis           Up 30s (healthy)
# vnquant-autonomous      Up 25s
# vnquant-api             Up 20s
# vnquant-trainer         Up 15s (background service)
```

**Step 4: Verify services**
```bash
# Check logs
docker-compose logs autonomous -f  # View trading logs
docker-compose logs api -f         # View API logs
docker-compose logs trainer -f     # View training logs

# Test API
curl http://localhost:8003/api/status

# Expected: {"status": "ok", "uptime": 123}
```

**Step 5: Access services**
```
Trading Dashboard: http://localhost:8001/autonomous
API Documentation: http://localhost:8003/docs
```

**Step 6: Stop services**
```bash
# Stop all containers
docker-compose down

# Stop and remove volumes (WARNING: deletes data!)
docker-compose down -v
```

### Docker Configuration

**docker-compose.yml structure:**
```yaml
version: '3.8'

services:
  autonomous:
    build: .
    ports:
      - "8000:8001"          # Trading UI
      - "8003:8003"          # REST API
    volumes:
      - ./models:/app/models         # Pre-trained models
      - ./data:/app/data             # Historical data
      - ./logs:/app/logs             # Log files
    environment:
      - TRADING_MODE=paper
      - ALLOW_REAL_TRADING=false
      - INITIAL_CAPITAL=100000000
      - LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8003/api/status"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Optional: PostgreSQL for production
  postgres:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=vnquant
      - POSTGRES_USER=vnquant
      - POSTGRES_PASSWORD=secure_password_change_me
    restart: unless-stopped

  # Optional: Redis for caching
  redis:
    image: redis:7
    restart: unless-stopped

  # Optional: Model training service
  trainer:
    build:
      context: .
      dockerfile: Dockerfile.training
    environment:
      - TRAINING_SCHEDULE=0 2 * * 0  # 2 AM Sunday
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    restart: unless-stopped

volumes:
  postgres_data:
```

### Environment Variables (.env)

**Essential Settings:**
```bash
# Trading mode
TRADING_MODE=paper              # Use "paper" for testing
ENVIRONMENT=production          # development, staging, production

# Database (optional - defaults work)
POSTGRES_PASSWORD=your_secure_password
REDIS_PASSWORD=your_redis_password

# API Keys (required for some features)
GEMINI_API_KEY=sk-...           # Google Gemini for AI analysis
TELEGRAM_BOT_TOKEN=...          # For alerts (optional)

# Trading parameters
INITIAL_CAPITAL=100000000       # Starting cash (VND)
MAX_POSITION_PCT=0.15           # 15% max per stock
MAX_POSITIONS=10                # Max concurrent positions
STOP_LOSS_PCT=0.07              # 7% stop loss
TAKE_PROFIT_PCT=0.15            # 15% profit target
MAX_DAILY_LOSS_PCT=0.05         # 5% daily loss limit

# Model Training
TRAINING_SCHEDULE=0 2 * * 0     # 2 AM Sunday
TIMEZONE=Asia/Ho_Chi_Minh
ENABLE_NOTIFICATIONS=true       # Email/Slack alerts
SLACK_WEBHOOK_URL=...           # For notifications
```

### Service Details

#### Autonomous Trading Service

**Container:** `vnquant-autonomous`

**What it does:**
- Scans markets every 3 minutes
- Runs 6 agent consensus
- Places and manages trades
- Monitors positions
- Logs all activity

**Monitor:**
```bash
# View logs
docker logs vnquant-autonomous -f --tail=50

# Sample output:
[09:15:23] 🔭 Scout: Scanning 102 stocks...
[09:15:45] 📊 Alex: Technical signals for HPG: BUY
[09:16:00] 🐂 Bull: Growth detected in VCB
[09:16:15] ⚖️  Chief: Consensus reached - VOTE BUY
[09:16:20] ✅ Order executed: BUY 1000 shares HPG @ 45,200 VND
```

#### Backend API Service

**Container:** `vnquant-api`

**Endpoints:**
```bash
# Trading endpoints
GET  /api/trades          # Get trade history
POST /api/trades          # Create new trade
GET  /api/positions       # Get open positions
DELETE /api/positions/:id # Close position

# Agent endpoints
GET /api/agents/status    # Current agent status
GET /api/agents/signals   # Recent signals

# Portfolio endpoints
GET /api/portfolio/summary
GET /api/portfolio/performance
```

**API Documentation:**
```
http://localhost:8003/docs
```

#### Model Training Service

**Container:** `vnquant-trainer`

**What it does:**
- Trains Stockformer models weekly
- Validates predictions
- Deploys new models
- Sends notifications
- Logs results

**View training log:**
```bash
docker logs vnquant-trainer -f
```

### Volume Mapping

**Data Persistence:**

| Container Path | Host Path | Purpose |
|---|---|---|
| `/app/models` | `./models` | Trained ML models |
| `/app/data` | `./data` | Historical price data |
| `/app/logs` | `./logs` | Application logs |
| `/var/lib/postgresql/data` | `postgres_data` | Database |
| `/data` | `redis_data` | Redis cache |

**Backup Data:**
```bash
# Backup models and data
zip -r vnquant_backup.zip models/ data/ logs/

# Restore backup
unzip vnquant_backup.zip
```

### Port Mapping

| Service | Container Port | Host Port | Purpose |
|---|---|---|---|
| API Server | 8003 | 8003 | REST API & Docs |
| Trading Dashboard | 8000 | 8001 | Web UI |
| PostgreSQL | 5432 | 5435 | Database |
| Redis | 6379 | 6380 | Cache |

**Change Ports:**

Edit `docker-compose.yml`:
```yaml
services:
  autonomous:
    ports:
      - "9000:8001"  # Change from 8001 to 9000
```

### Troubleshooting

**Container Won't Start**

Symptoms: Container exits immediately

Solution:
```bash
# 1. Check logs
docker-compose logs autonomous

# 2. Check if ports are in use
lsof -i :8001          # Check port 8001
netstat -tlnp | grep 8003  # Check port 8003

# 3. Free port and restart
docker-compose restart autonomous
```

**Database Connection Error**

Symptoms: "Cannot connect to postgres:5432"

Solution:
```bash
# 1. Check if PostgreSQL is healthy
docker-compose ps | grep postgres

# 2. If not healthy, restart
docker-compose restart postgres

# 3. Wait for health check (30 seconds)
docker-compose logs postgres

# 4. Check credentials in .env
grep POSTGRES_PASSWORD .env
```

**Out of Memory**

Symptoms: Container keeps restarting, OOMKilled

Solution:
```bash
# 1. Check memory usage
docker stats

# 2. Increase Docker memory limit (Windows/macOS):
# Settings → Resources → Memory: 8GB (from 2GB)

# 3. Reduce model training batch size
# Edit Dockerfile.training:
ENV BATCH_SIZE=16  # from 32
```

**Volumes Not Persisting**

Symptoms: Data lost after restart

Solution:
```bash
# 1. Verify volume exists
docker volume ls | grep vn-quant

# 2. Check volume mapping
docker inspect vnquant-postgres -f '{{json .Mounts}}'

# 3. Ensure .env has correct volume paths
grep "POSTGRES_DATA" docker-compose.yml
```

**API Not Responding**

Symptoms: "Connection refused" when accessing http://localhost:8003

Solution:
```bash
# 1. Check if container is running
docker-compose ps | grep api

# 2. Check if listening on port
docker exec vnquant-api netstat -tlnp | grep 8003

# 3. Check API logs for errors
docker-compose logs api

# 4. Restart API
docker-compose restart api
```

### Networking

**Internal Communication**

Services communicate via `vn-quant-network`:

```
autonomous ←→ postgres (DB)
autonomous ←→ redis (cache)
api ←→ postgres
api ←→ redis
trainer ←→ postgres
```

**External Communication**

Services connect to external APIs:

```
autonomous → CafeF API (market data)
autonomous → Finnhub API (news)
api → <client browser>
trainer → Email SMTP server (optional)
```

### Production Deployment

**Security Checklist:**

- [ ] Change all default passwords in `.env`
- [ ] Set `ALLOW_REAL_TRADING=false` (until tested)
- [ ] Enable `ENVIRONMENT=production`
- [ ] Use strong `JWT_SECRET`
- [ ] Configure HTTPS (reverse proxy with nginx)
- [ ] Set up firewall rules
- [ ] Enable audit logging
- [ ] Regular backups enabled

**Scaling:**

**Multiple Instances:**
```bash
# Run multiple autonomous traders
docker run -d --name vnquant-trader1 vnquant:latest
docker run -d --name vnquant-trader2 vnquant:latest

# Use load balancer (nginx)
# Each trader operates independently on different stock clusters
```

**Performance Tuning:**
```yaml
# In docker-compose.yml
services:
  autonomous:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### Backup & Disaster Recovery

```bash
# Daily backup script
#!/bin/bash
BACKUP_DIR="/backups/vnquant"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup volumes
docker run --rm \
  -v vnquant_postgres_data:/source \
  -v $BACKUP_DIR:/backup \
  ubuntu tar czf /backup/postgres_$DATE.tar.gz -C /source .

# Backup .env and configs
cp .env $BACKUP_DIR/env_$DATE.bak
cp docker-compose.yml $BACKUP_DIR/docker-compose_$DATE.yml

echo "Backup completed: $BACKUP_DIR"
```

---

## Configuration

### Environment Variables

**File:** `.env` (copy from `.env.example`)

**Trading Configuration:**
```bash
# Primary setting
TRADING_MODE=paper          # paper | live

# Capital & Risk (VND)
INITIAL_CAPITAL=100000000   # Starting balance
MAX_POSITION_PCT=0.125      # 12.5% per stock max
MAX_POSITIONS=10            # Max 10 holdings
STOP_LOSS_PCT=0.05          # 5% stop loss
TAKE_PROFIT_PCT=0.15        # 15% take profit

# Thresholds
MODEL_CONFIDENCE_THRESHOLD=0.7   # Min confidence for signal
MODEL_RETURN_THRESHOLD=0.03      # Min 3% expected return
```

**Scanning Configuration:**
```bash
# Model scanner (Path A)
MODEL_SCAN_INTERVAL=180              # Every 3 minutes
MODEL_RETURN_THRESHOLD=0.03          # 3% min return
MODEL_CONFIDENCE_THRESHOLD=0.7       # 70% min confidence

# News scanner (Path B)
NEWS_SCAN_INTERVAL=300               # Every 5 minutes
NEWS_SOURCES=cafef                   # CafeF RSS feeds
```

**API Configuration:**
```bash
API_HOST=0.0.0.0            # Listen on all interfaces
API_PORT=8000               # HTTP port
API_WORKERS=4               # Worker threads
CORS_ORIGINS=*              # CORS policy
```

**Security Configuration:**
```bash
# Critical: Paper trading protection
ALLOW_REAL_TRADING=false    # MUST be false for paper
ENVIRONMENT=development     # development | production
DEBUG=true                  # debug mode
LOG_LEVEL=INFO              # DEBUG | INFO | WARNING | ERROR
```

**Data Configuration:**
```bash
DATA_DIR=data                          # Data directory
HISTORICAL_DATA_DIR=data/historical   # Historical cache
CACHE_TTL=300                          # Cache 5 minutes
TIMEZONE=Asia/Ho_Chi_Minh             # VN timezone
```

**Broker Configuration (for future live trading):**
```bash
# Will be enabled when switching to live trading
SSI_USERNAME=your_username
SSI_PASSWORD=your_password
SSI_API_KEY=your_api_key
```

### Advanced Configuration

**config_manager.py (Python):**
```python
# Can also configure in code
from quantum_stock.core.config_manager import ConfigManager

config = ConfigManager()
config.set('trading.mode', 'paper')
config.set('risk.max_position_pct', 0.125)
config.set('scanning.model_interval', 180)

# Get configurations
mode = config.get('trading.mode')
```

**Per-stock overrides (planned):**
```python
# Override specific stocks
stock_config = {
    'ACB': {
        'max_position_pct': 0.15,  # 15% for this stock
        'stop_loss_pct': 0.03      # Tighter stop for ACB
    },
    'HPG': {
        'max_position_pct': 0.10   # 10% for HPG
    }
}
```

---

## Troubleshooting

### Common Issues

#### Issue: "Module not found" error

**Symptom:**
```
ModuleNotFoundError: No module named 'quantum_stock'
```

**Solution:**
```bash
# Ensure virtual environment is activated
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt

# Verify PYTHONPATH
python -c "import sys; print(sys.path)"
```

#### Issue: Port 8000 already in use

**Symptom:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Option 1: Find and kill process using port
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux
lsof -i :8000
kill -9 <PID>

# Option 2: Use different port
export API_PORT=8001
python run_autonomous_paper_trading.py

# Option 3: Update .env
echo "API_PORT=8001" >> .env
```

#### Issue: Models not loading

**Symptom:**
```
FileNotFoundError: models/stockformer_*.pkl not found
```

**Solution:**
```bash
# Check models directory
ls -la models/

# Expected: 102+ .pkl files

# If missing:
# 1. Verify download completed
# 2. Check disk space: df -h models/
# 3. Verify file permissions: chmod 644 models/*.pkl
```

#### Issue: Out of memory

**Symptom:**
```
MemoryError: Unable to allocate ...
```

**Solution:**
```bash
# Check system memory
# Windows: Task Manager
# macOS: Activity Monitor
# Linux: free -h

# Reduce model batch size
export MODEL_BATCH_SIZE=10  # Default 100

# Increase system swap (Linux)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### Issue: Dashboard not accessible

**Symptom:**
```
refused to connect / connection timeout
```

**Solution:**
```bash
# Check if server is running
curl http://localhost:8000/api/status

# Check logs
tail -f logs/autonomous_trading.log

# Verify firewall not blocking
# Windows Firewall: Check Python exceptions
# macOS: System Preferences → Security & Privacy

# Try different port
python run_autonomous_paper_trading.py --port 8001
```

#### Issue: Database errors

**Symptom:**
```
sqlite3.OperationalError: database is locked
```

**Solution:**
```bash
# Check if process already running
ps aux | grep python | grep autonomous

# Kill other instances
pkill -f autonomous

# Check database file
ls -lh data/*.db

# Repair database (if corrupted)
python -c "
import sqlite3
conn = sqlite3.connect('data/paper_trading.db')
conn.integrity_check()
conn.close()
print('✓ Database OK')
"
```

---

## Monitoring

### Log Files

**Location:** `logs/autonomous_trading.log`

**Log Levels:**
- DEBUG: Detailed diagnostic info (verbose)
- INFO: Operational confirmations
- WARNING: Suspicious but continuing
- ERROR: Function failure, can recover
- CRITICAL: System failure

**Monitoring log in real-time:**
```bash
# Windows
type logs\autonomous_trading.log

# macOS/Linux
tail -f logs/autonomous_trading.log

# Follow specific pattern
grep -i "order executed" logs/autonomous_trading.log
```

### Key Metrics to Monitor

```bash
# Check system startup
grep "STARTING AUTONOMOUS" logs/autonomous_trading.log

# Count successful scans
grep -c "Model scan completed" logs/autonomous_trading.log

# Count orders executed
grep -c "Order executed" logs/autonomous_trading.log

# Check for errors
grep "ERROR\|CRITICAL" logs/autonomous_trading.log

# Monitor specific stock
grep "ACB" logs/autonomous_trading.log
```

### Health Check API

```bash
# API status endpoint
curl http://localhost:8000/api/status

# Response:
{
  "status": "running",
  "uptime_seconds": 3600,
  "portfolio_value": 100542300,
  "total_trades_today": 5,
  "positions_open": 3,
  "last_scan": "2026-01-12T10:30:45"
}
```

### System Monitoring Dashboard

**Available at:** `http://localhost:8000/autonomous`

**Metrics displayed:**
- System status (Running/Stopped)
- Portfolio value (VND)
- Total P&L (VND & %)
- Open positions count
- Today's trades count
- Agent conversations (real-time stream)
- Current holdings table
- Order history

---

---

## Production Checklist

Before deploying to production:

- [ ] All tests passing: `pytest tests/`
- [ ] Configuration reviewed: `.env` settings verified
- [ ] Database optimized: Indexes created, old data cleaned
- [ ] Backups configured: Automated backup schedule
- [ ] Monitoring active: Log aggregation, alerts set
- [ ] Security hardened: Credentials rotated, firewall rules
- [ ] Team trained: Operations team familiar with system
- [ ] Dry run completed: Full trading cycle tested
- [ ] Rollback plan ready: Recovery procedures documented

---

*This deployment guide covers development, Docker, and production deployments. For detailed configuration, monitoring, and scaling strategies, see system-architecture.md.*
