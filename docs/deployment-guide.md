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

### Docker Quick Start

**Prerequisites:**
- Docker Engine 20.10+
- docker-compose 1.29+

**One-command deployment:**
```bash
# Navigate to project
cd D:\testpapertr  # Windows
cd ~/testpapertr   # macOS/Linux

# Deploy all services
docker-compose up -d

# Check status
docker-compose ps

# Expected output:
# NAME                STATUS              PORTS
# vn-quant-api        Up 2 minutes        0.0.0.0:8000->8000/tcp

# View logs
docker-compose logs -f vn-quant-api
```

### Docker Configuration

**docker-compose.yml structure:**
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"          # FastAPI
      - "8003:8003"          # Backend API (if separate)
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
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/status"]
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

volumes:
  postgres_data:
```

### Docker Deployment Steps

**Step 1: Build image**
```bash
# Build from Dockerfile
docker build -t vn-quant:latest .

# Expected output:
# Successfully tagged vn-quant:latest
```

**Step 2: Run container**
```bash
# Run single container
docker run -d \
  --name vn-quant \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -e TRADING_MODE=paper \
  vn-quant:latest

# Or use docker-compose (recommended)
docker-compose up -d
```

**Step 3: Verify deployment**
```bash
# Check container status
docker ps | grep vn-quant

# Check logs
docker logs -f vn-quant_vn-quant-api_1

# Test API
curl http://localhost:8000/api/status

# Access dashboard
# Open: http://localhost:8000/autonomous
```

**Step 4: Manage container**
```bash
# Stop container
docker-compose stop

# Restart container
docker-compose restart

# Remove container
docker-compose down

# View real-time logs
docker-compose logs -f --tail=100
```

### Docker Troubleshooting

**Port already in use:**
```bash
# Check what's using port 8000
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Use different port in docker-compose.yml
ports:
  - "8001:8000"  # External:Internal
```

**Out of memory:**
```bash
# Increase Docker memory limit
# Docker Desktop: Settings → Resources → Memory: 4GB+

# Or set resource limits in docker-compose
services:
  api:
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
```

**Models not loading:**
```bash
# Verify volume mount
docker exec vn-quant ls -la /app/models/

# If not found, check local models directory
ls -la models/

# Ensure Dockerfile copies models correctly
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

## Scaling

### Performance Optimization

**For faster predictions:**
```bash
# Increase worker threads
export API_WORKERS=8

# Batch predict more efficiently
export MODEL_BATCH_SIZE=50  # Default 10

# Cache more data
export CACHE_TTL=600  # 10 minutes (default 5)
```

**For lower memory usage:**
```bash
# Load models on demand
export LAZY_LOAD_MODELS=true

# Reduce position history
export MAX_HISTORY_DAYS=30  # Keep 30 days only
```

### Multi-Instance Deployment

**Planned: Load balancer (Nginx)**
```nginx
upstream vn_quant {
    server localhost:8000;
    server localhost:8001;
    server localhost:8002;
}

server {
    listen 80;
    server_name trading.example.com;

    location / {
        proxy_pass http://vn_quant;
    }
}
```

### Database Optimization (PostgreSQL)

**For production with 1M+ records:**
```bash
# Migrate from SQLite to PostgreSQL
DATABASE_URL=postgresql://user:pass@localhost/vnquant

# Index key queries
CREATE INDEX idx_orders_symbol ON orders(symbol);
CREATE INDEX idx_positions_active ON positions(status);
CREATE INDEX idx_conversations_timestamp ON agent_conversations(timestamp);

# Connection pooling
DATABASE_POOL_SIZE=20
```

---

## Maintenance

### Regular Maintenance Tasks

**Daily:**
- Check logs for errors: `grep ERROR logs/autonomous_trading.log`
- Monitor portfolio: http://localhost:8000/autonomous
- Verify broker connection (if live)

**Weekly:**
- Backup database: `cp data/paper_trading.db data/backup_$(date).db`
- Review P&L report
- Check disk usage: `du -sh .`
- Restart service for memory cleanup

**Monthly:**
- Update dependencies: `pip install --upgrade -r requirements.txt`
- Retrain ML models (future)
- Analyze trading performance
- Optimize configuration parameters

**Quarterly:**
- Full security audit
- Performance profiling
- Disaster recovery drill

### Backup & Recovery

**Backup command:**
```bash
# Backup all critical data
tar -czf vn-quant-backup-$(date +%Y%m%d).tar.gz \
  data/ logs/ models/ .env requirements.txt

# Restore from backup
tar -xzf vn-quant-backup-20260112.tar.gz
```

**Database backup:**
```bash
# SQLite
cp data/paper_trading.db data/backup-$(date +%s).db

# PostgreSQL (if used)
pg_dump vnquant > backup-$(date +%Y%m%d).sql
```

---

## Production Checklist

Before deploying to production:

- [ ] All tests passing: `pytest tests/`
- [ ] Configuration reviewed: `.env` settings verified
- [ ] Database optimized: Indexes created, old data cleaned
- [ ] Backups configured: Automated backup schedule
- [ ] Monitoring active: Log aggregation, alerts set
- [ ] Security hardened: Credentials rotated, firewall rules
- [ ] Documentation updated: Deployment guide current
- [ ] Team trained: Operations team familiar with system
- [ ] Dry run completed: Full trading cycle tested
- [ ] Rollback plan ready: Recovery procedures documented

---

*This deployment guide ensures smooth setup and operation of VN-Quant in any environment. For issues not covered, refer to system logs or contact the development team.*
