# Docker Deployment Guide

## Overview

VN-Quant runs in Docker containers with:
- **PostgreSQL** for data storage
- **Redis** for caching
- **Autonomous Trading Service** (main trading system)
- **Backend API** (REST endpoints)
- **Model Training Service** (weekly ML training)

This guide covers local development and production deployment.

---

## Prerequisites

### System Requirements

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

---

## Quick Start (5 Minutes)

### 1. Configure Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit .env with your settings
# Most defaults work fine for testing
```

### 2. Start All Services

```bash
# Build and start containers
docker-compose up -d

# Check status
docker-compose ps

# Output should show:
# NAME                    STATUS
# vnquant-postgres        Up 30s (healthy)
# vnquant-redis           Up 30s (healthy)
# vnquant-autonomous      Up 25s
# vnquant-api             Up 20s
# vnquant-trainer         Up 15s (background service)
```

### 3. Verify Services

```bash
# Check logs
docker-compose logs autonomous -f  # View trading logs
docker-compose logs api -f         # View API logs
docker-compose logs trainer -f     # View training logs

# Test API
curl http://localhost:8003/api/status

# Expected: {"status": "ok", "uptime": 123}
```

### 4. Access Dashboard

```
Trading Dashboard: http://localhost:8001/autonomous
API Documentation: http://localhost:8003/docs
```

### 5. Stop Services

```bash
# Stop all containers
docker-compose down

# Stop and remove volumes (WARNING: deletes data!)
docker-compose down -v
```

---

## Configuration

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

### Database Configuration

**PostgreSQL Settings** (in docker-compose.yml):

```yaml
environment:
  POSTGRES_DB: vnquant
  POSTGRES_USER: vnquant
  POSTGRES_PASSWORD: changeme    # Change this!
```

**Access Database:**

```bash
# Connect to PostgreSQL
docker exec -it vnquant-postgres psql -U vnquant -d vnquant

# Common commands:
\dt                    # List tables
SELECT * FROM trades;  # View trade history
\q                     # Quit
```

### Redis Configuration

**Access Cache:**

```bash
# Connect to Redis CLI
docker exec -it vnquant-redis redis-cli

# Common commands:
KEYS *                 # List all keys
GET key_name           # Get value
DEL key_name          # Delete key
FLUSHALL              # Clear all
QUIT                  # Exit
```

---

## Service Details

### Autonomous Trading Service

**Container:** `vnquant-autonomous`

**What it does:**
- Scans markets every 3 minutes
- Runs 6 agent consensus
- Places and manages trades
- Monitors positions
- Logs all activity

**Configuration:**
```bash
# Edit in docker-compose.yml or .env
AUTO_TRADE_ENABLED=false       # Start disabled (manual test first)
AUTO_SCAN_INTERVAL=300         # Seconds between scans
ALLOW_REAL_TRADING=false       # Extra safety: prevents real trades
```

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

### Backend API Service

**Container:** `vnquant-api`

**What it does:**
- REST API for dashboard
- WebSocket for real-time updates
- Order management
- Trade history
- Performance analytics

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

### Model Training Service

**Container:** `vnquant-trainer`

**What it does:**
- Trains Stockformer models weekly
- Validates predictions
- Deploys new models
- Sends notifications
- Logs results

**Scheduling:**
```bash
# Training runs automatically on schedule
# Default: Sundays 2:00 AM

# To run now (for testing):
docker exec vnquant-trainer python train_scheduler.py run-now

# View training log
docker logs vnquant-trainer -f
```

**Configuration:**
```bash
# In .env:
TRAINING_SCHEDULE=0 2 * * 0     # Cron format
TIMEZONE=Asia/Ho_Chi_Minh

# Notifications:
SLACK_WEBHOOK_URL=...            # Slack alerts
EMAIL_RECIPIENTS=you@example.com # Email reports
```

---

## Volume Mapping

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

---

## Port Mapping

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

---

## Building Images

### Build All Services

```bash
# Build from scratch
docker-compose build

# Build specific service
docker-compose build autonomous
docker-compose build api
docker-compose build model-trainer
```

### Custom Build

```bash
# Build with custom base image
docker build -t vnquant:custom --build-arg BASE_IMAGE=python:3.11 .
```

### Multi-Stage Build (Optimized)

For production, use multi-stage Dockerfile:

```dockerfile
# Build stage
FROM python:3.10 as builder
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.10-slim
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "run_autonomous_paper_trading.py"]
```

---

## Networking

### Internal Communication

Services communicate via `vn-quant-network`:

```
autonomous ←→ postgres (DB)
autonomous ←→ redis (cache)
api ←→ postgres
api ←→ redis
trainer ←→ postgres
```

### External Communication

Services connect to external APIs:

```
autonomous → CafeF API (market data)
autonomous → Finnhub API (news)
api → <client browser>
trainer → Email SMTP server (optional)
```

### Firewall/Network Issues

```bash
# Test connectivity
docker exec vnquant-api curl http://postgres:5432

# Check network
docker network ls
docker network inspect vn-quant-network

# DNS resolution
docker exec vnquant-api nslookup postgres
```

---

## Monitoring & Logs

### View Logs

```bash
# All services
docker-compose logs

# Specific service
docker-compose logs autonomous -f      # Follow (real-time)
docker-compose logs api --tail=100     # Last 100 lines
docker-compose logs trainer -f --since 1h  # Last hour

# Filter by level
docker-compose logs autonomous | grep ERROR
```

### Log Files

```bash
# Local log files
logs/autonomous.log       # Trading system
logs/api.log              # REST API
logs/trainer.log          # Model training

# View specific log
tail -f logs/autonomous.log
```

### Health Checks

```bash
# Check container health
docker-compose ps

# Detailed health info
docker inspect vnquant-autonomous --format='{{json .State.Health}}'

# Manual health test
docker exec vnquant-api curl http://localhost:8003/api/status
```

---

## Troubleshooting

### Container Won't Start

**Symptoms:** Container exits immediately

**Solution:**
```bash
# 1. Check logs
docker-compose logs autonomous

# 2. Check if ports are in use
lsof -i :8001          # Check port 8001
netstat -tlnp | grep 8003  # Check port 8003

# 3. Free port and restart
docker-compose restart autonomous
```

### Database Connection Error

**Symptoms:** "Cannot connect to postgres:5432"

**Solution:**
```bash
# 1. Check if PostgreSQL is healthy
docker-compose ps | grep postgres

# 2. If not healthy, restart
docker-compose restart postgres

# 3. Wait for health check (30 seconds)
docker-compose logs postgres

# 4. Check credentials in .env
grep POSTGRES_PASSWORD .env
grep DATABASE_URL .env
```

### Out of Memory

**Symptoms:** Container keeps restarting, OOMKilled

**Solution:**
```bash
# 1. Check memory usage
docker stats

# 2. Increase Docker memory limit (Windows/macOS):
# Settings → Resources → Memory: 8GB (from 2GB)

# 3. Reduce model training batch size
# Edit Dockerfile.training:
ENV BATCH_SIZE=16  # from 32
```

### Volumes Not Persisting

**Symptoms:** Data lost after restart

**Solution:**
```bash
# 1. Verify volume exists
docker volume ls | grep vn-quant

# 2. Check volume mapping
docker inspect vnquant-postgres -f '{{json .Mounts}}'

# 3. Ensure .env has correct volume paths
grep "POSTGRES_DATA" docker-compose.yml
```

### API Not Responding

**Symptoms:** "Connection refused" when accessing http://localhost:8003

**Solution:**
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

---

## Production Deployment

### Security Checklist

- [ ] Change all default passwords in `.env`
- [ ] Set `ALLOW_REAL_TRADING=false` (until tested)
- [ ] Enable `ENVIRONMENT=production`
- [ ] Use strong `JWT_SECRET`
- [ ] Configure HTTPS (reverse proxy with nginx)
- [ ] Set up firewall rules
- [ ] Enable audit logging
- [ ] Regular backups enabled

### Scaling

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

## Advanced Topics

### Custom Networks

```bash
# Create custom network
docker network create my-network

# Use in docker-compose.yml
networks:
  - my-network
```

### Secret Management

```bash
# Use Docker secrets (Swarm)
echo "my-secret-value" | docker secret create my-secret -

# Or environment files
docker-compose --env-file .env.production up
```

### CI/CD Integration

```yaml
# GitHub Actions example
- name: Build Docker images
  run: docker-compose build

- name: Run tests
  run: docker-compose run --rm autonomous pytest

- name: Deploy
  run: docker-compose -f docker-compose.prod.yml up -d
```

---

## Support

For Docker issues:

1. Check logs: `docker-compose logs`
2. Review this guide's troubleshooting section
3. Check Docker documentation: https://docs.docker.com
4. Create GitHub issue with full log output

**Last Updated**: 2025-01-12
**Version**: 1.0
