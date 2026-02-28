#!/bin/bash
# VN-Quant Docker Quick Start Script
# Run this script to start the complete system in Docker

set -e  # Exit on error

echo "================================"
echo "🚀 VN-QUANT Docker Quick Start"
echo "================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check prerequisites
echo -e "${BLUE}1. Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    echo "   Install from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed${NC}"
    exit 1
fi

DOCKER_VERSION=$(docker --version)
echo -e "${GREEN}✅ Docker: $DOCKER_VERSION${NC}"

# Check disk space
DISK_FREE=$(df . | tail -1 | awk '{print $4}')
if [ "$DISK_FREE" -lt 2097152 ]; then  # 2GB in KB
    echo -e "${YELLOW}⚠️  Warning: Less than 2GB free disk space${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if Docker daemon is running
echo -e "${BLUE}2. Checking Docker daemon...${NC}"
if ! docker ps &> /dev/null; then
    echo -e "${RED}❌ Docker daemon is not running${NC}"
    echo "   Start Docker Desktop and try again"
    exit 1
fi
echo -e "${GREEN}✅ Docker daemon is running${NC}"

# Configuration
echo ""
echo -e "${BLUE}3. Configuring environment...${NC}"

if [ ! -f .env ]; then
    echo "   Creating .env from .env.example..."
    cp .env.example .env
    echo -e "${GREEN}✅ .env created${NC}"
    echo ""
    echo -e "${YELLOW}   ⚠️  Please edit .env with your API keys${NC}"
    echo "   Edit these if needed:"
    echo "   - GEMINI_API_KEY"
    echo "   - TELEGRAM_BOT_TOKEN"
    echo ""
    read -p "Continue with defaults? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✅ .env already configured${NC}"
fi

# Build images
echo ""
echo -e "${BLUE}4. Building Docker images...${NC}"
echo "   This may take 5-10 minutes on first run"
echo ""

docker-compose build --progress=plain

# Start services
echo ""
echo -e "${BLUE}5. Starting services...${NC}"
echo "   Starting PostgreSQL, Redis, Trading, API, and Training..."
echo ""

docker-compose up -d

# Wait for services to be ready
echo -e "${BLUE}6. Waiting for services to be healthy...${NC}"
echo ""

RETRIES=30
RETRY=0

while [ $RETRY -lt $RETRIES ]; do
    if docker-compose ps | grep -q "postgres.*healthy"; then
        echo -e "${GREEN}✅ PostgreSQL is healthy${NC}"
        break
    fi
    echo "   Waiting for PostgreSQL... ($((RETRY+1))/$RETRIES)"
    sleep 1
    RETRY=$((RETRY+1))
done

RETRY=0
while [ $RETRY -lt $RETRIES ]; do
    if docker-compose ps | grep -q "redis.*healthy"; then
        echo -e "${GREEN}✅ Redis is healthy${NC}"
        break
    fi
    echo "   Waiting for Redis... ($((RETRY+1))/$RETRIES)"
    sleep 1
    RETRY=$((RETRY+1))
done

# Wait for other services
sleep 5

# Verify all services
echo ""
echo -e "${BLUE}7. Verifying services...${NC}"
echo ""

docker-compose ps

# Test API
echo ""
echo -e "${BLUE}8. Testing API connectivity...${NC}"

if curl -s http://localhost:8003/api/status > /dev/null 2>&1; then
    echo -e "${GREEN}✅ API is responding${NC}"
else
    echo -e "${YELLOW}⚠️  API may still be starting...${NC}"
fi

# Summary
echo ""
echo "================================"
echo -e "${GREEN}✅ VN-QUANT is ready!${NC}"
echo "================================"
echo ""
echo "📊 Dashboard:"
echo "   http://localhost:8001/autonomous"
echo ""
echo "📡 API Documentation:"
echo "   http://localhost:8003/docs"
echo ""
echo "📋 View Logs:"
echo "   docker-compose logs -f autonomous"
echo "   docker-compose logs -f api"
echo "   docker-compose logs -f model-trainer"
echo ""
echo "🛑 Stop System:"
echo "   docker-compose down"
echo ""
echo "📖 For more details, see docs/docker-deployment.md"
echo ""
