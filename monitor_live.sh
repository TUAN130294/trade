#!/bin/bash
# VN-QUANT Live Monitoring Script
# Shows agent signals, trades, and system status in real-time

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Clear screen
clear

echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║      VN-QUANT LIVE MONITORING DASHBOARD                        ║${NC}"
echo -e "${CYAN}║      Watching: Agent Signals • Trades • Status                 ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to show agent signals
show_agent_signals() {
    echo -e "${MAGENTA}━━━ AGENT SIGNALS (Last 30 events) ━━━${NC}"

    docker-compose logs --tail=200 autonomous 2>/dev/null | grep -E "Scout|Alex|Bull|Bear|Chief|RiskDoctor" | tail -30 | while read line; do
        if [[ $line == *"Scout"* ]]; then
            echo -e "${CYAN}🔭 $line${NC}"
        elif [[ $line == *"Alex"* ]]; then
            echo -e "${BLUE}📊 $line${NC}"
        elif [[ $line == *"Bull"* ]]; then
            echo -e "${GREEN}🐂 $line${NC}"
        elif [[ $line == *"Bear"* ]]; then
            echo -e "${RED}🐻 $line${NC}"
        elif [[ $line == *"RiskDoctor"* ]]; then
            echo -e "${YELLOW}🏥 $line${NC}"
        elif [[ $line == *"Chief"* ]]; then
            echo -e "${WHITE}⚖️  $line${NC}"
        else
            echo "$line"
        fi
    done
    echo ""
}

# Function to show trades
show_trades() {
    echo -e "${MAGENTA}━━━ TRADES EXECUTED (Last 20) ━━━${NC}"

    docker-compose logs --tail=100 autonomous 2>/dev/null | grep -i "order\|executed\|position" | tail -20 | while read line; do
        if [[ $line == *"ORDER EXECUTED"* ]]; then
            echo -e "${GREEN}✅ $line${NC}"
        elif [[ $line == *"POSITION EXIT"* ]]; then
            if [[ $line == *"+"* ]]; then
                echo -e "${GREEN}📈 $line${NC}"
            else
                echo -e "${RED}📉 $line${NC}"
            fi
        elif [[ $line == *"POSITION"* ]]; then
            echo -e "${YELLOW}📍 $line${NC}"
        else
            echo "$line"
        fi
    done
    echo ""
}

# Function to show system status
show_system_status() {
    echo -e "${MAGENTA}━━━ SYSTEM STATUS ━━━${NC}"

    # Service status
    echo -e "${BLUE}Services:${NC}"
    docker-compose ps --format "table {{.Names}}\t{{.Status}}" 2>/dev/null | grep -v "NAMES" | while read line; do
        if [[ $line == *"healthy"* ]]; then
            echo -e "  ${GREEN}✅${NC} $line"
        elif [[ $line == *"Up"* ]]; then
            echo -e "  ${YELLOW}⚙️ ${NC} $line"
        else
            echo -e "  ${RED}❌${NC} $line"
        fi
    done

    echo ""

    # Trading status
    echo -e "${BLUE}Trading Status:${NC}"
    status=$(curl -s http://localhost:5176/api/status 2>/dev/null || echo "{}")

    is_running=$(echo "$status" | grep -o '"is_running":[^,}]*' | cut -d: -f2 | tr -d ' ')
    paper_mode=$(echo "$status" | grep -o '"paper_trading":[^,}]*' | cut -d: -f2 | tr -d ' ')
    balance=$(echo "$status" | grep -o '"balance":[^,}]*' | cut -d: -f2 | tr -d ' ')
    opportunities=$(echo "$status" | grep -o '"opportunities_detected":[^,}]*' | cut -d: -f2 | tr -d ' ')

    if [[ "$is_running" == "true" ]]; then
        echo -e "  ${GREEN}✅${NC} System Running"
    else
        echo -e "  ${RED}❌${NC} System Stopped"
    fi

    if [[ "$paper_mode" == "true" ]]; then
        echo -e "  ${YELLOW}📄${NC} Paper Trading Mode"
    else
        echo -e "  ${RED}⚠️  LIVE TRADING${NC}"
    fi

    if [[ ! -z "$balance" ]]; then
        echo -e "  💰 Balance: ${balance} VND"
    fi

    if [[ ! -z "$opportunities" ]]; then
        echo -e "  🎯 Opportunities: ${opportunities}"
    fi

    echo ""
}

# Function to show recent errors
show_errors() {
    echo -e "${MAGENTA}━━━ WARNINGS & ERRORS (Last 10) ━━━${NC}"

    errors=$(docker-compose logs --tail=500 autonomous 2>/dev/null | grep -iE "error|warning|failed|exception" | tail -10)

    if [[ -z "$errors" ]]; then
        echo -e "  ${GREEN}✅ No errors detected${NC}"
    else
        echo "$errors" | while read line; do
            echo -e "  ${RED}⚠️  $line${NC}"
        done
    fi

    echo ""
}

# Main monitoring loop
echo -e "${BLUE}Updating every 5 seconds... Press Ctrl+C to stop${NC}"
echo ""

while true; do
    # Clear previous output (keep header)
    clear

    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║      VN-QUANT LIVE MONITORING DASHBOARD                        ║${NC}"
    echo -e "${CYAN}║      Updated: $(date '+%Y-%m-%d %H:%M:%S')                                   ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Show all sections
    show_system_status
    show_agent_signals
    show_trades
    show_errors

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Press Ctrl+C to stop | Refreshing in 5 seconds...${NC}"

    sleep 5
done
