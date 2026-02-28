#!/bin/bash
# Test script for Phase 3-4 interpretation endpoints
# Usage: ./test-interpretation-endpoints.sh

BASE_URL="http://localhost:8100/api/api"

echo "============================================================"
echo "Phase 3-4 Interpretation Endpoint Testing"
echo "============================================================"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

test_endpoint() {
    local name=$1
    local method=$2
    local endpoint=$3
    local data=$4

    echo -e "${YELLOW}Testing: ${name}${NC}"
    echo "Endpoint: ${method} ${endpoint}"

    if [ "$method" = "GET" ]; then
        response=$(curl -s "${BASE_URL}${endpoint}")
    else
        response=$(curl -s -X POST "${BASE_URL}${endpoint}" \
            -H "Content-Type: application/json" \
            -d "${data}")
    fi

    # Check if interpretation field exists
    if echo "$response" | jq -e '.interpretation' > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PASS - interpretation field present${NC}"
        echo "Interpretation preview:"
        echo "$response" | jq -r '.interpretation' | head -c 100
        echo "..."
    elif echo "$response" | jq -e '.llm_interpretation' > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PASS - llm_interpretation field present${NC}"
        echo "Interpretation preview:"
        echo "$response" | jq -r '.llm_interpretation' | head -c 100
        echo "..."
    else
        echo -e "${RED}❌ FAIL - no interpretation field${NC}"
        echo "Response preview:"
        echo "$response" | jq . | head -20
    fi

    echo ""
}

echo "1️⃣  Market Status"
test_endpoint "Market Status" "GET" "/market/status?interpret=true"

echo "2️⃣  Market Regime"
test_endpoint "Market Regime" "GET" "/market/regime?interpret=true"

echo "3️⃣  Smart Signals"
test_endpoint "Smart Signals" "GET" "/market/smart-signals?interpret=true"

echo "4️⃣  Technical Analysis (MWG)"
test_endpoint "Technical Analysis" "GET" "/analysis/technical/MWG?interpret=true"

echo "5️⃣  Data Stats"
test_endpoint "Data Stats" "GET" "/data/stats?interpret=true"

echo "6️⃣  News Market Mood"
test_endpoint "News Market Mood" "GET" "/news/market-mood?interpret=true"

echo "7️⃣  News Alerts"
test_endpoint "News Alerts" "GET" "/news/alerts?interpret=true"

echo "8️⃣  Backtest Run"
test_endpoint "Backtest" "POST" "/backtest/run?interpret=true" '{"strategy":"momentum","symbol":"MWG","days":365}'

echo "============================================================"
echo "Testing Complete"
echo "============================================================"
echo ""
echo "Note: If tests fail, restart backend:"
echo "  python start_backend_api.py"
echo ""
