# Full Broken Features Audit Report

**Date:** 2026-02-25
**Branch:** backup/before-refactor
**Scope:** Entire codebase -- routers, agents, services, scanners, frontend, data pipeline

---

## Executive Summary

The VN-QUANT platform has **7 Critical** and **6 High** priority issues that collectively render most features non-functional. The three user-reported issues (AI predict broken, auto trading broken, agent chat not calling LLM) are all confirmed, each caused by specific code bugs documented below.

**Root causes fall into 4 categories:**
1. **Route prefix doubling** -- market/data/news routers all 404 due to `/api/api/...` paths
2. **Wrong file paths** -- prediction model lookup searches wrong directory
3. **Missing imports/variables** -- `ConversationalQuant` not imported in market.py, `active_websockets` not defined
4. **LLM disabled by default** -- `USE_LLM_AGENTS` defaults to `false`, agent chat uses hardcoded strings

---

## CRITICAL Issues (7)

### C1. Route Prefix Doubling -- ALL Market/Data/News Endpoints Return 404

**Priority:** CRITICAL
**Impact:** Dashboard, AI Predict, News Intel, Data Hub, Backtest -- ALL broken
**Files:**
- `D:\testpapertr\run_autonomous_paper_trading.py` (lines 79-81)
- `D:\testpapertr\app\api\routers\market.py` (all endpoint definitions)
- `D:\testpapertr\app\api\routers\data.py` (all endpoint definitions)
- `D:\testpapertr\app\api\routers\news.py` (all endpoint definitions)

**Root Cause:**
The routers are included with `prefix="/api"`:
```python
# run_autonomous_paper_trading.py:79-81
app.include_router(market.router, prefix="/api")
app.include_router(data.router, prefix="/api")
app.include_router(news.router, prefix="/api")
```

But every endpoint in these routers ALREADY includes `/api/` in its path:
```python
# market.py:28
@router.get("/api/market/status")
# data.py:27
@router.get("/api/stock/{symbol}")
# news.py:33
@router.get("/api/news/status")
```

**Result:** Actual route becomes `/api/api/market/status`, `/api/api/stock/{symbol}`, etc. Frontend calls `/api/market/status` which returns 404.

**Affected Endpoints (ALL of these are unreachable):**
- `/api/market/status`, `/api/market/regime`, `/api/market/smart-signals`
- `/api/agents/status`, `/api/agents/chat`, `/api/agents/analyze`
- `/api/analysis/technical/{symbol}`
- `/api/stock/{symbol}`, `/api/predict/{symbol}`, `/api/data/stats`
- `/api/analyze/deep_flow`
- `/api/news/status`, `/api/news/alerts`, `/api/news/market-mood`
- `/api/news/watchlist`, `/api/news/scan`
- `/api/backtest/run`

**Fix:** Either remove `prefix="/api"` from `include_router()` calls, OR remove `/api/` prefix from all endpoint decorators in the router files. Recommended: remove the prefix from `include_router()` since the trading router works without it.

---

### C2. AI Prediction Endpoint Looks in Wrong Directory -- Always Returns "No Model"

**Priority:** CRITICAL
**Impact:** AI Predict feature completely broken
**File:** `D:\testpapertr\app\api\routers\data.py` (lines 113-114)

**Root Cause:**
```python
base_dir = Path(__file__).parent.resolve()
model_path = base_dir / "models" / f"{symbol}_stockformer_simple_best.pt"
```

`Path(__file__).parent.resolve()` = `D:\testpapertr\app\api\routers\`
So `model_path` = `D:\testpapertr\app\api\routers\models\{symbol}_stockformer_simple_best.pt`

But models are actually at `D:\testpapertr\models\{symbol}_stockformer_simple_best.pt` (100 models exist there).

`model_path.exists()` is ALWAYS `False`, so prediction ALWAYS returns the error fallback:
```json
{"error": "No trained model for {symbol}. Train model first."}
```

**Fix:**
```python
# Change to project root
base_dir = Path(__file__).parent.parent.parent.parent.resolve()  # D:\testpapertr
model_path = base_dir / "models" / f"{symbol}_stockformer_simple_best.pt"
```

---

### C3. Agent Chat Never Calls LLM -- `ConversationalQuant` NameError in market.py

**Priority:** CRITICAL
**Impact:** Agent Chat returns only hardcoded string responses
**File:** `D:\testpapertr\app\api\routers\market.py` (lines 747-756)

**Root Cause:**
```python
# market.py:747
if ConversationalQuant:
    conv = ConversationalQuant()
    result = conv.process_query(query)
```

`ConversationalQuant` is NEVER imported in `market.py`. It is defined as a global in `run_autonomous_paper_trading.py` (line 28-30), but Python module namespaces are isolated -- market.py cannot access globals from another module.

This causes a `NameError` at runtime, which is caught by the `except Exception` handler (line 755), and execution falls through to the hardcoded response map (lines 759-782) which returns canned strings like:
```
"MWG is a growth stock with strong momentum."
```

Even if `ConversationalQuant` were imported, it does NOT call any LLM. It is a pure regex/pattern-matching intent classifier (see `D:\testpapertr\quantum_stock\agents\conversational_quant.py`) that returns template strings like "Dang thuc hien phan tich..." -- placeholder text with no actual analysis.

**Fix:** Two options:
1. Import and use `AIAgentCoordinator` from `llm_agents.py` which actually calls LLM via the CCS proxy
2. Or import `ConversationalQuant` properly AND make it delegate to the LLM-powered agents

---

### C4. `active_websockets` Undefined in Main Runner -- WebSocket Broadcasting Crashes

**Priority:** CRITICAL
**Impact:** WebSocket real-time updates crash with NameError
**File:** `D:\testpapertr\run_autonomous_paper_trading.py` (lines 153-205)

**Root Cause:**
The `broadcast_messages()` function and `websocket_endpoint()` both reference `active_websockets` (bare variable), but this variable is never defined in the module scope. The module imports `from app.core import state` and `state.active_websockets` exists, but the code uses bare `active_websockets` instead.

```python
# Line 153: NameError -- active_websockets is not defined
logger.info(f"... to {len(active_websockets)} clients")
# Line 156
for ws in active_websockets[:]:
# Line 180
active_websockets.append(websocket)
```

**Fix:** Replace all bare `active_websockets` with `state.active_websockets` (8 occurrences in the file).

---

### C5. USE_LLM_AGENTS Defaults to False -- Autonomous System Never Uses AI

**Priority:** CRITICAL
**Impact:** Autonomous trading only uses rule-based agents, never LLM
**File:** `D:\testpapertr\quantum_stock\autonomous\orchestrator.py` (line 145)

**Root Cause:**
```python
self.use_llm_agents = os.getenv('USE_LLM_AGENTS', 'false').lower() == 'true'
```

Unless the environment variable `USE_LLM_AGENTS=true` is explicitly set, the system defaults to rule-based agents. The `.env` file likely does not contain this setting.

When `use_llm_agents` is `false`, agent discussions use the `AgentCoordinator` (pure algorithmic analysis) which produces template-based messages, not LLM-generated insights.

**Fix:** Set `USE_LLM_AGENTS=true` in `.env` and ensure `LLM_API_KEY` is configured. Also verify the CCS proxy at `localhost:8317` is running.

---

### C6. InterpretationService API Key Hardcoded

**Priority:** CRITICAL (Security)
**File:** `D:\testpapertr\quantum_stock\services\interpretation_service.py` (lines 24-25)

**Root Cause:**
```python
LLM_BASE_URL = "http://localhost:8317/v1"
LLM_API_KEY = "sk-***REDACTED***"
```

The API key is hardcoded as a class constant. This is a security violation -- API keys should come from environment variables.

**Fix:**
```python
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8317/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
```

---

### C7. Multi-Agent Analysis (`/api/agents/analyze`) Returns Hardcoded Strings, Not LLM Output

**Priority:** CRITICAL
**Impact:** The "Agent Radar" view shows template strings, not real AI analysis
**File:** `D:\testpapertr\app\api\routers\market.py` (lines 795-1079)

**Root Cause:**
The `analyze_with_agents` endpoint does NOT call any agent coordinator or LLM. It:
1. Fetches historical data (or generates synthetic data)
2. Computes RSI, MACD, Bollinger, volume indicators manually
3. Returns **hardcoded f-string templates** for each agent (Scout, Alex, Bull, Bear, Risk Doctor, Chief)

For example:
```python
# Line 998 -- Bull's "analysis" is just a hardcoded string template
"content": f"{'VOLUME BUG NO! Dong tien dang do vao manh' if vol_ratio > 2 else ...}"
```

```python
# Line 1035 -- Chief's "verdict" is a simple if-else
"{'KHUYEN NGHI: MUA' if current_rsi < 40 and macd_value > 0 else ...}"
```

None of the 6 agents actually "think" -- they are just formatted strings with conditional logic on RSI/MACD values.

**Fix:** Replace the hardcoded message templates with actual calls to either:
- `AgentCoordinator.analyze_stock()` (rule-based but uses proper agent logic)
- `AIAgentCoordinator.analyze_symbol()` (LLM-powered analysis)

---

## HIGH Priority Issues (6)

### H1. PASSED_STOCKS.txt Missing -- Scanner Cannot Prioritize Stocks

**Priority:** HIGH
**Impact:** Model prediction scanner cannot load priority stock list
**File:** `D:\testpapertr\quantum_stock\scanners\model_prediction_scanner.py` (lines 86-127)

The scanner expects `PASSED_STOCKS.txt` at project root. File does not exist. While the code handles this gracefully (falls back to scanning all), it logs a warning every scan cycle and loses the intended prioritization behavior.

**Fix:** Create `PASSED_STOCKS.txt` with the 8 best-performing stocks from backtesting.

---

### H2. Interpretation Template Name Mismatches

**Priority:** HIGH
**Impact:** LLM interpretation calls silently fail/return error strings
**Files:**
- `D:\testpapertr\app\api\routers\data.py` (line 174) -- calls `"data_stats"` template
- `D:\testpapertr\app\api\routers\news.py` (line 313) -- calls `"backtest_results"` template
- `D:\testpapertr\quantum_stock\services\interpretation_service.py` (PROMPT_TEMPLATES dict)

**Root Cause:**
- `data.py` calls `interp_service.interpret("data_stats", ...)` but no `"data_stats"` template exists
- `news.py` calls `interp_service.interpret("backtest_results", ...)` but template is named `"backtest_result"` (singular)

Both return `"[Template 'xxx' khong ton tai]"` to the frontend.

**Fix:** Either add missing templates to InterpretationService, or fix the caller template names.

---

### H3. PaperTradingBroker Uses Hardcoded/Random Prices Instead of Real Market Data

**Priority:** HIGH
**Impact:** Paper trading executes at wrong prices; P&L is fictional
**File:** `D:\testpapertr\quantum_stock\core\broker_api.py` (lines 538-573)

**Root Cause:**
`get_market_price()` returns hardcoded prices from a static dictionary (13 stocks) or random prices for unknown symbols:
```python
default_prices = {
    'VNM': {'last': 78500, ...},  # Static, never updates
    'HPG': {'last': 27800, ...},
}
# Unknown symbols:
base_price = random.randint(10, 150) * 1000  # Pure random!
```

These prices never change during the session and do not reflect actual market conditions.

**Fix:** Integrate with `RealTimeMarketConnector.get_stock_price()` for live price fetching. Use the VPS API or CafeF as the primary price source.

---

### H4. Deep Flow Analysis Returns Random Data

**Priority:** HIGH
**Impact:** Deep flow insights are meaningless
**File:** `D:\testpapertr\app\api\routers\data.py` (lines 188-212)

```python
"flow_score": round(np.random.uniform(60, 85), 1),
"recommendation": "WATCH" if np.random.random() > 0.5 else "ACCUMULATE",
```

The entire endpoint returns hardcoded insight descriptions with random scores. No actual flow analysis is performed.

**Fix:** Integrate with real flow analysis from `FlowAgent` or compute actual volume/foreign flow metrics.

---

### H5. Foreign Flow in Agent Analysis is Random

**Priority:** HIGH
**Impact:** Foreign investor flow data shown to users is fictional
**File:** `D:\testpapertr\app\api\routers\market.py` (line 946)

```python
foreign_net = np.random.uniform(-10, 10)  # billion VND -- RANDOM!
```

This random value feeds into the confidence scoring and agent messages. Users see fake foreign flow data.

**Fix:** Use `RealTimeMarketConnector.get_foreign_flow()` for real data.

---

### H6. `ConversationalQuant` is a Placeholder -- No Actual NLU

**Priority:** HIGH
**Impact:** Even if properly imported, agent chat is just regex pattern matching
**File:** `D:\testpapertr\quantum_stock\agents\conversational_quant.py` (entire file, 777 lines)

The `ConversationalQuant` class is a pure regex-based intent classifier that returns static Vietnamese template strings. Every handler method returns hardcoded text like:
```python
"Dang thuc hien phan tich da chieu..."
"Dang tai du lieu..."
"Dang phan tich..."
```

No actual analysis is performed. No LLM is called. The class serves as UI placeholder text only.

**Fix:** Replace with or delegate to `AIAgentCoordinator.analyze_symbol()` which actually calls LLM.

---

## MEDIUM Priority Issues (5)

### M1. LLM Agents Mock Mode When No API Key

**File:** `D:\testpapertr\quantum_stock\agents\llm_agents.py` (lines 88-90, 119-130)

When `LLM_API_KEY` is empty (likely the case without explicit `.env` config), the `LLMClient.chat()` silently falls back to `_mock_response()` which returns canned Vietnamese strings. No error/warning is surfaced to the user.

**Fix:** Surface a clear warning in the API response when LLM is in mock mode.

---

### M2. vn_quant_api.py (Port 8003 Server) Uses Relative Imports That Break

**File:** `D:\testpapertr\quantum_stock\web\vn_quant_api.py` (lines 24-31)

```python
from agents.agent_coordinator import AgentCoordinator
from agents.conversational_quant import ConversationalQuant
```

These relative imports only work if the working directory is `quantum_stock/`. If `start_backend_api.py` is run from the project root, these imports fail because Python looks for `agents` at the top-level, not inside `quantum_stock/`.

**Fix:** Use absolute imports: `from quantum_stock.agents.agent_coordinator import AgentCoordinator`

---

### M3. Market Status Returns Hardcoded VN-Index Default

**File:** `D:\testpapertr\app\api\routers\market.py` (lines 37-39)

```python
vnindex = 1249.05  # Hardcoded default
change = 0.0
change_pct = 0.0
```

If CafeF API call fails (which it often does due to rate limiting), the endpoint returns this stale hardcoded value.

**Fix:** Add VPS API as fallback; cache last known good value; indicate data staleness in response.

---

### M4. Synthetic Data Generation in Agent Analysis

**File:** `D:\testpapertr\app\api\routers\market.py` (lines 838-861)

When no historical data is available, the endpoint generates synthetic/random price data and runs technical analysis on it. This produces meaningless RSI/MACD values that look real to users.

**Fix:** Return clear error when no real data available. Do not generate synthetic data for production analysis.

---

### M5. Multiple InterpretationService Instances Initialized at Module Level

**Files:**
- `D:\testpapertr\app\api\routers\data.py` (line 16)
- `D:\testpapertr\app\api\routers\news.py` (line 15)
- `D:\testpapertr\app\api\routers\market.py` (line 17)

Each router creates its own `InterpretationService()` instance at import time, each establishing an AsyncOpenAI client connection. This wastes resources and creates separate caches.

**Fix:** Use the singleton `get_interpretation_service()` function from the service module.

---

## LOW Priority Issues (3)

### L1. Hundreds of `tmpclaude-*-cwd` Temp Directories Cluttering Repo

Over 300 untracked `tmpclaude-*-cwd` directories exist in the project root. These are Claude Code working directory artifacts that should be gitignored.

**Fix:** Add `tmpclaude-*` to `.gitignore`. Delete existing ones.

---

### L2. Duplicate Pydantic Models Across Routers

`QueryRequest` and `AnalyzeRequest` are defined identically in multiple router files (`data.py`, `news.py`, `market.py`, and the main runner). This violates DRY.

**Fix:** Define shared models in `app/api/models.py` and import.

---

### L3. `news.py` Router Contains Backtest Endpoint

**File:** `D:\testpapertr\app\api\routers\news.py` (lines 211-332)

The `/api/backtest/run` endpoint is defined inside the news router, which is confusing for maintainability. It should be in its own router or in the data/analysis router.

---

## Summary Table

| # | Issue | Severity | Category | Affected Feature |
|---|-------|----------|----------|------------------|
| C1 | Route prefix doubling `/api/api/...` | CRITICAL | Routing | ALL endpoints (market, data, news) |
| C2 | Prediction model path wrong directory | CRITICAL | Path | AI Predict |
| C3 | ConversationalQuant not imported | CRITICAL | Import | Agent Chat |
| C4 | active_websockets undefined | CRITICAL | Variable | WebSocket/Broadcasting |
| C5 | USE_LLM_AGENTS defaults false | CRITICAL | Config | Autonomous Trading + LLM |
| C6 | API key hardcoded in source | CRITICAL | Security | InterpretationService |
| C7 | Agent analysis returns hardcoded strings | CRITICAL | Logic | Agent Radar/Analysis |
| H1 | PASSED_STOCKS.txt missing | HIGH | Data | Model Scanner Priority |
| H2 | Interpretation template name mismatches | HIGH | String | LLM Interpretations |
| H3 | Broker uses hardcoded/random prices | HIGH | Data | Paper Trading |
| H4 | Deep flow returns random data | HIGH | Logic | Deep Flow Analysis |
| H5 | Foreign flow is random number | HIGH | Logic | Agent Analysis |
| H6 | ConversationalQuant is regex placeholder | HIGH | Logic | Agent Chat |
| M1 | LLM mock mode silent fallback | MEDIUM | UX | LLM Agents |
| M2 | vn_quant_api.py relative imports | MEDIUM | Import | Backend API (8003) |
| M3 | Market status hardcoded default | MEDIUM | Data | Market Overview |
| M4 | Synthetic data in analysis | MEDIUM | Logic | Agent Analysis |
| M5 | Multiple InterpretationService instances | MEDIUM | Resource | All routers |
| L1 | Temp directories cluttering repo | LOW | Cleanup | Repository |
| L2 | Duplicate Pydantic models | LOW | DRY | All routers |
| L3 | Backtest endpoint in news router | LOW | Organization | Code Structure |

---

## Recommended Fix Order

1. **C1** -- Fix route prefix (unblocks ALL other endpoints) -- 5 min
2. **C2** -- Fix model path (unblocks AI Predict) -- 2 min
3. **C4** -- Fix active_websockets (unblocks WebSocket) -- 2 min
4. **C3 + C7 + H6** -- Wire up LLM agents to chat and analysis endpoints -- 2-4 hours
5. **C5** -- Configure USE_LLM_AGENTS=true in .env -- 1 min
6. **C6** -- Move API key to env var -- 5 min
7. **H2** -- Fix template name mismatches -- 5 min
8. **H3** -- Integrate real prices into broker -- 1-2 hours
9. **H4 + H5** -- Replace random data with real flow/foreign data -- 2-3 hours

Fixes 1-3 alone would restore basic dashboard functionality. Fixes 4-5 would enable real AI-powered analysis.

---

## Unresolved Questions

1. Is the CCS proxy at `localhost:8317` actually running and reachable? If not, all LLM calls will fall back to mock responses even with proper configuration.
2. What are the correct env var values for `LLM_API_KEY` and `LLM_BASE_URL`?
3. Should `BYPASS_MARKET_HOURS=true` be set for testing/paper trading outside market hours?
4. Are the 100 trained Stockformer models up-to-date, or do they need retraining with recent data?
5. The `vn_quant_api.py` (port 8003) appears to be an older/separate server. Is it still used or should all traffic go through the autonomous runner (port 8100)?
