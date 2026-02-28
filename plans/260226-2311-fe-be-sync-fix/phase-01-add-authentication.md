# Phase 01: Add Basic Authentication

**Priority:** P0 (CRITICAL)
**Effort:** 2h
**Status:** Pending

## Context Links
- Backend: `run_autonomous_paper_trading.py` (line 85-96: CORS config)
- Backend: `app/api/routers/trading.py` (line 90-306: dangerous endpoints /test/*, /reset, /stop)
- Audit report: Identified 0.0.0.0 binding with no auth on destructive endpoints

## Overview
Current state: FastAPI bound to 0.0.0.0:8100 with ZERO authentication. Dangerous endpoints exposed:
- `/api/reset` - Resets entire trading state
- `/api/stop` - Stops orchestrator
- `/api/test/opportunity` - Triggers test trades
- `/api/test/trade` - Injects fake trades

**Goal:** Add lightweight API key auth (X-API-Key header). Keep simple - this is paper trading, not production banking.

## Key Insights
- YAGNI: Don't need OAuth2/JWT complexity for paper trading
- Simple API key in .env, validate via FastAPI dependency
- Protect only dangerous endpoints (read-only endpoints stay open for dashboard)
- Frontend stores key in localStorage, sends in headers

## Requirements

### Functional
- Generate random API key on first run, save to .env
- Validate X-API-Key header on protected routes
- Return 401 Unauthorized if missing/invalid
- Frontend prompts for API key, stores in localStorage

### Non-Functional
- Add <1ms latency per request
- No external dependencies (use built-in secrets module)
- Backward compatible (existing endpoints without auth still work)

## Architecture

```
┌─────────────────┐
│ React Frontend  │
│ localStorage:   │
│ - apiKey        │
└────────┬────────┘
         │ X-API-Key: abc123...
         ↓
┌─────────────────┐
│ FastAPI         │
│ Dependency:     │
│ verify_api_key()│
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Protected Route │
│ /api/reset      │
│ /api/stop       │
│ /api/test/*     │
└─────────────────┘
```

## Related Code Files

**Backend (Modify):**
- `run_autonomous_paper_trading.py` - Add API key dependency, protect routes
- `.env` - Add API_KEY variable
- `.env.example` - Document API_KEY

**Backend (Create):**
- `app/core/auth.py` - API key validation dependency (new file, ~30 lines)

**Frontend (Modify):**
- `vn-quant-web/src/App.jsx` - Add API key prompt, include in fetch headers
- `vn-quant-web/.env.production` - Add VITE_REQUIRE_AUTH=true

## Implementation Steps

### Step 1: Generate API Key (Backend)
1. Create `app/core/auth.py`:
```python
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
import os
import secrets

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key():
    """Get API key from env, generate if missing"""
    key = os.getenv("API_KEY")
    if not key:
        key = secrets.token_urlsafe(32)
        print(f"\n⚠️  Generated new API_KEY: {key}")
        print("Add to .env: API_KEY={key}\n")
    return key

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Dependency to validate API key"""
    correct_key = get_api_key()
    if api_key != correct_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )
    return api_key
```

2. Update `.env.example`:
```bash
# Authentication (generated on first run)
API_KEY=your_api_key_here_generate_with_secrets_module
```

### Step 2: Protect Dangerous Endpoints (Backend)
Modify `app/api/routers/trading.py`:

```python
from app.core.auth import verify_api_key

# Add dependency to protected routes:
@router.post("/api/reset", dependencies=[Depends(verify_api_key)])
async def reset_system():
    ...

@router.post("/api/stop", dependencies=[Depends(verify_api_key)])
async def stop_system():
    ...

@router.post("/api/test/opportunity", dependencies=[Depends(verify_api_key)])
async def trigger_test_opportunity(symbol: str = "ACB"):
    ...

@router.post("/api/test/trade", dependencies=[Depends(verify_api_key)])
async def trigger_test_trade(request: TestTradeRequest):
    ...
```

### Step 3: Frontend API Key Prompt
Modify `vn-quant-web/src/App.jsx`:

Add state at top of App component:
```javascript
const [apiKey, setApiKey] = useState(localStorage.getItem('vn-quant-api-key') || null)
const [showKeyPrompt, setShowKeyPrompt] = useState(false)

// Helper to add auth header
const fetchWithAuth = (url, options = {}) => {
  const headers = { ...options.headers }
  if (apiKey) {
    headers['X-API-Key'] = apiKey
  }
  return fetch(url, { ...options, headers })
}
```

Add key prompt UI (before main render):
```javascript
if (showKeyPrompt) {
  return (
    <div className="flex items-center justify-center h-screen bg-[#0a0e17]">
      <div className="glass-panel p-8 rounded-xl max-w-md">
        <h2 className="text-xl font-bold mb-4">API Key Required</h2>
        <p className="text-slate-400 mb-4">Enter your API key to access protected features.</p>
        <input
          type="password"
          placeholder="API Key"
          className="w-full p-3 bg-black/30 border border-white/10 rounded-lg mb-4"
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              const key = e.target.value.trim()
              localStorage.setItem('vn-quant-api-key', key)
              setApiKey(key)
              setShowKeyPrompt(false)
            }
          }}
        />
      </div>
    </div>
  )
}
```

Handle 401 errors:
```javascript
// In fetch calls for protected endpoints, check for 401:
fetchWithAuth(`${API_URL}/reset`, { method: 'POST' })
  .then(r => {
    if (r.status === 401) {
      setShowKeyPrompt(true)
      throw new Error('Unauthorized')
    }
    return r.json()
  })
```

### Step 4: Test Auth Flow
1. Start backend: `python run_autonomous_paper_trading.py`
2. Copy generated API_KEY from console
3. Add to .env: `API_KEY=<copied_key>`
4. Restart backend
5. Open frontend, try accessing /api/reset → should prompt for key
6. Enter key, retry → should succeed

## Todo List
- [ ] Create app/core/auth.py with verify_api_key dependency
- [ ] Update .env.example with API_KEY docs
- [ ] Protect 4 dangerous endpoints in trading.py
- [ ] Add apiKey state to App.jsx
- [ ] Add fetchWithAuth helper function
- [ ] Add API key prompt UI
- [ ] Handle 401 errors, show prompt
- [ ] Test: /api/reset without key → 401
- [ ] Test: /api/reset with key → success
- [ ] Test: Read-only endpoints (status, orders) work without key

## Success Criteria
- [ ] API key auto-generated on first backend run
- [ ] 401 returned for protected endpoints without valid key
- [ ] Frontend prompts for key on 401
- [ ] Key stored in localStorage, sent in X-API-Key header
- [ ] Protected endpoints work with valid key
- [ ] Read-only endpoints work without key (dashboard usability)

## Risk Assessment
- **Risk:** Key leaked in frontend code → Mitigated: localStorage, not hardcoded
- **Risk:** CORS bypass allows external access → Mitigated: CORS already restricts to localhost
- **Risk:** Key visible in browser DevTools → Acceptable: paper trading only, not production

## Security Considerations
- API key is NOT encrypted in localStorage (acceptable for paper trading)
- No rate limiting (YAGNI for single-user system)
- No key rotation (manual via .env if needed)
- Production deployment should use proper OAuth2/JWT

## Next Steps
After completing this phase:
1. Update plan.md with completion status
2. Proceed to Phase 02 (WebSocket client)
3. Ensure auth works with WebSocket connection (add key to WS query params if needed)
