# Phase 04: Fix Schema Mismatches

**Priority:** P1 (HIGH)
**Effort:** 1.5h
**Status:** Pending

## Context Links
- Backend: `app/api/routers/market.py` (line 130-150: /api/market/regime returns market_regime)
- Frontend: `vn-quant-web/src/App.jsx` (line 312-317: reads regime + hurst_exponent)
- Backend: `app/api/routers/market.py` (line 221: smart-signals emits "WARNING" severity)
- Frontend: `vn-quant-web/src/App.jsx` (line 280: mapper only handles HIGH|MEDIUM|INFO)
- Backend: `app/api/routers/data.py` (line 95, 141: returns {error: ...} with HTTP 200)
- Frontend: `vn-quant-web/src/App.jsx` (line 750: expects proper HTTP error codes)

## Overview
Three schema drift issues causing runtime errors:

1. **Market Regime**: BE returns `market_regime`, FE expects `regime` + `hurst_exponent`
2. **Smart Signals Severity**: BE emits "WARNING", FE mapper missing this enum value
3. **Error Response Shape**: BE returns `{error: "..."}` with HTTP 200, FE expects 4xx/5xx codes

**Goal:** Align FE and BE schemas to prevent runtime errors and undefined values.

## Key Insights
- Fix #1: Change FE to read `market_regime` instead of `regime`
- Fix #2: Add "WARNING" case to FE severity mapper
- Fix #3: Update BE to return proper HTTP error codes (400, 404, 500)
- Alternative Fix #3: Update FE to check for `error` field in 200 responses (less clean)
- Prefer **fixing BE** for error codes (proper REST API design)

## Requirements

### Functional
- Market regime displays correctly without "undefined"
- Smart signals with WARNING severity display with correct color/icon
- Error responses trigger error UI, not silent failures

### Non-Functional
- Zero breaking changes to existing working features
- Backward compatible (handle both old and new schemas during transition)
- Proper HTTP semantics (4xx client errors, 5xx server errors)

## Architecture

### Fix 1: Market Regime Schema
```
BEFORE:
BE: {market_regime: "BULL", ...}
FE: regime?.regime → undefined ❌

AFTER:
BE: {market_regime: "BULL", ...}
FE: regime?.market_regime → "BULL" ✅
```

### Fix 2: Smart Signals Severity
```
BEFORE:
BE: {type: "...", severity: "WARNING", ...}
FE: mapper("WARNING") → undefined ❌ (falls through, no case)

AFTER:
BE: {type: "...", severity: "WARNING", ...}
FE: mapper("WARNING") → {color: "text-amber-400", icon: "warning"} ✅
```

### Fix 3: Error Response Format
```
BEFORE:
BE: HTTP 200 + {error: "No data for symbol"}
FE: res.ok === true → treats as success, renders error message in UI ❌

AFTER (Option A - Fix BE):
BE: HTTP 404 + {error: "No data for symbol"}
FE: res.ok === false → triggers .catch(), shows error toast ✅

AFTER (Option B - Fix FE):
BE: HTTP 200 + {error: "..."}
FE: Check data.error, throw if present ✅
```

**Recommendation:** Option A (fix BE) for proper REST semantics.

## Related Code Files

**Frontend (Modify):**
- `vn-quant-web/src/App.jsx` - Fix market regime field, add WARNING severity case

**Backend (Modify):**
- `app/api/routers/data.py` - Return HTTP 404 instead of {error: ...} with 200
- `app/api/routers/market.py` - Ensure consistent error handling

## Implementation Steps

### Step 1: Fix Market Regime Schema (Frontend)
Modify `vn-quant-web/src/App.jsx`:

```javascript
// BEFORE (line ~312-317):
<h3 className="text-2xl font-bold text-white tracking-tight">{regime?.regime || '---'}</h3>
...
<span className="text-slate-400 text-xs">Hurst: {regime?.hurst_exponent?.toFixed(3) || '---'}</span>

// AFTER:
<h3 className="text-2xl font-bold text-white tracking-tight">{regime?.market_regime || '---'}</h3>
...
<span className="text-slate-400 text-xs">Conf: {((regime?.confidence || 0) * 100).toFixed(0)}%</span>
```

**Rationale:** Backend returns `market_regime`, not `regime`. Use `confidence` instead of `hurst_exponent` (which BE doesn't return consistently).

### Step 2: Add WARNING Severity to Mapper (Frontend)
Modify `vn-quant-web/src/App.jsx` (around line 280):

```javascript
// Find the severity mapper function, add WARNING case:
const getSeverityColor = (severity) => {
  switch (severity) {
    case 'HIGH': return 'text-red-400'
    case 'MEDIUM': return 'text-amber-400'
    case 'WARNING': return 'text-amber-400' // Add this line
    case 'INFO': return 'text-blue-400'
    default: return 'text-slate-400'
  }
}

const getSeverityIcon = (severity) => {
  switch (severity) {
    case 'HIGH': return 'error'
    case 'MEDIUM': return 'warning'
    case 'WARNING': return 'warning' // Add this line
    case 'INFO': return 'info'
    default: return 'circle'
  }
}
```

### Step 3: Fix Error Response Format (Backend)
Modify `app/api/routers/data.py`:

```python
# BEFORE (line ~95):
return {"error": f"No data available for {symbol}. Please download data first."}

# AFTER:
raise HTTPException(
    status_code=404,
    detail=f"No data available for {symbol}. Please download data first."
)

# BEFORE (line ~141):
return {
    "error": f"No trained model for {symbol}. Train model first or add to watchlist.",
    "symbol": symbol,
    "hint": "Run: python train_stockformer.py --symbol " + symbol
}

# AFTER:
raise HTTPException(
    status_code=404,
    detail={
        "error": f"No trained model for {symbol}. Train model first or add to watchlist.",
        "symbol": symbol,
        "hint": f"Run: python train_stockformer.py --symbol {symbol}"
    }
)
```

Check other routers for similar patterns:

```bash
# Search for error patterns in backend
grep -n 'return {"error"' app/api/routers/*.py
```

Fix all instances to use `raise HTTPException(status_code=4xx/5xx, detail=...)`.

### Step 4: Update Frontend Error Handling (Optional Enhancement)
If keeping some {error: ...} patterns in BE, update FE to handle both:

```javascript
// In App.jsx, create helper:
const fetchJSON = async (url) => {
  const res = await fetch(url)
  const data = await res.json()

  // Check for error field even in 200 responses (backward compat)
  if (data.error) {
    throw new Error(data.error)
  }

  // Also check HTTP status
  if (!res.ok) {
    throw new Error(data.detail || data.error || 'Request failed')
  }

  return data
}

// Use everywhere:
fetchJSON(`${API_URL}/stock/${symbol}`)
  .then(setStockData)
  .catch(err => {
    console.error(err)
    setError(err.message)
  })
```

### Step 5: Test All Fixed Schemas
1. **Market Regime Test:**
   ```bash
   curl http://localhost:8100/api/market/regime
   # Should return: {"market_regime": "BULL", "confidence": 0.85, ...}
   ```
   - Open frontend, verify Market Regime card shows "BULL" (not "---")
   - Verify Confidence displays (not "undefined")

2. **Smart Signals WARNING Test:**
   ```bash
   curl http://localhost:8100/api/market/smart-signals
   # Should include: {"type": "...", "severity": "WARNING", ...}
   ```
   - Verify WARNING signals display with amber color and warning icon
   - No "undefined" text

3. **Error Response Test:**
   ```bash
   curl -i http://localhost:8100/api/stock/FAKESYMBOL
   # Should return: HTTP 404
   # Body: {"detail": "No data available for FAKESYMBOL..."}
   ```
   - Frontend should show error message, not blank screen
   - Console should log proper error, not silent failure

## Todo List
- [ ] Update regime field from `regime?.regime` to `regime?.market_regime` in App.jsx
- [ ] Update hurst_exponent field to use `confidence` instead
- [ ] Add WARNING case to severity color mapper
- [ ] Add WARNING case to severity icon mapper
- [ ] Search all routers for `return {"error"` patterns
- [ ] Replace with `raise HTTPException(status_code=4xx, detail=...)`
- [ ] Test market regime displays correctly
- [ ] Test WARNING severity displays correctly
- [ ] Test error responses return proper HTTP codes
- [ ] Test FE error handling shows user-friendly messages

## Success Criteria
- [ ] Market regime card shows correct value (no "undefined")
- [ ] Confidence displays instead of hurst_exponent
- [ ] WARNING severity signals render with amber color
- [ ] Error responses return HTTP 404/400/500 (not 200)
- [ ] Frontend catches errors and displays user-friendly messages
- [ ] No console errors related to undefined schema fields
- [ ] All existing features still work (backward compatible)

## Risk Assessment
- **Risk:** Breaking changes affect existing features → Mitigated: Test all views after changes
- **Risk:** Missed error patterns in other endpoints → Mitigated: Grep search across all routers
- **Risk:** FE expects old schema → Mitigated: Use optional chaining (?.) for safety

## Security Considerations
- Proper HTTP error codes improve API security (don't leak info in 200 responses)
- Error messages should not expose sensitive paths or internal details

## Next Steps
After completing this phase:
1. Run full regression test on all views
2. Update plan.md with completion status
3. Proceed to Phase 05 (Integrate unused core endpoints)
4. Consider adding TypeScript for compile-time schema validation (future enhancement)
