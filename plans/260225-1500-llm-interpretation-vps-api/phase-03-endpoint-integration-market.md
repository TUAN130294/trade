---
title: "Phase 3: Endpoint Integration - Market & Analysis"
status: pending
priority: P1
effort: 3h
---

# Phase 3: Endpoint Integration - Market & Analysis

## Context Links

- InterpretationService: `quantum_stock/services/interpretation_service.py` (Phase 2)
- Market router: `app/api/routers/market.py` (~725 lines, 8 endpoints)
- Data router: `app/api/routers/data.py` (~199 lines, 4 endpoints)
- VPS connector: `quantum_stock/dataconnector/vps_market.py` (Phase 1)

## Overview

Add `?interpret=true` query param to market and analysis endpoints. When enabled, append an `interpretation` field with LLM-generated Vietnamese insight. Also wire VPS connector as primary data source where applicable.

## Key Insights

- `market.py` already has `generate_signal_interpretation()` (rule-based, lines 426-449) — LLM interpretation is a richer replacement but should be additive (keep existing `interpretation` field, add `ai_interpretation`)
- `get_technical_analysis()` is the highest-value endpoint for AI insight (RSI/MACD numbers need MUA/BAN/CHO conclusion)
- `get_agents_analyze()` already uses LLM agents for deep analysis — avoid duplicate calls
- VPS should replace CafeF for real-time price fetches but NOT for historical OHLCV (VPS has no historical API)

## Requirements

### Functional
- 6 endpoints get `?interpret=true` support in this phase:
  1. `GET /api/market/status` — market narrative
  2. `GET /api/market/regime` — regime explanation
  3. `GET /api/market/smart-signals` — signal synthesis
  4. `GET /api/analysis/technical/{symbol}` — MUA/BAN/CHO conclusion
  5. `POST /api/analyze/deep_flow` — flow interpretation
  6. `POST /api/agents/analyze` — already has agents, add summary interpretation
- Each endpoint adds `ai_interpretation: str | null` to response
- Interpretation is null/absent when `interpret` param is false (default)

### Non-Functional
- No additional latency when `interpret=false`
- Interpretation runs AFTER data collection (non-blocking to base response if possible)

## Architecture

Pattern for each endpoint:
```python
@router.get("/api/market/status")
async def get_market_status(interpret: bool = False):
    # ... existing logic to build response_data ...

    if interpret:
        from quantum_stock.services.interpretation_service import get_interpretation_service
        svc = get_interpretation_service()
        response_data["ai_interpretation"] = await svc.interpret_market_status(response_data)

    return response_data
```

## Related Code Files

### Files to Modify
- `app/api/routers/market.py` — 4 endpoints: status, regime, smart-signals, technical
- `app/api/routers/data.py` — 1 endpoint: deep_flow

### Dependencies (from earlier phases)
- `quantum_stock/services/interpretation_service.py` (Phase 2)
- `quantum_stock/dataconnector/vps_market.py` (Phase 1)

## Implementation Steps

### 1. `/api/market/status` — Market Status Interpretation

**File**: `app/api/routers/market.py`, function `get_market_status()`

**Changes**:
- Add `interpret: bool = False` parameter
- After building response dict, if interpret=True:
  ```python
  svc = get_interpretation_service()
  response_data["ai_interpretation"] = await svc.interpret_market_status({
      "vnindex": vnindex,
      "change": change,
      "change_pct": change_pct,
      "is_open": is_open,
      "session_info": session_info
  })
  ```
- Also: replace inline CafeF fetch with VPS connector for real-time VN-Index price

### 2. `/api/market/regime` — Regime Explanation

**File**: `app/api/routers/market.py`, function `get_market_regime()`

**Changes**:
- Add `interpret: bool = False` parameter
- After computing regime, if interpret=True:
  ```python
  response_data["ai_interpretation"] = await svc.interpret_market_regime({
      "market_regime": market_regime,
      "volatility_regime": "NORMAL",
      "confidence": confidence,
      "recommended_strategies": strategies
  })
  ```

### 3. `/api/market/smart-signals` — Signal Synthesis

**File**: `app/api/routers/market.py`, function `get_smart_signals()`

**Changes**:
- Add `interpret: bool = False` parameter
- Build `signals_summary` string from signals list
- If interpret=True:
  ```python
  signals_text = "\n".join([f"- {s['name']}: {s['description']}" for s in signals[:5]])
  response_data["ai_interpretation"] = await svc.interpret_smart_signals({
      "signals_summary": signals_text
  })
  ```
- Also: use VPS `get_foreign_flow()` for more accurate khoi ngoai data

### 4. `/api/analysis/technical/{symbol}` — Technical Analysis (SONNET)

**File**: `app/api/routers/market.py`, function `get_technical_analysis()`

**Changes**:
- Add `interpret: bool = False` parameter
- This is the HIGHEST VALUE interpretation — use sonnet model
- If interpret=True:
  ```python
  response_data["ai_interpretation"] = await svc.interpret_technical({
      "symbol": symbol,
      "current_price": current_price,
      "rsi": current_rsi,
      "support_levels": support_levels,
      "resistance_levels": resistance_levels,
      "patterns": patterns
  })
  ```
- Also: try VPS for current price instead of CafeF

### 5. `/api/analyze/deep_flow` — Deep Flow (SONNET)

**File**: `app/api/routers/data.py`, function `analyze_deep_flow()`

**Changes**:
- Add `interpret: bool = False` parameter
- Use sonnet for deeper analysis
- Build insights summary from insights list
  ```python
  insights_text = "\n".join([f"- {i['type']}: {i['description']} (conf: {i['confidence']})" for i in insights])
  response_data["ai_interpretation"] = await svc.interpret_deep_flow({
      "symbol": symbol,
      "insights_summary": insights_text,
      "flow_score": flow_score,
      "recommendation": recommendation
  })
  ```

### 6. `/api/agents/analyze` — Agent Analysis Summary

**File**: `app/api/routers/market.py`, function `analyze_with_agents()`

**Changes**:
- Add `interpret: bool = False` parameter
- This endpoint already has agent opinions; add a brief LLM summary
- Use haiku (agents already provide deep analysis)
- Build summary from existing verdict/action fields

## Todo List

- [ ] Add `interpret: bool = False` param to `get_market_status()`
- [ ] Add VPS connector for real-time price in `get_market_status()`
- [ ] Add `interpret` param to `get_market_regime()`
- [ ] Add `interpret` param to `get_smart_signals()`
- [ ] Add VPS foreign flow data to `get_smart_signals()`
- [ ] Add `interpret` param to `get_technical_analysis()` (sonnet)
- [ ] Add `interpret` param to `analyze_deep_flow()` (sonnet)
- [ ] Add `interpret` param to `analyze_with_agents()`
- [ ] Verify no regressions when `interpret=false` (default path unchanged)
- [ ] Test each endpoint with `?interpret=true`

## Success Criteria

- All 6 endpoints accept `?interpret=true` query param
- Default behavior (no param) returns identical response as before
- With `interpret=true`, response includes `ai_interpretation` field with Vietnamese text
- Technical analysis interpretation includes MUA/BAN/CHO verdict
- VPS prices used where applicable, with CafeF fallback

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Interpretation slows down response | Medium | Only runs when explicitly requested |
| LLM proxy timeout | Low | LLMClient has 30s timeout + mock fallback |
| Breaking existing API contract | High | `interpret` defaults to False, no change to default response |
| Prompt template data mismatch | Medium | Try/except on format, return empty string on failure |
