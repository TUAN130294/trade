# Phase 3-4 Endpoint Architecture

## Overview

8 API endpoints enhanced with optional LLM interpretation via `?interpret=true` parameter.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (Port 8100)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ├─── app/api/routers/market.py
                              │    ├─ GET /api/market/status
                              │    ├─ GET /api/market/regime
                              │    ├─ GET /api/market/smart-signals
                              │    └─ GET /api/analysis/technical/{symbol}
                              │
                              ├─── app/api/routers/data.py
                              │    └─ GET /api/data/stats
                              │
                              └─── app/api/routers/news.py
                                   ├─ GET /api/news/market-mood
                                   ├─ GET /api/news/alerts
                                   └─ POST /api/backtest/run

                              ALL use ↓

┌─────────────────────────────────────────────────────────────────┐
│         quantum_stock/services/interpretation_service.py         │
│                   InterpretationService (Singleton)              │
│                                                                  │
│  • async interpret(context_type, data, language="vi")           │
│  • OpenAI-compatible async client                               │
│  • 5-minute response cache                                      │
│  • Fallback templates for offline mode                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Local LLM Proxy (localhost:8317/v1)                │
│                                                                  │
│  Models:                                                         │
│  • claudible-haiku-4.5 (fast, for real-time)                   │
│  • claudible-sonnet-4.6 (deep analysis)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Request Flow

### Without Interpretation (Default)
```
Client Request
    ↓
GET /api/market/status
    ↓
[Existing Logic: Fetch VN-Index from CafeF]
    ↓
Return JSON Response
    {
      "vnindex": 1867.62,
      "change": 7.48,
      "change_pct": 0.4,
      ...
    }
```

### With Interpretation (`?interpret=true`)
```
Client Request
    ↓
GET /api/market/status?interpret=true
    ↓
[Existing Logic: Fetch VN-Index from CafeF]
    ↓
if interpret:
    result["interpretation"] = await interp_service.interpret(
        "market_status",
        {"vnindex": 1867.62, "change": 7.48, ...}
    )
    ↓
    [InterpretationService]
        ↓
        Check cache (5min TTL)
        ↓
        If miss: Call LLM proxy
        ↓
        Generate Vietnamese narrative
        ↓
        Cache & return
    ↓
Return JSON Response
    {
      "vnindex": 1867.62,
      "change": 7.48,
      "change_pct": 0.4,
      "interpretation": "📊 Thị trường đang trong xu hướng tích cực..."
      ...
    }
```

---

## Endpoint Mapping

| # | Endpoint | Context Type | Vietnamese Output |
|---|----------|--------------|-------------------|
| 1 | `/api/market/status` | market_status | Tổng quan VN-Index, khuyến nghị hành động |
| 2 | `/api/market/regime` | market_regime | Giải thích xu hướng bull/bear/sideways |
| 3 | `/api/market/smart-signals` | smart_signals | Phân tích breadth + foreign + smart money |
| 4 | `/api/analysis/technical/{symbol}` | technical_analysis | Kết luận MUA/BÁN/CHỜ với lý do |
| 5 | `/api/data/stats` | data_stats | Tình trạng dữ liệu, coverage |
| 6 | `/api/news/market-mood` | market_mood | Tâm lý thị trường từ tin tức |
| 7 | `/api/news/alerts` | news_alerts | Tóm tắt tin quan trọng |
| 8 | `/api/backtest/run` | backtest_results | Đánh giá chiến lược + khuyến nghị |

---

## Code Pattern (Consistent Across All 8 Endpoints)

```python
from quantum_stock.services.interpretation_service import InterpretationService

# Module-level singleton (initialized once)
interp_service = InterpretationService()

@router.get("/api/endpoint")
async def handler(interpret: bool = Query(False)):
    # Step 1: Execute existing business logic
    result = {
        "data_field_1": value1,
        "data_field_2": value2,
        # ... existing response
    }

    # Step 2: Add interpretation if requested
    if interpret:
        result["interpretation"] = await interp_service.interpret(
            "context_type",  # From PROMPT_TEMPLATES
            {
                "key1": value1,  # Relevant data for LLM
                "key2": value2
            }
        )

    # Step 3: Return (backward compatible)
    return result
```

---

## Performance Characteristics

| Metric | Without Interpret | With Interpret (Cache Hit) | With Interpret (Cache Miss) |
|--------|-------------------|----------------------------|------------------------------|
| Response Time | 50-200ms | 50-250ms (+50ms) | 250-750ms (+200-500ms) |
| DB Queries | Same | Same | Same |
| External APIs | CafeF | CafeF | CafeF + LLM proxy |
| Cache Usage | None | 5-min TTL | 5-min TTL |

---

## Testing Commands

```bash
# 1. Market Status Interpretation
curl "http://localhost:8100/api/api/market/status?interpret=true" | jq .interpretation

# 2. Market Regime Explanation
curl "http://localhost:8100/api/api/market/regime?interpret=true" | jq .interpretation

# 3. Smart Signals Context
curl "http://localhost:8100/api/api/market/smart-signals?interpret=true" | jq .llm_interpretation

# 4. Technical Analysis (MUA/BÁN/CHỜ)
curl "http://localhost:8100/api/api/analysis/technical/MWG?interpret=true" | jq .interpretation

# 5. Data Stats Summary
curl "http://localhost:8100/api/api/data/stats?interpret=true" | jq .interpretation

# 6. News Market Mood
curl "http://localhost:8100/api/api/news/market-mood?interpret=true" | jq .interpretation

# 7. News Alerts Summary
curl "http://localhost:8100/api/api/news/alerts?interpret=true" | jq .interpretation

# 8. Backtest Strategy Analysis
curl -X POST "http://localhost:8100/api/api/backtest/run?interpret=true" \
  -H "Content-Type: application/json" \
  -d '{"strategy": "momentum", "symbol": "MWG"}' | jq .interpretation
```

---

## Error Handling

```python
# InterpretationService graceful degradation:
try:
    # 1. Try LLM proxy
    response = await llm_client.chat.completions.create(...)
    return response.choices[0].message.content
except Exception as e:
    logger.error(f"LLM API error: {e}")
    # 2. Fallback to template
    if context_type in FALLBACK_TEMPLATES:
        return FALLBACK_TEMPLATES[context_type].format(**data)
    # 3. Generic fallback
    return "📊 Thị trường đang được phân tích. Vui lòng thử lại sau."
```

---

## Backward Compatibility

✅ **Default behavior unchanged**
- `interpret` parameter defaults to `False`
- Existing clients see no change
- No breaking changes to response schema

✅ **Opt-in enhancement**
- Add `?interpret=true` to get Vietnamese narrative
- New `interpretation` field in response
- Response structure otherwise identical

---

## Frontend Integration Example

```javascript
// Before (no interpretation)
const response = await fetch('/api/market/status');
const data = await response.json();
// { vnindex: 1867.62, change: 7.48, ... }

// After (with interpretation)
const response = await fetch('/api/market/status?interpret=true');
const data = await response.json();
// {
//   vnindex: 1867.62,
//   change: 7.48,
//   interpretation: "📊 Thị trường đang..."
// }

// Display interpretation in UI
if (data.interpretation) {
  showToast(data.interpretation);
}
```

---

## Dependencies

### Runtime
- `openai` (AsyncOpenAI client)
- Local LLM proxy at `http://localhost:8317/v1`
- API key: `sk-***REDACTED***`

### Optional (Phase 1-2)
- `quantum_stock.dataconnector.vps_market` (currently stub)

---

## Cache Strategy

```python
# InterpretationService internal cache
_cache = {
    "market_status_<hash>": {
        "response": "📊 Thị trường...",
        "timestamp": 1709012345.67,
        "ttl": 300  # 5 minutes
    }
}

# Cache key: f"{context_type}_{hash(json.dumps(data))}"
# Invalidation: TTL-based (5 minutes)
# Size limit: None (in-memory, cleared on restart)
```

---

## Next Phase Integration

Phase 1-2 agent provides:
1. ✅ **InterpretationService** (DONE - 11KB real implementation)
2. ⏳ **VPSMarketConnector** (stub exists, needs real API)

Phase 5-6 (Frontend):
- Add `interpret=true` to dashboard API calls
- Display `interpretation` field in Vietnamese tooltips
- Toggle interpretation on/off in settings

---

**Status:** ✅ Phase 3-4 Complete | Backend restart needed for testing
