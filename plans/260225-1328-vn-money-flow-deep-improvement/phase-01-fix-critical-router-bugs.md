# Phase 01: Fix Critical Router Bugs

**Priority:** P0 CRITICAL
**Status:** Pending
**Blocks:** Phase 04, 05, 06

---

## Context

Codex audit found routers crash at runtime due to missing imports and undefined symbols. System returns fallback/fake data instead of real analysis.

## Key Insights

- `data.py`: `datetime` not imported, `np` not imported → NameError on `get_market_status()`, `analyze_deep_flow()`
- `market.py`: Missing imports at top, `get_market_status()` crashes, deep flow endpoints return simulated data
- `news.py`: `get_news_alerts()` and `run_backtest()` can crash NameError
- These bugs mean the dashboard shows fake data while appearing functional

## Related Code Files

**Modify:**
- `app/api/routers/data.py:1,56,186` - Add missing imports (datetime, numpy)
- `app/api/routers/market.py:1,23,26,333` - Add missing imports, fix undefined refs
- `app/api/routers/news.py:1,58,210,283` - Add missing imports

## Implementation Steps

1. Read each router file, identify all missing imports
2. Add `import datetime`, `import numpy as np`, and any other missing deps
3. Fix undefined symbol references (variable names, function calls)
4. Test each endpoint returns real data, not fallback
5. Run `python -c "from app.api.routers import data, market, news"` to verify no import errors

## Success Criteria

- [ ] All 3 router files import without error
- [ ] `get_market_status()` returns real datetime
- [ ] `analyze_deep_flow()` uses numpy correctly
- [ ] `get_news_alerts()` doesn't crash
- [ ] No NameError in any endpoint

## Risk Assessment

- Low risk: purely fixing missing imports
- May uncover deeper logic issues once imports are fixed
