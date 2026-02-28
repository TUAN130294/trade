---
title: "Phase 4: Endpoint Integration - News & Trading"
status: pending
priority: P2
effort: 2h
---

# Phase 4: Endpoint Integration - News & Trading

## Context Links

- InterpretationService: `quantum_stock/services/interpretation_service.py` (Phase 2)
- News router: `app/api/routers/news.py` (~300 lines)
- Trading router: `app/api/routers/trading.py` (~315 lines)
- VPS connector: `quantum_stock/dataconnector/vps_market.py` (Phase 1)

## Overview

Add `?interpret=true` to remaining news and trading endpoints. Also update trading endpoints to use VPS for real-time price data.

## Requirements

### Functional
- 4 endpoints get `?interpret=true` support:
  1. `GET /api/news/market-mood` — sentiment summary
  2. `GET /api/news/alerts` — news synthesis
  3. `POST /api/backtest/run` — strategy recommendation (sonnet)
  4. `POST /api/test/trade` — use VPS price (no interpretation needed)
- Update price fetching in trading router to use VPS

### Non-Functional
- Same pattern as Phase 3 endpoints
- No breaking changes to default responses

## Related Code Files

### Files to Modify
- `app/api/routers/news.py` — 2 endpoints: market-mood, alerts
- `app/api/routers/trading.py` — 1 endpoint: test/trade (VPS price), no interpretation

### Dependencies
- `quantum_stock/services/interpretation_service.py` (Phase 2)
- `quantum_stock/dataconnector/vps_market.py` (Phase 1)

## Implementation Steps

### 1. `/api/news/market-mood` — Sentiment Interpretation

**File**: `app/api/routers/news.py`, function `get_market_mood()`

**Changes**:
- Add `interpret: bool = False` parameter
- If interpret=True:
  ```python
  svc = get_interpretation_service()
  response_data["ai_interpretation"] = await svc.interpret_market_mood({
      "current_mood": mood,
      "positive_news": positive_count,
      "negative_news": negative_count,
      "neutral_news": neutral_count
  })
  ```

### 2. `/api/news/alerts` — News Synthesis

**File**: `app/api/routers/news.py`, function `get_news_alerts()`

**Changes**:
- Add `interpret: bool = False` parameter
- Build alerts summary from first 5 alerts
- If interpret=True:
  ```python
  alerts_text = "\n".join([
      f"- [{a['priority']}] {a['title']} (sentiment: {a.get('sentiment', 0.5):.1f})"
      for a in alerts[:5]
  ])
  response_data["ai_interpretation"] = await svc.interpret_news_alerts({
      "alerts_summary": alerts_text,
      "total": len(alerts)
  })
  ```

### 3. `/api/backtest/run` — Strategy Recommendation (SONNET)

**File**: `app/api/routers/news.py`, function `run_backtest()`

**Changes**:
- Add `interpret: bool = False` to request handling
- Extract interpret from request dict: `interpret = request.get("interpret", False)`
- After computing backtest results, if interpret:
  ```python
  svc = get_interpretation_service()
  result_data["ai_interpretation"] = await svc.interpret_backtest({
      "strategy": strategy_name,
      "symbol": symbol,
      "total_return_pct": result.total_return_pct,
      "sharpe_ratio": result.sharpe_ratio,
      "max_drawdown_pct": result.max_drawdown_pct,
      "win_rate": result.win_rate,
      "profit_factor": result.profit_factor
  })
  ```
- This uses sonnet because backtest evaluation requires deeper analysis

### 4. `/api/test/trade` — VPS Price Integration

**File**: `app/api/routers/trading.py`, function `trigger_test_trade()`

**Changes**:
- Replace CafeF `get_stock_price()` call with VPS connector:
  ```python
  # Try VPS first for accurate price
  try:
      from quantum_stock.dataconnector.vps_market import get_vps_connector
      vps = get_vps_connector()
      real_price = vps.get_stock_price(symbol)
  except Exception:
      real_price = None

  # Fallback to CafeF
  if not real_price:
      from quantum_stock.dataconnector.realtime_market import get_realtime_connector
      connector = get_realtime_connector()
      real_price = connector.get_stock_price(symbol)
  ```
- No interpretation needed for trade execution endpoints

### 5. Additional Endpoints Audit

Other endpoints that could benefit from VPS data (lower priority):
- `GET /api/stock/{symbol}` in `data.py` — historical data, keep CafeF (VPS has no history)
- `GET /api/predict/{symbol}` in `data.py` — model prediction, no interpretation needed (model already provides prediction)
- `GET /api/agents/status` in `market.py` — agent metadata, no interpretation needed

## Todo List

- [ ] Add `interpret` param to `get_market_mood()`
- [ ] Add `interpret` param to `get_news_alerts()`
- [ ] Add `interpret` to `run_backtest()` (from request body)
- [ ] Update `trigger_test_trade()` to use VPS for price
- [ ] Verify default responses unchanged for all 4 endpoints
- [ ] Test with `?interpret=true` for news endpoints

## Success Criteria

- 3 endpoints return `ai_interpretation` when requested
- Backtest interpretation compares strategy to bank deposit benchmark
- Trade endpoint uses VPS price (more accurate)
- All default responses remain identical to pre-change behavior

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Backtest is POST, interpret needs to be in body | Low | Extract from request dict |
| News alerts may be empty | Low | Template handles empty list gracefully |
| VPS price unavailable for trading | Medium | CafeF fallback retained |
