# Code Review: Backend API Routers Audit

**Date:** 2026-02-25 16:10
**Reviewer:** code-reviewer
**Scope:** trading.py, data.py, market.py, news.py, state.py
**Focus:** Fake data, crashes, format mismatches, missing error handling, silent fallbacks

---

## ISSUE TABLE

| ID | SEV | FILE:LINE | DESCRIPTION | FIX NEEDED |
|----|-----|-----------|-------------|------------|
| C1 | CRITICAL | trading.py:192-196 | **Price format mismatch in test trade.** `get_stock_price()` returns full VND (e.g. 86000) after `* 1000` conversion (realtime_market.py:278). Comment on line 192 says "Already in thousands format" which is WRONG. Fallback prices on line 195 use thousands format (86.0, 26.2, etc.) while real connector returns full VND. When real data works, quantity calc at line 215 `position_value / current_price / 100 * 100` produces correct qty. When fallback fires, price=86.0 instead of 86000 means qty is ~1000x too large, executing a massively oversized trade. | Fix fallback prices to full VND: `{"MWG": 86000, "HPG": 26200, "SSI": 30350, ...}`. Fix comment. |
| C2 | CRITICAL | trading.py:290-292 | **`datetime` not imported -- server crash on reset.** `trading.py` has NO `datetime` import (only imports from fastapi, typing, pydantic, logging, state). Line 292 calls `datetime.now().isoformat()` which raises `NameError: name 'datetime' is not defined`. The `/api/reset` endpoint will crash every time. | Add `from datetime import datetime` to top of trading.py. |
| C3 | CRITICAL | market.py:141-153 | **Synthetic VN-Index data in market regime endpoint.** When VNINDEX parquet has < 50 rows or is missing, falls back to `np.random.seed(42)` generated synthetic prices starting at 1200. Regime detection (UPTREND/DOWNTREND/SIDEWAYS) is then computed on fake data. Frontend receives this as real analysis with no indicator it is synthetic. | Return error response when no real data, or add `"data_source": "synthetic"` field so frontend can show warning. |
| H1 | HIGH | market.py:44 | **Hardcoded VN-Index default `vnindex = 1249.05`.** If CafeF API fails AND parquet fallback fails, endpoint returns this stale hardcoded value (lines 44, 121-122) with `change=0, change_pct=0`. Frontend displays it as current market data. No flag indicates data is stale/default. | Add `"data_source": "default_stale"` when both sources fail. Use `None` or 0 with an error field instead of a misleading number. |
| H2 | HIGH | market.py:514-582 | **Agents status returns hardcoded fake data.** `get_agents_status()` tries `get_radar_agent_status()` but on any exception (bare `except:` at line 510) or if it returns falsy, falls back to 6 hardcoded agent objects with fabricated accuracy values (0.78-0.95), `signals_today: 0`. Response includes `"data_source": "real-time"` (line 590) even when using the hardcoded fallback. Frontend shows fake accuracy metrics as real. | Set `"data_source": "fallback_static"` when using hardcoded list. Remove fake accuracy values or label them as defaults. |
| H3 | HIGH | market.py:625-651 | **Technical analysis generates synthetic price data.** When CafeF and parquet both fail, generates 100 days of `np.random.randn` synthetic OHLCV (line 640-651). All technical indicators (RSI, support/resistance, patterns) are computed on this fake data. `data_source` is set to `"Synthetic (demo)"` which is good, but the computed indicators are meaningless and could mislead users. | Consider returning an error instead of fake technical analysis. At minimum, ensure frontend checks `data_source` and shows warning. |
| H4 | HIGH | market.py:870-893 | **Agent analysis also generates synthetic data.** Same pattern: when no historical data, generates synthetic prices with `np.random.randn` (line 873-893). Agent messages (buy/sell recommendations) are then based on fake indicators. `data_source` field exists but frontend may not check it. | Same fix: prefer error response or prominent warning when data is synthetic. |
| H5 | HIGH | data.py:168 | **`last_update` is hardcoded to current time.** `get_data_stats()` returns `"last_update": datetime.now().strftime("%Y-%m-%d 17:30")` -- always shows "today at 17:30" regardless of when data was actually last updated. This is misleading since parquet files may be days old. | Use actual file modification time: `max(f.stat().st_mtime for f in parquet_files)` converted to datetime. |
| H6 | HIGH | news.py:37 | **News status always returns `is_running: True`.** There is no actual running scanner process -- this is a static hardcoded value. Frontend shows "Scanner Active" when nothing is actually scanning. | Track actual scanner state, or change to `"is_running": False` with `"mode": "on_demand"`. |
| H7 | HIGH | market.py:252 | **`get_foreign_flow()` called with no args on RealTimeMarketConnector.** This is the CafeF-based connector. But in data.py:203, `vps.get_foreign_flow([symbol])` is called on VPS connector with symbol list (correct). Both APIs have different return schemas (CafeF returns `flow_type` as BUY/SELL, VPS returns more detailed `stocks` array). Code at data.py:205 accesses `flow_data.get('net_value_billion')` which exists in VPS response but not in CafeF fallback. | Standardize access patterns or ensure fallback data includes all expected keys. |
| H8 | HIGH | market.py:102 | **`source` always says "CafeF Real-time"** in market status, even when data came from parquet fallback or hardcoded default. Only set inside the result dict at line 102, never updated during fallback paths (lines 77-88). | Track which source actually provided data and reflect it in the response. |
| M1 | MEDIUM | market.py:210-211 | **Creates new `RealTimeMarketConnector()` instances instead of using singleton.** Three places (lines 211, 631, 838) create `RealTimeMarketConnector()` directly. Singleton `get_realtime_connector()` exists but is not used. Each new instance re-fetches market data from CafeF, wasting API calls and bypassing cache. | Use `get_realtime_connector()` everywhere instead of `RealTimeMarketConnector()`. |
| M2 | MEDIUM | data.py:164-165 | **`total_available: 1730` is hardcoded.** The total number of stocks on Vietnam exchanges may change over time. Coverage percentage is computed against this constant. | Acceptable if periodically verified. Consider fetching from exchange API or making configurable. |
| M3 | MEDIUM | news.py:27-31 | **In-memory news cache with hardcoded watchlist.** `_news_cache` uses a module-level dict with hardcoded `["MWG", "HPG", "FPT", "VNM", "VIC"]`. On server restart, all cached news is lost and watchlist resets. The POST endpoint at line 152 updates the watchlist but it does not persist. | Document this is ephemeral, or persist watchlist to file/DB. |
| M4 | MEDIUM | news.py:119 | **`confidence: 0.72` is a hardcoded constant** in market mood endpoint. Not derived from any calculation. Misleading as it implies statistical confidence. | Either compute from data quality/quantity or remove. |
| M5 | MEDIUM | market.py:510 | **Bare `except:` catches all exceptions silently.** `get_agents_status()` has `except:` with no logging when `get_radar_agent_status()` fails. ImportError, RuntimeError, etc. are all silently swallowed. | Change to `except Exception as e: logger.warning(f"...")`. |
| M6 | MEDIUM | market.py:139 | **Bare `except: pass`** in market regime (line 139) and market status (line 88). Failures are completely silent. | Add logging to identify root causes. |
| M7 | MEDIUM | trading.py:21 | **Non-standard error responses.** `/api/status` returns `{"error": "..."}` with HTTP 200 (line 21). Same pattern in lines 34, 45, 56, 75, 86, 130, 261, 301, 311. FastAPI convention is to raise `HTTPException` for errors. Frontend must check both HTTP status and response body for errors. | Use `raise HTTPException(status_code=503, detail="Orchestrator not initialized")` for true errors. |
| M8 | MEDIUM | market.py:1174 | **Division by zero risk in Risk Doctor message.** `int(2000000 / risk)` where `risk = current_price - support`. If `current_price == support` (possible with synthetic data), `risk = 0` causes `ZeroDivisionError`. | Add guard: `int(2000000 / risk) if risk > 0 else 0`. |
| M9 | MEDIUM | data.py:10 | **Module-level service initialization.** `interp_service = get_interpretation_service()` at import time in data.py, market.py, news.py. If the service requires env vars or API keys not yet loaded, import fails and crashes the entire FastAPI app on startup. | Lazy-init inside endpoints, or handle import errors gracefully. |
| M10 | MEDIUM | trading.py:100 | **Test endpoint imports from scanner at runtime.** `from quantum_stock.scanners.model_prediction_scanner import ModelPrediction` inside endpoint. If scanner module has import errors (e.g., missing torch), error is caught but generic `{"error": str(e)}` is returned with HTTP 200. | Not inherently broken but could give confusing errors. Document dependencies. |
| L1 | LOW | state.py:8-17 | **`ALLOWED_ORIGINS` is hardcoded localhost only.** No production origins. If deployed, CORS will block all requests from production frontend. | Add production origins or make configurable via env vars. |
| L2 | LOW | market.py:1230 | **market.py is 1230 lines.** Exceeds the 200-line file size guideline significantly. Contains market status, regime, smart signals, agents status, technical analysis, agent chat, and agent analysis -- all in one file. | Split into: `market_status.py`, `market_signals.py`, `market_analysis.py`, `agent_endpoints.py`. |
| L3 | LOW | data.py:10-16, news.py:9-15 | **Duplicate `QueryRequest` and `AnalyzeRequest` models** defined in trading.py, data.py, market.py, and news.py. Four identical copies. | Extract to shared `app/api/schemas.py` module. |

---

## SUMMARY

**Total Issues Found:** 22
- **Critical:** 3 (C1-C3)
- **High:** 8 (H1-H8)
- **Medium:** 10 (M1-M10)
- **Low:** 3 (L1-L3)

### Most Dangerous Issues

1. **C2 -- Reset endpoint crashes (NameError):** Every call to `/api/reset` will throw `NameError: name 'datetime' is not defined`. This is a guaranteed crash.

2. **C1 -- Price format mismatch in test trades:** When CafeF API is down, fallback prices are 1000x too small (86.0 vs 86000). This would execute trades with ~1000x the intended quantity, potentially "buying" the entire portfolio value in a single stock.

3. **C3 + H3 + H4 -- Synthetic data masquerading as analysis:** Multiple endpoints silently fall back to random-generated data and run full technical analysis on it, producing meaningless buy/sell signals that the frontend displays as real analysis.

### Positive Observations

- Data endpoints (stock, predict) correctly refuse to return fake data and return errors instead -- good pattern
- CafeF -> Parquet -> Error fallback chain in data.py is well-structured
- `data_source` field is included in many responses, enabling frontend to distinguish data quality
- Foreign flow integration with VPS API provides real data with proper error handling
- LLM agent analysis with parallel execution is well-architected
- Market breadth, bull trap detection, circuit breaker alerts use real CafeF data

---

## Unresolved Questions

1. Is there a frontend check for `data_source: "Synthetic (demo)"` to warn users? If not, H3/H4 become critical.
2. Does the deployment environment have all Python dependencies (torch, pandas, numpy, requests) installed? Missing any would cause cascading import failures.
3. Are the VPS API credentials configured in production? `get_vps_connector()` behavior when unconfigured is unclear.
