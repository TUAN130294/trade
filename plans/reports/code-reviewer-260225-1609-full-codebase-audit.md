# Full Codebase Audit Report
**Date:** 2026-02-25 16:09 | **Branch:** backup/before-refactor | **Audited by:** 3 parallel code-reviewer agents

## Executive Summary

**Total Issues Found: 44** | 9 Critical | 12 High | 16 Medium | 7 Low

The connectors DO fetch real data from genuine Vietnamese market APIs (CafeF, VPS, RSS). However, many endpoints silently fall back to synthetic/hardcoded data when APIs fail, and the frontend can't distinguish real vs fake. Three bugs will crash the server immediately.

---

## CRITICAL Issues (9) — Must Fix Before Deploy

| # | ID | Location | Issue | Fix |
|---|-----|----------|-------|-----|
| 1 | C1 | `trading.py:195` | **Price format mismatch.** Fallback prices in thousands (`86.0`) but `get_stock_price()` returns full VND (`86000`). Creates 1000x oversized trades when CafeF is down. | Change to `{"MWG": 86000, "HPG": 26200, ...}` |
| 2 | C2 | `trading.py:290` | **Reset endpoint crashes.** No `datetime` import in `trading.py`. `datetime.now()` raises `NameError`. | Add `from datetime import datetime` |
| 3 | C3 | `market.py:141-153` | **Synthetic VN-Index in regime detection.** Uses `np.random.seed(42)` fake data when parquet missing. Frontend shows as real analysis. | Return error instead of fake data |
| 4 | DC-01 | `realtime_market.py:129` | **VPS fallback dead code.** Checks `self.vps_connector` but attr is `self._vps_connector`. VPS data never used. | Fix attribute name |
| 5 | DC-02 | `realtime_market.py:133` | **Async in sync context.** `run_until_complete()` inside running FastAPI loop = `RuntimeError`. | Make method async or use sync VPS call |
| 6 | DC-03 | `requirements.txt` | **Missing `openai` package.** Interpretation service imports it but not in requirements. Fresh deploy crashes. | Add `openai>=1.6.0` |
| 7 | F-01 | `App.jsx:749-758` | **Fake predictions shown as real.** PredictView renders fabricated data (price=25000, direction=UP, confidence=0.75) when API fails. Users may trade on fake signals. | Show error state, not fake data |
| 8 | F-02 | `App.jsx:319-325` | **Hardcoded "Circuit Breaker: NORMAL".** No API call. Always shows green even during market halt. | Fetch from smart-signals endpoint |
| 9 | F-03 | `App.jsx:330-336` | **Hardcoded "Active Agents: 8/8".** No API call. Always shows all agents online. | Fetch from `/api/agents/status` |

---

## HIGH Priority Issues (12)

| # | ID | Location | Issue |
|---|-----|----------|-------|
| 1 | H1 | `market.py:44` | Hardcoded `vnindex = 1249.05` when both API sources fail |
| 2 | H2 | `market.py:514-582` | Agent status returns fake accuracy (0.78-0.95) labeled as `"data_source": "real-time"` |
| 3 | H3 | `market.py:625-651` | Technical analysis runs on synthetic random data when stock data missing |
| 4 | H4 | `market.py:870-893` | Agent analysis generates recommendations on synthetic data |
| 5 | H5 | `data.py:168` | `last_update` always shows "today 17:30" regardless of actual data age |
| 6 | H6 | `news.py:37` | `is_running: True` hardcoded — no scanner actually running |
| 7 | H7 | `market.py:252` | CafeF vs VPS foreign flow return different schemas |
| 8 | H8 | `market.py:102` | `source` always says "CafeF Real-time" even when data is from parquet/hardcoded |
| 9 | DC-04 | `trading.py:192` | Price *1000 from connector but comment says "thousands format" — corrupts P&L |
| 10 | DC-05 | `market.py:632` | Double multiplication for penny stocks: `raw_price * 1000 if < 1000` but already VND |
| 11 | DC-06 | `interpretation_service.py:26` | Hardcoded API key in source code (credential leak) |
| 12 | F-05 | `App.jsx:1134` | Trading iframe hardcodes `localhost:8001` — breaks in production |

---

## MEDIUM Priority Issues (16)

| # | ID | Location | Issue |
|---|-----|----------|-------|
| 1 | F-06 | `App.jsx:361` | Smart signals render `undefined` for missing `action` field |
| 2 | F-08 | `App.jsx:268` | No `.ok` check on smart-signals fetch |
| 3 | F-09 | `App.jsx:448` | No error handling on `/api/agents/status` fetch |
| 4 | F-10 | `App.jsx:517` | No `.ok` check on `/api/agents/analyze` POST |
| 5 | F-11 | `App.jsx:1112` | No `.catch()` on market status/regime fetches — unhandled rejection |
| 6 | F-12 | `App.jsx:1118` | Stock data fetch passes raw response without shape validation |
| 7 | F-13 | `App.jsx:49` | Chart filter drops candles where OHLC = 0 (truthiness check) |
| 8 | F-14 | `App.jsx:96` | Technical panel crashes on missing nested fields (no null checks) |
| 9 | F-15 | `App.jsx:696` | Win rate shows decimal as % (0.65% instead of 65%) |
| 10 | F-16 | `App.jsx:432` | Deep flow uses `alert()` — blocks UI thread |
| 11 | M1 | `market.py:211,631,838` | Creates new connector instances instead of singleton |
| 12 | M5 | `market.py:510` | Bare `except:` swallows all errors silently |
| 13 | M7 | `trading.py:21+` | Returns `{"error": ...}` with HTTP 200 instead of proper status codes |
| 14 | M8 | `market.py:1174` | Division by zero: `2000000 / risk` when risk=0 |
| 15 | DC-08 | `realtime_market.py:44` | Cache TTL uses `.seconds` not `.total_seconds()` — stale data after 24h |
| 16 | DC-09 | `rss_news_fetcher.py:68` | No caching — 6 sequential HTTP requests every call |

---

## Positive Observations

- All data connectors fetch from genuine Vietnamese market APIs (CafeF, VPS, VietStock RSS)
- `data.py` stock/predict endpoints correctly refuse fake data and return errors
- `data_source` field included in many responses for differentiation
- Backtest engine uses real OHLCV data with comprehensive metrics
- Frontend API_URL uses relative `/api` path (works through nginx proxy)
- All frontend fetch URLs match existing backend routes (no 404s)
- LLM agent parallel execution well-architected
- Chart component has proper cleanup on unmount

---

## Unresolved Questions

1. Does frontend check `data_source: "Synthetic (demo)"` to warn users?
2. Is the CafeF undocumented endpoint (`banggia.cafef.vn`) stable long-term?
3. VPS API rate limits / auth requirements?
4. LLM proxy at `localhost:8317` — expected in production?
5. Are all Python dependencies (torch, scipy, openai) installed in deploy?
