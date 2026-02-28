# Code Review: Data Connectors & Services - Real Data Pipeline

**Date:** 2026-02-25
**Scope:** 5 files, ~1250 LOC, real data connectors + services
**Branch:** backup/before-refactor

---

## Overall Assessment

The data connectors DO fetch real data from external APIs (CafeF, VPS, RSS feeds). No hardcoded fake responses masquerading as real data. However, there is one CRITICAL attribute name mismatch that silently disables the VPS foreign flow fallback path inside the CafeF connector, and the `openai` package is missing from `requirements.txt`, which will crash the interpretation service on fresh deploys.

---

## Issues

| ID | SEV | FILE:LINE | DESCRIPTION | FIX |
|----|-----|-----------|-------------|-----|
| DC-01 | **CRITICAL** | `realtime_market.py:129` | VPS fallback in `get_foreign_flow()` checks `self.vps_connector` but the attribute is stored as `self._vps_connector` (line 31). The `hasattr(self, 'vps_connector')` check always returns `False`, so VPS foreign flow data is **never used** even when VPS connector is available. The entire VPS-primary code path (lines 129-148) is dead code. | Change `hasattr(self, 'vps_connector') and self.vps_connector` to `self._vps_connector` and `self._vps_connector.get_foreign_flow(...)` on line 134. |
| DC-02 | **CRITICAL** | `realtime_market.py:133-134` | Even if DC-01 is fixed, calling `asyncio.get_event_loop().run_until_complete()` inside a sync method that will be called from an async FastAPI handler will raise `RuntimeError: This event loop is already running`. The VPS `get_foreign_flow()` is `async`, but `RealTimeMarketConnector.get_foreign_flow()` is sync. | Either make `RealTimeMarketConnector.get_foreign_flow()` async or use `_get_cached_or_fetch` from VPS synchronously (VPS `_get_cached_or_fetch` is sync). |
| DC-03 | **CRITICAL** | `requirements.txt` (missing) | `openai` package is not in `requirements.txt` but `interpretation_service.py:12` does `from openai import AsyncOpenAI`. Fresh `pip install -r requirements.txt` will crash the interpretation service. | Add `openai>=1.6.0` to `requirements.txt`. |
| DC-04 | **HIGH** | `realtime_market.py:278` + `trading.py:192` | `get_stock_price()` returns `float(price) * 1000` (VND). But `trading.py:192` comment says "Already in thousands format from CafeF" and treats it as thousands (fallback on line 196 uses values like `86.0`). This means: if CafeF returns `l=86` (meaning 86,000 VND), `get_stock_price()` returns `86000`, but the trading router expects ~`86.0`. The broker will record prices 1000x too high, corrupting paper trading P&L. | The comment in `trading.py` is wrong. The price IS in VND (already multiplied). Either (a) remove the *1000 in `get_stock_price()` to return raw CafeF format, or (b) fix all callers to expect VND. Recommend (b) since `broker_api.py:563` already expects VND. But `trading.py` fallback prices (line 195-196) need updating to VND (86000 not 86.0). |
| DC-05 | **HIGH** | `market.py:632-633` | Double multiplication risk: `raw_price = connector.get_stock_price(symbol)` returns VND (already *1000), then `base_price = raw_price * 1000 if raw_price and raw_price < 1000 else raw_price`. If CafeF returns `l=86`, `get_stock_price` returns `86000`, which is >1000, so the guard works. But if a penny stock has `l=0.8`, `get_stock_price` returns `800`, which IS <1000, so it becomes `800*1000=800000` -- wrong. | Remove the `*1000` conditional. `get_stock_price()` already returns VND. Just use `base_price = raw_price`. |
| DC-06 | **HIGH** | `interpretation_service.py:26-27` | Hardcoded default API key `sk-***REDACTED***` is committed to source. This is a credential leak, even if it's for a local proxy. | Move to `.env` file, remove default value: `LLM_API_KEY = os.getenv("LLM_API_KEY")`. Validate at init and disable service if not set. |
| DC-07 | **HIGH** | `backtest_engine.py:572` | `from scipy import stats` is a lazy import inside `_calculate_metrics()`. While `scipy` IS in `requirements.txt`, this import only triggers when `total_trades >= 30 && sharpe_ratio != 0`. A user with enough trades will hit this code path, and if scipy is somehow not installed, the entire backtest result calculation fails (not just PSR). The exception is not caught. | Wrap lines 572-573 in try/except or import scipy at module level to fail fast. |
| DC-08 | **MEDIUM** | `realtime_market.py:44` | Cache TTL check uses `(now - self._cache_time).seconds` which only returns the seconds component of the timedelta, not total seconds. If cache is >1 day old, `.seconds` resets and may return a small value, serving stale data. | Use `.total_seconds()` instead of `.seconds` (same fix as VPS connector line 64 already does correctly). |
| DC-09 | **MEDIUM** | `rss_news_fetcher.py:68-84` | `fetch_all_feeds()` has no cache. Every call fetches all 6 RSS feeds sequentially. If called frequently by the API, this creates 6 HTTP requests per call with no throttling. | Add a cache with TTL (e.g., 5 min) similar to other connectors. The `self.cache` dict exists but is never used. |
| DC-10 | **MEDIUM** | `rss_news_fetcher.py:137` | Bare `except:` on date parsing silently swallows errors and defaults to `datetime.now()`. This means unparseable dates won't be noticed and news sorting by date will be inaccurate. | Use `except (TypeError, ValueError, AttributeError):` and log a warning. |
| DC-11 | **MEDIUM** | `interpretation_service.py:186` | `tuple[datetime, str]` type hint uses lowercase `tuple` which requires Python 3.9+. If running on 3.8, this will raise `TypeError`. | Use `from __future__ import annotations` or `Tuple[datetime, str]` from typing (already imported). |
| DC-12 | **MEDIUM** | `backtest_engine.py:343` | `for i, row in df.iterrows()` iterates over entire DataFrame row by row. For large datasets (1000+ rows), this is extremely slow. Pandas `iterrows()` is an anti-pattern for performance. | Use vectorized operations or `df.itertuples()` (5-10x faster). For backtesting, consider vectorized signal processing. |
| DC-13 | **MEDIUM** | `vps_market.py:315-330` | `get_market_depth()` fabricates bid/ask data as `price * 0.99` and `price * 1.01` with volume 1000. This is NOT real order book data but is presented as a function returning market depth. Callers may not realize it's synthetic. | Add `"synthetic": True` flag or rename to `get_estimated_spread()`. The `note` field exists but callers may not check it. |
| DC-14 | **MEDIUM** | `vps_market.py:332-349` | `get_intraday_data()` returns a single current snapshot pretending to be intraday tick data. Same issue as DC-13 -- synthetic data masquerading as real. | Add `"synthetic": True` flag or rename method. |
| DC-15 | **LOW** | `realtime_market.py:303` | Bare `except:` on line 303 (VN-Index change parsing). Should catch specific exceptions. | Use `except (ValueError, IndexError, TypeError):`. |

---

## Positive Observations

1. **Real external APIs**: CafeF (`banggia.cafef.vn`), VPS (`bgapidatafeed.vps.com.vn`), and RSS feeds are genuine Vietnam market data sources. No mock/fake data generators in the main code paths.
2. **Fallback chains**: Connectors have layered fallback (VPS -> CafeF -> Parquet -> Synthetic). Synthetic data is clearly labeled with `data_source` field.
3. **Caching**: Both CafeF and VPS connectors implement TTL-based caching to avoid hammering external APIs.
4. **Error handling**: Most external calls are wrapped in try/except with logging. Fallbacks return sensible defaults.
5. **Price unit documentation**: Comments explain CafeF price format (x1000) throughout.
6. **Backtest engine**: Comprehensive metrics (Sharpe, Sortino, Calmar, SQN, PSR). Well-structured dataclasses. Multiple strategy implementations.
7. **Interpretation service**: Clean template system, caching, graceful fallback messages when LLM is unavailable.

---

## Data Format Compatibility Check

| Connector | Returns | Router Expects | Match? |
|-----------|---------|----------------|--------|
| `realtime_market.get_market_breadth()` | Dict with advancing/declining/etc | `market.py` reads same keys | YES |
| `realtime_market.get_foreign_flow()` | Dict with total_buy/sell/net_value | `market.py` reads same keys | YES |
| `realtime_market.get_stock_price()` | Float in VND (x1000 applied) | `trading.py` expects raw CafeF format | **NO** (DC-04) |
| `realtime_market.get_stock_historical()` | List[Dict] with OHLCV in VND | `market.py/news.py` builds DataFrame | YES |
| `vps_market.get_foreign_flow()` | Dict with net_value_billion | `market.py:984` reads same key | YES |
| `rss_news_fetcher.fetch_all_feeds()` | List[Dict] with headline/sentiment/etc | `news.py` router reads same keys | YES |
| `BacktestEngine.run()` | BacktestResult with to_dict() | `news.py:220` calls strategy methods | YES |

---

## Recommended Actions (Priority Order)

1. **Fix DC-01 + DC-02** -- VPS foreign flow is completely broken in the CafeF connector. Fix attribute name and async handling.
2. **Fix DC-03** -- Add `openai` to `requirements.txt` before any fresh deployment.
3. **Fix DC-04 + DC-05** -- Audit all `get_stock_price()` callers and standardize on VND. Fix `trading.py` fallback prices.
4. **Fix DC-06** -- Remove hardcoded API key from source code.
5. **Fix DC-08** -- Cache TTL bug (`.seconds` vs `.total_seconds()`).
6. **Fix DC-09** -- Add caching to RSS fetcher.

---

## Unresolved Questions

1. Is the CafeF `banggia.cafef.vn/stockhandler.ashx` endpoint still publicly accessible? It's an undocumented API that could break without notice.
2. Does the VPS API require authentication or rate limiting? Current code has no auth headers.
3. Is the LLM proxy at `localhost:8317` expected to be running in production, or is there a remote endpoint for deployment?
4. The `market_flow.py:101-190` proprietary flow estimation is entirely heuristic-based (no real data). Should this be flagged to the UI as "estimated"?
