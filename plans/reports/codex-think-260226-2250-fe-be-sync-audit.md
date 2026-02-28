# FE/BE Sync Audit Report - Codex + Claude Peer Debate

**Date:** 2026-02-26
**Method:** Codex Think-About (2 rounds peer debate)
**Confidence:** HIGH
**Branch:** backup/before-refactor

---

## Summary

FE→BE endpoint sync: **17/17 matched** (all FE calls have BE routes).
BE→FE coverage: **17/29** (12 endpoints unused by React SPA).
Features working E2E: **~60%** (schema mismatches, WebSocket gap, iframe broken).

---

## CRITICAL (P0)

### 1. No Authentication
- `0.0.0.0` binding + Docker port mapping
- `/api/reset`, `/api/stop`, `/api/test/trade` exposed without auth
- CORS allowlist != authentication
- **Impact:** Anyone on network can wipe portfolio, kill system, place trades

### 2. WebSocket Not Implemented in FE
- BE broadcasts 5 event types via `/ws/autonomous`
- FE has 0 lines of `new WebSocket()`
- Vite proxy configured but unused
- **Impact:** Dashboard not real-time, users miss trades/exits

### 3. `/autonomous` iframe Broken
- BE has NO `/autonomous` route (only `/` and `/ws/autonomous`)
- Nginx SPA fallback serves React shell
- iframe loads wrong content
- **Impact:** Trading tab unusable

---

## HIGH (P1)

### 4. Market Regime Schema Mismatch
- BE returns `market_regime`, FE reads `regime` + `hurst_exponent`
- **Files:** market.py:130, App.jsx:312-317

### 5. Smart-signals Severity Enum Drift
- BE emits `"WARNING"`, FE mapper only handles `HIGH|MEDIUM|INFO`
- **Files:** market.py:221, App.jsx:280

### 6. Error-shape Mismatch on 200 Responses
- BE returns `{error: ...}` with HTTP 200
- FE parses as success object → silent failures
- **Files:** data.py:95,141, App.jsx:750

### 7. 12 BE Endpoints Unused by SPA
- Core: orders, positions, trades, discussions
- Admin: reset, stop
- Debug: test/opportunity, test/trade
- Unused: agents/chat, discussion/{id}, order/{id}/discussion, status

### 8. No Production Deployment Config
- `API_URL = '/api'` hardcoded
- No `.env.production`
- Vite proxy = dev only

---

## MEDIUM (P2)

### 9. Monolithic App.jsx (1156 lines)
### 10. Backtest endpoint in news.py router (misplaced)
### 11. No FE state persistence (no localStorage)
### 12. CafeF single point of failure

---

## LOW (P3)

### 13. No React Router (state-based switching)
### 14. No FE tests
### 15. Docs/code drift (README says /api/portfolio exists - it doesn't)

---

## Endpoint Sync Matrix

| Endpoint | FE Uses | Status |
|----------|---------|--------|
| GET /api/market/status | YES | Working |
| GET /api/market/regime | YES | Schema mismatch |
| GET /api/market/smart-signals | YES | Severity enum drift |
| GET /api/stock/{symbol} | YES | Working |
| GET /api/predict/{symbol} | YES | Error handling issue |
| GET /api/data/stats | YES | Working |
| GET /api/analysis/technical/{symbol} | YES | Working |
| GET /api/agents/status | YES | Working |
| GET /api/news/status | YES | Working |
| GET /api/news/alerts | YES | Working |
| GET /api/news/market-mood | YES | Working |
| GET /api/news/watchlist | YES | Working |
| POST /api/analyze/deep_flow | YES | Working |
| POST /api/agents/analyze | YES | Working |
| POST /api/backtest/run | YES | Working |
| POST /api/news/scan | YES | Working |
| POST /api/news/watchlist | YES | Working |
| GET /api/status | NO | Unused |
| GET /api/orders | NO | iframe only |
| GET /api/positions | NO | iframe only |
| GET /api/trades | NO | iframe only |
| GET /api/discussions | NO | Unused |
| GET /api/discussion/{id} | NO | Unused |
| GET /api/order/{id}/discussion | NO | Unused |
| POST /api/agents/chat | NO | Unused |
| POST /api/test/opportunity | NO | Debug |
| POST /api/test/trade | NO | Debug |
| POST /api/reset | NO | No UI |
| POST /api/stop | NO | No UI |
| WS /ws/autonomous | NO | Zero WS code |

---

## Open Questions

1. Is `/autonomous` a separate Flask page or should it integrate into React SPA?
2. Keep or remove `POST /api/agents/chat`?
3. Auth strategy for production? (JWT, API key, session?)
4. RBAC for admin endpoints (reset, stop)?
