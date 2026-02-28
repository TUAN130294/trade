---
title: "VN-Quant Frontend-Backend Sync Fix"
description: "Fix critical sync issues between React FE and FastAPI BE (auth, WebSocket, schema mismatches, iframe removal)"
status: pending
priority: P1
effort: 12h
branch: backup/before-refactor
tags: [critical, frontend, backend, websocket, authentication, integration]
created: 2026-02-26
---

# VN-Quant Frontend-Backend Sync Fix

## Overview
Fix 11 critical/high/medium sync issues between React FE (1156-line App.jsx) and FastAPI BE (29 endpoints). Focus: security (auth), real-time (WebSocket), schema alignment, and component modularization.

## Quick Links
- **Phase 01**: [Add Basic Authentication](phase-01-add-authentication.md) - P0 Critical
- **Phase 02**: [Implement WebSocket Client](phase-02-websocket-client.md) - P0 Critical
- **Phase 03**: [Remove iframe, Integrate Trading View](phase-03-remove-iframe-integrate-trading.md) - P0 Critical
- **Phase 04**: [Fix Schema Mismatches](phase-04-fix-schema-mismatches.md) - P1 High
- **Phase 05**: [Integrate Core Endpoints](phase-05-integrate-core-endpoints.md) - P1 High
- **Phase 06**: [Production Config](phase-06-production-config.md) - P1 High
- **Phase 07**: [Modularize Frontend](phase-07-modularize-frontend.md) - P2 Medium
- **Phase 08**: [State Persistence](phase-08-state-persistence.md) - P2 Medium

## Current State
- **Backend**: FastAPI on :8100, 29 endpoints across 4 routers (trading.py, market.py, data.py, news.py)
- **Frontend**: React 19 + Vite + TailwindCSS, single 1156-line App.jsx, 17 API calls
- **WebSocket**: BE has `/ws/autonomous` (5 event types), FE has ZERO WebSocket code
- **Auth**: 0.0.0.0 binding, exposed dangerous endpoints (/api/reset, /api/stop)
- **Trading View**: iframe to `/autonomous` (broken) instead of proper React integration

## Priority Breakdown
- **P0 (CRITICAL)**: Phases 01-03 (auth, WebSocket, iframe removal) - 6h
- **P1 (HIGH)**: Phases 04-06 (schema fixes, endpoint integration, prod config) - 4h
- **P2 (MEDIUM)**: Phases 07-08 (modularization, persistence) - 2h

## Key Constraints
- YAGNI/KISS/DRY principles (don't over-engineer auth for paper trading)
- Keep files under 200 lines
- kebab-case naming
- Prioritize WORKING features over code beauty
- Windows 11, bash shell

## Phase Status
| Phase | Priority | Status | Effort | Completion |
|-------|----------|--------|--------|------------|
| 01 - Authentication | P0 | Pending | 2h | 0% |
| 02 - WebSocket | P0 | Pending | 2h | 0% |
| 03 - Remove iframe | P0 | Pending | 2h | 0% |
| 04 - Schema fixes | P1 | Pending | 1.5h | 0% |
| 05 - Endpoint integration | P1 | Pending | 1.5h | 0% |
| 06 - Production config | P1 | Pending | 1h | 0% |
| 07 - Modularization | P2 | Pending | 1.5h | 0% |
| 08 - State persistence | P2 | Pending | 0.5h | 0% |

## Dependencies
```
Phase 01 (Auth) ──┐
                  ├──> Phase 04 (Schema) ──> Phase 05 (Endpoints) ──> Phase 06 (Prod Config)
Phase 02 (WS) ────┤                                                        │
                  │                                                        V
Phase 03 (iframe)─┘──> Phase 07 (Modularization) ──────────────────> Phase 08 (Persistence)
```

## Success Criteria
- [ ] Basic auth protects /api/reset, /api/stop, /api/test/*
- [ ] WebSocket client connects to /ws/autonomous, receives 5 event types
- [ ] Trading view integrated into React SPA (no iframe)
- [ ] Market regime schema matches BE (market_regime, not regime+hurst_exponent)
- [ ] Smart-signals severity handles WARNING from BE
- [ ] HTTP error codes proper (not {error: ...} with 200 OK)
- [ ] /api/orders, /api/positions, /api/trades, /api/discussions wired into UI
- [ ] .env.production created with API_URL config
- [ ] App.jsx split into 14+ component files
- [ ] localStorage persists user preferences

## Related Files
**Backend:**
- `run_autonomous_paper_trading.py` - Main FastAPI app, WebSocket endpoint
- `app/api/routers/trading.py` - 11 endpoints (status, orders, positions, trades, discussions, test)
- `app/api/routers/market.py` - 7 endpoints (status, regime, smart-signals, agents, technical, chat, analyze)
- `app/api/routers/data.py` - 5 endpoints (stock/{symbol}, predict/{symbol}, stats, deep_flow)
- `app/api/routers/news.py` - 6 endpoints (status, alerts, market-mood, watchlist, scan, backtest)

**Frontend:**
- `vn-quant-web/src/App.jsx` - 1156 lines, 14 components, 17 API calls
- `vn-quant-web/src/main.jsx` - Entry point

## Next Steps
1. Read phase-01-add-authentication.md, implement basic API key auth
2. Read phase-02-websocket-client.md, implement WebSocket client
3. Read phase-03-remove-iframe-integrate-trading.md, integrate trading view
4. Continue sequentially through remaining phases
5. Update this plan.md with completion % after each phase
