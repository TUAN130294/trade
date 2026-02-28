# VN-Quant Frontend-Backend Sync Fix - Implementation Plan

**Date:** 2026-02-26
**Planner:** planner-260226-2311
**Status:** Ready for Implementation
**Total Effort:** 12h
**Priority:** P1 (Critical + High issues)

---

## Executive Summary

Created comprehensive 8-phase implementation plan to fix critical sync issues between VN-Quant React frontend (1156-line monolith) and FastAPI backend (29 endpoints). Plan addresses security gaps, missing WebSocket integration, schema mismatches, and code quality issues.

**Plan Location:** `D:\testpapertr\plans\260226-2311-fe-be-sync-fix/`

---

## Issues Addressed

### P0 - Critical (6h)
1. **No Authentication** - 0.0.0.0 binding, exposed /api/reset, /api/stop, /api/test/* endpoints
2. **WebSocket Missing** - Backend broadcasts 5 event types, FE has zero WebSocket code
3. **iframe Broken** - /autonomous iframe instead of native React integration

### P1 - High (4h)
4. **Market Regime Schema Mismatch** - BE returns `market_regime`, FE reads `regime` + `hurst_exponent`
5. **Smart-signals Severity Drift** - BE emits "WARNING", FE mapper only handles HIGH|MEDIUM|INFO
6. **Error-shape Mismatch** - BE returns `{error: ...}` with HTTP 200 instead of proper 4xx/5xx codes
7. **Unused Core Endpoints** - /api/orders, /api/positions, /api/trades, /api/discussions not wired to UI
8. **Production Config Missing** - No .env.production, nginx config, or deployment scripts

### P2 - Medium (2h)
9. **App.jsx Monolith** - 1156 lines, 14 components in single file
10. **No State Persistence** - User preferences reset on page reload

---

## Phase Breakdown

### Phase 01: Add Basic Authentication (P0, 2h)
**File:** `phase-01-add-authentication.md`

**Deliverables:**
- `app/core/auth.py` - API key validation dependency
- `.env` - API_KEY generation
- `app/api/routers/trading.py` - Protect 4 dangerous endpoints
- `vn-quant-web/src/App.jsx` - API key prompt, localStorage storage, X-API-Key header

**Success Criteria:**
- 401 returned for protected endpoints without key
- Frontend prompts for key, stores in localStorage
- Read-only endpoints work without auth

---

### Phase 02: Implement WebSocket Client (P0, 2h)
**File:** `phase-02-websocket-client.md`

**Deliverables:**
- `vn-quant-web/src/hooks/use-websocket.js` - Reusable hook with auto-reconnect
- `vn-quant-web/src/components/websocket-feed.jsx` - Live event feed component
- Integration into DashboardView

**Event Types Supported:**
1. opportunity_detected
2. agent_discussion
3. order_executed
4. position_exited
5. system_reset

**Success Criteria:**
- WebSocket connects on mount, shows connection status
- All 5 event types render correctly
- Auto-reconnect with exponential backoff (1s → 30s max)
- Max 100 events limit prevents memory leak

---

### Phase 03: Remove iframe, Integrate Trading View (P0, 2h)
**File:** `phase-03-remove-iframe-integrate-trading.md`

**Deliverables:**
- `vn-quant-web/src/components/trading-view.jsx` - Main trading view
- `vn-quant-web/src/components/portfolio-stats.jsx` - Stats panel (/api/status)
- `vn-quant-web/src/components/positions-table.jsx` - Positions (/api/positions)
- `vn-quant-web/src/components/orders-table.jsx` - Orders (/api/orders)

**Success Criteria:**
- No iframe in App.jsx
- Native React components using backend endpoints
- Auto-refresh every 5s
- WebSocket feed integrated in same view

---

### Phase 04: Fix Schema Mismatches (P1, 1.5h)
**File:** `phase-04-fix-schema-mismatches.md`

**Fixes:**
1. **Market Regime:** FE reads `regime?.market_regime` (not `regime?.regime`)
2. **Smart Signals:** Add WARNING severity case to FE mapper
3. **Error Responses:** BE returns HTTP 404/400/500 (not 200 + {error: ...})

**Files Modified:**
- `vn-quant-web/src/App.jsx` - Fix field names, add WARNING case
- `app/api/routers/data.py` - Use `raise HTTPException(status_code=4xx)`
- `app/api/routers/market.py` - Consistent error handling

**Success Criteria:**
- Market regime displays correctly (no "undefined")
- WARNING signals render with amber color
- Error responses return proper HTTP codes
- FE shows user-friendly error messages

---

### Phase 05: Integrate Core Endpoints (P1, 1.5h)
**File:** `phase-05-integrate-core-endpoints.md`

**Deliverables:**
- `vn-quant-web/src/components/discussions-view.jsx` - Discussion list
- `vn-quant-web/src/components/discussion-detail-modal.jsx` - Full agent breakdown
- `vn-quant-web/src/components/agent-votes-table.jsx` - 6-agent votes table
- Enhancement to OrdersTable with "View Discussion" button

**Success Criteria:**
- Discussions view shows recent 20 discussions
- Click opens modal with agent votes, Chief verdict, confidence scores
- Order → discussion linking works
- Full audit trail transparency

---

### Phase 06: Production Deployment Config (P1, 1h)
**File:** `phase-06-production-config.md`

**Deliverables:**
- `vn-quant-web/.env.production` - VITE_API_URL, VITE_WS_URL
- `nginx/vn-quant.conf` - Reverse proxy config (static files + /api + /ws)
- `.env.production` (root) - Backend CORS, ALLOWED_ORIGINS
- `scripts/deploy-production.sh` - Automated deployment script
- `scripts/vn-quant-backend.service` - Systemd service file

**Architecture:**
```
User → Nginx (80/443)
  ├─ / → Static files (FE build)
  ├─ /api → Proxy to localhost:8100 (BE)
  └─ /ws → WebSocket proxy to localhost:8100
```

**Success Criteria:**
- Frontend builds with `npm run build`
- Nginx proxies /api and /ws correctly
- CORS allows production domain
- Zero code changes between dev and prod

---

### Phase 07: Modularize Frontend (P2, 1.5h)
**File:** `phase-07-modularize-frontend.md`

**Structure:**
```
vn-quant-web/src/
├── App.jsx (~80 lines)
├── components/
│   ├── sidebar.jsx
│   ├── websocket-feed.jsx
│   ├── trading-view.jsx
│   ├── portfolio-stats.jsx
│   ├── positions-table.jsx
│   ├── orders-table.jsx
│   ├── discussions-view.jsx
│   ├── discussion-detail-modal.jsx
│   ├── agent-votes-table.jsx
│   └── stock-chart.jsx
├── views/
│   ├── dashboard-view.jsx
│   ├── analysis-view.jsx
│   ├── radar-view.jsx
│   ├── command-view.jsx
│   ├── backtest-view.jsx
│   ├── predict-view.jsx
│   ├── data-hub-view.jsx
│   └── news-intel-view.jsx
├── hooks/
│   └── use-websocket.js
└── utils/
    └── formatters.js
```

**Success Criteria:**
- App.jsx under 100 lines
- Each file under 200 lines
- All views work identically
- Fast HMR in dev mode

---

### Phase 08: State Persistence (P2, 0.5h)
**File:** `phase-08-state-persistence.md`

**Deliverables:**
- `vn-quant-web/src/utils/storage.js` - localStorage helpers
- `vn-quant-web/src/hooks/use-local-storage.js` - Reusable persistence hook
- `vn-quant-web/src/components/preferences-panel.jsx` - Settings UI

**Persisted State:**
- Active view
- Analysis symbol
- API key
- User preferences (autoRefresh, refreshInterval)

**Success Criteria:**
- State persists across page reloads
- Quota exceeded handled gracefully
- Clear data button works

---

## Implementation Order

### Sequential (Must Complete in Order)
**Phase 01 (Auth) → Phase 04 (Schema) → Phase 05 (Endpoints) → Phase 06 (Prod Config)**

### Parallel (Can Run Simultaneously)
- Phase 02 (WebSocket) + Phase 03 (iframe removal) can run parallel with Phase 01
- Phase 07 (Modularization) + Phase 08 (Persistence) can run parallel after Phase 06

### Recommended Execution Plan
```
Day 1 (6h):
├─ Morning: Phase 01 (Auth) + Phase 02 (WebSocket) - 4h
└─ Afternoon: Phase 03 (iframe removal) - 2h

Day 2 (4h):
├─ Morning: Phase 04 (Schema fixes) - 1.5h
├─ Late Morning: Phase 05 (Endpoints) - 1.5h
└─ Afternoon: Phase 06 (Prod config) - 1h

Day 3 (2h):
├─ Morning: Phase 07 (Modularization) - 1.5h
└─ Afternoon: Phase 08 (Persistence) - 0.5h
```

---

## Key Design Decisions

### 1. Authentication Approach (KISS Principle)
**Decision:** Simple API key in X-API-Key header, not OAuth2/JWT
**Rationale:** Paper trading system, single user, YAGNI - don't over-engineer
**Risk Mitigation:** Recommend proper auth for production deployment

### 2. WebSocket Auto-Reconnect Strategy
**Decision:** Exponential backoff (1s → 2s → 4s → max 30s)
**Rationale:** Balance between responsiveness and server load
**Alternative Considered:** Fixed 5s interval (rejected - too slow on first reconnect)

### 3. Error Response Format
**Decision:** Fix backend to use proper HTTP codes (404, 400, 500)
**Rationale:** Proper REST API semantics, better error handling in FE
**Alternative Considered:** FE checks for {error: ...} field (rejected - not RESTful)

### 4. Component Split Strategy
**Decision:** views/ for full pages, components/ for reusable widgets
**Rationale:** Clear separation, easy navigation for LLMs and developers
**File Naming:** kebab-case for files, PascalCase for React components

### 5. State Persistence Scope
**Decision:** Only non-sensitive data in localStorage (preferences, symbol, API key)
**Rationale:** Paper trading acceptable, production needs encryption
**Security:** Clear on logout, no passwords/tokens

---

## Testing Strategy

### Phase 01 - Auth
- [ ] /api/reset without key → 401
- [ ] /api/reset with key → success
- [ ] Read-only endpoints work without key

### Phase 02 - WebSocket
- [ ] Connection on mount
- [ ] Auto-reconnect after backend restart
- [ ] All 5 event types render correctly
- [ ] Max 100 events enforced

### Phase 03 - Trading View
- [ ] Portfolio stats load
- [ ] Positions table displays correctly
- [ ] Orders table displays correctly
- [ ] Auto-refresh works

### Phase 04 - Schema
- [ ] Market regime shows correct value
- [ ] WARNING severity renders
- [ ] Error responses proper HTTP codes

### Phase 05 - Endpoints
- [ ] Discussions list loads
- [ ] Discussion detail modal opens
- [ ] Order → discussion linking works

### Phase 06 - Production
- [ ] Frontend builds successfully
- [ ] Nginx proxies work locally
- [ ] CORS allows production domain

### Phase 07 - Modularization
- [ ] All 10 views load correctly
- [ ] No broken imports
- [ ] HMR works fast

### Phase 08 - Persistence
- [ ] Active view persists on reload
- [ ] Symbol persists on reload
- [ ] Preferences saved

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Breaking changes in existing features | Medium | High | Full regression testing after each phase |
| CORS issues in production | Low | Medium | Test with real domain before go-live |
| WebSocket fails through nginx | Low | High | Dedicated WS proxy testing |
| Large payloads slow UI | Low | Medium | Max events limit, pagination |
| localStorage quota exceeded | Low | Low | Quota error handling, clear old data |

---

## Unresolved Questions

1. **SSL Certificate:** Use Let's Encrypt or custom SSL for production?
   - Recommendation: Let's Encrypt (free, automated renewal)

2. **Backend Deployment:** Keep on localhost:8100 or separate server?
   - Recommendation: Keep localhost, nginx proxies (security)

3. **Monitoring:** Add logging/monitoring for production?
   - Recommendation: Phase 09+ enhancement (Sentry, LogRocket)

4. **TypeScript Migration:** Worth the effort for type safety?
   - Recommendation: Phase 10+ enhancement (gradual migration)

5. **Mobile App:** Need native iOS/Android apps?
   - Recommendation: Defer, responsive web sufficient

---

## Success Metrics

**Technical:**
- [ ] Zero critical bugs (P0)
- [ ] Zero high bugs (P1)
- [ ] All 8 phases completed
- [ ] 100% existing features working
- [ ] All tests passing

**Code Quality:**
- [ ] All files under 200 lines
- [ ] No duplicate code
- [ ] Clear component hierarchy
- [ ] Consistent naming conventions

**Performance:**
- [ ] API response time < 500ms
- [ ] WebSocket latency < 500ms
- [ ] Frontend build time < 30s
- [ ] HMR update < 500ms

**Security:**
- [ ] Dangerous endpoints protected
- [ ] CORS properly configured
- [ ] No sensitive data in localStorage
- [ ] Proper HTTP error codes

---

## Next Steps

1. **Review plan with team** - Get approval on approach
2. **Set up branch** - `feat/fe-be-sync-fix` from `backup/before-refactor`
3. **Execute Phase 01** - Start with authentication (2h)
4. **Create progress tracker** - Update plan.md after each phase
5. **Final review** - Full regression test after Phase 08

---

## Files Created

**Plan Files:**
- `plan.md` - Overview and phase status tracker
- `phase-01-add-authentication.md`
- `phase-02-websocket-client.md`
- `phase-03-remove-iframe-integrate-trading.md`
- `phase-04-fix-schema-mismatches.md`
- `phase-05-integrate-core-endpoints.md`
- `phase-06-production-config.md`
- `phase-07-modularize-frontend.md`
- `phase-08-state-persistence.md`

**Report:**
- `plans/reports/planner-260226-2311-fe-be-sync-fix-summary.md` (this file)

---

**Plan Status:** ✅ Ready for Implementation
**Estimated Completion:** 3 days (12h total effort)
**Priority:** P1 - High (includes P0 critical issues)

---

*Generated by planner agent on 2026-02-26 23:11*
