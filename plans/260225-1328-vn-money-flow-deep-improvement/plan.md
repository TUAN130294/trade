# VN Money Flow Deep Improvement Plan

**Created:** 2026-02-25
**Sources:** Claude review + Codex audit (combined)
**Goal:** Transform from Western-style technical system → VN money-flow-driven quant

---

## Phase Overview

| Phase | Name | Priority | Status |
|-------|------|----------|--------|
| 01 | Stabilize - Fix Critical Bugs | P0 CRITICAL | ✅ COMPLETED |
| 02 | Safety Gate - Block Mock Trading | P0 CRITICAL | ✅ COMPLETED |
| 03 | Price Unit Standardization | P0 CRITICAL | ✅ COMPLETED |
| 04 | Real Data Layer - Replace Stubs | P1 HIGH | ✅ COMPLETED |
| 05 | Wire Existing Indicators to Agents | P1 HIGH (Quick Win) | ✅ COMPLETED |
| 06 | FlowAgent Integration into Decision Pipeline | P1 HIGH | ✅ COMPLETED |
| 07 | VN FOMO/Behavioral Detection Engine | P2 MEDIUM | ✅ COMPLETED |
| 08 | Enhanced Confidence Scoring | P2 MEDIUM | ✅ COMPLETED |
| 09 | Money Flow-aware Exit Strategy | P2 MEDIUM | ✅ COMPLETED |
| 10 | T+ Compliance & Session Logic Fix | P2 MEDIUM | ✅ COMPLETED |
| 11 | Wyckoff Smart Money Patterns | P3 LATER | ✅ COMPLETED |
| 12 | Test Coverage & Validation | P3 LATER | ✅ COMPLETED |

---

## Dependency Chain

```
Phase 01 (Fix bugs) ──┐
Phase 02 (Mock gate) ──┼──→ Phase 04 (Real data) ──→ Phase 06 (FlowAgent) ──→ Phase 07 (FOMO)
Phase 03 (Price unit) ─┘         │                         │                        │
                                  └──→ Phase 05 (Wire) ────┘                        │
                                                                                     ▼
Phase 10 (T+ fix) ──→ Phase 09 (Exit strategy) ──→ Phase 08 (Confidence) ──→ Phase 12 (Tests)
                                                                                     ▲
                                                            Phase 11 (Wyckoff) ──────┘
```

---

## Phase Details → See individual files:
- [phase-01-fix-critical-router-bugs.md](phase-01-fix-critical-router-bugs.md)
- [phase-02-block-mock-trading.md](phase-02-block-mock-trading.md)
- [phase-03-price-unit-standardization.md](phase-03-price-unit-standardization.md)
- [phase-04-real-data-layer.md](phase-04-real-data-layer.md)
- [phase-05-wire-existing-indicators.md](phase-05-wire-existing-indicators.md)
- [phase-06-flow-agent-pipeline.md](phase-06-flow-agent-pipeline.md)
- [phase-07-vn-fomo-behavioral-engine.md](phase-07-vn-fomo-behavioral-engine.md)
- [phase-08-enhanced-confidence-scoring.md](phase-08-enhanced-confidence-scoring.md)
- [phase-09-money-flow-exit-strategy.md](phase-09-money-flow-exit-strategy.md)
- [phase-10-t-plus-compliance-fix.md](phase-10-t-plus-compliance-fix.md)
- [phase-11-wyckoff-smart-money.md](phase-11-wyckoff-smart-money.md)
- [phase-12-test-coverage-validation.md](phase-12-test-coverage-validation.md)
