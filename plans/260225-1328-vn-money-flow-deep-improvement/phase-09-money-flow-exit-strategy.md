# Phase 09: Money Flow-aware Exit Strategy

**Priority:** P2 MEDIUM
**Status:** ✅ COMPLETED (2026-02-25)
**Depends on:** Phase 06, 07

---

## Context

Current exits are purely price-based (TP/SL/trailing). No exit triggers from flow signals like smart money distribution, foreign panic sell, or FOMO exhaustion.

## New Exit Signals to Add

Add to `PositionExitScheduler._should_exit()` after existing checks:

```
5. SMART_MONEY_DISTRIBUTION: volume high + price sideways + close location declining 3+ days
6. FOREIGN_PANIC_SELL: foreign net sell > 2x 20-day avg for the stock
7. FOMO_EXHAUSTION_EXIT: FOMO signal transitions from PEAK → EXHAUSTION
8. LIQUIDITY_DRY_UP: volume < 30% of 20-day avg after 3+ days held (tay to rút)
```

## Related Code Files

**Modify:**
- `quantum_stock/autonomous/position_exit_scheduler.py:284-327` - Add 4 new exit conditions
- `quantum_stock/autonomous/orchestrator.py` - Pass flow data to exit scheduler

## Implementation Steps

1. Add flow data fetcher to PositionExitScheduler (inject MarketFlowConnector)
2. After T+2 check and before existing exits, add flow-based exit checks
3. Priority order: STOP_LOSS > ATR_STOP > TRAILING > **FOREIGN_PANIC** > **SMART_DISTRIBUTION** > **FOMO_EXHAUSTION** > TAKE_PROFIT > **LIQUIDITY_DRY** > TIME_DECAY
4. Each flow exit logs clearly with evidence data

## Success Criteria

- [x] 4 new flow-based exit signals operational
- [x] Exits triggered by real flow data, not simulated
- [x] Exit logs include flow evidence (foreign net, volume ratio, FOMO state)

## Implementation Notes (2026-02-25)

**Files Modified:**
- `position_exit_scheduler.py` (615 lines): Added flow_fetcher param, async _should_exit(), 4 new flow exits, _get_volume_ratio() helper
- `orchestrator.py` (1252 lines): Integrated MarketFlowConnector, injected flow_fetcher callback

**Exit Signals Added:**
1. FOREIGN_PANIC_SELL: Foreign net sell > 200k shares
2. SMART_MONEY_DISTRIBUTION: Distribution status + held ≥3 days
3. FOMO_EXHAUSTION_EXIT: Peak drop >3% + distribution flow
4. LIQUIDITY_DRY_UP: Volume <30% of 20-day avg

**Priority Order:** STOP_LOSS > ATR_STOP > TRAILING > FOREIGN_PANIC > SMART_DISTRIBUTION > FOMO_EXHAUSTION > TAKE_PROFIT > LIQUIDITY_DRY > TIME_DECAY

**Tests:** All 5 test cases pass (test_flow_exit_signals.py)
