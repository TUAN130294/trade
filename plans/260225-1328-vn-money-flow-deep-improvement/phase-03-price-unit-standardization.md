# Phase 03: Price Unit Standardization

**Priority:** P0 CRITICAL
**Status:** Pending

---

## Context

Codex: Price units mix VND vs thousands-VND across system. CafeF returns price/1000 (e.g. 25.5 = 25,500 VND). Some modules multiply, others don't → wrong position sizing, PnL.

## Related Code Files

- `quantum_stock/dataconnector/realtime_market.py:241` - CafeF conversion
- `quantum_stock/core/broker_api.py:545,565` - Order execution prices
- `quantum_stock/autonomous/orchestrator.py:809,813` - Price used for orders

## Implementation Steps

1. Audit all price entry points: CafeF, vnstock3, parquet files
2. Establish single convention: ALL prices in VND (not thousands)
3. Ensure CafeF `*1000` conversion happens ONCE at data ingestion layer
4. Add assertion/validation: `assert price > 1000, "Price likely in thousands, not VND"`
5. Fix broker_api position avg_price calculation to match

## Success Criteria

- [ ] Single price unit (VND) across entire pipeline
- [ ] CafeF conversion only at `realtime_market.py` ingestion
- [ ] Position sizing uses correct VND prices
- [ ] PnL calculations accurate
