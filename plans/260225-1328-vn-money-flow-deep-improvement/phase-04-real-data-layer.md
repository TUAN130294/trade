# Phase 04: Real Data Layer - Replace Stubs

**Priority:** P1 HIGH
**Status:** Pending
**Depends on:** Phase 01, 03
**Blocks:** Phase 05, 06, 07

---

## Context

Both reviews found MarketFlowConnector is mostly stub/hardcoded. `get_foreign_flow()`, `get_proprietary_flow()`, `get_market_liquidity()` return fake data. Deep flow endpoints in routers also return simulated/random values.

## Related Code Files

**Modify:**
- `quantum_stock/dataconnector/market_flow.py:37,40` - Replace hardcoded returns with real CafeF data
- `quantum_stock/dataconnector/realtime_market.py` - Enhance CafeF field extraction (tb/ts)
- `app/api/routers/data.py:175,186` - Remove simulated deep flow
- `app/api/routers/market.py:568,872` - Remove hardcoded flow data
- `quantum_stock/web/vn_quant_api.py:1327` - Remove simulated data

## Implementation Steps

1. **Fix `get_foreign_flow()`**: Parse CafeF fields `tb` (foreign buy) / `ts` (foreign sell) per stock. Accumulate 5D/10D/20D rolling sums.
2. **Fix `get_market_liquidity()`**: Sum total traded value from CafeF `n` (volume) * `l` (price) across all stocks.
3. **Enhance `detect_smart_money_footprint()`**: Expand from 3 → 8+ patterns (accumulation, distribution, spring, upthrust, absorption, stopping volume, effort vs result).
4. **Remove random/simulated data** from all production endpoints. If real data unavailable, return `null`/`N/A` with `"data_quality": "unavailable"` flag instead of fake numbers.
5. **Add data quality tracking**: Each response includes `data_source: "cafef_live" | "parquet_historical" | "unavailable"`.

## CafeF Fields to Extract

```
'a':  Symbol              (already used)
'l':  Current price       (already used)
'n':  Total volume        (already used)
'tb': Foreign buy volume  (CRITICAL - partially parsed, spotty)
'ts': Foreign sell volume (CRITICAL - partially parsed)
'b':  Ceiling price       (already used)
'd':  Floor price         (already used)
'g5': Bid price 1         (NEW - for order book analysis)
'g6': Bid vol 1           (NEW)
'g7': Ask price 1         (NEW)
'g8': Ask vol 1           (NEW)
```

## Success Criteria

- [ ] `get_foreign_flow()` returns real per-stock foreign buy/sell from CafeF
- [ ] `get_market_liquidity()` returns real total traded value
- [ ] Zero random/simulated values in production endpoints
- [ ] Data quality flag on every response
- [ ] Smart money detection uses 8+ patterns
