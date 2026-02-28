# Phase 08: Enhanced Confidence Scoring

**Priority:** P2 MEDIUM
**Status:** ✅ COMPLETED (2026-02-25)
**Depends on:** Phase 06, 07

---

## Context

Current 6-factor confidence scoring has 0% money flow weight. Need to add 3 new factors.

## Current vs Proposed Weights

```
Factor           Current  Proposed
─────────────────────────────────
return           20%      15%
model_accuracy   20%      15%
volatility       15%      10%
volume           15%      10%
technical        15%      10%
market_regime    15%      10%
money_flow        0%      15%   ← NEW: smart money + Wyckoff
foreign_flow      0%      10%   ← NEW: khối ngoại net
fomo_score        0%       5%   ← NEW: inverse FOMO (high FOMO = lower confidence for BUY)
```

## Related Code Files

**Modify:**
- `quantum_stock/core/confidence_scoring.py:83-91` - Add 3 new factors
- `quantum_stock/scanners/model_prediction_scanner.py` - Pass flow data to confidence calc

## Implementation Steps

1. Add `money_flow_score()`: composite of smart_money_index + cumulative_delta + absorption signals. Range 0-1.
2. Add `foreign_flow_score()`: based on 5D accumulated foreign net. Positive net buy = higher score. Range 0-1.
3. Add `fomo_penalty()`: FOMO_PEAK → 0.2 (strong penalty), FOMO_BUILDING → 0.7, NO_FOMO → 1.0. Applied inversely to BUY confidence.
4. Update `MultiFactorConfidence.calculate()` to include all 9 factors.
5. Ensure total weights = 100%.

## Success Criteria

- [x] 9-factor scoring operational
- [x] money_flow factor uses real smart money data
- [x] FOMO penalty reduces BUY confidence at peaks
- [x] Foreign selling reduces confidence for BUY signals

## Implementation Notes (2026-02-25)

**Files Modified:**
- `confidence_scoring.py` (263 lines added): Upgraded from 6 to 9 factors with money_flow (15%), foreign_flow (10%), fomo_penalty (5%)

**New Methods:**
- `_calculate_money_flow_factor()`: Composite of smart_money_index + cumulative_delta + absorption signals
- `_calculate_foreign_flow_factor()`: Based on 5D accumulated foreign net buy/sell
- `_calculate_fomo_penalty_factor()`: FOMO_PEAK → 0.2, FOMO_BUILDING → 0.7, NO_FOMO → 1.0

**Weight Distribution (9 factors, total 100%):**
- return: 15%, model_accuracy: 15%, volatility: 10%, volume: 10%
- technical: 10%, market_regime: 10%, money_flow: 15%, foreign_flow: 10%, fomo_penalty: 5%

**Report:** D:\testpapertr\plans\reports\fullstack-260225-1333-phase-07-08-fomo-confidence.md
