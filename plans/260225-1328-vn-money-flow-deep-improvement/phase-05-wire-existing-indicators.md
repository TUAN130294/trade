# Phase 05: Wire Existing Indicators to Agents (Quick Win)

**Priority:** P1 HIGH (Quick Win - no new code needed)
**Status:** Pending
**Depends on:** Phase 04

---

## Context

Both reviews found 14+ indicators already coded but NOT used by any agent. Just wiring them in gives immediate analysis improvement.

## Unused Indicators to Wire

| File | Function | Wire to Agent | Priority |
|------|----------|---------------|----------|
| `orderflow.py` | `cumulative_delta()` | AnalystAgent, FlowAgent | CRITICAL |
| `orderflow.py` | `absorption_exhaustion()` | FlowAgent | CRITICAL |
| `orderflow.py` | `foreign_flow_analysis()` | FlowAgent | CRITICAL |
| `orderflow.py` | `smart_money_index()` | FlowAgent, ChiefAgent | CRITICAL |
| `orderflow.py` | `vwap_bands()` | AnalystAgent | HIGH |
| `custom.py` | `foreign_flow_indicator()` | FlowAgent | CRITICAL |
| `custom.py` | `smart_money_index()` | FlowAgent | CRITICAL |
| `custom.py` | `vn_market_strength()` | BullAgent, BearAgent | HIGH |
| `custom.py` | `accumulation_distribution_zone()` | FlowAgent | HIGH |
| `custom.py` | `vn_sector_rotation()` | FlowAgent | HIGH |
| `custom.py` | `ceiling_floor_detector()` | BearAgent | MEDIUM |
| `volume.py` | `twiggs_money_flow()` | AnalystAgent | HIGH |
| `volume.py` | `klinger_oscillator()` | AnalystAgent | MEDIUM |
| `volume.py` | `volume_zone_oscillator()` | AnalystAgent | MEDIUM |

## Related Code Files

**Modify:**
- `quantum_stock/agents/analyst_agent.py:35` - Add money_flow dimension (currently 0%)
- `quantum_stock/agents/bull_agent.py` - Add vn_market_strength check
- `quantum_stock/agents/bear_agent.py` - Add ceiling_floor_detector, distribution check
- `quantum_stock/agents/flow_agent.py:48` - Wire all flow indicators

## Implementation Steps

1. In `analyst_agent.py`, add `_analyze_money_flow()` method calling orderflow + custom indicators
2. Update weights: trend 20%, momentum 20%, volume 15%, pattern 10%, levels 10%, **money_flow 25%**
3. In `bull_agent.py`, add `vn_market_strength()` to `_analyze_volume()`
4. In `bear_agent.py`, add `ceiling_floor_detector()` and `accumulation_distribution_zone()` (inverse for distribution warning)
5. Ensure FlowAgent calls all 10 indicators listed above

## Success Criteria

- [ ] AnalystAgent uses money_flow in scoring (25% weight)
- [ ] All 14 indicators wired to at least one agent
- [ ] Agent discussions reference flow data in their analysis text
