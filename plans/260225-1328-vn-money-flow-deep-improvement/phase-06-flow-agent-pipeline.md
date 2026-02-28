# Phase 06: FlowAgent Integration into Decision Pipeline

**Priority:** P1 HIGH
**Status:** Pending
**Depends on:** Phase 04, 05

---

## Context

Both reviews: FlowAgent exists in code (`flow_agent.py:48`) but is NOT registered in AgentCoordinator. It doesn't participate in the discussion or voting pipeline. Deep flow intelligence module also disconnected.

## Related Code Files

**Modify:**
- `quantum_stock/agents/agent_coordinator.py:59,210` - Register FlowAgent in agent list + voting
- `quantum_stock/agents/flow_agent.py:48` - Ensure analyze() returns proper AgentSignal
- `quantum_stock/agents/deep_flow_intelligence.py:162` - Wire as data provider to FlowAgent

**Reference:**
- `quantum_stock/agents/base_agent.py` - AgentSignal dataclass
- `quantum_stock/agents/chief_agent.py` - How Chief weighs agents

## Implementation Steps

1. Register FlowAgent in `AgentCoordinator.__init__()` agents dict
2. Set FlowAgent weight = 1.3 (highest advisory, above Alex's 1.0)
3. FlowAgent.analyze() must return AgentSignal with:
   - signal_type from flow analysis (not technical)
   - confidence based on: smart_money_index, foreign_flow, cumulative_delta, absorption
   - message explaining flow behavior in Vietnamese
4. Update Chief agent to explicitly consider FlowAgent opinion
5. **Decision gating**: If FlowAgent reports `data_quality = "unavailable"`, Chief defaults to HOLD regardless of other agents
6. Update agent weights:
   ```
   FlowAgent:   1.3  (NEW - highest advisory)
   Alex:        1.0  (reduced from 1.2)
   Bull:        0.8  (reduced from 1.0)
   Bear:        0.8  (reduced from 1.0)
   RiskDoctor:  0.9  (increased from 0.8)
   Chief:       1.5  (unchanged)
   ```

## Success Criteria

- [ ] FlowAgent appears in agent discussions on dashboard
- [ ] FlowAgent weight 1.3 in voting
- [ ] Chief considers flow opinion before verdict
- [ ] Bad data quality → forced HOLD (no trading on garbage)
- [ ] Agent coordinator runs FlowAgent in correct sequence
