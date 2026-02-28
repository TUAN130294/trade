# Phase 02: Block Mock Trading - Safety Gate

**Priority:** P0 CRITICAL
**Status:** Pending
**Blocks:** Phase 06

---

## Context

Codex audit: When LLM agents timeout or error, orchestrator falls back to "mock discussion" which still triggers `place_order()`. System can enter real positions based on fake analysis.

## Key Insights

- `orchestrator.py:466-505`: timeout/error → mock discussion → still calls place_order()
- `orchestrator.py:894-966`: mock fallback generates BUY signals with hardcoded confidence
- No guard between "mock analysis" vs "real analysis" before execution

## Related Code Files

**Modify:**
- `quantum_stock/autonomous/orchestrator.py:466,474,505,557,894,908,966`

## Implementation Steps

1. Add `is_mock: bool` flag to `TeamDiscussion` dataclass
2. In mock fallback path, set `discussion.is_mock = True`
3. Before `_execute_signal()`, check: if `discussion.is_mock` → log warning, skip execution, return HOLD
4. Add env var `ALLOW_MOCK_TRADING=false` (default false) as override for testing
5. Log clearly: "BLOCKED: Mock discussion cannot trigger orders"

## Success Criteria

- [ ] Mock discussions NEVER trigger place_order() in production
- [ ] Clear log message when mock trading is blocked
- [ ] `ALLOW_MOCK_TRADING=true` can override for paper testing only
- [ ] Real LLM/rule-based analysis still executes normally
