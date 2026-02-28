# Phase 07: VN FOMO/Behavioral Detection Engine

**Priority:** P2 MEDIUM
**Status:** Pending
**Depends on:** Phase 04, 06

---

## Context

VN market retail = ~85% liquidity. FOMO extremely high. Neither review found any FOMO detection. This is the core differentiator for VN-specific quant.

## New Components to Create

### 7A. FOMODetector class

```python
# quantum_stock/indicators/fomo_detector.py
class FOMODetector:
    Signals: FOMO_BUILDING, FOMO_PEAK, FOMO_EXHAUSTION, FOMO_TRAP

    Metrics:
    1. ceiling_chase_velocity: price_change_rate towards ceiling price
    2. volume_acceleration: vol[t] / vol[t-1] > 2.0 for 2+ consecutive bars
    3. rsi_volume_divergence: RSI > 80 + volume spike = peak FOMO
    4. consecutive_gap_ups: 3+ gap-up sessions = retail chase
    5. bid_dominance_ratio: total_bid_vol / total_ask_vol near ceiling
    6. breadth_fomo: advancing% > 80% + turnover spike = market-wide FOMO
```

### 7B. VN Session Analyzer

```python
# quantum_stock/indicators/session_analyzer.py
SESSIONS = {
    'ATO':       (9:00, 9:15),   # Institutional positioning
    'MORNING':   (9:15, 11:30),  # Trend formation (60-65% volume)
    'AFTERNOON': (13:00, 14:30), # Confirmation/reversal
    'ATC':       (14:30, 14:45)  # Smart money final moves
}

Detect:
- ATO volume burst (institutional "dẫn giá")
- Morning vs Afternoon flow reversal (warning signal)
- ATC manipulation: "đập giá cuối phiên" or "kéo giá cuối phiên"
- Queue imbalance near ceiling/floor at ATO/ATC
```

### 7C. Retail Panic Index

```python
# Add to existing custom.py or new file
retail_panic_index():
    Components:
    - breadth crash: declining% > 70%
    - turnover spike: value > 2x 20-day avg
    - floor hit count: stocks hitting floor > 20
    - foreign net sell acceleration
    Output: 0-100 panic score
```

## Related Code Files

**Create:**
- `quantum_stock/indicators/fomo-detector.py`
- `quantum_stock/indicators/session-analyzer.py`

**Modify:**
- `quantum_stock/agents/flow_agent.py` - Import and use FOMO + session signals
- `quantum_stock/agents/bear_agent.py` - Use FOMO_PEAK as sell signal
- `quantum_stock/indicators/custom.py` - Add retail_panic_index

## Success Criteria

- [ ] FOMO detector returns 4 distinct signal states
- [ ] Session analyzer identifies ATO/ATC manipulation
- [ ] FlowAgent uses FOMO signals in analysis
- [ ] BearAgent warns on FOMO_PEAK / FOMO_EXHAUSTION
- [ ] Retail panic index available for exit decisions
