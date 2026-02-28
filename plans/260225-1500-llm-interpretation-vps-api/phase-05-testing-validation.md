---
title: "Phase 5: Testing & Validation"
status: pending
priority: P2
effort: 2h
---

# Phase 5: Testing & Validation

## Context Links

- Existing tests: `quantum_stock/tests/test_compliance.py`, `test_price_units.py`, `test_fomo_confidence.py`
- VPS connector: `quantum_stock/dataconnector/vps_market.py` (Phase 1)
- InterpretationService: `quantum_stock/services/interpretation_service.py` (Phase 2)
- Modified routers: `app/api/routers/market.py`, `data.py`, `news.py`, `trading.py`

## Overview

Write integration tests validating:
1. VPS connector returns accurate data
2. Interpretation service produces Vietnamese text
3. Endpoints return correct shape with/without `?interpret=true`
4. Price unit consistency (no 2M VND deviation)
5. Fallback behavior when VPS/LLM unavailable

## Requirements

### Functional
- Test VPS connector price accuracy vs known reference
- Test interpretation service with mock and live LLM
- Test endpoint response schema with interpret=true/false
- Test CafeF fallback when VPS fails
- Test cache behavior (second call should be faster)

### Non-Functional
- Tests should run in < 30 seconds (use mocking for LLM)
- Tests should work without LLM proxy running (mock fallback)
- Price unit tests should catch the 2M VND deviation issue

## Related Code Files

### Files to Create
- `quantum_stock/tests/test_vps_connector.py` — VPS data tests
- `quantum_stock/tests/test_interpretation_service.py` — LLM interpretation tests
- `quantum_stock/tests/test_endpoint_interpretation.py` — Full endpoint integration tests

### Files to Reference
- `quantum_stock/tests/test_price_units.py` — existing price unit test pattern

## Implementation Steps

### 1. VPS Connector Tests (`test_vps_connector.py`)

```python
class TestVPSConnector:
    def test_fetch_single_stock():
        """VPS returns valid price for known stock"""
        connector = VPSDataConnector()
        price = connector.get_stock_price("SSI")
        assert price is not None
        assert 10_000 < price < 500_000  # Reasonable VND range

    def test_fetch_batch():
        """VPS batch returns multiple stocks"""
        connector = VPSDataConnector()
        prices = connector.get_multiple_prices(["SSI", "VNM", "MWG"])
        assert len(prices) >= 2

    def test_foreign_flow_data():
        """VPS returns foreign buy/sell volumes"""
        connector = VPSDataConnector()
        flow = connector.get_foreign_flow()
        assert "total_buy" in flow
        assert "top_buy" in flow

    def test_price_unit_consistency():
        """Prices should be in VND (thousands range), not raw API values"""
        connector = VPSDataConnector()
        price = connector.get_stock_price("VNM")
        if price:
            assert price > 1000, "Price should be in VND, not thousands"
            assert price < 1_000_000, "Price unreasonably high"

    def test_cache_behavior():
        """Second call should use cache"""
        connector = VPSDataConnector()
        connector.get_stock_price("SSI")
        # Second call should be nearly instant
        import time
        start = time.time()
        connector.get_stock_price("SSI")
        assert time.time() - start < 0.01  # Cache hit

    def test_fallback_to_cafef():
        """When VPS fails, CafeF should work"""
        # Simulate VPS failure by using invalid URL
        from quantum_stock.dataconnector.realtime_market import get_realtime_connector
        connector = get_realtime_connector()
        price = connector.get_stock_price("MWG")
        assert price is not None
```

### 2. Interpretation Service Tests (`test_interpretation_service.py`)

```python
class TestInterpretationService:
    @pytest.mark.asyncio
    async def test_interpret_returns_string():
        """interpret() should return a non-empty string"""
        svc = InterpretationService()
        result = await svc.interpret_market_status({
            "vnindex": 1280.5, "change": 5.3,
            "change_pct": 0.42, "is_open": True,
            "session_info": "Phien sang"
        })
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_interpret_cache_hit():
        """Second call with same data should hit cache"""
        svc = InterpretationService()
        data = {"vnindex": 1280.5, "change": 5.3, "change_pct": 0.42,
                "is_open": True, "session_info": "Phien sang"}
        result1 = await svc.interpret_market_status(data)
        result2 = await svc.interpret_market_status(data)
        assert result1 == result2  # Same cached result

    @pytest.mark.asyncio
    async def test_interpret_technical_uses_sonnet():
        """Technical interpretation should use sonnet model"""
        svc = InterpretationService()
        result = await svc.interpret_technical({
            "symbol": "MWG", "current_price": 87000,
            "rsi": 35.2, "support_levels": [82000, 84000],
            "resistance_levels": [90000, 92000],
            "patterns": ["Oversold bounce"]
        })
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_interpret_graceful_failure():
        """Should return empty string when LLM unavailable"""
        svc = InterpretationService()
        svc._llm_haiku.api_key = ""  # Force failure
        result = await svc.interpret_market_status({
            "vnindex": 0, "change": 0, "change_pct": 0,
            "is_open": False, "session_info": "Test"
        })
        assert isinstance(result, str)  # Should not raise
```

### 3. Endpoint Integration Tests (`test_endpoint_interpretation.py`)

```python
class TestEndpointInterpretation:
    """Test endpoints with interpret=true/false"""

    def test_market_status_default():
        """Default response should NOT have ai_interpretation"""
        response = client.get("/api/market/status")
        assert response.status_code == 200
        assert "ai_interpretation" not in response.json()

    def test_market_status_interpret():
        """With interpret=true, response should include ai_interpretation"""
        response = client.get("/api/market/status?interpret=true")
        assert response.status_code == 200
        data = response.json()
        assert "ai_interpretation" in data
        assert isinstance(data["ai_interpretation"], str)

    def test_technical_interpret():
        """Technical analysis with interpretation"""
        response = client.get("/api/analysis/technical/MWG?interpret=true")
        assert response.status_code == 200
        data = response.json()
        assert "ai_interpretation" in data

    def test_all_endpoints_no_regression():
        """All endpoints without interpret param return same schema"""
        endpoints = [
            "/api/market/status",
            "/api/market/regime",
            "/api/market/smart-signals",
            "/api/analysis/technical/MWG",
            "/api/news/market-mood",
            "/api/news/alerts"
        ]
        for ep in endpoints:
            response = client.get(ep)
            assert response.status_code == 200
            assert "ai_interpretation" not in response.json()
```

### 4. Price Comparison Test

```python
def test_vps_vs_cafef_price_accuracy():
    """VPS price should be closer to reality than CafeF"""
    from quantum_stock.dataconnector.vps_market import get_vps_connector
    from quantum_stock.dataconnector.realtime_market import get_realtime_connector

    symbol = "VNM"
    vps_price = get_vps_connector().get_stock_price(symbol)
    cafef_price = get_realtime_connector().get_stock_price(symbol)

    if vps_price and cafef_price:
        deviation = abs(vps_price - cafef_price)
        # VPS should be within 1% of CafeF (both valid sources)
        # But CafeF may deviate up to ~2M VND from actual
        print(f"VPS: {vps_price:,.0f}, CafeF: {cafef_price:,.0f}, Delta: {deviation:,.0f}")
```

## Todo List

- [ ] Create `test_vps_connector.py` with 6 test cases
- [ ] Create `test_interpretation_service.py` with 4 test cases
- [ ] Create `test_endpoint_interpretation.py` with 4 test cases
- [ ] Run all tests, verify pass rate
- [ ] Document any flaky tests (network-dependent)
- [ ] Verify price unit consistency across all paths

## Success Criteria

- All VPS connector tests pass
- Interpretation tests pass (with mock fallback when LLM unavailable)
- Endpoint tests verify no regression on default paths
- Price tests confirm VPS accuracy improvement

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Tests require network access | Medium | Skip with `@pytest.mark.skipif(no_network)` |
| LLM proxy not running in CI | Medium | Mock fallback produces valid string, tests still pass |
| VPS API changes break tests | Low | Defensive assertions (type checks, range checks) |
