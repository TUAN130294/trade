---
title: "Phase 1: VPS Data Connector"
status: pending
priority: P1
effort: 2h
---

# Phase 1: VPS Data Connector

## Context Links

- Current CafeF connector: `quantum_stock/dataconnector/realtime_market.py`
- VPS API: `https://bgapidatafeed.vps.com.vn/getliststockdata/{SYMBOLS}`
- Existing cache utility: `quantum_stock/utils/cache.py`

## Overview

Create `VPSDataConnector` as primary price/market data source. VPS API returns accurate real-time prices with foreign flow data that CafeF lacks. Keep `RealTimeMarketConnector` as fallback.

## Key Insights

- VPS API response fields (confirmed via live fetch):
  - `sym`: symbol, `lastPrice`: real-time price (in thousands, e.g., 32.15 = 32,150 VND)
  - `fBVol`: foreign buy volume (string), `fSVolume`: foreign sell volume (string)
  - `lot`: total traded volume, `changePc`: % change (string)
  - `highPrice`, `lowPrice`, `openPrice`: OHLC (strings)
  - `c`: ceiling, `f`: floor, `r`: reference price
  - `g1`-`g7`: bid/ask depth as pipe-delimited strings
- VPS supports comma-separated symbols: `/getliststockdata/SSI,VNM,MWG`
- No auth required, public endpoint
- Response is a JSON array of objects

## Requirements

### Functional
- Fetch real-time stock data from VPS API
- Parse all numeric fields from strings correctly
- Support batch fetching (multiple symbols in one call)
- Provide same interface methods as `RealTimeMarketConnector` for drop-in replacement
- Extract foreign buy/sell volume (not available in CafeF)

### Non-Functional
- 60-second cache TTL (same as CafeF connector)
- 10-second request timeout
- Graceful fallback to CafeF on VPS failure

## Architecture

```
VPSDataConnector
├── __init__(cache_ttl=60)
├── _fetch_batch(symbols: List[str]) -> List[Dict]    # raw VPS API call
├── _parse_stock(raw: Dict) -> Dict                    # normalize field types
├── get_stock_price(symbol: str) -> Optional[float]    # single stock price
├── get_multiple_prices(symbols: List[str]) -> Dict    # batch prices
├── get_foreign_flow() -> Dict                         # khoi ngoai from fBVol/fSVolume
├── get_market_breadth() -> Dict                       # from all-stock fetch
├── get_stock_realtime(symbol: str) -> Dict            # full real-time data
└── get_full_market_signals() -> Dict                  # combined signals
```

## Related Code Files

### Files to Create
- `quantum_stock/dataconnector/vps_market.py` — VPS connector (~180 lines)

### Files to Modify
- `quantum_stock/dataconnector/realtime_market.py` — add `get_realtime_connector()` to return VPS-first, CafeF-fallback wrapper
- `app/api/routers/data.py` — update `get_stock_data()` to try VPS first
- `app/api/routers/market.py` — update `get_market_status()` to use VPS for VN-Index

## Implementation Steps

1. **Create `quantum_stock/dataconnector/vps_market.py`**
   - Class `VPSDataConnector` with `VPS_BASE_URL = "https://bgapidatafeed.vps.com.vn"`
   - `_fetch_batch(symbols)`: GET `/getliststockdata/{comma_separated}`, parse JSON
   - `_parse_stock(raw)`: Convert string fields to proper types:
     ```python
     {
         "symbol": raw["sym"],
         "last_price": float(raw.get("lastPrice", 0)) * 1000,  # to VND
         "open": float(raw.get("openPrice", "0").replace(",", "")) * 1000,
         "high": float(raw.get("highPrice", "0").replace(",", "")) * 1000,
         "low": float(raw.get("lowPrice", "0").replace(",", "")) * 1000,
         "volume": int(raw.get("lot", 0)),
         "change_pct": float(raw.get("changePc", "0")),
         "ceiling": float(raw.get("c", 0)) * 1000,
         "floor": float(raw.get("f", 0)) * 1000,
         "reference": float(raw.get("r", 0)) * 1000,
         "foreign_buy_vol": int(raw.get("fBVol", "0")),
         "foreign_sell_vol": int(raw.get("fSVolume", "0")),
         "foreign_buy_value": float(raw.get("fBValue", "0")),
         "foreign_sell_value": float(raw.get("fSValue", "0")),
         "foreign_room": float(raw.get("fRoom", "0")),
     }
     ```
   - Add same caching pattern as `RealTimeMarketConnector._get_cached_or_fetch()`
   - Implement `get_stock_price(symbol)` returning price in VND
   - Implement `get_foreign_flow()` using real `fBVol`/`fSVolume` data
   - Implement `get_market_breadth()` from all-stock data

2. **Create unified connector factory**
   - In `realtime_market.py`, modify `get_realtime_connector()` to return a `HybridMarketConnector` that tries VPS first, falls back to CafeF
   - Or: simpler approach — create `get_vps_connector()` singleton and update callers individually

3. **Update routers to prefer VPS**
   - `market.py` `get_market_status()`: use VPS for stock prices
   - `data.py` `get_stock_data()`: try VPS connector first for current price
   - Keep CafeF for historical OHLCV (VPS only has real-time, not historical)

## Todo List

- [ ] Create `quantum_stock/dataconnector/vps_market.py`
- [ ] Implement `VPSDataConnector._fetch_batch()`
- [ ] Implement `VPSDataConnector._parse_stock()`
- [ ] Implement `get_stock_price()`, `get_multiple_prices()`
- [ ] Implement `get_foreign_flow()` with real VPS data
- [ ] Implement `get_market_breadth()`
- [ ] Add singleton `get_vps_connector()` function
- [ ] Update `market.py` to use VPS for real-time prices
- [ ] Update `data.py` to try VPS first for current price
- [ ] Keep CafeF as fallback everywhere

## Success Criteria

- VPS connector returns accurate prices matching market (no 2M VND deviation)
- Foreign flow data populated with real `fBVol`/`fSVolume` values
- All existing endpoints still work when VPS is down (CafeF fallback)
- Response time under 2 seconds for batch requests

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| VPS API rate limits | Medium | 60s cache, batch requests |
| VPS API down | Medium | CafeF fallback retained |
| Price unit confusion (thousands vs VND) | High | Consistent `* 1000` conversion, unit tests |
| VPS field format changes | Low | Defensive parsing with defaults |

## Security Considerations

- VPS API is public (no auth needed)
- No sensitive data stored
- User-Agent header should be set to avoid blocking
