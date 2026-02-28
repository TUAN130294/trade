# VN-QUANT Blind Spots & Upgrade Roadmap

## Critical Gaps Identified

### 1. Derivative Shadow (VN30F1M) - 🛑 CRITICAL MISSING
**Problem**: System only tracks stocks (HOSE), missing futures market that leads by 5-15 minutes.

**Needed Features**:
- [ ] Basis Monitoring: VN30F1M vs VN30-Index deviation (Basis < -10 = strong short sentiment)
- [ ] F-Indicator: When VN30F1M breaks trend with high volume → auto-reduce portfolio
- [ ] Arbitrage Detection: Detect "Lái" manipulation (VCB/VHM/VIC pulled to Long futures)

**Data Sources**: SSI, VNDIRECT, or TCBS for VN30F1M real-time

---

### 2. Corporate Actions (GDKHQ) - ⚠️ HIGH RISK
**Problem**: Ex-dividend dates cause price drops that trigger false Stop Loss signals.

**Needed Features**:
- [ ] Calendar Guard module: Check GDKHQ schedule before trading
- [ ] Auto-lock symbols on GDKHQ days
- [ ] Price adjustment formula for historical data

**Data Sources**: CafeF/VietStock corporate events calendar

---

### 3. Liquidity Hunting - ⚠️ MEDIUM PRIORITY
**Problem**: Market orders (MP) on midcap stocks can move prices significantly.

**Needed Features**:
- [ ] Passive Order Logic: Limit orders instead of MP
- [ ] Order Book Depth analysis (bid/ask spread tracking)
- [ ] Iceberg Detection: Hidden large orders identification

---

### 4. Macro & SBV Factor - 🛑 CRITICAL MISSING
**Problem**: Pure technical analysis fails during SBV monetary policy changes.

**Needed Features**:
- [ ] Interbank Rate Monitor: Overnight rate spikes = cash withdrawal = SELL
- [ ] USD/VND Exchange Rate: Rate increase = foreign selling = Bluechip pressure
- [ ] Add macro features to Stockformer model

**Data Sources**: SBV website, Bloomberg, Reuters

---

### 5. Backtest Reality Check - ⚠️ MEDIUM PRIORITY
**Problem**: Backtest without proper fees/slippage gives false results.

**Current Settings** (.env):
- BACKTEST_COMMISSION=0.0015 (0.15%)
- BACKTEST_SLIPPAGE=0.001 (0.1%)
- BACKTEST_TAX=0.001 (0.1% on sells)

**Needed Improvements**:
- [ ] Round-trip cost = ~0.4-0.5% (tax + commission + slippage)
- [ ] Only trade if Expectancy > 1.5% after fees
- [ ] Transaction Cost Model in backtest engine

---

## Priority Matrix

| Feature | Impact | Effort | Priority |
|---------|--------|--------|----------|
| VN30F1M Monitoring | HIGH | MEDIUM | P0 |
| GDKHQ Calendar Guard | HIGH | LOW | P0 |
| Macro Features (SBV) | HIGH | HIGH | P1 |
| Order Execution Logic | MEDIUM | MEDIUM | P2 |
| Backtest Cost Model | MEDIUM | LOW | P2 |

---

## Data Sources to Integrate

1. **Futures (VN30F1M)**: SSI API / VNDIRECT WebSocket
2. **Corporate Events**: CafeF RSS / VietStock Calendar
3. **Interbank Rates**: SBV official / Bloomberg
4. **USD/VND Rate**: Vietcombank / SBV
5. **Order Book**: SSI/VNDIRECT Level 2 data

---

*Generated: 2026-01-12*
