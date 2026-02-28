# Codebase Review: Money Flow Analysis for Vietnamese Market

**Date:** 2026-02-25
**Reviewer:** AI Code Analyst
**Scope:** Full codebase review - analysis logic evaluation & money flow depth assessment

---

## Executive Summary

Hệ thống quant hiện tại có **nền tảng kỹ thuật tốt** (80+ indicators, 102 ML models, multi-agent architecture) nhưng **thiếu chiều sâu phân tích hành vi dòng tiền** - yếu tố quyết định trên thị trường VN nơi retail chiếm ~85% thanh khoản và FOMO rất cao.

**Verdict:** Hệ thống đang phân tích theo kiểu "kỹ thuật phương Tây" (EMA, RSI, MACD) thay vì "đọc dòng tiền kiểu VN" (gom hàng, xả hàng, đập ATC, lái gom, FOMO chase).

---

## 1. GAP ANALYSIS: Hiện trạng vs. Yêu cầu

### 1.1 MarketFlowConnector - CÒN LÀ STUB

**File:** `quantum_stock/dataconnector/market_flow.py`

| Function | Status | Issue |
|---|---|---|
| `get_foreign_flow()` | ❌ FAKE | Returns hardcoded data (`5_000_000_000`), không crawl thực |
| `get_proprietary_flow()` | ❌ FAKE | Returns hardcoded data, chưa implement |
| `get_market_liquidity()` | ❌ FAKE | Returns hardcoded data |
| `detect_smart_money_footprint()` | ⚠️ BASIC | Chỉ check 3 pattern đơn giản (volume spike + close location) |

**Impact:** Tất cả agent decisions đang dựa trên technical indicators truyền thống, KHÔNG có real money flow data.

### 1.2 Agents KHÔNG dùng Money Flow trong quyết định

**File:** `quantum_stock/agents/analyst_agent.py:40-123`

AnalystAgent import MarketFlowConnector nhưng **KHÔNG GỌI** `detect_smart_money_footprint()` trong `analyze()`. Weights hiện tại:

```
trend:      25%  ← EMA/ADX (pure technical)
momentum:   25%  ← RSI/MACD/Stochastic (pure technical)
volume:     20%  ← Basic volume ratio (shallow)
pattern:    15%  ← BB/divergence (pure technical)
levels:     15%  ← S/R (pure technical)
money_flow:  0%  ← KHÔNG CÓ
```

### 1.3 Confidence Scoring thiếu Money Flow Factor

**File:** `quantum_stock/core/confidence_scoring.py:83-91`

```
return:         20%  ← Expected return magnitude
model_accuracy: 20%  ← Historical model accuracy
volatility:     15%  ← ATR-based
volume:         15%  ← Basic volume ratio
technical:      15%  ← EMA/RSI/MACD composite
market_regime:  15%  ← Bull/Bear/Neutral
money_flow:      0%  ← KHÔNG CÓ
foreign_flow:    0%  ← KHÔNG CÓ
```

### 1.4 Thiếu hoàn toàn VN-specific Behavioral Patterns

| Pattern cần có | Status | Giải thích |
|---|---|---|
| ATO manipulation | ❌ | Tổ chức thường đặt ATO để "dẫn giá" |
| ATC đập giá | ❌ | Đập ATC để gom phiên sau |
| Gom hàng âm thầm (Iceberg) | ❌ | Lệnh nhỏ lặp lại cùng giá |
| FOMO chase detection | ❌ | Retail đuổi giá trần |
| Xanh vỏ đỏ lòng (Bull trap) | ⚠️ Có nhưng k tích hợp | Có detect nhưng k feed vào agent |
| Dãn cầu (Demand spacing) | ❌ | Spread order book rộng |
| Block deal detection | ❌ | Giao dịch thỏa thuận lớn |
| Morning vs Afternoon flow | ❌ | Sáng gom, chiều xả (hoặc ngược lại) |
| Ceiling chase velocity | ❌ | Tốc độ giá tiến đến trần |
| Smart distribution | ❌ | Volume cao + giá sideway = xả hàng |

---

## 2. ĐIỂM MẠNH CẦN GIỮ

1. **Multi-agent architecture** - Framework tốt, chỉ cần thêm Money Flow Agent
2. **102 Stockformer models** - ML prediction backbone solid
3. **80+ technical indicators** - Toolkit đầy đủ
4. **VN market compliance** - T+2.5, ceiling/floor, tick size
5. **6-factor confidence scoring** - Dễ mở rộng thêm factor
6. **Real-time CafeF connector** - Đã có data source, cần khai thác sâu hơn
7. **Volume Profile, VWAP, Cumulative Delta** - Đã implement nhưng chưa dùng sâu

---

## 3. ĐỀ XUẤT CẢI THIỆN - PHÂN TÍCH HÀNH VI DÒNG TIỀN

### Phase 1: Data Layer - Khai thác sâu data hiện có

#### 1A. Fix MarketFlowConnector - Implement REAL data

```
Thay thế hardcoded data bằng:
- CafeF API: foreign buy/sell per stock (fields 'tb'/'ts')
- Accumulated foreign flow: 5-day, 10-day rolling
- Proprietary trading: crawl từ HOSE/HNX reports
- Market liquidity: total value, morning vs afternoon split
```

#### 1B. Intraday Session Analysis

```python
# Chia phiên thành 4 giai đoạn:
SESSIONS = {
    'ATO':     (9, 0, 9, 15),    # Mở cửa - institutional positioning
    'MORNING': (9, 15, 11, 30),  # Phiên sáng - trend formation
    'AFTERNOON': (13, 0, 14, 30), # Phiên chiều - confirmation/reversal
    'ATC':     (14, 30, 14, 45)   # Đóng cửa - smart money final moves
}
```

**Insight quan trọng:** Trên thị trường VN:
- ATO: Tổ chức đặt lệnh "dẫn giá" → detect volume burst at open
- Phiên sáng chiếm 60-65% thanh khoản → nếu chiều đảo ngược = warning
- ATC: "Đập giá" hoặc "kéo giá" cuối phiên = tín hiệu mạnh nhất

#### 1C. Enhanced CafeF Data Extraction

CafeF API fields chưa khai thác:
```
'tb': Foreign buy volume     ← CRITICAL, đang parse nhưng data spotty
'ts': Foreign sell volume    ← CRITICAL
'b':  Ceiling price          ← Đang dùng
'd':  Floor price            ← Đang dùng
'n':  Total volume           ← Đang dùng
'k':  Price change           ← Đang dùng
```

### Phase 2: Money Flow Analysis Engine (MỚI)

#### 2A. Smart Money Footprint Detector (nâng cấp)

Nâng cấp `detect_smart_money_footprint()` từ 3 patterns → 10+ patterns:

```
HIỆN TẠI (3 patterns):
1. CLIMAX_BUYING: vol_spike + strong_close
2. CLIMAX_SELLING: vol_spike + weak_close
3. CHURNING: vol_spike + narrow_spread

CẦN THÊM:
4. ACCUMULATION: vol tăng dần + price sideway + close location tăng
5. DISTRIBUTION: vol tăng dần + price sideway + close location giảm
6. SPRING/SHAKEOUT: xuyên support rồi revert (Wyckoff)
7. UPTHRUST: xuyên resistance rồi revert
8. ABSORPTION: dư bán lớn nhưng giá không giảm
9. INITIATIVE_BUYING: gap up + vol spike + hold
10. INITIATIVE_SELLING: gap down + vol spike + no bounce
11. STOPPING_VOLUME: vol cực lớn ở đáy + long lower shadow
12. EFFORT_VS_RESULT: vol to nhưng price move nhỏ (divergence)
```

#### 2B. FOMO Detection Engine (MỚI - Critical cho VN)

```python
class FOMODetector:
    """
    Detect FOMO behavior - đặc trưng thị trường VN

    VN market FOMO indicators:
    1. Ceiling chase: Tốc độ giá tiến đến trần
    2. Volume acceleration: Vol tăng theo cấp số nhân
    3. RSI > 80 + Volume spike: Đỉnh FOMO
    4. Gap up liên tục 3+ phiên: Retail chase
    5. Bid dominance: Dư mua >> Dư bán (tất cả muốn mua)
    """

    Signals:
    - FOMO_BUILDING: Early stage, safe to ride
    - FOMO_PEAK: Maximum euphoria, danger zone
    - FOMO_EXHAUSTION: Smart money exiting, retail stuck
    - FOMO_TRAP: Price reversal after FOMO peak
```

#### 2C. Foreign Flow Intelligence (MỚI)

```python
class ForeignFlowAnalyzer:
    """
    Phân tích sâu hành vi khối ngoại

    Levels:
    1. Daily net: Mua/bán ròng hôm nay
    2. Accumulated 5D: Xu hướng ngắn hạn
    3. Accumulated 20D: Xu hướng trung hạn
    4. Flow velocity: Tốc độ tăng/giảm mua ròng
    5. Concentration: Tập trung vào mấy mã hay dàn trải
    6. Timing: Mua sáng hay chiều, ATO hay ATC
    """

    Signals:
    - STRONG_ACCUMULATION: 5D mua ròng tăng tốc
    - STEALTH_BUYING: Volume thấp nhưng foreign net buy tăng
    - PANIC_SELL: Foreign bán ròng đột biến
    - ROTATION: Chuyển dòng tiền từ sector này sang sector khác
```

### Phase 3: New Money Flow Agent

#### 3A. Thêm MoneyFlowAgent vào hệ thống multi-agent

```python
class MoneyFlowAgent(BaseAgent):
    """
    Agent chuyên phân tích hành vi dòng tiền

    Weight: 1.3 (cao hơn Bull/Bear vì VN market = money flow driven)

    Analysis dimensions:
    1. Smart Money Flow (30%)
       - Volume-Price analysis (Wyckoff)
       - Close location value
       - Effort vs Result

    2. Foreign & Institutional Flow (25%)
       - Net foreign 1D/5D/20D
       - Proprietary trading
       - Block deals

    3. FOMO/Panic Behavior (25%)
       - Ceiling chase velocity
       - Volume acceleration
       - Retail vs Smart money divergence

    4. Session Flow Analysis (20%)
       - ATO vs ATC patterns
       - Morning vs Afternoon divergence
       - Late session manipulation
    """
```

#### 3B. Update Agent Weights

```
HIỆN TẠI:
  Bull:       1.0
  Bear:       1.0
  Alex:       1.2  (Technical analyst)
  RiskDoctor: 0.8
  Chief:      1.5  (Decision maker)

ĐỀ XUẤT:
  MoneyFlow:  1.3  ← MỚI - highest advisory weight
  Alex:       1.0  ← Giảm vì VN market k phải technical-driven
  Bull:       0.8  ← Giảm bias
  Bear:       0.8  ← Giảm bias
  RiskDoctor: 0.9  ← Tăng nhẹ
  Chief:      1.5  (giữ nguyên)
```

### Phase 4: Update Confidence Scoring

#### 4A. Thêm Money Flow Factors

```
ĐỀ XUẤT WEIGHTS MỚI:
  return:         15%  (giảm từ 20%)
  model_accuracy: 15%  (giảm từ 20%)
  volatility:     10%  (giảm từ 15%)
  volume:         10%  (giảm từ 15%)
  technical:      10%  (giảm từ 15%)
  market_regime:  10%  (giảm từ 15%)
  money_flow:     15%  ← MỚI: Smart money + Wyckoff
  foreign_flow:   10%  ← MỚI: Khối ngoại
  fomo_score:      5%  ← MỚI: FOMO detection (inverse)
```

### Phase 5: Enhanced Exit Strategy

#### 5A. Money Flow-based Exits

Thêm vào `PositionExitScheduler._should_exit()`:

```python
# 5. Money Flow Exit Signals (MỚI)
# Smart money distribution detected
if money_flow.is_distribution(position.symbol):
    return "SMART_MONEY_DISTRIBUTION"

# Foreign selling acceleration
if foreign_flow.is_panic_sell(position.symbol):
    return "FOREIGN_PANIC_SELL"

# FOMO exhaustion (đã lên đỉnh FOMO, retail stuck)
if fomo.is_exhaustion(position.symbol):
    return "FOMO_EXHAUSTION_EXIT"

# Volume dry up after pump (tay to rút, thanh khoản cạn)
if volume_ratio < 0.3 and days_held > 3:
    return "LIQUIDITY_DRY_UP"
```

---

## 4. IMPLEMENTATION PRIORITY

| Priority | Task | Impact | Effort |
|---|---|---|---|
| 🔴 P0 | Fix MarketFlowConnector (real data) | Critical | Medium |
| 🔴 P0 | Create MoneyFlowAgent | Critical | Medium |
| 🟡 P1 | FOMO Detection Engine | High | Medium |
| 🟡 P1 | Foreign Flow Intelligence | High | Medium |
| 🟡 P1 | Update Confidence Scoring (add MF factors) | High | Low |
| 🟢 P2 | Session Analysis (ATO/ATC patterns) | Medium | Medium |
| 🟢 P2 | Smart Money Exit Signals | Medium | Low |
| 🟢 P2 | Wyckoff Pattern Recognition | Medium | High |
| 🔵 P3 | Block Deal Detection | Low | Medium |
| 🔵 P3 | Sector Rotation Flow | Low | Medium |

---

## 5. CÁC INDICATOR HIỆN CÓ NHƯNG CHƯA TÍCH HỢP VÀO AGENTS

Những indicator đã code nhưng KHÔNG được agent nào sử dụng:

| File | Indicator | Relevance cho Money Flow |
|---|---|---|
| `orderflow.py` | `cumulative_delta()` | HIGH - Buy vs sell pressure |
| `orderflow.py` | `absorption_exhaustion()` | HIGH - Detect institutional absorption |
| `orderflow.py` | `vwap_bands()` | MEDIUM - Institutional price level |
| `orderflow.py` | `foreign_flow_analysis()` | CRITICAL - Chưa dùng! |
| `orderflow.py` | `smart_money_index()` | CRITICAL - Chưa dùng! |
| `volume.py` | `twiggs_money_flow()` | HIGH - Advanced money flow |
| `volume.py` | `klinger_oscillator()` | MEDIUM - Volume force |
| `volume.py` | `volume_zone_oscillator()` | MEDIUM - Buy/sell zones |
| `custom.py` | `vn_market_strength()` | HIGH - Market breadth |
| `custom.py` | `foreign_flow_indicator()` | CRITICAL - Chưa dùng! |
| `custom.py` | `smart_money_index()` | CRITICAL - Chưa dùng! |
| `custom.py` | `vn_sector_rotation()` | HIGH - Sector flow |
| `custom.py` | `accumulation_distribution_zone()` | HIGH - A/D detection |
| `custom.py` | `ceiling_floor_detector()` | MEDIUM - VN-specific |

**Quick Win:** Chỉ cần tích hợp các indicator đã có vào agents là tăng đáng kể chất lượng phân tích mà không cần code mới.

---

## 6. KIẾN TRÚC ĐỀ XUẤT

```
                     ┌─────────────────┐
                     │  Market Data     │
                     │  (CafeF API)     │
                     └────────┬────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
    ┌─────────────┐  ┌──────────────┐  ┌──────────────┐
    │ Price/OHLCV  │  │ Foreign Flow │  │ Volume Data  │
    │ (existing)   │  │ (upgrade)    │  │ (existing)   │
    └──────┬──────┘  └──────┬───────┘  └──────┬───────┘
           │                │                  │
           ▼                ▼                  ▼
    ┌──────────────────────────────────────────────────┐
    │           Money Flow Analysis Engine (MỚI)        │
    │                                                    │
    │  ┌─────────┐ ┌──────────┐ ┌────────┐ ┌────────┐  │
    │  │ Wyckoff  │ │ Foreign  │ │  FOMO  │ │Session │  │
    │  │ Patterns │ │ Flow     │ │ Detect │ │Analysis│  │
    │  └─────────┘ └──────────┘ └────────┘ └────────┘  │
    └──────────────────────┬───────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌─────────────┐ ┌──────────────┐ ┌──────────────┐
    │ MoneyFlow   │ │ Alex         │ │ Bull/Bear    │
    │ Agent (NEW) │ │ (Technical)  │ │ Agents       │
    │ weight: 1.3 │ │ weight: 1.0  │ │ weight: 0.8  │
    └──────┬──────┘ └──────┬───────┘ └──────┬───────┘
           │                │                │
           ▼                ▼                ▼
    ┌──────────────────────────────────────────────────┐
    │         Chief Agent (Weighted Consensus)          │
    │         + Enhanced Confidence Scoring              │
    └──────────────────────┬───────────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────────┐
    │    Execution + Money Flow-aware Exit Strategy     │
    └──────────────────────────────────────────────────┘
```

---

## 7. UNRESOLVED QUESTIONS

1. **Data quality:** CafeF fields `tb`/`ts` (foreign buy/sell) có available đầy đủ cho tất cả stocks không? Cần test.
2. **Intraday data:** CafeF API hiện tại chỉ cho snapshot, không cho tick-by-tick. Có cần upgrade data source (SSI iBoard, VPS)?
3. **Proprietary trading data:** Data tự doanh có thể crawl từ đâu realtime? HOSE chỉ publish cuối ngày.
4. **ATO/ATC analysis:** Cần intraday time-series data, CafeF API có hỗ trợ không?
5. **Block deal data:** Giao dịch thỏa thuận ngoài sàn - nguồn data nào?
