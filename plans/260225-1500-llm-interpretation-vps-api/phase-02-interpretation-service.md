---
title: "Phase 2: LLM Interpretation Service"
status: pending
priority: P1
effort: 3h
---

# Phase 2: LLM Interpretation Service

## Context Links

- Existing LLM client: `quantum_stock/agents/llm_agents.py` (LLMClient class)
- Cache utility: `quantum_stock/utils/cache.py` (MemoryCache, CacheManager)
- LLM proxy: `http://localhost:8317/v1` with key `sk-***REDACTED***`
- Models: `claudible-haiku-4.5` (fast/bulk), `claudible-sonnet-4.6` (deep analysis)

## Overview

Create a shared `InterpretationService` that wraps the existing `LLMClient` with:
- Prompt template registry for each endpoint type
- Model routing (haiku for fast, sonnet for deep)
- Response caching (5 min TTL)
- Graceful fallback when LLM unavailable

This is the core module that all endpoints will use to add `interpretation` fields.

## Key Insights

- Existing `LLMClient` already handles CCS proxy communication, error handling, and mock fallback
- `MemoryCache` from `quantum_stock/utils/cache.py` provides TTL-based caching without Redis
- Each endpoint produces a different data shape; prompt templates must be data-aware
- Vietnamese output is mandatory; system prompt should enforce this
- Max tokens should be kept low (200-400) to minimize latency and cost

## Requirements

### Functional
- Accept any dict of endpoint data + template key, return Vietnamese interpretation string
- Support 8 prompt templates (one per endpoint category)
- Route to appropriate model based on analysis depth
- Cache by endpoint + data hash, 5 min TTL
- Return empty string (not error) when LLM unavailable

### Non-Functional
- Interpretation latency < 3 seconds (haiku), < 8 seconds (sonnet)
- Memory cache only (no Redis dependency)
- Thread-safe singleton pattern

## Architecture

```
InterpretationService (singleton)
├── _llm_haiku: LLMClient        # claudible-haiku-4.5
├── _llm_sonnet: LLMClient       # claudible-sonnet-4.6
├── _cache: MemoryCache           # 5min TTL
├── _templates: Dict[str, str]    # prompt templates
│
├── interpret(data, template_key, use_sonnet=False) -> str
│     |-- build cache key from template_key + hash(data)
│     |-- check cache → return if hit
│     |-- select model (haiku or sonnet)
│     |-- format prompt from template + data
│     |-- call LLMClient.chat()
│     |-- cache response
│     |-- return interpretation
│
├── interpret_market_status(data) -> str
├── interpret_market_regime(data) -> str
├── interpret_smart_signals(data) -> str
├── interpret_technical(data) -> str
├── interpret_market_mood(data) -> str
├── interpret_news_alerts(data) -> str
├── interpret_backtest(data) -> str
└── interpret_deep_flow(data) -> str
```

## Related Code Files

### Files to Create
- `quantum_stock/services/interpretation_service.py` — Main service (~180 lines)
- `quantum_stock/services/__init__.py` — Package init

### Files to Modify
- None in this phase (endpoints updated in Phase 3-4)

## Implementation Steps

1. **Create `quantum_stock/services/__init__.py`**
   - Empty init or export `get_interpretation_service`

2. **Create `quantum_stock/services/interpretation_service.py`**

3. **Define prompt templates** — each is a format string that receives data keys:

   ```python
   TEMPLATES = {
       "market_status": """Dựa trên dữ liệu thị trường VN-Index:
   - VN-Index: {vnindex} ({change:+.2f}, {change_pct:+.2f}%)
   - Phiên: {session_info}
   - Trạng thái: {"Đang mở" if is_open else "Đóng cửa"}

   Viết 2-3 câu nhận định ngắn gọn bằng tiếng Việt về tình hình thị trường hiện tại.
   Đưa ra gợi ý hành động: MUA / BÁN / CHỜ.""",

       "market_regime": """Dữ liệu regime thị trường:
   - Regime: {market_regime}
   - Volatility: {volatility_regime}
   - Confidence: {confidence:.0%}
   - Strategies: {recommended_strategies}

   Giải thích TẠI SAO thị trường đang ở regime này bằng tiếng Việt.
   Gợi ý chiến lược phù hợp cho nhà đầu tư cá nhân.""",

       "smart_signals": """Tín hiệu thị trường:
   {signals_summary}

   Tổng hợp các tín hiệu trên thành 3-4 câu nhận định bằng tiếng Việt.
   Kết luận: thị trường đang ở trạng thái nào và nên làm gì.""",

       "technical": """Phân tích kỹ thuật mã {symbol}:
   - Giá hiện tại: {current_price:,.0f} VND
   - RSI(14): {rsi:.1f}
   - Hỗ trợ: {support_levels}
   - Kháng cự: {resistance_levels}
   - Pattern: {patterns}

   Đưa ra khuyến nghị MUA / BÁN / CHỜ với lý do ngắn gọn bằng tiếng Việt.
   Nêu mức giá mục tiêu và mức cắt lỗ.""",

       "market_mood": """Tâm lý thị trường:
   - Mood: {current_mood}
   - Tin tích cực: {positive_news}
   - Tin tiêu cực: {negative_news}
   - Tin trung tính: {neutral_news}

   Tóm tắt tin tức đang ảnh hưởng thị trường bằng tiếng Việt.
   Nhận xét tâm lý nhà đầu tư hiện tại.""",

       "news_alerts": """Danh sách tin tức mới nhất:
   {alerts_summary}

   Tổng hợp {total} tin tức trên thành nhận định ngắn gọn bằng tiếng Việt.
   Nêu tin nào quan trọng nhất và ảnh hưởng thế nào.""",

       "backtest": """Kết quả backtest chiến lược {strategy} cho mã {symbol}:
   - Lợi nhuận: {total_return_pct}%
   - Sharpe: {sharpe_ratio}
   - Max Drawdown: {max_drawdown_pct}%
   - Win Rate: {win_rate}%
   - Profit Factor: {profit_factor}

   Đánh giá chiến lược này bằng tiếng Việt.
   So sánh với benchmark (gửi tiết kiệm 6%/năm).
   Khuyến nghị có nên áp dụng chiến lược này không.""",

       "deep_flow": """Phân tích dòng tiền mã {symbol}:
   - Insights: {insights_summary}
   - Flow Score: {flow_score}
   - Recommendation: {recommendation}

   Giải thích dòng tiền đang chảy như thế nào bằng tiếng Việt.
   Đưa ra nhận định: smart money đang tích lũy hay phân phối."""
   }
   ```

4. **Implement `InterpretationService`**
   ```python
   class InterpretationService:
       def __init__(self):
           api_key = os.getenv('LLM_API_KEY', 'sk-***REDACTED***')
           base_url = os.getenv('LLM_BASE_URL', 'http://localhost:8317/v1')

           self._llm_haiku = LLMClient(
               api_key=api_key, base_url=base_url,
               model='claudible-haiku-4.5', provider='claudible'
           )
           self._llm_sonnet = LLMClient(
               api_key=api_key, base_url=base_url,
               model='claudible-sonnet-4.6', provider='claudible'
           )
           self._cache = MemoryCache()
           self._cache_ttl = 300  # 5 minutes

       async def interpret(self, data: dict, template_key: str,
                           use_sonnet: bool = False) -> str:
           # 1. Build cache key
           cache_key = f"interp:{template_key}:{_hash_data(data)}"
           cached = self._cache.get(cache_key)
           if cached:
               return cached

           # 2. Select model
           llm = self._llm_sonnet if use_sonnet else self._llm_haiku

           # 3. Build prompt
           template = TEMPLATES.get(template_key, "")
           try:
               user_msg = template.format(**data)
           except KeyError:
               user_msg = template  # use as-is if format fails

           # 4. Call LLM
           messages = [
               {"role": "system", "content": SYSTEM_PROMPT_VI},
               {"role": "user", "content": user_msg}
           ]
           result = await llm.chat(messages, temperature=0.5, max_tokens=300)

           # 5. Cache and return
           self._cache.set(cache_key, result, self._cache_ttl)
           return result
   ```

5. **Add convenience methods** that pre-select template and model:
   - `interpret_market_status(data)` → haiku
   - `interpret_market_regime(data)` → haiku
   - `interpret_smart_signals(data)` → haiku
   - `interpret_technical(data)` → sonnet (deep analysis)
   - `interpret_market_mood(data)` → haiku
   - `interpret_news_alerts(data)` → haiku
   - `interpret_backtest(data)` → sonnet (deep analysis)
   - `interpret_deep_flow(data)` → sonnet

6. **Add singleton accessor**
   ```python
   _instance = None
   def get_interpretation_service() -> InterpretationService:
       global _instance
       if _instance is None:
           _instance = InterpretationService()
       return _instance
   ```

7. **System prompt constant**
   ```python
   SYSTEM_PROMPT_VI = """Bạn là chuyên gia phân tích chứng khoán Việt Nam.
   Quy tắc:
   - Trả lời hoàn toàn bằng tiếng Việt
   - Ngắn gọn, đi thẳng vào vấn đề (2-4 câu)
   - Dùng thuật ngữ chứng khoán VN: MUA, BÁN, CHỜ, hỗ trợ, kháng cự
   - Đưa ra hành động cụ thể, không mơ hồ
   - Nếu dữ liệu không rõ ràng, nói "cần theo dõi thêm"
   """
   ```

## Todo List

- [ ] Create `quantum_stock/services/__init__.py`
- [ ] Create `quantum_stock/services/interpretation_service.py`
- [ ] Define all 8 prompt templates
- [ ] Implement `InterpretationService.__init__()` with dual LLM clients
- [ ] Implement `interpret()` with cache + model routing
- [ ] Implement 8 convenience methods
- [ ] Add singleton `get_interpretation_service()`
- [ ] Add `_hash_data()` utility for cache key generation
- [ ] Test with mock data to verify prompt formatting

## Success Criteria

- `interpret()` returns Vietnamese string for each template key
- Cache hit returns in < 5ms
- Cache miss with haiku returns in < 3s
- Cache miss with sonnet returns in < 8s
- Returns empty string (not exception) when LLM proxy is down
- All 8 templates produce valid formatted prompts

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM proxy down | Low | Mock fallback in LLMClient, return empty string |
| Prompt format errors | Medium | Try/except on `.format()`, use raw template as fallback |
| Cache key collisions | Low | Use hashlib.md5 on sorted JSON |
| Response too long/short | Low | max_tokens=300, temperature=0.5 |

## Security Considerations

- API key should come from env var, not hardcoded in production
- LLM responses should not be trusted for trading decisions (informational only)
- No user-provided data injected directly into prompts (only server-side data)
