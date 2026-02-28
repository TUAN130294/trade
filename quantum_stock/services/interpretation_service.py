"""
Interpretation Service - LLM-based Vietnamese narrative generation for trading data
Uses OpenAI-compatible API through local LLM proxy for actionable insights
"""

import logging
import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class InterpretationService:
    """
    Shared LLM interpretation service for generating Vietnamese trading narratives

    Uses claudible-haiku-4.5 (fast) and claudible-sonnet-4.6 (deep analysis)
    through local proxy at http://localhost:8317/v1
    """

    # LLM Configuration
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8317/v1")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    MODEL_FAST = "claude-sonnet-4-6"
    MODEL_DEEP = "claude-sonnet-4-6"

    # Cache TTL
    CACHE_TTL = 300  # 5 minutes

    # Vietnamese prompt templates (max 200 words output, actionable, with emoji)
    PROMPT_TEMPLATES = {
        "market_status": """Bạn là chuyên gia phân tích thị trường chứng khoán Việt Nam.

Dữ liệu thị trường:
{data}

Hãy tóm tắt tổng quan thị trường bằng tiếng Việt (tối đa 200 từ):
- Tình trạng VN-Index (tăng/giảm/đi ngang)
- Độ rộng thị trường (mã tăng vs mã giảm)
- Khối ngoại (mua ròng/bán ròng)
- Khuyến nghị hành động cụ thể

Dùng emoji để dễ đọc. Kết thúc bằng 1 câu khuyến nghị rõ ràng.""",

        "market_regime": """Bạn là chuyên gia phân tích xu hướng thị trường.

Market Regime: {regime}
Dữ liệu bổ sung:
{data}

Giải thích ngắn gọn bằng tiếng Việt (tối đa 200 từ):
- Ý nghĩa của regime này (bull/bear/neutral/sideways)
- Điều gì sẽ xảy ra tiếp theo
- Chiến lược giao dịch phù hợp
- Mức rủi ro cần lưu ý

Dùng emoji, ngôn ngữ đời thường, dễ hiểu.""",

        "smart_signals": """Bạn là chuyên gia phân tích tín hiệu thông minh.

Tín hiệu phát hiện:
{data}

Diễn giải tín hiệu bằng tiếng Việt (tối đa 200 từ):
- Loại tín hiệu (breakout, reversal, momentum, etc.)
- Độ tin cậy của tín hiệu
- Hành động cụ thể: MUA/BÁN/CHỜ
- Stop-loss và take-profit đề xuất

Ngắn gọn, dễ hiểu, có emoji, kết thúc bằng khuyến nghị.""",

        "technical_analysis": """Bạn là chuyên gia phân tích kỹ thuật.

Mã cổ phiếu: {symbol}
Indicators:
{data}

Phân tích kỹ thuật bằng tiếng Việt (tối đa 200 từ):
- RSI, MACD, MA đang nói gì?
- Xu hướng hiện tại (uptrend/downtrend/sideways)
- Kết luận rõ ràng: MUA/BÁN/CHỜ
- Điểm vào lệnh và stop-loss

Dùng emoji, ngôn ngữ đơn giản, kết thúc bằng VERDICT rõ ràng.""",

        "news_mood": """Bạn là chuyên gia phân tích sentiment tin tức.

Tin tức gần đây:
{data}

Tóm tắt sentiment bằng tiếng Việt (tối đa 200 từ):
- Tâm lý chung (tích cực/tiêu cực/trung tính)
- Chủ đề nóng đang được quan tâm
- Tác động lên thị trường
- Cổ phiếu nào được đề cập nhiều

Ngắn gọn, có emoji, kết thúc bằng khuyến nghị.""",

        "news_alerts": """Bạn là chuyên gia lọc tin tức quan trọng.

Tin tức:
{data}

Tóm tắt tin quan trọng bằng tiếng Việt (tối đa 200 từ):
- Top 3-5 tin nóng nhất
- Tác động ngay lập tức lên giá
- Cổ phiếu bị ảnh hưởng
- Hành động nên làm

Dùng emoji, bullet points, ngắn gọn.""",

        "backtest_result": """Bạn là chuyên gia phân tích backtest.

Kết quả backtest:
{data}

Phân tích kết quả bằng tiếng Việt (tối đa 200 từ):
- Hiệu suất chiến lược (win rate, profit, drawdown)
- Điểm mạnh và điểm yếu
- Có nên sử dụng chiến lược này không?
- Đề xuất cải thiện hoặc điều chỉnh tham số

Dùng emoji, ngôn ngữ thực tế, kết thúc bằng khuyến nghị rõ ràng.""",

        "deep_flow": """Bạn là chuyên gia phân tích dòng tiền.

Dữ liệu dòng tiền:
{data}

Diễn giải dòng tiền bằng tiếng Việt (tối đa 200 từ):
- Tiền đang chảy vào/ra khỏi thị trường
- Nhóm ngành nào đang hút tiền
- Smart money vs Retail money
- Cổ phiếu nào đáng chú ý

Dùng emoji, ngắn gọn, kết thúc bằng danh sách top 3-5 mã đáng mua.""",

        "data_stats": """Bạn là chuyên gia phân tích dữ liệu thị trường.

Dữ liệu hệ thống:
{data}

Tóm tắt tình trạng dữ liệu bằng tiếng Việt (tối đa 200 từ):
- Chất lượng dữ liệu (đầy đủ/thiếu)
- Nguồn dữ liệu đang hoạt động
- Khuyến nghị cải thiện

Dùng emoji, ngắn gọn.""",

        "agent_chat": """Bạn là trợ lý phân tích chứng khoán Việt Nam thông minh.

Câu hỏi của người dùng: {query}

Dữ liệu thị trường hiện tại:
{data}

Trả lời bằng tiếng Việt (tối đa 300 từ):
- Phân tích dựa trên dữ liệu thực
- Đưa ra khuyến nghị cụ thể
- Sử dụng emoji cho dễ đọc

Nếu không có đủ dữ liệu, hãy nói rõ và đưa ra phân tích tổng quan.""",

        "agent_analysis": """Bạn đang đóng vai {role} trong team phân tích cổ phiếu.

Vai trò: {role_description}
Mã cổ phiếu: {symbol}

Dữ liệu kỹ thuật:
{data}

Đưa ra phân tích ngắn gọn (tối đa 150 từ) theo góc nhìn của vai trò.
Kết thúc bằng verdict: MUA / BÁN / CHỜ với confidence %.
Dùng emoji."""
    }

    def __init__(self):
        """Initialize interpretation service with LLM client"""
        self.client = AsyncOpenAI(
            base_url=self.LLM_BASE_URL,
            api_key=self.LLM_API_KEY
        )
        self.cache: Dict[str, tuple[datetime, str]] = {}  # key -> (timestamp, result)
        self.enabled = True
        logger.info(f"InterpretationService initialized: {self.LLM_BASE_URL}")

    def _get_cache_key(self, template_name: str, data: Dict[str, Any]) -> str:
        """Generate cache key from template and data"""
        # Use template + hash of data for cache key
        data_str = json.dumps(data, sort_keys=True)
        return f"{template_name}:{hash(data_str)}"

    def _check_cache(self, cache_key: str) -> Optional[str]:
        """Check if cached result exists and is still valid"""
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < self.CACHE_TTL:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_result
        return None

    def _set_cache(self, cache_key: str, result: str):
        """Store result in cache"""
        self.cache[cache_key] = (datetime.now(), result)

        # Clean old cache entries (simple cleanup)
        if len(self.cache) > 100:
            # Remove oldest 20 entries
            sorted_keys = sorted(self.cache.keys(), key=lambda k: self.cache[k][0])
            for key in sorted_keys[:20]:
                del self.cache[key]

    async def interpret(
        self,
        template_name: str,
        data: Dict[str, Any],
        model: str = None,
        language: str = "vi"
    ) -> str:
        """
        Generate interpretation using LLM

        Args:
            template_name: Name of prompt template (market_status, market_regime, etc.)
            data: Data to interpret
            model: Model to use (default: MODEL_FAST)
            language: Output language (only 'vi' supported for now)

        Returns:
            Vietnamese interpretation text (max 200 words)
        """
        if not self.enabled:
            return f"[Interpretation service chưa sẵn sàng]"

        # Check cache
        cache_key = self._get_cache_key(template_name, data)
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        # Get prompt template
        if template_name not in self.PROMPT_TEMPLATES:
            logger.warning(f"Unknown template: {template_name}")
            return f"[Template '{template_name}' không tồn tại]"

        prompt_template = self.PROMPT_TEMPLATES[template_name]

        # Format prompt with data
        try:
            # Convert data to readable format
            data_str = json.dumps(data, indent=2, ensure_ascii=False)

            # Special handling for different templates
            if template_name == "technical_analysis" and "symbol" in data:
                prompt = prompt_template.format(
                    symbol=data.get("symbol", "N/A"),
                    data=data_str
                )
            elif template_name == "market_regime" and "regime" in data:
                prompt = prompt_template.format(
                    regime=data.get("regime", "N/A"),
                    data=data_str
                )
            elif template_name == "agent_chat":
                prompt = prompt_template.format(
                    query=data.get("query", "N/A"),
                    data=data_str
                )
            elif template_name == "agent_analysis":
                prompt = prompt_template.format(
                    role=data.get("role", "N/A"),
                    role_description=data.get("role_description", "N/A"),
                    symbol=data.get("symbol", "N/A"),
                    data=data_str
                )
            else:
                prompt = prompt_template.replace("{data}", data_str)

        except Exception as e:
            logger.error(f"Failed to format prompt: {e}")
            return f"[Lỗi format prompt: {e}]"

        # Call LLM
        try:
            model_to_use = model or self.MODEL_FAST

            response = await self.client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là chuyên gia phân tích thị trường chứng khoán Việt Nam. Trả lời ngắn gọn, súc tích, có emoji, tối đa 200 từ."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=500
            )

            result = response.choices[0].message.content.strip()

            # Cache result
            self._set_cache(cache_key, result)

            logger.info(f"✅ LLM interpretation: {template_name} ({len(result)} chars)")
            return result

        except Exception as e:
            logger.error(f"LLM API error: {e}")
            # Return fallback message
            return self._fallback_interpretation(template_name, data)

    def _fallback_interpretation(self, template_name: str, data: Dict[str, Any]) -> str:
        """Fallback interpretation when LLM fails"""
        fallbacks = {
            "market_status": "📊 Thị trường đang được phân tích. Vui lòng thử lại sau.",
            "market_regime": "📈 Xu hướng thị trường: Dữ liệu đang được xử lý.",
            "smart_signals": "🎯 Tín hiệu giao dịch đang được phân tích.",
            "technical_analysis": "📉 Phân tích kỹ thuật: Đang kết nối LLM service.",
            "news_mood": "📰 Sentiment tin tức: Đang xử lý.",
            "news_alerts": "🚨 Tin tức quan trọng: Đang tổng hợp.",
            "backtest_result": "🔬 Kết quả backtest: Đang phân tích.",
            "deep_flow": "💰 Dòng tiền: Đang theo dõi.",
            "agent_chat": "🤖 Trợ lý AI đang xử lý câu hỏi của bạn. Vui lòng thử lại sau.",
            "agent_analysis": "📊 Agent đang phân tích. LLM service tạm thời không khả dụng."
        }
        return fallbacks.get(template_name, "⏳ Đang xử lý dữ liệu...")

    async def batch_interpret(
        self,
        items: List[tuple[str, Dict[str, Any]]],
        model: str = None
    ) -> List[str]:
        """
        Batch interpretation for multiple items

        Args:
            items: List of (template_name, data) tuples
            model: Model to use for all items

        Returns:
            List of interpretation strings
        """
        results = []
        for template_name, data in items:
            result = await self.interpret(template_name, data, model=model)
            results.append(result)
        return results


# Singleton instance
_interpretation_service: Optional[InterpretationService] = None


def get_interpretation_service() -> InterpretationService:
    """Get or create interpretation service singleton"""
    global _interpretation_service
    if _interpretation_service is None:
        _interpretation_service = InterpretationService()
    return _interpretation_service
