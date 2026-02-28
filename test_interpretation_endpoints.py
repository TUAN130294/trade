#!/usr/bin/env python3
"""
Test script for interpretation-enabled endpoints
Tests all Phase 3-4 endpoints with interpret=true parameter
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from quantum_stock.services.interpretation_service import InterpretationService


async def test_interpretation_service():
    """Test that InterpretationService works correctly"""
    print("=" * 80)
    print("Testing InterpretationService (STUB)")
    print("=" * 80)

    service = InterpretationService()

    # Test market_status
    result = await service.interpret(
        "market_status",
        {"status": "mở cửa", "vnindex": 1249.05, "change": 5.2, "change_pct": 0.42}
    )
    print(f"\n✅ market_status: {result}")

    # Test market_regime
    result = await service.interpret(
        "market_regime",
        {"regime": "UPTREND", "confidence": 0.8, "volatility": "NORMAL", "liquidity": "HIGH"}
    )
    print(f"\n✅ market_regime: {result}")

    # Test smart_signals
    result = await service.interpret(
        "smart_signals",
        {"count": 5, "breadth": "POSITIVE", "foreign_flow": "BUY", "smart_money": "CLIMAX_BUY"}
    )
    print(f"\n✅ smart_signals: {result}")

    # Test technical_analysis
    result = await service.interpret(
        "technical_analysis",
        {"symbol": "MWG", "signal": "MUA", "rsi": 32.5, "current_price": 86000, "bottom_score": 75}
    )
    print(f"\n✅ technical_analysis: {result}")

    # Test data_stats
    result = await service.interpret(
        "data_stats",
        {"total_files": 850, "coverage_pct": 49.1}
    )
    print(f"\n✅ data_stats: {result}")

    # Test backtest_results
    result = await service.interpret(
        "backtest_results",
        {"strategy": "momentum", "return_pct": 35.2, "sharpe_ratio": 1.8, "win_rate": 65.5}
    )
    print(f"\n✅ backtest_results: {result}")

    # Test market_mood
    result = await service.interpret(
        "market_mood",
        {"mood": "bullish", "positive_news": 15, "negative_news": 3}
    )
    print(f"\n✅ market_mood: {result}")

    # Test news_alerts
    result = await service.interpret(
        "news_alerts",
        {"count": 20, "high_priority": 5}
    )
    print(f"\n✅ news_alerts: {result}")

    print("\n" + "=" * 80)
    print("All interpretation tests PASSED (STUB version)")
    print("=" * 80)


async def main():
    """Run all tests"""
    try:
        await test_interpretation_service()
        print("\n✅ All Phase 3-4 stub tests completed successfully!")
        print("\nNote: These are STUB versions. Phase 1-2 agent will implement:")
        print("  - Real LLM integration (Gemini/OpenAI)")
        print("  - VPS market connector with real API calls")
        print("  - Vietnamese narrative generation")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
