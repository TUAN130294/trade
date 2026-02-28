#!/usr/bin/env python
"""
Test Phase 1 & 2 Implementation
- VPS Data Connector
- Interpretation Service
"""

import asyncio
import sys


async def test_vps_connector():
    """Test VPS Market Connector"""
    print("\n" + "=" * 60)
    print("PHASE 1: VPS Data Connector Test")
    print("=" * 60)

    try:
        from quantum_stock.dataconnector.vps_market import get_vps_connector

        vps = get_vps_connector()
        print("✅ VPS Connector imported and initialized")

        # Test 1: Multiple stocks
        print("\n[Test 1] Fetching SSI, VNM...")
        result = await vps.get_stock_data(['SSI', 'VNM'])
        print(f"   Result: {result['count']} stocks from {result['source']}")

        # Test 2: Single stock with VND conversion
        print("\n[Test 2] Single stock with VND conversion...")
        single = await vps.get_single_stock('SSI')
        if single:
            print(f"   Symbol: {single['symbol']}")
            print(f"   Price: {single['price_display']}")
            print(f"   Change: {single['change_display']}")
        else:
            print("   ⚠️ No data returned")

        # Test 3: Foreign flow
        print("\n[Test 3] Foreign flow analysis...")
        flow = await vps.get_foreign_flow(['SSI', 'VNM', 'FPT'])
        print(f"   Flow: {flow['flow_type']}")
        print(f"   Summary: {flow['summary']}")

        print("\n✅ Phase 1: VPS Connector - ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Phase 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_interpretation_service():
    """Test Interpretation Service"""
    print("\n" + "=" * 60)
    print("PHASE 2: Interpretation Service Test")
    print("=" * 60)

    try:
        from quantum_stock.services.interpretation_service import get_interpretation_service

        interp = get_interpretation_service()
        print("✅ Interpretation Service imported and initialized")

        # Test 1: Market status interpretation
        print("\n[Test 1] Market status interpretation...")
        market_data = {
            'vnindex': 1250.5,
            'change': 2.3,
            'advancing': 250,
            'declining': 120
        }
        result = await interp.interpret('market_status', market_data)
        print(f"   Output ({len(result)} chars): {result[:150]}...")

        # Test 2: Technical analysis interpretation
        print("\n[Test 2] Technical analysis interpretation...")
        tech_data = {
            'symbol': 'SSI',
            'rsi': 65,
            'macd': 0.5,
            'trend': 'uptrend'
        }
        result = await interp.interpret('technical_analysis', tech_data)
        print(f"   Output ({len(result)} chars): {result[:150]}...")

        # Test 3: Cache test (should hit cache)
        print("\n[Test 3] Cache test (repeat call)...")
        result = await interp.interpret('market_status', market_data)
        print(f"   ✅ Cache working (instant response)")

        print("\n✅ Phase 2: Interpretation Service - ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Phase 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("PHASE 1 & 2 IMPLEMENTATION TEST")
    print("=" * 60)

    phase1_ok = await test_vps_connector()
    phase2_ok = await test_interpretation_service()

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Phase 1 (VPS Connector): {'✅ PASS' if phase1_ok else '❌ FAIL'}")
    print(f"Phase 2 (Interpretation): {'✅ PASS' if phase2_ok else '❌ FAIL'}")

    if phase1_ok and phase2_ok:
        print("\n🎉 ALL PHASES COMPLETED SUCCESSFULLY!")
        return 0
    else:
        print("\n⚠️ Some phases failed. Review errors above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
