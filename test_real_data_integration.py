"""
Test script to verify real data integration fixes
Tests H3, H4, H5 fixes with actual VPS/CafeF API calls
"""

import asyncio
import sys
sys.path.insert(0, '.')

from quantum_stock.core.broker_api import PaperTradingBroker
from quantum_stock.dataconnector.vps_market import get_vps_connector
from quantum_stock.dataconnector.realtime_market import get_realtime_connector


async def test_h3_broker_real_prices():
    """Test H3: PaperTradingBroker uses real prices"""
    print("\n=== TEST H3: PaperTradingBroker Real Prices ===")

    broker = PaperTradingBroker(initial_balance=1_000_000_000)

    test_symbols = ['SSI', 'VNM', 'HPG', 'FPT']

    for symbol in test_symbols:
        try:
            price_data = await broker.get_market_price(symbol)
            print(f"✅ {symbol}: {price_data['last']:,.0f} VND (bid: {price_data['bid']:,.0f}, ask: {price_data['ask']:,.0f})")

            # Verify not zero (real data)
            if price_data['last'] == 0:
                print(f"   ⚠️ WARNING: Zero price detected for {symbol}")

        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")

    print("✅ H3 test completed\n")


async def test_h4_deep_flow_real_data():
    """Test H4: Deep flow uses real foreign flow"""
    print("\n=== TEST H4: Deep Flow Real Data ===")

    # Import the endpoint function
    from app.api.routers.data import analyze_deep_flow

    test_symbols = ['SSI', 'VNM']

    for symbol in test_symbols:
        try:
            request = {"symbol": symbol, "days": 60}
            result = await analyze_deep_flow(request)

            print(f"✅ {symbol}:")
            print(f"   Flow Score: {result['flow_score']}")
            print(f"   Recommendation: {result['recommendation']}")
            print(f"   Insights: {len(result['insights'])} items")

            for insight in result['insights']:
                print(f"     - {insight['type']}: {insight['description']}")

        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")

    print("✅ H4 test completed\n")


async def test_h5_foreign_flow_real():
    """Test H5: Foreign flow in agent analysis uses real data"""
    print("\n=== TEST H5: Foreign Flow Real Data ===")

    vps = get_vps_connector()

    test_symbols = ['SSI', 'VNM', 'HPG']

    for symbol in test_symbols:
        try:
            flow_data = await vps.get_foreign_flow([symbol])

            net_bn = flow_data.get('net_value_billion', 0.0)
            flow_type = flow_data.get('flow_type', 'NEUTRAL')

            print(f"✅ {symbol}: {net_bn:+.2f}B VND ({flow_type})")

            # Show details if available
            stocks_flow = flow_data.get('stocks', [])
            if stocks_flow:
                stock = stocks_flow[0]
                buy_vol = stock.get('foreign_buy_volume', 0)
                sell_vol = stock.get('foreign_sell_volume', 0)
                print(f"   Buy: {buy_vol:,.0f} | Sell: {sell_vol:,.0f}")

        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")

    print("✅ H5 test completed\n")


async def test_vps_api_connectivity():
    """Test VPS API is working"""
    print("\n=== TEST: VPS API Connectivity ===")

    vps = get_vps_connector()

    try:
        # Test single stock
        stock_data = await vps.get_single_stock('SSI')
        if stock_data:
            print(f"✅ VPS API working - SSI: {stock_data['price_vnd']:,.0f} VND")
            print(f"   Source: {stock_data.get('source', 'vps')}")
            print(f"   Change: {stock_data['change_percent']:+.2f}%")
        else:
            print("⚠️ VPS API returned no data")
    except Exception as e:
        print(f"❌ VPS API error: {e}")

    print("✅ Connectivity test completed\n")


async def test_cafef_fallback():
    """Test CafeF fallback works"""
    print("\n=== TEST: CafeF Fallback ===")

    connector = get_realtime_connector()

    try:
        price = connector.get_stock_price('SSI')
        if price and price > 0:
            print(f"✅ CafeF working - SSI: {price:,.0f} VND")
        else:
            print("⚠️ CafeF returned no price")
    except Exception as e:
        print(f"❌ CafeF error: {e}")

    print("✅ Fallback test completed\n")


async def main():
    """Run all tests"""
    print("=" * 60)
    print("REAL DATA INTEGRATION TEST SUITE")
    print("Testing fixes: H3, H4, H5")
    print("=" * 60)

    try:
        await test_vps_api_connectivity()
        await test_cafef_fallback()
        await test_h3_broker_real_prices()
        await test_h4_deep_flow_real_data()
        await test_h5_foreign_flow_real()

        print("=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
