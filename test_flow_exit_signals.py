#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test flow-based exit signals in PositionExitScheduler
"""

import asyncio
from datetime import datetime, timedelta
from quantum_stock.autonomous.position_exit_scheduler import PositionExitScheduler, Position

async def mock_flow_fetcher(symbol: str):
    """Mock flow data for testing"""
    # Simulate different flow scenarios
    if symbol == "TEST_FOREIGN_PANIC":
        return {
            'net_buy_vol_1d': -250_000,  # Large foreign net sell
            'status': 'STRONG_DISTRIBUTION',
            'data_source': 'test'
        }
    elif symbol == "TEST_SMART_DISTRIBUTION":
        return {
            'net_buy_vol_1d': -50_000,
            'status': 'DISTRIBUTION',
            'data_source': 'test'
        }
    elif symbol == "TEST_ACCUMULATION":
        return {
            'net_buy_vol_1d': 100_000,
            'status': 'ACCUMULATION',
            'data_source': 'test'
        }
    else:
        return {
            'net_buy_vol_1d': 0,
            'status': 'NEUTRAL',
            'data_source': 'test'
        }

async def test_flow_exits():
    """Test flow-based exit signals"""

    print("=" * 70)
    print("Testing Flow-Based Exit Signals")
    print("=" * 70)

    # Create scheduler with flow fetcher
    scheduler = PositionExitScheduler(
        check_interval=5,
        flow_fetcher=mock_flow_fetcher
    )

    # Test 1: FOREIGN_PANIC_SELL
    print("\n[Test 1] FOREIGN_PANIC_SELL")
    print("-" * 70)
    position1 = Position(
        symbol="TEST_FOREIGN_PANIC",
        quantity=500,
        avg_price=26_500,
        entry_date=datetime.now() - timedelta(days=3),  # T+3 (can sell)
        current_price=27_000,  # Small profit
        take_profit_pct=0.15,
        trailing_stop_pct=0.05,
        stop_loss_pct=-0.05
    )
    position1.update_price(27_000)

    # Fetch flow data
    flow_data = await mock_flow_fetcher("TEST_FOREIGN_PANIC")
    scheduler.flow_data_cache["TEST_FOREIGN_PANIC"] = flow_data
    print(f"Flow data: {flow_data}")

    # Check exit
    exit_reason = await scheduler._should_exit(position1)
    print(f"Exit reason: {exit_reason}")
    assert exit_reason == "FOREIGN_PANIC_SELL", f"Expected FOREIGN_PANIC_SELL, got {exit_reason}"
    print("✓ FOREIGN_PANIC_SELL detected correctly")

    # Test 2: SMART_MONEY_DISTRIBUTION
    print("\n[Test 2] SMART_MONEY_DISTRIBUTION")
    print("-" * 70)
    position2 = Position(
        symbol="TEST_SMART_DISTRIBUTION",
        quantity=500,
        avg_price=26_500,
        entry_date=datetime.now() - timedelta(days=4),  # T+4 (held >= 3 days)
        current_price=26_800,
        take_profit_pct=0.15,
        trailing_stop_pct=0.05,
        stop_loss_pct=-0.05
    )
    position2.update_price(26_800)

    flow_data = await mock_flow_fetcher("TEST_SMART_DISTRIBUTION")
    scheduler.flow_data_cache["TEST_SMART_DISTRIBUTION"] = flow_data
    print(f"Flow data: {flow_data}")

    exit_reason = await scheduler._should_exit(position2)
    print(f"Exit reason: {exit_reason}")
    assert exit_reason == "SMART_MONEY_DISTRIBUTION", f"Expected SMART_MONEY_DISTRIBUTION, got {exit_reason}"
    print("✓ SMART_MONEY_DISTRIBUTION detected correctly")

    # Test 3: FOMO_EXHAUSTION_EXIT
    print("\n[Test 3] FOMO_EXHAUSTION_EXIT")
    print("-" * 70)
    position3 = Position(
        symbol="TEST_SMART_DISTRIBUTION",  # Reuse distribution flow
        quantity=500,
        avg_price=26_500,
        entry_date=datetime.now() - timedelta(days=3),  # T+3
        current_price=27_200,  # Below peak
        peak_price=28_500,  # Peak was much higher (4.6% drop from peak)
        take_profit_pct=0.15,
        trailing_stop_pct=0.05,
        stop_loss_pct=-0.05
    )
    position3.update_price(27_200)

    flow_data = await mock_flow_fetcher("TEST_SMART_DISTRIBUTION")
    scheduler.flow_data_cache["TEST_SMART_DISTRIBUTION"] = flow_data
    print(f"Flow data: {flow_data}")
    print(f"Peak: {position3.peak_price:,.0f}, Current: {position3.current_price:,.0f}")
    print(f"Drop from peak: {((position3.peak_price - position3.current_price) / position3.peak_price * 100):.2f}%")

    exit_reason = await scheduler._should_exit(position3)
    print(f"Exit reason: {exit_reason}")
    # Should trigger FOREIGN_PANIC or SMART_DISTRIBUTION or FOMO_EXHAUSTION
    assert exit_reason in ["FOREIGN_PANIC_SELL", "SMART_MONEY_DISTRIBUTION", "FOMO_EXHAUSTION_EXIT"], \
        f"Expected flow-based exit, got {exit_reason}"
    print(f"✓ Flow-based exit triggered: {exit_reason}")

    # Test 4: Priority order - STOP_LOSS should override flow exits
    print("\n[Test 4] Priority: STOP_LOSS > flow exits")
    print("-" * 70)
    position4 = Position(
        symbol="TEST_FOREIGN_PANIC",
        quantity=500,
        avg_price=26_500,
        entry_date=datetime.now() - timedelta(days=3),
        current_price=25_000,  # -5.7% (below stop loss)
        take_profit_pct=0.15,
        trailing_stop_pct=0.05,
        stop_loss_pct=-0.05
    )
    position4.update_price(25_000)

    flow_data = await mock_flow_fetcher("TEST_FOREIGN_PANIC")
    scheduler.flow_data_cache["TEST_FOREIGN_PANIC"] = flow_data
    print(f"Flow data: {flow_data} (should be ignored)")
    print(f"PnL: {position4.unrealized_pnl_pct*100:.2f}% (below {position4.stop_loss_pct*100:.0f}% SL)")

    exit_reason = await scheduler._should_exit(position4)
    print(f"Exit reason: {exit_reason}")
    assert exit_reason == "STOP_LOSS", f"Expected STOP_LOSS to override flow exits, got {exit_reason}"
    print("✓ STOP_LOSS correctly prioritized over flow exits")

    # Test 5: T+2 compliance - should not exit before T+2
    print("\n[Test 5] T+2 compliance - no exit before T+2")
    print("-" * 70)
    position5 = Position(
        symbol="TEST_FOREIGN_PANIC",
        quantity=500,
        avg_price=26_500,
        entry_date=datetime.now() - timedelta(hours=12),  # Less than 1 trading day
        current_price=25_000,  # Large loss but can't sell yet
        take_profit_pct=0.15,
        trailing_stop_pct=0.05,
        stop_loss_pct=-0.05
    )
    position5.update_price(25_000)

    flow_data = await mock_flow_fetcher("TEST_FOREIGN_PANIC")
    scheduler.flow_data_cache["TEST_FOREIGN_PANIC"] = flow_data
    print(f"Trading days held: {position5.trading_days_held}")
    print(f"Can sell: {position5.can_sell}")
    print(f"PnL: {position5.unrealized_pnl_pct*100:.2f}% (would trigger SL if T+2 met)")

    exit_reason = await scheduler._should_exit(position5)
    print(f"Exit reason: {exit_reason}")
    assert exit_reason is None, f"Expected None (T+2 not satisfied), got {exit_reason}"
    print("✓ T+2 compliance enforced correctly (blocks even stop-loss exit)")

    print("\n" + "=" * 70)
    print("All flow-based exit signal tests passed!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(test_flow_exits())
