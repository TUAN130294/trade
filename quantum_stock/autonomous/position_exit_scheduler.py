# -*- coding: utf-8 -*-
"""
Position Exit Scheduler
========================
Tự động exit positions dựa trên:
1. Max profit target (take-profit)
2. Trailing stop (bảo vệ lợi nhuận)
3. Stop loss (cắt lỗ)

QUAN TRỌNG:
- KHÔNG tự động exit sau T+2.5
- NHƯNG vẫn tuân thủ luật T+2.5 (không được bán trước T+2)
- Chỉ exit khi ĐỦ T+2 VÀ đạt điều kiện profit/stop
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
import logging
logger = logging.getLogger(__name__)

# Import holidays from single source of truth (DRY principle)
from quantum_stock.core.vn_market_rules import VN_HOLIDAYS

# Build flat set of all holiday dates from vn_market_rules
VN_HOLIDAY_DATES = set()
for year_holidays in VN_HOLIDAYS.values():
    VN_HOLIDAY_DATES.update(year_holidays)


def count_trading_days(start_date: datetime, end_date: datetime) -> int:
    """
    Count trading days between two dates (EXCLUDES start date, includes end date)

    Vietnam Stock Exchange (HOSE/HNX) operates Monday-Friday only.
    This function excludes:
    - Weekends (Saturday/Sunday)
    - VN public holidays (Tết, Quốc khánh, etc.)
    - Start date (buy date itself is T+0, not counted)

    Args:
        start_date: Start datetime (buy date, excluded from count)
        end_date: End datetime (included in count)

    Returns:
        Number of trading days AFTER start_date (Mon-Fri only, excluding holidays)

    Example:
        Buy Monday (T+0) → Can sell Wednesday (T+2)
        Buy Friday (T+0) → Can sell Tuesday (T+2, skips weekend)
        Buy Jan 27 2025 (Mon) → Cannot sell until Feb 4 (after Tết)

    CRITICAL: Start date is EXCLUDED to match T+2 rule correctly.
              If buy on Monday, Tuesday is T+1, Wednesday is T+2.
    """
    # Start counting from day AFTER buy date (exclude start_date)
    current = (start_date + timedelta(days=1)).date()
    end = end_date.date()
    trading_days = 0

    while current <= end:
        is_weekend = current.weekday() >= 5  # Saturday = 5, Sunday = 6
        is_holiday = current in VN_HOLIDAY_DATES

        if not is_weekend and not is_holiday:
            trading_days += 1
        current += timedelta(days=1)

    return trading_days


@dataclass
class Position:
    """Trading position with exit tracking"""
    symbol: str
    quantity: int
    avg_price: float
    entry_date: datetime

    # Current state
    current_price: float = 0.0
    peak_price: float = 0.0  # Highest price since entry

    # P&L
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    # Exit parameters
    take_profit_pct: float = 0.15  # 15% profit target
    trailing_stop_pct: float = 0.05  # 5% trailing stop
    stop_loss_pct: float = -0.05  # -5% stop loss

    # Trailing stop state
    has_trailing_stop: bool = True
    trailing_stop_price: float = 0.0

    # Dynamic Exit State
    atr: float = 0.0          # Volatility (ATR) for dynamic exits
    time_decay_threshold: int = 5  # Days to hold before checking time decay

    # Entry metadata
    entry_reason: str = ""  # Reason for entering position

    # Exit metadata (set by update_price and exit logic)
    trading_days_held: int = 0
    can_sell: bool = False
    exit_reason: str = ""

    def __post_init__(self):
        if self.peak_price == 0:
            self.peak_price = self.avg_price
        if self.trailing_stop_price == 0:
            self.trailing_stop_price = self.avg_price * (1 - self.trailing_stop_pct)

    def update_price(self, price: float):
        """Update current price and recalculate P&L"""
        self.current_price = price

        # Update P&L
        self.unrealized_pnl = (price - self.avg_price) * self.quantity
        self.unrealized_pnl_pct = (price - self.avg_price) / self.avg_price if self.avg_price > 0 else 0

        # Update trading days held (excludes weekends)
        self.trading_days_held = count_trading_days(self.entry_date, datetime.now())

        # Check T+2 compliance (2 TRADING days, not calendar days)
        self.can_sell = self.trading_days_held >= 2

    def update_trailing_stop(self):
        """
        Update trailing stop logic:
        1. Base: Peak price * (1 - pct)
        2. Dynamic: Tighten stop if profit > 5%
        3. ATR: Use ATR-based stop if available
        """
        if self.current_price > self.peak_price:
            self.peak_price = self.current_price
            
            # Dynamic Tightening: If profit > 5%, tighten trailing stop to capture range
            current_pct = self.unrealized_pnl_pct
            dynamic_trailing_pct = self.trailing_stop_pct
            
            if current_pct > 0.05: # > 5% profit
                # Tighten by half (e.g., 5% -> 2.5%)
                dynamic_trailing_pct = max(0.025, self.trailing_stop_pct * 0.5)
            
            # ATR-based trailing (if ATR provided)
            if self.atr > 0:
                # Trailing = 2 * ATR
                atr_pct = (self.atr * 2) / self.peak_price
                dynamic_trailing_pct = atr_pct

            self.trailing_stop_price = self.peak_price * (1 - dynamic_trailing_pct)

            logger.debug(
                f"Trailing stop updated {self.symbol}: "
                f"Peak {self.peak_price:,.0f} | "
                f"Stop {self.trailing_stop_price:,.0f} ({dynamic_trailing_pct*100:.1f}%)"
            )

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_price': self.avg_price,
            'entry_date': self.entry_date.isoformat(),
            'current_price': self.current_price,
            'peak_price': self.peak_price,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'take_profit_pct': self.take_profit_pct,
            'trailing_stop_pct': self.trailing_stop_pct,
            'stop_loss_pct': self.stop_loss_pct,
            'trailing_stop_price': self.trailing_stop_price,
            'trading_days_held': self.trading_days_held,
            'can_sell': self.can_sell,
            'entry_reason': self.entry_reason,
            'exit_reason': self.exit_reason
        }


class PositionExitScheduler:
    """
    Monitor positions và tự động exit khi đạt điều kiện

    Exit logic:
    1. CHỈ exit nếu >= T+2 (tuân thủ luật VN)
    2. Exit khi:
       - Take profit hit (đạt max profit)
       - Trailing stop hit (bảo vệ lợi nhuận)
       - Stop loss hit (cắt lỗ)

    KHÔNG tự động exit chỉ vì đủ T+2.5
    """

    def __init__(
        self,
        check_interval: int = 60,  # Check mỗi 1 phút
        price_fetcher: Optional[Callable] = None,
        flow_fetcher: Optional[Callable] = None
    ):
        self.check_interval = check_interval
        self.price_fetcher = price_fetcher or self._mock_price_fetcher
        self.flow_fetcher = flow_fetcher  # MarketFlowConnector injected from orchestrator

        # Positions
        self.positions: Dict[str, Position] = {}

        # Callbacks
        self.on_exit_callbacks: List[Callable] = []

        # State
        self.is_running = False

        # Flow data cache
        self.flow_data_cache: Dict[str, Dict] = {}  # symbol -> flow data

    def add_exit_callback(self, callback: Callable):
        """Add callback for position exits"""
        self.on_exit_callbacks.append(callback)

    def add_position(self, position: Position):
        """Add position to monitor"""
        self.positions[position.symbol] = position
        logger.info(
            f"➕ Position added: {position.symbol} | "
            f"Qty: {position.quantity} @ {position.avg_price:,.0f} | "
            f"TP: +{position.take_profit_pct*100:.0f}% | "
            f"Trail: {position.trailing_stop_pct*100:.0f}% | "
            f"SL: {position.stop_loss_pct*100:.0f}%"
        )

    def remove_position(self, symbol: str):
        """Remove position from monitoring"""
        if symbol in self.positions:
            del self.positions[symbol]
            logger.info(f"➖ Position removed: {symbol}")

    async def start(self):
        """Start monitoring positions"""
        self.is_running = True
        logger.info(
            f"Position exit scheduler started\n"
            f"  - Check interval: {self.check_interval}s\n"
            f"  - T+2 compliance: ENFORCED\n"
            f"  - Auto exit after T+2.5: DISABLED"
        )

        while self.is_running:
            try:
                await self.check_all_positions()
                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Position monitor error: {e}")
                await asyncio.sleep(60)

    def stop(self):
        """Stop monitoring"""
        self.is_running = False
        logger.info("Position exit scheduler stopped")

    async def check_all_positions(self):
        """Check all positions for exit conditions"""
        if not self.positions:
            return

        for symbol, position in list(self.positions.items()):
            try:
                # 1. Update current price
                current_price = await self.price_fetcher(symbol)
                position.update_price(current_price)

                # 2. Update trailing stop if price increased
                position.update_trailing_stop()

                # 3. Update flow data if flow_fetcher available
                if self.flow_fetcher:
                    try:
                        flow_data = await self.flow_fetcher(symbol)
                        self.flow_data_cache[symbol] = flow_data
                    except Exception as e:
                        logger.warning(f"Flow data fetch failed for {symbol}: {e}")

                # 4. Check exit conditions
                exit_reason = await self._should_exit(position)

                if exit_reason:
                    # Exit triggered
                    await self._execute_exit(position, exit_reason)

            except Exception as e:
                logger.error(f"Error checking position {symbol}: {e}")

    async def _should_exit(self, position: Position) -> Optional[str]:
        """
        Determine if position should exit

        Priority order: STOP_LOSS > ATR_STOP > TRAILING > FOREIGN_PANIC > SMART_DISTRIBUTION > FOMO_EXHAUSTION > TAKE_PROFIT > LIQUIDITY_DRY > TIME_DECAY

        Returns:
            exit_reason (str) if should exit, None otherwise
        """

        # CRITICAL: Chỉ exit nếu đã đủ T+2 (tuân thủ luật VN)
        if not position.can_sell:
            logger.debug(
                f"{position.symbol}: Cannot sell yet (T+{position.trading_days_held} trading days < T+2)"
            )
            return None

        # 1. STOP_LOSS (highest priority - protect capital)
        if position.unrealized_pnl_pct <= position.stop_loss_pct:
            logger.info(
                f"{position.symbol}: Stop loss hit ({position.unrealized_pnl_pct*100:.2f}% <= {position.stop_loss_pct*100:.2f}%)"
            )
            return "STOP_LOSS"

        # 2. ATR_STOP (dynamic stop based on volatility)
        if position.atr > 0:
            # SL = 2 * ATR
            atr_sl_price = position.avg_price - (position.atr * 2)
            if position.current_price <= atr_sl_price:
                logger.info(
                    f"{position.symbol}: ATR stop hit (price {position.current_price:,.0f} <= ATR stop {atr_sl_price:,.0f})"
                )
                return "ATR_STOP_LOSS"

        # 3. TRAILING_STOP (protect profits)
        if position.has_trailing_stop:
            if position.current_price <= position.trailing_stop_price:
                logger.info(
                    f"{position.symbol}: Trailing stop hit (price {position.current_price:,.0f} <= stop {position.trailing_stop_price:,.0f})"
                )
                return "TRAILING_STOP"

        # ========== FLOW-BASED EXIT SIGNALS (NEW) ==========

        flow_data = self.flow_data_cache.get(position.symbol, {})

        # 4. FOREIGN_PANIC_SELL (khối ngoại bán ròng mạnh)
        foreign_net_vol = flow_data.get('net_buy_vol_1d', 0)
        if foreign_net_vol is not None and foreign_net_vol < 0:
            # Check if foreign net sell > 2x 20-day average
            # Simplified: if net sell volume > 200k shares (proxy for 2x avg)
            if abs(foreign_net_vol) > 200_000:
                logger.info(
                    f"{position.symbol}: FOREIGN_PANIC_SELL detected (net sell: {foreign_net_vol:,} shares)"
                )
                return "FOREIGN_PANIC_SELL"

        # 5. SMART_MONEY_DISTRIBUTION (volume cao + giá sideway + close location giảm)
        # Need historical data to check 3+ days pattern
        # Simplified check: if flow status is STRONG_DISTRIBUTION
        flow_status = flow_data.get('status', 'NEUTRAL')
        if flow_status in ['STRONG_DISTRIBUTION', 'DISTRIBUTION'] and position.trading_days_held >= 3:
            logger.info(
                f"{position.symbol}: SMART_MONEY_DISTRIBUTION detected (status: {flow_status}, held {position.trading_days_held} days)"
            )
            return "SMART_MONEY_DISTRIBUTION"

        # 6. FOMO_EXHAUSTION_EXIT (FOMO signal transition from PEAK → EXHAUSTION)
        # Check for FOMO exhaustion pattern: high volume spike but price declining
        # Proxy: volume > 2x avg but unrealized PnL dropping from peak
        if position.trading_days_held >= 2:
            # Check if current price significantly below peak (>3% drop from peak)
            peak_drop_pct = (position.peak_price - position.current_price) / position.peak_price
            if peak_drop_pct > 0.03:  # 3% drop from peak
                # If also seeing distribution flow status
                if flow_status in ['DISTRIBUTION', 'STRONG_DISTRIBUTION']:
                    logger.info(
                        f"{position.symbol}: FOMO_EXHAUSTION_EXIT detected (peak drop: {peak_drop_pct*100:.2f}%, flow: {flow_status})"
                    )
                    return "FOMO_EXHAUSTION_EXIT"

        # 7. LIQUIDITY_DRY_UP (volume < 30% of 20-day avg after 3+ days)
        if position.trading_days_held >= 3:
            # Try to get volume ratio from historical data
            volume_ratio = await self._get_volume_ratio(position.symbol)
            if volume_ratio is not None and volume_ratio < 0.3:
                logger.info(
                    f"{position.symbol}: LIQUIDITY_DRY_UP detected (volume ratio: {volume_ratio*100:.1f}% of 20-day avg)"
                )
                return "LIQUIDITY_DRY_UP"

        # ========== END FLOW-BASED EXITS ==========

        # 8. TAKE_PROFIT (standard profit target)
        if position.unrealized_pnl_pct >= position.take_profit_pct:
            logger.info(
                f"{position.symbol}: Take profit hit ({position.unrealized_pnl_pct*100:.2f}% >= {position.take_profit_pct*100:.2f}%)"
            )
            return "TAKE_PROFIT"

        # 9. TIME_DECAY_ROTATION (weak momentum after T+5)
        if position.trading_days_held >= position.time_decay_threshold:
            if position.unrealized_pnl_pct < 0.02:
                logger.info(
                    f"{position.symbol}: Weak momentum (T+{position.trading_days_held}, PnL < 2%) -> Rotate"
                )
                return "TIME_DECAY_ROTATION"

        return None

    async def _execute_exit(self, position: Position, exit_reason: str):
        """Execute position exit"""
        position.exit_reason = exit_reason

        logger.info(
            f"🔄 AUTO-EXIT: {position.symbol} [{exit_reason}]\n"
            f"   Entry: {position.avg_price:,.0f} @ {position.entry_date.strftime('%Y-%m-%d')}\n"
            f"   Exit: {position.current_price:,.0f} (T+{position.trading_days_held} trading days)\n"
            f"   P&L: {position.unrealized_pnl:,.0f} VND ({position.unrealized_pnl_pct*100:+.2f}%)\n"
            f"   Peak: {position.peak_price:,.0f}"
        )

        # Notify callbacks
        for callback in self.on_exit_callbacks:
            try:
                await callback(position, exit_reason)
            except Exception as e:
                logger.error(f"Exit callback error: {e}")

        # Remove from monitoring
        self.remove_position(position.symbol)

    async def _mock_price_fetcher(self, symbol: str) -> float:
        """
        Price fetcher with multi-layer fallback

        Priority:
        1. vnstock3 (real-time) - preferred
        2. Parquet file (historical) - reliable fallback
        3. Default prices dict (common stocks)
        4. Position avg_price with small variation (last resort)
        """
        # Default prices for common VN stocks (as of Dec 2024)
        DEFAULT_PRICES = {
            'ACB': 26.5, 'VCB': 92.5, 'TCB': 23.5, 'MBB': 25.3,
            'HDB': 32.8, 'STB': 18.5, 'TPB': 39.5, 'VPB': 19.8,
            'BID': 48.0, 'CTG': 35.5, 'SSI': 45.2, 'HPG': 27.8,
            'FPT': 128.0, 'VNM': 78.5, 'VIC': 45.6, 'VHM': 48.2,
            'MWG': 52.0, 'GAS': 85.0, 'PLX': 42.5, 'POW': 13.8,
            'MSN': 85.0, 'VRE': 28.5, 'VJC': 98.0, 'SAB': 58.0,
            'GVR': 32.0, 'BCM': 65.0, 'BVH': 45.0, 'NVL': 13.5,
        }

        # 1. Try vnstock3 first (real-time)
        try:
            from vnstock3 import Vnstock

            stock = Vnstock().stock(symbol=symbol, source='VCI')
            df = stock.quote.history(
                start='2024-12-01',
                end=datetime.now().strftime('%Y-%m-%d')
            )

            if len(df) > 0:
                price = float(df.iloc[-1]['close'])
                logger.debug(f"Price for {symbol}: {price:,.0f} VND (vnstock)")
                return price

        except Exception as e:
            logger.debug(f"vnstock fetch failed for {symbol}: {e}")

        # 2. Fallback: Try parquet file
        try:
            import pandas as pd
            from pathlib import Path

            parquet_path = Path(f"data/historical/{symbol}.parquet")
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                df = df.sort_values('date')
                price = float(df.iloc[-1]['close'])
                logger.debug(f"Price for {symbol}: {price:,.0f} VND (parquet)")
                return price

        except Exception as e:
            logger.debug(f"Parquet fetch failed for {symbol}: {e}")

        # 3. Fallback: Default prices
        if symbol in DEFAULT_PRICES:
            # Add small random variation to simulate market movement
            import random
            base_price = DEFAULT_PRICES[symbol]
            price = base_price * (1 + random.uniform(-0.01, 0.01))
            logger.debug(f"Price for {symbol}: {price:,.2f} VND (default)")
            return price

        # 4. Last resort: Position avg_price with variation
        if symbol in self.positions:
            import random
            base_price = self.positions[symbol].avg_price
            price = base_price * (1 + random.uniform(-0.02, 0.02))
            logger.warning(f"Using position avg_price for {symbol}: {price:,.0f} VND")
            return price

        logger.error(f"No price available for {symbol}")
        return 0.0

    def get_all_positions(self) -> List[Position]:
        """Get all monitored positions"""
        return list(self.positions.values())

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get specific position"""
        return self.positions.get(symbol)

    async def _get_volume_ratio(self, symbol: str) -> Optional[float]:
        """
        Get current volume ratio vs 20-day average

        Returns:
            Volume ratio (current / 20-day avg) or None if data unavailable
        """
        try:
            import pandas as pd
            from pathlib import Path

            # Try to read from parquet file
            parquet_path = Path(f"data/historical/{symbol}.parquet")
            if not parquet_path.exists():
                return None

            df = pd.read_parquet(parquet_path)
            if len(df) < 20:
                return None

            df = df.sort_values('date')

            # Get last 20 days of volume
            last_20 = df.tail(20)
            avg_volume = last_20['volume'].mean()

            # Get current day volume (last row)
            current_volume = df.iloc[-1]['volume']

            if avg_volume > 0:
                return current_volume / avg_volume

            return None

        except Exception as e:
            logger.debug(f"Volume ratio fetch failed for {symbol}: {e}")
            return None


# Example usage
if __name__ == "__main__":
    async def on_exit(position: Position, reason: str):
        print(f"Position exited: {position.symbol}")
        print(f"  Reason: {reason}")
        print(f"  P&L: {position.unrealized_pnl:,.0f} ({position.unrealized_pnl_pct*100:+.2f}%)")

    scheduler = PositionExitScheduler(check_interval=5)
    scheduler.add_exit_callback(on_exit)

    # Add test position
    test_position = Position(
        symbol="ACB",
        quantity=500,
        avg_price=26500,
        entry_date=datetime.now() - timedelta(days=2.5),  # T+2.5
        take_profit_pct=0.15,
        trailing_stop_pct=0.05,
        stop_loss_pct=-0.05
    )

    scheduler.add_position(test_position)

    # Run for 30 seconds
    async def test():
        task = asyncio.create_task(scheduler.start())
        await asyncio.sleep(30)
        scheduler.stop()
        await task

    asyncio.run(test())
