# -*- coding: utf-8 -*-
"""
Vietnam Stock Market Rules & Regulations
========================================
Comprehensive implementation of VN market-specific rules

Includes:
1. T+2.5 Settlement rules
2. Tick size by price range
3. Ceiling/Floor limits (±7% HOSE, ±10% HNX)
4. ATO/ATC auction sessions
5. Trading hours validation
6. Lot size requirements
7. Holiday calendar
"""

from datetime import datetime, date, time, timedelta
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
import pandas as pd


class Exchange(Enum):
    """Vietnam stock exchanges"""
    HOSE = "HOSE"  # Ho Chi Minh Stock Exchange
    HNX = "HNX"    # Hanoi Stock Exchange
    UPCOM = "UPCOM"


class SessionType(Enum):
    """Trading session types"""
    PRE_OPEN = "pre_open"           # 8:30-9:00
    ATO = "ato"                      # 9:00-9:15 (Opening auction)
    CONTINUOUS_AM = "continuous_am"  # 9:15-11:30
    LUNCH_BREAK = "lunch"           # 11:30-13:00
    CONTINUOUS_PM = "continuous_pm"  # 13:00-14:30
    ATC = "atc"                      # 14:30-14:45 (Closing auction)
    POST_CLOSE = "post_close"        # 14:45-15:00
    CLOSED = "closed"


@dataclass
class TradingSession:
    """Trading session information"""
    session_type: SessionType
    start_time: time
    end_time: time
    can_place_order: bool
    can_cancel_order: bool
    order_types_allowed: List[str]


# Vietnam Holidays (configurable)
VN_HOLIDAYS = {
    2025: [
        date(2025, 1, 1),   # Tết Dương lịch
        date(2025, 1, 28),  # Tết Nguyên đán
        date(2025, 1, 29),
        date(2025, 1, 30),
        date(2025, 1, 31),
        date(2025, 2, 1),
        date(2025, 2, 2),
        date(2025, 2, 3),
        date(2025, 4, 7),   # Giỗ Tổ Hùng Vương
        date(2025, 4, 30),  # Giải phóng miền Nam
        date(2025, 5, 1),   # Quốc tế Lao động
        date(2025, 9, 2),   # Quốc khánh
        date(2025, 9, 3),
    ],
    2026: [
        date(2026, 1, 1),   # Tết Dương lịch
        date(2026, 2, 16),  # Tết Nguyên đán (dự kiến)
        date(2026, 2, 17),
        date(2026, 2, 18),
        date(2026, 2, 19),
        date(2026, 2, 20),
        date(2026, 4, 2),   # Giỗ Tổ Hùng Vương
        date(2026, 4, 30),  # Giải phóng miền Nam
        date(2026, 5, 1),   # Quốc tế Lao động
        date(2026, 9, 2),   # Quốc khánh
    ],
    2027: [
        date(2027, 1, 1),   # Tết Dương lịch
        date(2027, 2, 5),   # Tết Nguyên đán (dự kiến)
        date(2027, 2, 6),
        date(2027, 2, 7),
        date(2027, 2, 8),
        date(2027, 2, 9),
        date(2027, 2, 10),
        date(2027, 2, 11),
        date(2027, 4, 21),  # Giỗ Tổ Hùng Vương
        date(2027, 4, 30),  # Giải phóng miền Nam
        date(2027, 5, 1),   # Quốc tế Lao động
        date(2027, 5, 3),   # Compensatory day
        date(2027, 9, 2),   # Quốc khánh
        date(2027, 9, 3),
    ]
}


class VNMarketRules:
    """
    Vietnam Stock Market Rules Implementation

    Key rules:
    - T+2.5: Can sell from ATO on T+3, or ATC on T+2
    - Tick size varies by price range
    - Ceiling/Floor: ±7% HOSE, ±10% HNX, ±15% UPCOM
    - Lot size: 100 shares (standard), 10 shares (odd lot)
    """

    # Price limits by exchange
    PRICE_LIMITS = {
        Exchange.HOSE: 0.07,   # ±7%
        Exchange.HNX: 0.10,    # ±10%
        Exchange.UPCOM: 0.15,  # ±15%
    }

    # Tick sizes for HOSE (VND)
    TICK_SIZES_HOSE = [
        (10000, 10),      # Price < 10,000: tick = 10
        (50000, 50),      # 10,000 <= Price < 50,000: tick = 50
        (float('inf'), 100),  # Price >= 50,000: tick = 100
    ]

    # Tick sizes for HNX
    TICK_SIZES_HNX = [
        (10000, 10),
        (float('inf'), 100),
    ]

    # Standard lot size
    STANDARD_LOT = 100
    ODD_LOT_MIN = 1
    ODD_LOT_MAX = 99

    # Trading sessions
    SESSIONS = [
        TradingSession(SessionType.PRE_OPEN, time(8, 30), time(9, 0),
                      can_place_order=True, can_cancel_order=True,
                      order_types_allowed=['LO', 'ATO']),
        TradingSession(SessionType.ATO, time(9, 0), time(9, 15),
                      can_place_order=False, can_cancel_order=False,
                      order_types_allowed=[]),
        TradingSession(SessionType.CONTINUOUS_AM, time(9, 15), time(11, 30),
                      can_place_order=True, can_cancel_order=True,
                      order_types_allowed=['LO', 'MP', 'ATC']),
        TradingSession(SessionType.LUNCH_BREAK, time(11, 30), time(13, 0),
                      can_place_order=True, can_cancel_order=True,
                      order_types_allowed=['LO']),
        TradingSession(SessionType.CONTINUOUS_PM, time(13, 0), time(14, 30),
                      can_place_order=True, can_cancel_order=True,
                      order_types_allowed=['LO', 'MP', 'ATC']),
        TradingSession(SessionType.ATC, time(14, 30), time(14, 45),
                      can_place_order=True, can_cancel_order=False,
                      order_types_allowed=['ATC']),
        TradingSession(SessionType.POST_CLOSE, time(14, 45), time(15, 0),
                      can_place_order=False, can_cancel_order=False,
                      order_types_allowed=[]),
    ]

    def __init__(self):
        self._holidays_cache = {}

    # ==========================================
    # Tick Size
    # ==========================================

    def get_tick_size(self, price: float, exchange: Exchange = Exchange.HOSE) -> float:
        """
        Get tick size for a given price

        Args:
            price: Current price in VND (already in thousands, e.g., 26.5 = 26,500 VND)
            exchange: Stock exchange

        Returns:
            Tick size in VND
        """
        # Convert to full VND if needed (handle both formats)
        if price < 1000:  # Likely in thousands (e.g., 26.5)
            price_vnd = price * 1000
        else:
            price_vnd = price

        tick_sizes = self.TICK_SIZES_HOSE if exchange == Exchange.HOSE else self.TICK_SIZES_HNX

        for threshold, tick in tick_sizes:
            if price_vnd < threshold:
                return tick

        return 100  # Default

    def round_to_tick(self, price: float, exchange: Exchange = Exchange.HOSE) -> float:
        """Round price to nearest valid tick"""
        tick = self.get_tick_size(price, exchange)

        # Convert to VND for calculation
        if price < 1000:
            price_vnd = price * 1000
            result = round(price_vnd / tick) * tick
            return result / 1000  # Back to thousands
        else:
            return round(price / tick) * tick

    # ==========================================
    # Price Limits (Ceiling/Floor)
    # ==========================================

    def get_price_limits(self, reference_price: float,
                         exchange: Exchange = Exchange.HOSE) -> Tuple[float, float]:
        """
        Calculate ceiling and floor prices

        Args:
            reference_price: Reference price (previous close)
            exchange: Stock exchange

        Returns:
            Tuple of (floor_price, ceiling_price)
        """
        limit_pct = self.PRICE_LIMITS.get(exchange, 0.07)

        ceiling = reference_price * (1 + limit_pct)
        floor = reference_price * (1 - limit_pct)

        # Round to tick
        ceiling = self.round_to_tick(ceiling, exchange)
        floor = self.round_to_tick(floor, exchange)

        return (floor, ceiling)

    def validate_price(self, price: float, reference_price: float,
                       exchange: Exchange = Exchange.HOSE) -> Tuple[bool, str]:
        """
        Validate if price is within limits

        Returns:
            Tuple of (is_valid, error_message)
        """
        floor, ceiling = self.get_price_limits(reference_price, exchange)

        if price > ceiling:
            return False, f"Price {price} exceeds ceiling {ceiling}"
        if price < floor:
            return False, f"Price {price} below floor {floor}"

        # Check tick size
        tick = self.get_tick_size(price, exchange)
        if price < 1000:
            price_vnd = price * 1000
        else:
            price_vnd = price

        if price_vnd % tick != 0:
            return False, f"Price {price} not aligned to tick size {tick}"

        return True, ""

    # ==========================================
    # Lot Size
    # ==========================================

    def validate_quantity(self, quantity: int, is_odd_lot: bool = False) -> Tuple[bool, str]:
        """
        Validate order quantity

        Returns:
            Tuple of (is_valid, error_message)
        """
        if quantity <= 0:
            return False, "Quantity must be positive"

        if is_odd_lot:
            if quantity > self.ODD_LOT_MAX:
                return False, f"Odd lot quantity must be <= {self.ODD_LOT_MAX}"
        else:
            if quantity % self.STANDARD_LOT != 0:
                return False, f"Quantity must be multiple of {self.STANDARD_LOT}"

        return True, ""

    def round_to_lot(self, quantity: int) -> int:
        """Round quantity to nearest valid lot size"""
        return (quantity // self.STANDARD_LOT) * self.STANDARD_LOT

    # ==========================================
    # T+2.5 Settlement
    # ==========================================

    def is_trading_day(self, d: date) -> bool:
        """Check if date is a trading day"""
        # Weekend check
        if d.weekday() >= 5:
            return False

        # Holiday check
        year = d.year
        holidays = VN_HOLIDAYS.get(year, [])

        return d not in holidays

    def get_next_trading_day(self, d: date) -> date:
        """Get next trading day from given date"""
        next_day = d + timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)
        return next_day

    def count_trading_days(self, start_date: date, end_date: date) -> int:
        """Count trading days between two dates (exclusive of start)"""
        if end_date <= start_date:
            return 0

        count = 0
        current = start_date + timedelta(days=1)

        while current <= end_date:
            if self.is_trading_day(current):
                count += 1
            current += timedelta(days=1)

        return count

    def can_sell_position(self, entry_date: date, current_datetime: datetime = None) -> Tuple[bool, str]:
        """
        Check if position can be sold based on T+2.5 rule

        T+2.5 means:
        - Can sell from ATO on T+3
        - OR can sell in ATC session on T+2 (after 14:30)

        Args:
            entry_date: Date when position was opened
            current_datetime: Current datetime (default: now)

        Returns:
            Tuple of (can_sell, reason)
        """
        if current_datetime is None:
            current_datetime = datetime.now()

        current_date = current_datetime.date()
        current_time = current_datetime.time()

        # Count trading days since entry
        trading_days = self.count_trading_days(entry_date, current_date)

        # T+3 or later: Can sell anytime
        if trading_days >= 3:
            return True, f"T+{trading_days}: Có thể bán bất kỳ lúc nào"

        # T+2: Can only sell in ATC session (14:30-14:45)
        if trading_days == 2:
            if time(14, 30) <= current_time <= time(14, 45):
                return True, "T+2 ATC: Có thể bán trong phiên ATC"
            else:
                return False, f"T+2: Chưa đến phiên ATC (cần đợi sau 14:30 hoặc đợi T+3)"

        # T+0 or T+1: Cannot sell
        return False, f"T+{trading_days}: Chưa đủ thời gian thanh toán (cần T+2.5)"

    def get_earliest_sell_datetime(self, entry_date: date) -> datetime:
        """
        Get earliest datetime when position can be sold

        Returns datetime of ATC session on T+2 or ATO on T+3
        """
        # Find T+2 date
        t_plus_2 = entry_date
        trading_days_counted = 0

        while trading_days_counted < 2:
            t_plus_2 = self.get_next_trading_day(t_plus_2)
            trading_days_counted += 1

        # Return ATC time on T+2 (14:30)
        return datetime.combine(t_plus_2, time(14, 30))

    # ==========================================
    # Trading Sessions
    # ==========================================

    def get_current_session(self, dt: datetime = None) -> TradingSession:
        """Get current trading session"""
        if dt is None:
            dt = datetime.now()

        current_time = dt.time()
        current_date = dt.date()

        # Check if trading day
        if not self.is_trading_day(current_date):
            return TradingSession(
                SessionType.CLOSED, time(0, 0), time(23, 59),
                can_place_order=False, can_cancel_order=False,
                order_types_allowed=[]
            )

        # Find current session
        for session in self.SESSIONS:
            if session.start_time <= current_time < session.end_time:
                return session

        # Outside trading hours
        return TradingSession(
            SessionType.CLOSED, time(0, 0), time(23, 59),
            can_place_order=False, can_cancel_order=False,
            order_types_allowed=[]
        )

    def is_market_open(self, dt: datetime = None) -> bool:
        """Check if market is currently open for trading"""
        session = self.get_current_session(dt)
        return session.session_type in [
            SessionType.CONTINUOUS_AM,
            SessionType.CONTINUOUS_PM,
            SessionType.LUNCH_BREAK  # Can place orders during lunch
        ]

    def can_place_order(self, order_type: str, dt: datetime = None) -> Tuple[bool, str]:
        """
        Check if order can be placed now

        Args:
            order_type: Order type (LO, ATO, ATC, MP)
            dt: Datetime to check

        Returns:
            Tuple of (can_place, reason)
        """
        session = self.get_current_session(dt)

        if not session.can_place_order:
            return False, f"Cannot place orders during {session.session_type.value}"

        if order_type not in session.order_types_allowed:
            return False, f"Order type {order_type} not allowed during {session.session_type.value}"

        return True, ""

    # ==========================================
    # Order Validation (Complete)
    # ==========================================

    def validate_order(self, symbol: str, side: str, quantity: int,
                       price: float, order_type: str,
                       reference_price: float,
                       exchange: Exchange = Exchange.HOSE,
                       position_entry_date: date = None,
                       current_datetime: datetime = None) -> Tuple[bool, List[str]]:
        """
        Complete order validation

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # 1. Validate quantity
        is_valid, msg = self.validate_quantity(quantity)
        if not is_valid:
            errors.append(msg)

        # 2. Validate price
        is_valid, msg = self.validate_price(price, reference_price, exchange)
        if not is_valid:
            errors.append(msg)

        # 3. Validate trading session
        is_valid, msg = self.can_place_order(order_type, current_datetime)
        if not is_valid:
            errors.append(msg)

        # 4. For SELL orders, validate T+2.5
        if side.upper() == "SELL" and position_entry_date:
            is_valid, msg = self.can_sell_position(position_entry_date, current_datetime)
            if not is_valid:
                errors.append(msg)

        return len(errors) == 0, errors


# Singleton instance
_vn_rules = None


def get_vn_market_rules() -> VNMarketRules:
    """Get singleton VN market rules instance"""
    global _vn_rules
    if _vn_rules is None:
        _vn_rules = VNMarketRules()
    return _vn_rules


# Convenience functions
def get_tick_size(price: float, exchange: str = "HOSE") -> float:
    """Get tick size for price"""
    return get_vn_market_rules().get_tick_size(price, Exchange[exchange])


def validate_vn_order(symbol: str, side: str, quantity: int, price: float,
                      reference_price: float, order_type: str = "LO") -> Tuple[bool, List[str]]:
    """Validate order against VN market rules"""
    return get_vn_market_rules().validate_order(
        symbol, side, quantity, price, order_type, reference_price
    )


def can_sell_t2(entry_date: date) -> Tuple[bool, str]:
    """Check T+2.5 sell eligibility"""
    return get_vn_market_rules().can_sell_position(entry_date)
