#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test T+2 compliance edge cases
"""

from datetime import datetime, date, time
from quantum_stock.core.vn_market_rules import VNMarketRules

rules = VNMarketRules()

# Test Case 1: Friday buy → Tuesday sell (skip weekend)
print("=" * 70)
print("Test Case 1: Friday buy → Tuesday sell (skip weekend)")
print("=" * 70)
friday = date(2026, 2, 20)  # Friday
monday = date(2026, 2, 23)  # Monday
tuesday = date(2026, 2, 24)  # Tuesday

trading_days_mon = rules.count_trading_days(friday, monday)
trading_days_tue = rules.count_trading_days(friday, tuesday)

print(f"Friday buy: {friday}")
print(f"Monday (T+1): {monday} - Trading days: {trading_days_mon}")
print(f"Tuesday (T+2): {tuesday} - Trading days: {trading_days_tue}")

# Can sell on Tuesday ATC?
can_sell_tue_atc, msg_tue = rules.can_sell_position(
    friday,
    datetime.combine(tuesday, time(14, 35))  # ATC session
)
print(f"Can sell Tuesday ATC (14:35)? {can_sell_tue_atc} - {msg_tue}")

# Can sell on Tuesday morning?
can_sell_tue_am, msg_tue_am = rules.can_sell_position(
    friday,
    datetime.combine(tuesday, time(10, 0))  # Morning session
)
print(f"Can sell Tuesday morning (10:00)? {can_sell_tue_am} - {msg_tue_am}")

# Can sell on Wednesday anytime?
wednesday = date(2026, 2, 25)
can_sell_wed, msg_wed = rules.can_sell_position(
    friday,
    datetime.combine(wednesday, time(10, 0))
)
trading_days_wed = rules.count_trading_days(friday, wednesday)
print(f"Wednesday (T+3): {wednesday} - Trading days: {trading_days_wed}")
print(f"Can sell Wednesday morning (10:00)? {can_sell_wed} - {msg_wed}")

# Test Case 2: Pre-Tết buy → Post-Tết sell
print("\n" + "=" * 70)
print("Test Case 2: Pre-Tết buy → Post-Tết sell (2026)")
print("=" * 70)
pre_tet = date(2026, 2, 13)  # Friday before Tết (2/16-2/20 are holidays)
post_tet = date(2026, 2, 23)  # Monday after Tết

trading_days_post_tet = rules.count_trading_days(pre_tet, post_tet)
print(f"Pre-Tết buy: {pre_tet} (Friday)")
print(f"Post-Tết: {post_tet} (Monday) - Trading days: {trading_days_post_tet}")

can_sell_post_tet, msg_post_tet = rules.can_sell_position(
    pre_tet,
    datetime.combine(post_tet, time(10, 0))
)
print(f"Can sell post-Tết Monday (10:00)? {can_sell_post_tet} - {msg_post_tet}")

# Test Case 3: Thursday buy → Monday sell
print("\n" + "=" * 70)
print("Test Case 3: Thursday buy → Monday sell (T+2)")
print("=" * 70)
thursday = date(2026, 2, 26)  # Thursday
next_monday = date(2026, 3, 2)  # Next Monday

trading_days = rules.count_trading_days(thursday, next_monday)
print(f"Thursday buy: {thursday}")
print(f"Next Monday (T+2): {next_monday} - Trading days: {trading_days}")

can_sell_mon_atc, msg_mon_atc = rules.can_sell_position(
    thursday,
    datetime.combine(next_monday, time(14, 35))  # ATC
)
print(f"Can sell Monday ATC (14:35)? {can_sell_mon_atc} - {msg_mon_atc}")

can_sell_mon_am, msg_mon_am = rules.can_sell_position(
    thursday,
    datetime.combine(next_monday, time(10, 0))  # Morning
)
print(f"Can sell Monday morning (10:00)? {can_sell_mon_am} - {msg_mon_am}")

# Test Case 4: ATC session order placement
print("\n" + "=" * 70)
print("Test Case 4: ATC session sell order placement")
print("=" * 70)
test_date = date(2026, 2, 24)
atc_time = datetime.combine(test_date, time(14, 35))

session = rules.get_current_session(atc_time)
print(f"Session at 14:35: {session.session_type.value}")
print(f"Can place order: {session.can_place_order}")
print(f"Allowed order types: {session.order_types_allowed}")

can_place_atc, msg_atc = rules.can_place_order('ATC', atc_time)
print(f"Can place ATC order at 14:35? {can_place_atc} - {msg_atc}")

print("\n" + "=" * 70)
print("All T+2 compliance tests completed!")
print("=" * 70)
