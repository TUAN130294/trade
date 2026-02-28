# -*- coding: utf-8 -*-
"""
Market Flow Data Connector
==========================
Kết nối dữ liệu dòng tiền thông minh (Smart Money Flow):
1. Foreign Investor (Khối ngoại)
2. Proprietary Trading (Tự doanh)
3. Market Breadth (Độ rộng thị trường)

Fallback Strategy:
- Nếu không có Premium API, sử dụng "Price Action Proxy" để ước lượng dòng tiền.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)

class MarketFlowConnector:
    """
    Connects to liquidity data sources
    """
    
    def __init__(self):
        self.cache = {}
        self.last_update = None
        
    async def get_foreign_flow(self, symbol: str) -> Dict:
        """
        Get Foreign Investor Net Buy/Sell from CafeF API
        (Mua ròng/Bán ròng khối ngoại)
        """
        try:
            from quantum_stock.dataconnector.realtime_market import get_realtime_connector
            connector = get_realtime_connector()

            # Get current session data
            stocks = connector._get_cached_or_fetch()

            # Find symbol in data
            symbol_data = None
            for stock in stocks:
                if stock.get('a', '').upper() == symbol.upper():
                    symbol_data = stock
                    break

            if not symbol_data:
                return {
                    'net_buy_value_1d': None,
                    'net_buy_vol_1d': None,
                    'accumulated_5d': None,
                    'status': 'NO_DATA',
                    'data_source': 'unavailable'
                }

            # Extract real foreign data from CafeF fields
            foreign_buy = symbol_data.get('tb', 0) or 0  # Foreign buy volume
            foreign_sell = symbol_data.get('ts', 0) or 0  # Foreign sell volume
            price = symbol_data.get('l', 0) or 0  # Current price (in thousands)

            # Convert to VND value
            net_buy_vol = foreign_buy - foreign_sell
            net_buy_value = net_buy_vol * price * 1000  # Price in thousands

            # Determine status based on real data
            if net_buy_vol > 100000:  # >100k shares net buy
                status = 'STRONG_ACCUMULATION'
            elif net_buy_vol > 0:
                status = 'ACCUMULATION'
            elif net_buy_vol < -100000:
                status = 'STRONG_DISTRIBUTION'
            elif net_buy_vol < 0:
                status = 'DISTRIBUTION'
            else:
                status = 'NEUTRAL'

            return {
                'net_buy_value_1d': net_buy_value,
                'net_buy_vol_1d': net_buy_vol,
                'foreign_buy': foreign_buy,
                'foreign_sell': foreign_sell,
                'accumulated_5d': None,  # Need historical data for this
                'status': status,
                'data_source': 'cafef_live'
            }

        except Exception as e:
            logger.error(f"Error getting foreign flow for {symbol}: {e}")
            return {
                'net_buy_value_1d': None,
                'net_buy_vol_1d': None,
                'accumulated_5d': None,
                'status': 'ERROR',
                'data_source': 'unavailable'
            }

    async def get_proprietary_flow(self, symbol: str) -> Dict:
        """
        Get Proprietary Trading Net Buy/Sell
        (Tự doanh CTCK)

        Estimates proprietary flow from price/volume patterns since
        real data unavailable without premium API.
        """
        try:
            from quantum_stock.dataconnector.realtime_market import get_realtime_connector
            connector = get_realtime_connector()

            # Get current session data
            stocks = connector._get_cached_or_fetch()

            # Find symbol in data
            symbol_data = None
            for stock in stocks:
                if stock.get('a', '').upper() == symbol.upper():
                    symbol_data = stock
                    break

            if not symbol_data:
                return {
                    'net_buy_value_1d': None,
                    'status': 'NO_DATA',
                    'data_source': 'unavailable'
                }

            # Estimate proprietary flow from OHLCV patterns
            volume = symbol_data.get('n', 0) or 0
            price = symbol_data.get('l', 0) or 0  # Current price
            open_price = symbol_data.get('o', 0) or price
            high = symbol_data.get('h', 0) or price
            low = symbol_data.get('g', 0) or price

            # Calculate metrics for estimation
            price_change_pct = ((price - open_price) / open_price) if open_price > 0 else 0
            range_total = high - low
            close_position = ((price - low) / range_total) if range_total > 0 else 0.5

            # Estimate institutional flow:
            # - Strong buying with volume + price at high = accumulation
            # - Selling pressure with volume + price at low = distribution

            # Use volume and price position to estimate flow direction
            net_buy_vol_estimate = 0

            if close_position > 0.7 and price_change_pct > 0.01:
                # Price closed high in range + positive = buying pressure
                net_buy_vol_estimate = volume * 0.3  # Estimate 30% institutional
            elif close_position < 0.3 and price_change_pct < -0.01:
                # Price closed low in range + negative = selling pressure
                net_buy_vol_estimate = -volume * 0.3
            elif abs(price_change_pct) < 0.005:
                # Neutral/sideways
                net_buy_vol_estimate = 0
            else:
                # Mixed signals - proportional estimate
                net_buy_vol_estimate = volume * price_change_pct * close_position * 0.2

            # Convert to VND value
            net_buy_value = net_buy_vol_estimate * price * 1000  # Price in thousands

            # Determine status based on estimated flow
            if net_buy_value > 500_000_000:  # >500M VND net buy
                status = 'STRONG_ACCUMULATION'
            elif net_buy_value > 0:
                status = 'ACCUMULATION'
            elif net_buy_value < -500_000_000:
                status = 'STRONG_DISTRIBUTION'
            elif net_buy_value < 0:
                status = 'DISTRIBUTION'
            else:
                status = 'NEUTRAL'

            return {
                'net_buy_value_1d': net_buy_value,
                'net_buy_vol_estimate': net_buy_vol_estimate,
                'status': status,
                'data_source': 'estimated'  # Flag as estimated
            }

        except Exception as e:
            logger.error(f"Error estimating proprietary flow for {symbol}: {e}")
            return {
                'net_buy_value_1d': None,
                'status': 'ERROR',
                'data_source': 'unavailable'
            }

    async def get_market_liquidity(self) -> Dict:
        """
        Get whole market liquidity snapshot from real CafeF data
        """
        try:
            from quantum_stock.dataconnector.realtime_market import get_realtime_connector
            connector = get_realtime_connector()

            # Get all stocks data
            stocks = connector._get_cached_or_fetch()

            if not stocks:
                return {
                    'total_volume': None,
                    'total_value': None,
                    'morning_session_ratio': None,
                    'data_source': 'unavailable'
                }

            # Calculate real market totals
            total_volume = 0
            total_value = 0

            for stock in stocks:
                volume = stock.get('n', 0) or stock.get('totalvolume', 0) or 0  # Total volume
                price = stock.get('l', 0) or 0  # Current price (in thousands)

                total_volume += volume
                total_value += volume * price * 1000  # Convert to VND

            return {
                'total_volume': total_volume,
                'total_value': total_value,
                'total_value_billion': round(total_value / 1_000_000_000, 2),
                'morning_session_ratio': None,  # Not available in real-time data
                'data_source': 'cafef_live'
            }

        except Exception as e:
            logger.error(f"Error getting market liquidity: {e}")
            return {
                'total_volume': None,
                'total_value': None,
                'morning_session_ratio': None,
                'data_source': 'unavailable'
            }
        
    async def detect_smart_money_footprint(self, ohlcv_df: pd.DataFrame) -> Dict:
        """
        Detect Smart Money using 8+ advanced patterns
        """
        if len(ohlcv_df) < 20:
            return {'detected': False, 'data_source': 'insufficient_data'}

        latest = ohlcv_df.iloc[-1]
        avg_vol = ohlcv_df['volume'].rolling(20).mean().iloc[-1]
        avg_spread = (ohlcv_df['high'] - ohlcv_df['low']).rolling(20).mean().iloc[-1]

        # Calculate metrics
        vol_ratio = latest['volume'] / avg_vol if avg_vol > 0 else 1
        spread = latest['high'] - latest['low']
        spread_ratio = spread / avg_spread if avg_spread > 0 else 1
        close_loc = (latest['close'] - latest['low']) / (0.001 + latest['high'] - latest['low'])

        # Recent price trend
        recent_5 = ohlcv_df.iloc[-5:]
        price_change_5d = (recent_5['close'].iloc[-1] - recent_5['close'].iloc[0]) / recent_5['close'].iloc[0] * 100

        patterns_detected = []

        # Pattern 1: CLIMAX_BUYING - Volume spike + strong close high
        if vol_ratio > 2.5 and close_loc > 0.8:
            patterns_detected.append({
                'type': 'CLIMAX_BUYING',
                'confidence': 0.9,
                'description': 'Cao trào mua - Volume spike với giá đóng cửa cao',
                'direction': 'bullish',
                'evidence': {
                    'vol_ratio': round(vol_ratio, 2),
                    'close_location': round(close_loc, 2)
                }
            })

        # Pattern 2: CLIMAX_SELLING - Volume spike + close low
        elif vol_ratio > 2.5 and close_loc < 0.2:
            patterns_detected.append({
                'type': 'CLIMAX_SELLING',
                'confidence': 0.9,
                'description': 'Xả hàng mạnh - Volume cao với giá đóng cửa thấp',
                'direction': 'bearish',
                'evidence': {
                    'vol_ratio': round(vol_ratio, 2),
                    'close_location': round(close_loc, 2)
                }
            })

        # Pattern 3: CHURNING - High volume but tight range
        if vol_ratio > 2.0 and spread_ratio < 0.5:
            patterns_detected.append({
                'type': 'CHURNING',
                'confidence': 0.75,
                'description': 'Quay tay - Volume cao nhưng giá không đổi nhiều',
                'direction': 'neutral',
                'evidence': {
                    'vol_ratio': round(vol_ratio, 2),
                    'spread_ratio': round(spread_ratio, 2)
                }
            })

        # Pattern 4: ACCUMULATION - Gradual volume increase with sideways price
        if abs(price_change_5d) < 2 and recent_5['volume'].is_monotonic_increasing:
            patterns_detected.append({
                'type': 'ACCUMULATION',
                'confidence': 0.8,
                'description': 'Tích lũy ngầm - Volume tăng dần với giá sideway',
                'direction': 'bullish',
                'evidence': {
                    'price_change_5d': round(price_change_5d, 2),
                    'vol_increasing': True
                }
            })

        # Pattern 5: DISTRIBUTION - Sideways at top with high volume
        if price_change_5d < -1 and vol_ratio > 1.5 and close_loc < 0.4:
            patterns_detected.append({
                'type': 'DISTRIBUTION',
                'confidence': 0.8,
                'description': 'Phân phối - Giá sideway/giảm nhẹ với volume cao',
                'direction': 'bearish',
                'evidence': {
                    'price_change_5d': round(price_change_5d, 2),
                    'vol_ratio': round(vol_ratio, 2),
                    'close_location': round(close_loc, 2)
                }
            })

        # Pattern 6: SPRING - Sharp drop then recovery (Wyckoff)
        if len(ohlcv_df) >= 3:
            prev_low = ohlcv_df['low'].iloc[-3:-1].min()
            if latest['low'] < prev_low and latest['close'] > prev_low and close_loc > 0.6:
                patterns_detected.append({
                    'type': 'SPRING',
                    'confidence': 0.7,
                    'description': 'Spring (Wyckoff) - Test support rồi recovery mạnh',
                    'direction': 'bullish',
                    'evidence': {
                        'prev_low': round(prev_low, 2),
                        'current_low': round(latest['low'], 2),
                        'close_location': round(close_loc, 2)
                    }
                })

        # Pattern 7: UPTHRUST - Brief spike above resistance then failure
        if len(ohlcv_df) >= 3:
            prev_high = ohlcv_df['high'].iloc[-3:-1].max()
            if latest['high'] > prev_high and latest['close'] < prev_high and close_loc < 0.3:
                patterns_detected.append({
                    'type': 'UPTHRUST',
                    'confidence': 0.7,
                    'description': 'Upthrust - Đột phá giả, giá fail quay đầu',
                    'direction': 'bearish',
                    'evidence': {
                        'prev_high': round(prev_high, 2),
                        'current_high': round(latest['high'], 2),
                        'close_location': round(close_loc, 2)
                    }
                })

        # Pattern 8: ABSORPTION - High volume with small range = buying/selling absorption
        if vol_ratio > 2.0 and spread_ratio < 0.6:
            if close_loc > 0.5:
                patterns_detected.append({
                    'type': 'ABSORPTION',
                    'confidence': 0.75,
                    'description': 'Hấp thụ bán - Sellers bị hấp thụ bởi buyers',
                    'direction': 'bullish',
                    'evidence': {
                        'vol_ratio': round(vol_ratio, 2),
                        'spread_ratio': round(spread_ratio, 2),
                        'close_location': round(close_loc, 2)
                    }
                })
            else:
                patterns_detected.append({
                    'type': 'ABSORPTION',
                    'confidence': 0.75,
                    'description': 'Hấp thụ mua - Buyers bị hấp thụ bởi sellers',
                    'direction': 'bearish',
                    'evidence': {
                        'vol_ratio': round(vol_ratio, 2),
                        'spread_ratio': round(spread_ratio, 2),
                        'close_location': round(close_loc, 2)
                    }
                })

        # Pattern 9: STOPPING_VOLUME - Volume spike halts trend
        if vol_ratio > 3.0:
            if price_change_5d > 3 and close_loc < 0.4:
                patterns_detected.append({
                    'type': 'STOPPING_VOLUME',
                    'confidence': 0.8,
                    'description': 'Volume dừng xu hướng - Có thể đảo chiều',
                    'direction': 'bearish',
                    'evidence': {
                        'vol_ratio': round(vol_ratio, 2),
                        'price_change_5d': round(price_change_5d, 2),
                        'close_location': round(close_loc, 2)
                    }
                })

        # Pattern 10: EFFORT_VS_RESULT - High volume but low price move
        if vol_ratio > 2.5 and abs(latest['close'] - latest['open']) / latest['open'] * 100 < 0.5:
            patterns_detected.append({
                'type': 'EFFORT_VS_RESULT',
                'confidence': 0.7,
                'description': 'Nỗ lực không tương xứng kết quả - Cảnh báo yếu',
                'direction': 'neutral',
                'evidence': {
                    'vol_ratio': round(vol_ratio, 2),
                    'price_move_pct': round(abs(latest['close'] - latest['open']) / latest['open'] * 100, 2)
                }
            })

        # Pattern 11: INITIATIVE_BUYING - Gap up with volume + holds price
        if len(ohlcv_df) >= 2:
            prev_close = ohlcv_df['close'].iloc[-2]
            gap_up = (latest['open'] - prev_close) / prev_close * 100
            hold_strength = (latest['close'] - latest['open']) / (latest['open'] + 0.001) * 100

            if gap_up > 1.5 and vol_ratio > 1.5 and hold_strength > -0.5:
                patterns_detected.append({
                    'type': 'INITIATIVE_BUYING',
                    'confidence': 0.85,
                    'description': 'Mua chủ động - Gap up với volume và giữ giá tốt',
                    'direction': 'bullish',
                    'evidence': {
                        'gap_pct': round(gap_up, 2),
                        'vol_ratio': round(vol_ratio, 2),
                        'hold_pct': round(hold_strength, 2)
                    }
                })

        # Pattern 12: INITIATIVE_SELLING - Gap down with volume + no bounce
        if len(ohlcv_df) >= 2:
            prev_close = ohlcv_df['close'].iloc[-2]
            gap_down = (latest['open'] - prev_close) / prev_close * 100
            bounce_strength = (latest['close'] - latest['low']) / (latest['high'] - latest['low'] + 0.001)

            if gap_down < -1.5 and vol_ratio > 1.5 and bounce_strength < 0.3:
                patterns_detected.append({
                    'type': 'INITIATIVE_SELLING',
                    'confidence': 0.85,
                    'description': 'Bán chủ động - Gap down với volume và không có phục hồi',
                    'direction': 'bearish',
                    'evidence': {
                        'gap_pct': round(gap_down, 2),
                        'vol_ratio': round(vol_ratio, 2),
                        'bounce_ratio': round(bounce_strength, 2)
                    }
                })

        if patterns_detected:
            return {
                'detected': True,
                'patterns': patterns_detected,
                'primary_pattern': patterns_detected[0]['type'],
                'confidence': patterns_detected[0]['confidence'],
                'data_source': 'real_analysis',
                'metrics': {
                    'volume_ratio': vol_ratio,
                    'spread_ratio': spread_ratio,
                    'close_location': close_loc
                }
            }

        return {
            'detected': False,
            'data_source': 'real_analysis'
        }
