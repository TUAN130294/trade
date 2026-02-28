# -*- coding: utf-8 -*-
"""
Market Regime Detector
======================
Phát hiện trạng thái thị trường VN-Index để điều chỉnh chiến thuật trading.

Logic đặc thù VN:
1. Xanh vỏ đỏ lòng (Divergence): Index tăng nhưng đa số cổ phiếu giảm -> Rủi ro Bull trap
2. Trend Alignment: Giá nằm trên/dưới MA50/MA200
3. Volatility Check: ATR của Index cao -> Thị trường biến động mạnh -> Giảm size
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """
    Phát hiện Regime thị trường:
    - BULL: Uptrend vững chắc, lan tỏa tốt
    - BEAR: Downtrend, nên cash heavy
    - VOLATILE: Biến động mạnh, rủi ro cao
    - DIVERGENT (Xanh vỏ đỏ lòng): Cực kỳ nguy hiểm, bull trap
    """
    
    def __init__(self, data_source=None):
        self.data_source = data_source
        self.last_regime = None
    
    async def analyze_market(self, vn_index_data: pd.DataFrame, market_breadth: Dict = None) -> Dict:
        """
        Phân tích toàn diện Market Regime
        
        Args:
            vn_index_data: DF chứa dữ liệu VNINDEX (OHLCV)
            market_breadth: Dict chứa số mã tăng/giảm {'advancing': 150, 'declining': 200}
        """
        if vn_index_data is None or len(vn_index_data) < 200:
            logger.warning("Not enough VN-Index data for analysis")
            return self._default_regime()

        current_price = vn_index_data.iloc[-1]['close']
        
        # 1. Trend Analysis
        ema20 = vn_index_data['close'].ewm(span=20).mean().iloc[-1]
        ma50 = vn_index_data['close'].rolling(window=50).mean().iloc[-1]
        ma200 = vn_index_data['close'].rolling(window=200).mean().iloc[-1]
        
        # 2. Volatility Analysis (ATR check)
        # Tính ATR chuẩn hóa theo giá (ATR %)
        high_low = vn_index_data['high'] - vn_index_data['low']
        high_close = np.abs(vn_index_data['high'] - vn_index_data['close'].shift())
        low_close = np.abs(vn_index_data['low'] - vn_index_data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr_14 = true_range.rolling(14).mean().iloc[-1]
        
        atr_pct = (atr_14 / current_price) * 100
        is_volatile = atr_pct > 1.2  # VNINDEX biến động > 1.2% trung bình là cao
        
        # 3. Market Breadth / Divergence Analysis (Xanh vỏ đỏ lòng)
        is_divergent = False
        breadth_score = 0.5
        
        if market_breadth:
            advancing = market_breadth.get('advancing', 1)
            declining = market_breadth.get('declining', 1)
            total = advancing + declining
            
            breadth_ratio = advancing / total if total > 0 else 0.5
            change_pct = (current_price - vn_index_data.iloc[-2]['close']) / vn_index_data.iloc[-2]['close']
            
            # Logic: Index tăng mạnh (>0.3%) nhưng số mã giảm áp đảo (>60%)
            if change_pct > 0.003 and breadth_ratio < 0.4:
                is_divergent = True
                logger.warning(f"⚠️ DETECTED: Xanh vỏ đỏ lòng (Index +{change_pct*100:.2f}%, Breadth {breadth_ratio:.2f})")
            
            breadth_score = breadth_ratio

        # 4. Determine Regime
        regime = "NEUTRAL"
        signal = "HOLD"
        risk_factor = 1.0 # 1.0 = normal size
        
        # Determine base trend
        if current_price > ma50 and ma50 > ma200:
            base_trend = "BULL"
        elif current_price < ma50 and ma50 < ma200:
            base_trend = "BEAR"
        else:
            base_trend = "SIDEWAYS"
            
        # Logic combine
        if is_divergent:
            regime = "DIVERGENT_BEAR"  # Bull trap danger
            signal = "DEFENSE_ONLY"
            risk_factor = 0.0 # Stop buy
            
        elif base_trend == "BULL":
            if is_volatile:
                regime = "VOLATILE_BULL"
                signal = "BUY_DIPS"
                risk_factor = 0.7 # Giảm size vì biến động
            elif breadth_score > 0.6:
                regime = "STRONG_BULL"
                signal = "AGGRESSIVE_BUY"
                risk_factor = 1.2 # Tăng size
            else:
                regime = "BULL"
                signal = "BUY"
                risk_factor = 1.0
                
        elif base_trend == "BEAR":
            if current_price < ma50:
                regime = "BEAR"
                signal = "CASH_IS_KING"
                risk_factor = 0.3 # Đánh rất nhỏ hoặc nghỉ
            if is_volatile:
                regime = "CRASH_RISK"
                signal = "SELL_ALL"
                risk_factor = 0.0
                
        else: # Sideways
            regime = "SIDEWAYS"
            signal = "SWING_TRADING" # Mua hỗ trợ bán kháng cự
            risk_factor = 0.6

        result = {
            'timestamp': datetime.now().isoformat(),
            'regime': regime,
            'signal': signal,
            'risk_factor': risk_factor,
            'vn_index': current_price,
            'trend_status': {
                'price_vs_ma50': 'ABOVE' if current_price > ma50 else 'BELOW',
                'price_vs_ma200': 'ABOVE' if current_price > ma200 else 'BELOW',
                'atr_pct': atr_pct
            },
            'warnings': []
        }
        
        if is_divergent:
            result['warnings'].append("XANH VỎ ĐỎ LÒNG - Rủi ro Bull Trap cao")
        if is_volatile:
            result['warnings'].append("Biến động thị trường cao")
            
        return result

    def _default_regime(self):
        return {
            'regime': 'NEUTRAL',
            'signal': 'HOLD', 
            'risk_factor': 0.5,
            'warnings': ['Insufficient data']
        }
