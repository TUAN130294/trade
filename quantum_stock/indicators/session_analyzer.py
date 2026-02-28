"""
VN Trading Session Analyzer
Analyzes intraday session patterns specific to Vietnamese stock market

VN Trading Sessions:
- ATO (Auction Opening): 09:00-09:15 - Institutional positioning
- Morning: 09:15-11:30 - Trend formation (60-65% daily volume)
- Afternoon: 13:00-14:30 - Confirmation/reversal
- ATC (Auction Closing): 14:30-14:45 - Smart money final moves
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import time
from enum import Enum


class SessionSignal(Enum):
    """Session analysis signals"""
    ATO_INSTITUTIONAL_BUY = "ato_institutional_buy"
    ATO_INSTITUTIONAL_SELL = "ato_institutional_sell"
    MORNING_AFTERNOON_REVERSAL = "morning_afternoon_reversal"
    ATC_MANIPULATION_UP = "atc_dap_gia"  # Đập giá cuối phiên (push down)
    ATC_MANIPULATION_DOWN = "atc_keo_gia"  # Kéo giá cuối phiên (pull up)
    NORMAL_SESSION = "normal_session"


class SessionType(Enum):
    """VN trading session types"""
    ATO = "ato"
    MORNING = "morning"
    AFTERNOON = "afternoon"
    ATC = "atc"


class SessionAnalyzer:
    """
    Analyzes VN market session patterns to detect:
    - ATO institutional positioning
    - Morning vs afternoon flow reversals
    - ATC manipulation (đập giá / kéo giá cuối phiên)
    """

    # VN session times
    SESSION_TIMES = {
        SessionType.ATO: (time(9, 0), time(9, 15)),
        SessionType.MORNING: (time(9, 15), time(11, 30)),
        SessionType.AFTERNOON: (time(13, 0), time(14, 30)),
        SessionType.ATC: (time(14, 30), time(14, 45))
    }

    def __init__(self):
        self.ato_volume_threshold = 1.5  # ATO volume vs normal bar
        self.morning_volume_pct = 0.60  # Expected morning volume %
        self.atc_manipulation_threshold = 0.015  # 1.5% move in ATC

    def analyze_session(
        self,
        df: pd.DataFrame,
        has_intraday_data: bool = False
    ) -> Tuple[SessionSignal, float, Dict[str, any]]:
        """
        Main session analysis method

        Args:
            df: OHLCV DataFrame
                If has_intraday_data=True, expects 'timestamp' column with datetime
                If has_intraday_data=False, uses daily data to estimate session behavior
            has_intraday_data: Whether intraday timestamp data is available

        Returns:
            Tuple of (SessionSignal, confidence 0-1, details dict)
        """
        if len(df) < 10:
            return SessionSignal.NORMAL_SESSION, 0.5, {'reason': 'insufficient_data'}

        metrics = {}

        if has_intraday_data and 'timestamp' in df.columns:
            # Full intraday analysis
            session_data = self._analyze_intraday_sessions(df)
            metrics.update(session_data)
        else:
            # Estimate from daily OHLCV
            session_data = self._estimate_session_behavior_daily(df)
            metrics.update(session_data)

        # Generate signal
        signal, confidence = self._generate_session_signal(metrics)

        return signal, confidence, metrics

    def _analyze_intraday_sessions(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze intraday data by session

        Requires 'timestamp' column with datetime
        """
        metrics = {}

        # Separate by session
        df['time'] = pd.to_datetime(df['timestamp']).dt.time
        df['date'] = pd.to_datetime(df['timestamp']).dt.date

        # Get most recent trading day
        latest_date = df['date'].max()
        today_df = df[df['date'] == latest_date].copy()

        if len(today_df) < 10:
            return {'session_quality': 'insufficient_intraday_data'}

        # ATO analysis
        ato_df = today_df[
            (today_df['time'] >= self.SESSION_TIMES[SessionType.ATO][0]) &
            (today_df['time'] < self.SESSION_TIMES[SessionType.ATO][1])
        ]

        if len(ato_df) > 0:
            ato_volume = ato_df['volume'].sum()
            ato_price_change = (ato_df['close'].iloc[-1] - ato_df['open'].iloc[0]) / ato_df['open'].iloc[0]
            metrics['ato_volume'] = ato_volume
            metrics['ato_price_change'] = ato_price_change
            metrics['ato_direction'] = 'BUY' if ato_price_change > 0.005 else 'SELL' if ato_price_change < -0.005 else 'NEUTRAL'

        # Morning session
        morning_df = today_df[
            (today_df['time'] >= self.SESSION_TIMES[SessionType.MORNING][0]) &
            (today_df['time'] < self.SESSION_TIMES[SessionType.MORNING][1])
        ]

        if len(morning_df) > 0:
            morning_volume = morning_df['volume'].sum()
            morning_price_change = (morning_df['close'].iloc[-1] - morning_df['open'].iloc[0]) / morning_df['open'].iloc[0]
            metrics['morning_volume'] = morning_volume
            metrics['morning_price_change'] = morning_price_change

        # Afternoon session
        afternoon_df = today_df[
            (today_df['time'] >= self.SESSION_TIMES[SessionType.AFTERNOON][0]) &
            (today_df['time'] < self.SESSION_TIMES[SessionType.AFTERNOON][1])
        ]

        if len(afternoon_df) > 0:
            afternoon_volume = afternoon_df['volume'].sum()
            afternoon_price_change = (afternoon_df['close'].iloc[-1] - afternoon_df['open'].iloc[0]) / afternoon_df['open'].iloc[0]
            metrics['afternoon_volume'] = afternoon_volume
            metrics['afternoon_price_change'] = afternoon_price_change

        # ATC analysis
        atc_df = today_df[
            (today_df['time'] >= self.SESSION_TIMES[SessionType.ATC][0]) &
            (today_df['time'] <= self.SESSION_TIMES[SessionType.ATC][1])
        ]

        if len(atc_df) > 0:
            pre_atc_close = afternoon_df['close'].iloc[-1] if len(afternoon_df) > 0 else atc_df['open'].iloc[0]
            atc_close = atc_df['close'].iloc[-1]
            atc_volume = atc_df['volume'].sum()
            atc_price_change = (atc_close - pre_atc_close) / pre_atc_close

            metrics['atc_volume'] = atc_volume
            metrics['atc_price_change'] = atc_price_change
            metrics['atc_manipulation_suspected'] = abs(atc_price_change) > self.atc_manipulation_threshold

        # Session reversal detection
        if 'morning_price_change' in metrics and 'afternoon_price_change' in metrics:
            morning_dir = 1 if metrics['morning_price_change'] > 0 else -1
            afternoon_dir = 1 if metrics['afternoon_price_change'] > 0 else -1

            if morning_dir != afternoon_dir:
                metrics['session_reversal'] = True
                metrics['reversal_magnitude'] = abs(metrics['morning_price_change']) + abs(metrics['afternoon_price_change'])
            else:
                metrics['session_reversal'] = False

        return metrics

    def _estimate_session_behavior_daily(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Estimate session behavior from daily OHLCV when intraday data unavailable

        Uses heuristics:
        - Large gap at open = ATO institutional activity
        - Open-to-high-to-close pattern suggests morning/afternoon dynamics
        - Close vs high/low suggests ATC manipulation
        """
        metrics = {}
        recent = df.iloc[-1]
        prev_close = df.iloc[-2]['close'] if len(df) > 1 else recent['open']

        # ATO estimation: gap analysis
        gap = (recent['open'] - prev_close) / prev_close
        metrics['ato_gap'] = gap
        metrics['ato_gap_significant'] = abs(gap) > 0.01  # 1% gap

        if gap > 0.01:
            metrics['ato_direction'] = 'BUY'
        elif gap < -0.01:
            metrics['ato_direction'] = 'SELL'
        else:
            metrics['ato_direction'] = 'NEUTRAL'

        # Morning/afternoon estimation from OHLC pattern
        # Assume high happens in morning if close < high
        # Assume low happens in morning if close > low

        body = recent['close'] - recent['open']
        upper_wick = recent['high'] - max(recent['open'], recent['close'])
        lower_wick = min(recent['open'], recent['close']) - recent['low']
        range_total = recent['high'] - recent['low']

        # Guard against division by zero when range_total = 0 (locked at ceiling/floor)
        if range_total > 0:
            close_position = (recent['close'] - recent['low']) / range_total
        else:
            # Price locked (high == low) - neutral position
            close_position = 0.5

        if range_total > 0:
            # If close near low and opened higher -> morning up, afternoon down (reversal)
            if close_position < 0.3 and body < 0:
                metrics['session_reversal'] = True
                metrics['reversal_type'] = 'morning_up_afternoon_down'

            # If close near high and opened lower -> morning down, afternoon up (reversal)
            elif close_position > 0.7 and body > 0:
                metrics['session_reversal'] = True
                metrics['reversal_type'] = 'morning_down_afternoon_up'
            else:
                metrics['session_reversal'] = False
        else:
            # Locked price - no reversal pattern
            metrics['session_reversal'] = False

        # ATC manipulation estimation
        # Large move at close with volume spike suggests ATC activity
        avg_volume = df['volume'].iloc[-20:].mean()
        volume_spike = recent['volume'] > avg_volume * 1.3

        # If volume spike + close at extreme
        if volume_spike and range_total > 0:
            if close_position > 0.9:
                # Close at high -> possible "kéo giá" (pull up at ATC)
                metrics['atc_manipulation_suspected'] = True
                metrics['atc_type'] = 'keo_gia'  # Pull up
                # Guard against division by zero
                if recent['low'] > 0:
                    metrics['atc_price_change'] = (recent['close'] - recent['low']) / recent['low']
                else:
                    metrics['atc_price_change'] = 0

            elif close_position < 0.1:
                # Close at low -> possible "đập giá" (push down at ATC)
                metrics['atc_manipulation_suspected'] = True
                metrics['atc_type'] = 'dap_gia'  # Push down
                # Guard against division by zero
                if recent['high'] > 0:
                    metrics['atc_price_change'] = -(recent['high'] - recent['close']) / recent['high']
                else:
                    metrics['atc_price_change'] = 0
        else:
            metrics['atc_manipulation_suspected'] = False

        return metrics

    def _generate_session_signal(self, metrics: Dict[str, any]) -> Tuple[SessionSignal, float]:
        """
        Generate session signal from metrics

        Priority:
        1. ATC manipulation (most actionable)
        2. ATO institutional positioning
        3. Session reversal
        4. Normal
        """
        # ATC manipulation signals
        if metrics.get('atc_manipulation_suspected', False):
            atc_type = metrics.get('atc_type', '')

            if atc_type == 'keo_gia':
                return SessionSignal.ATC_MANIPULATION_DOWN, 0.75

            elif atc_type == 'dap_gia':
                return SessionSignal.ATC_MANIPULATION_UP, 0.75

        # Session reversal
        if metrics.get('session_reversal', False):
            reversal_mag = metrics.get('reversal_magnitude', 0.01)
            confidence = min(0.85, 0.6 + reversal_mag * 10)
            return SessionSignal.MORNING_AFTERNOON_REVERSAL, confidence

        # ATO institutional activity
        ato_direction = metrics.get('ato_direction', 'NEUTRAL')

        if ato_direction == 'BUY' and metrics.get('ato_gap_significant', False):
            ato_gap = metrics.get('ato_gap', 0)
            confidence = min(0.80, 0.6 + abs(ato_gap) * 20)
            return SessionSignal.ATO_INSTITUTIONAL_BUY, confidence

        elif ato_direction == 'SELL' and metrics.get('ato_gap_significant', False):
            ato_gap = metrics.get('ato_gap', 0)
            confidence = min(0.80, 0.6 + abs(ato_gap) * 20)
            return SessionSignal.ATO_INSTITUTIONAL_SELL, confidence

        # Normal session
        return SessionSignal.NORMAL_SESSION, 0.50


# Convenience function
def analyze_vn_session(
    df: pd.DataFrame,
    has_intraday_data: bool = False
) -> Dict[str, any]:
    """
    Convenience function for VN session analysis

    Returns dict with signal, confidence, and metrics
    """
    analyzer = SessionAnalyzer()
    signal, confidence, metrics = analyzer.analyze_session(df, has_intraday_data)

    return {
        'signal': signal,
        'signal_name': signal.value,
        'confidence': confidence,
        'metrics': metrics
    }
