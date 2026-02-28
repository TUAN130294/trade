import pandas as pd
import numpy as np

def calculate_vn_market_features_simple(df: pd.DataFrame, vn_index_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Calculate simple features for VN market prediction.
    Features (15):
    1. Returns
    2. Log Returns
    3. Volatility (20d)
    4. RSI (14)
    5. MACD
    6. MACD Signal
    7. Bollinger Upper
    8. Bollinger Lower
    9. SMA 20
    10. SMA 50
    11. EMA 20
    12. Volume Change
    13. Close to SMA20
    14. Close to SMA50
    15. Momentum (10d)
    """
    df = df.copy()
    
    # 1. Returns
    df['returns'] = df['close'].pct_change()
    
    # 2. Log Returns
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # 3. Volatility
    df['volatility_20'] = df['returns'].rolling(window=20).std()
    
    # 4. RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # 5-6. MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # 7-8. Bollinger Bands
    sma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['bb_upper'] = sma20 + (std20 * 2)
    df['bb_lower'] = sma20 - (std20 * 2)
    
    # 9-11. Moving Averages
    df['sma_20'] = sma20
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # 12. Volume Change
    df['volume_change'] = df['volume'].pct_change()
    
    # 13-14. Distance to MA
    df['close_to_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
    df['close_to_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
    
    # 15. Momentum
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    
    # 16. Williams %R
    highest_high = df['high'].rolling(window=14).max()
    lowest_low = df['low'].rolling(window=14).min()
    df['williams_r'] = ((highest_high - df['close']) / (highest_high - lowest_low)) * -100
    
    # 17. ATR & ATR Percent (Volatility)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr_14'] = true_range.rolling(14).mean()
    df['atr_percent'] = (df['atr_14'] / df['close']) * 100
    
    # 18. Bollinger Band Width (Squeeze detection)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma_20']
    
    # 19. Volume Shock (Active Trading)
    # Volume > 2x trung bình 20 phiên
    df['volume_shock'] = (df['volume'] > df['volume'].rolling(20).mean() * 2).astype(int)
    
    # 20. Price/Volume Divergence (Simple)
    # Giá tăng nhưng Vol giảm (hoặc ngược lại)
    rank_price = df['close'].rolling(10).rank()
    rank_vol = df['volume'].rolling(10).rank()
    df['pv_divergence'] = rank_price - rank_vol # High positive = Price high/Vol low (Weakness)
    
    # 21. Market Relative Strength (cần VN-Index, tạm thời dùng so với sma50)
    # Stocks above SMA50 are generally stronger
    df['trend_strength'] = np.where(df['close'] > df['sma_50'], 1, -1)

    # 22. ADX (Average Directional Index) - Simplified
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr14 = df['atr_14']
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / tr14)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean().abs() / tr14)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['adx_14'] = dx.rolling(14).mean()

    # Fill NaN
    df = df.fillna(0)
    
    return df

def normalize_features_simple(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize features using Z-score standardization
    """
    return (df - df.mean()) / df.std()
