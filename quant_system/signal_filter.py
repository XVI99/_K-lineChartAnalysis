"""
Signal Quality Filter Module

Implements multi-condition confirmation to improve win rate:
1. Pattern quality score filter (only trade high-quality patterns)
2. Trend alignment filter (price > MA20 > MA60)
3. Volume confirmation filter
4. RSI filter (avoid overbought conditions)
5. Market regime filter (avoid bear markets)

Target: Win rate 85%+
"""
import pandas as pd
import numpy as np


def check_trend_alignment(df, ma_short=20, ma_long=60, idx=-1):
    """
    Check if price is in uptrend (Close > MA_short > MA_long).
    
    Args:
        df: DataFrame with 'Close' column
        ma_short: Short MA period (default 20)
        ma_long: Long MA period (default 60)
        idx: Row index to check (default -1 = latest)
    
    Returns:
        tuple: (is_aligned: bool, ma_short_val: float, ma_long_val: float)
    """
    if len(df) < ma_long:
        return True, 0, 0  # Not enough data, pass through
    
    close = df['Close'].iloc[idx]
    ma_s = df['Close'].rolling(ma_short).mean().iloc[idx]
    ma_l = df['Close'].rolling(ma_long).mean().iloc[idx]
    
    if pd.isna(ma_s) or pd.isna(ma_l):
        return True, ma_s, ma_l
    
    # Uptrend: Close > MA20 > MA60
    is_aligned = (close > ma_s) and (ma_s > ma_l)
    
    return is_aligned, ma_s, ma_l


def check_volume_confirmation(df, vol_ma_period=20, vol_ratio_threshold=1.0, idx=-1):
    """
    Check if volume confirms the signal (volume > average).
    
    For buy signals, we want above-average volume (institutional buying).
    
    Args:
        df: DataFrame with 'Volume' column
        vol_ma_period: Volume MA period (default 20)
        vol_ratio_threshold: Minimum volume ratio (default 1.0 = average)
        idx: Row index to check
    
    Returns:
        tuple: (is_confirmed: bool, vol_ratio: float)
    """
    if 'Volume' not in df.columns or len(df) < vol_ma_period:
        return True, 1.0  # No volume data, pass through
    
    current_vol = df['Volume'].iloc[idx]
    avg_vol = df['Volume'].rolling(vol_ma_period).mean().iloc[idx]
    
    if pd.isna(avg_vol) or avg_vol == 0:
        return True, 1.0
    
    vol_ratio = current_vol / avg_vol
    is_confirmed = vol_ratio >= vol_ratio_threshold
    
    return is_confirmed, round(vol_ratio, 2)


def check_rsi_filter(df, rsi_period=14, overbought=70, oversold=30, idx=-1):
    """
    Check RSI filter conditions.
    
    For buy: RSI should not be overbought (< 70)
    For sell: RSI should not be oversold (> 30)
    
    Args:
        df: DataFrame with 'Close' column
        rsi_period: RSI calculation period
        overbought: Overbought threshold (default 70)
        oversold: Oversold threshold (default 30)
        idx: Row index to check
    
    Returns:
        tuple: (buy_ok: bool, sell_ok: bool, rsi_value: float)
    """
    if len(df) < rsi_period + 1:
        return True, True, 50.0  # Not enough data
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    current_rsi = rsi.iloc[idx]
    
    if pd.isna(current_rsi):
        return True, True, 50.0
    
    # Buy OK if not overbought
    buy_ok = current_rsi < overbought
    # Sell OK if not oversold
    sell_ok = current_rsi > oversold
    
    return buy_ok, sell_ok, round(current_rsi, 2)


def check_price_position(df, ma_period=20, idx=-1):
    """
    Check price position relative to MA.
    
    Good buy zone: Price near or below MA (within 5%)
    Good sell zone: Price well above MA (> 10%)
    
    Args:
        df: DataFrame with 'Close' column
        ma_period: MA period
        idx: Row index
    
    Returns:
        tuple: (buy_zone: bool, sell_zone: bool, distance_pct: float)
    """
    if len(df) < ma_period:
        return True, True, 0.0
    
    close = df['Close'].iloc[idx]
    ma = df['Close'].rolling(ma_period).mean().iloc[idx]
    
    if pd.isna(ma) or ma == 0:
        return True, True, 0.0
    
    distance = (close - ma) / ma
    
    # Buy zone: within 5% below MA or up to 3% above
    buy_zone = distance < 0.03
    # Sell zone: more than 10% above MA
    sell_zone = distance > 0.10
    
    return buy_zone, sell_zone, round(distance, 4)


def check_macd_confirmation(df, idx=-1):
    """
    Check MACD for trend confirmation.
    
    Buy confirmation: MACD histogram turning positive or rising
    Sell confirmation: MACD histogram turning negative or falling
    
    Args:
        df: DataFrame with MACD columns (MACD, Signal, Histogram)
        idx: Row index
    
    Returns:
        tuple: (buy_ok: bool, sell_ok: bool)
    """
    if 'MACD' not in df.columns or 'Signal' not in df.columns:
        return True, True
    
    if idx < 1:
        return True, True
    
    # Calculate histogram if not present
    if 'Histogram' not in df.columns:
        df['Histogram'] = df['MACD'] - df['Signal']
    
    hist = df['Histogram'].iloc[idx]
    prev_hist = df['Histogram'].iloc[idx - 1]
    
    if pd.isna(hist) or pd.isna(prev_hist):
        return True, True
    
    # Buy: histogram rising or turning positive
    buy_ok = (hist > prev_hist) or (hist > 0 and prev_hist <= 0)
    # Sell: histogram falling or turning negative
    sell_ok = (hist < prev_hist) or (hist < 0 and prev_hist >= 0)
    
    return buy_ok, sell_ok


class SignalFilter:
    """
    Multi-condition signal filter for improving win rate.
    """
    
    def __init__(self, config=None):
        """
        Initialize with optional configuration.
        
        Args:
            config: dict with filter settings
        """
        self.config = config or {
            'min_pattern_quality': 70,      # Minimum pattern quality score
            'require_trend_alignment': True, # Require Close > MA20 > MA60
            'require_volume_confirm': True,  # Require volume > average
            'require_rsi_filter': True,      # Require RSI < 70 for buys
            'require_price_position': False, # Check price vs MA position
            'require_macd_confirm': False,   # Check MACD confirmation
            'vol_ratio_threshold': 1.0,      # Minimum volume ratio
            'rsi_overbought': 70,            # RSI overbought level
            'rsi_oversold': 30,              # RSI oversold level
        }
    
    def filter_buy_signal(self, df, pattern_quality=100, idx=-1):
        """
        Apply all filters to a buy signal.
        
        Args:
            df: DataFrame with OHLCV and indicator columns
            pattern_quality: Quality score of the detected pattern (0-100)
            idx: Row index to check
        
        Returns:
            tuple: (pass: bool, reasons: list[str], details: dict)
        """
        reasons = []
        details = {}
        passes = True
        
        # 1. Pattern quality filter
        if pattern_quality < self.config['min_pattern_quality']:
            passes = False
            reasons.append(f"Pattern quality {pattern_quality} < {self.config['min_pattern_quality']}")
        details['pattern_quality'] = pattern_quality
        
        # 2. Trend alignment
        if self.config['require_trend_alignment']:
            trend_ok, ma_s, ma_l = check_trend_alignment(df, idx=idx)
            details['trend_aligned'] = trend_ok
            details['ma_short'] = ma_s
            details['ma_long'] = ma_l
            if not trend_ok:
                passes = False
                reasons.append("Trend not aligned (Close < MA20 or MA20 < MA60)")
        
        # 3. Volume confirmation
        if self.config['require_volume_confirm']:
            vol_ok, vol_ratio = check_volume_confirmation(
                df, 
                vol_ratio_threshold=self.config['vol_ratio_threshold'],
                idx=idx
            )
            details['volume_confirmed'] = vol_ok
            details['volume_ratio'] = vol_ratio
            if not vol_ok:
                passes = False
                reasons.append(f"Volume ratio {vol_ratio} < {self.config['vol_ratio_threshold']}")
        
        # 4. RSI filter
        if self.config['require_rsi_filter']:
            rsi_buy_ok, _, rsi_val = check_rsi_filter(
                df, 
                overbought=self.config['rsi_overbought'],
                idx=idx
            )
            details['rsi_ok'] = rsi_buy_ok
            details['rsi_value'] = rsi_val
            if not rsi_buy_ok:
                passes = False
                reasons.append(f"RSI overbought ({rsi_val} >= {self.config['rsi_overbought']})")
        
        # 5. Price position (optional)
        if self.config['require_price_position']:
            buy_zone, _, distance = check_price_position(df, idx=idx)
            details['price_position_ok'] = buy_zone
            details['ma_distance'] = distance
            if not buy_zone:
                # This is a warning, not a hard filter
                reasons.append(f"Warning: Price {distance:.1%} above MA20")
        
        # 6. MACD confirmation (optional)
        if self.config['require_macd_confirm']:
            macd_buy_ok, _ = check_macd_confirmation(df, idx=idx)
            details['macd_confirmed'] = macd_buy_ok
            if not macd_buy_ok:
                passes = False
                reasons.append("MACD not confirming (histogram falling)")
        
        return passes, reasons, details
    
    def filter_sell_signal(self, df, pattern_quality=100, idx=-1):
        """
        Apply filters to a sell signal.
        
        Sell signals are less strict - we want to exit quickly when conditions deteriorate.
        
        Args:
            df: DataFrame with OHLCV and indicator columns
            pattern_quality: Quality score of the detected pattern
            idx: Row index
        
        Returns:
            tuple: (pass: bool, reasons: list[str], details: dict)
        """
        reasons = []
        details = {}
        passes = True
        
        # For sells, we only check RSI oversold (don't sell at bottom)
        if self.config['require_rsi_filter']:
            _, rsi_sell_ok, rsi_val = check_rsi_filter(
                df,
                oversold=self.config['rsi_oversold'],
                idx=idx
            )
            details['rsi_ok'] = rsi_sell_ok
            details['rsi_value'] = rsi_val
            if not rsi_sell_ok:
                # Warning only - still allow sell but note it
                reasons.append(f"Warning: RSI oversold ({rsi_val}), may be near bottom")
        
        details['pattern_quality'] = pattern_quality
        
        return passes, reasons, details
    
    def get_filter_stats(self):
        """Return current filter configuration."""
        return self.config.copy()


# Default filter instance - Balanced for win rate and trade frequency
DEFAULT_FILTER = SignalFilter(config={
    'min_pattern_quality': 50,       # Lower threshold to allow more signals
    'require_trend_alignment': False, # Relax trend requirement initially
    'require_volume_confirm': False,  # Relax volume requirement
    'require_rsi_filter': True,       # Keep RSI filter to avoid overbought
    'require_price_position': False,  # Don't require specific price position
    'require_macd_confirm': False,    # Don't require MACD confirmation
    'vol_ratio_threshold': 0.8,       # Lower volume threshold
    'rsi_overbought': 75,             # Standard overbought level
    'rsi_oversold': 25,               # Standard oversold level
})

# Strict filter for higher win rate (may reduce trade frequency)
STRICT_FILTER = SignalFilter(config={
    'min_pattern_quality': 70,
    'require_trend_alignment': True,
    'require_volume_confirm': False,
    'require_rsi_filter': True,
    'require_price_position': False,
    'require_macd_confirm': False,
    'vol_ratio_threshold': 1.0,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
})

# Relaxed filter for more trades (may lower win rate)
RELAXED_FILTER = SignalFilter(config={
    'min_pattern_quality': 40,
    'require_trend_alignment': False,
    'require_volume_confirm': False,
    'require_rsi_filter': False,
    'require_price_position': False,
    'require_macd_confirm': False,
    'vol_ratio_threshold': 0.5,
    'rsi_overbought': 80,
    'rsi_oversold': 20,
})
