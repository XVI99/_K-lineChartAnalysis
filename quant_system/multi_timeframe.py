"""
Multi-Timeframe Confirmation Module

Implements higher timeframe trend confirmation for daily signals.
A daily buy signal is only valid if the weekly trend is bullish.

This significantly improves win rate by filtering counter-trend trades.
"""
import pandas as pd
import numpy as np


def resample_to_weekly(df):
    """
    Resample daily OHLCV data to weekly.
    
    Args:
        df: DataFrame with DatetimeIndex and OHLCV columns
    
    Returns:
        DataFrame: Weekly OHLCV data
    """
    if df.empty:
        return df
    
    weekly = df.resample('W').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    
    return weekly.dropna()


def check_weekly_trend(weekly_df, ma_period=20, idx=-1):
    """
    Check if weekly trend is bullish (Close > MA).
    
    Args:
        weekly_df: Weekly OHLCV DataFrame
        ma_period: MA period (default 20 weeks = ~5 months)
        idx: Index to check (default -1 = latest)
    
    Returns:
        tuple: (is_bullish: bool, ma_value: float, close_value: float)
    """
    if weekly_df.empty or len(weekly_df) < ma_period:
        return True, 0, 0  # Not enough data, pass through
    
    close = weekly_df['Close'].iloc[idx]
    ma = weekly_df['Close'].rolling(ma_period).mean().iloc[idx]
    
    if pd.isna(ma):
        return True, 0, close
    
    is_bullish = close > ma
    
    return is_bullish, ma, close


def check_weekly_ma_cross(weekly_df, ma_short=10, ma_long=30, idx=-1):
    """
    Check if weekly MA short is above MA long (golden cross zone).
    
    Args:
        weekly_df: Weekly OHLCV DataFrame
        ma_short: Short MA period (default 10 weeks)
        ma_long: Long MA period (default 30 weeks)
        idx: Index to check
    
    Returns:
        tuple: (is_bullish: bool, ma_short_val: float, ma_long_val: float)
    """
    if weekly_df.empty or len(weekly_df) < ma_long:
        return True, 0, 0
    
    ma_s = weekly_df['Close'].rolling(ma_short).mean().iloc[idx]
    ma_l = weekly_df['Close'].rolling(ma_long).mean().iloc[idx]
    
    if pd.isna(ma_s) or pd.isna(ma_l):
        return True, ma_s, ma_l
    
    is_bullish = ma_s > ma_l
    
    return is_bullish, ma_s, ma_l


def get_weekly_rsi(weekly_df, period=14, idx=-1):
    """
    Calculate RSI on weekly data.
    
    Args:
        weekly_df: Weekly OHLCV DataFrame
        period: RSI period
        idx: Index to check
    
    Returns:
        float: RSI value
    """
    if weekly_df.empty or len(weekly_df) < period + 1:
        return 50.0  # Neutral
    
    delta = weekly_df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    current_rsi = rsi.iloc[idx]
    
    return current_rsi if not pd.isna(current_rsi) else 50.0


class MultiTimeframeConfirm:
    """
    Multi-timeframe confirmation for trading signals.
    """
    
    def __init__(self, config=None):
        """
        Initialize with configuration.
        
        Args:
            config: dict with settings
        """
        self.config = config or {
            'require_weekly_trend': True,      # Require weekly Close > MA20
            'require_weekly_ma_cross': False,  # Require weekly MA10 > MA30
            'weekly_trend_ma': 20,             # Weekly trend MA period
            'weekly_rsi_filter': True,         # Check weekly RSI
            'weekly_rsi_max': 70,              # Max weekly RSI for buys
        }
        
        # Cache for weekly data
        self._weekly_cache = {}
    
    def get_weekly_data(self, daily_df):
        """
        Get or compute weekly data from daily data.
        Uses length-based cache key to avoid recomputing when only
        a few bars change (e.g., during daily updates).
        
        Args:
            daily_df: Daily OHLCV DataFrame
        
        Returns:
            DataFrame: Weekly OHLCV
        """
        if daily_df.empty:
            return daily_df
        
        # Use (start_date, end_date, length) as cache key — not just end_date
        cache_key = (str(daily_df.index[0]), str(daily_df.index[-1]), len(daily_df))
        
        if cache_key not in self._weekly_cache:
            # Clear old entries to avoid unbounded growth
            if len(self._weekly_cache) > 5:
                self._weekly_cache.clear()
            self._weekly_cache[cache_key] = resample_to_weekly(daily_df)
        
        return self._weekly_cache[cache_key]
    
    def confirm_buy_signal(self, daily_df, idx=-1):
        """
        Confirm a daily buy signal with weekly timeframe.
        
        Args:
            daily_df: Daily OHLCV DataFrame
            idx: Index of the signal (default -1 = latest)
        
        Returns:
            tuple: (confirmed: bool, reasons: list, details: dict)
        """
        reasons = []
        details = {}
        confirmed = True
        
        # Get weekly data
        weekly_df = self.get_weekly_data(daily_df)
        
        if weekly_df.empty:
            return True, ['No weekly data'], {'weekly_available': False}
        
        details['weekly_available'] = True
        details['weekly_bars'] = len(weekly_df)
        
        # 1. Weekly trend check
        if self.config['require_weekly_trend']:
            is_bullish, ma_val, close_val = check_weekly_trend(
                weekly_df, 
                ma_period=self.config['weekly_trend_ma'],
                idx=-1
            )
            details['weekly_trend_bullish'] = is_bullish
            details['weekly_ma'] = ma_val
            details['weekly_close'] = close_val
            
            if not is_bullish:
                confirmed = False
                reasons.append(f"Weekly trend bearish (Close {close_val:.2f} < MA {ma_val:.2f})")
        
        # 2. Weekly MA cross check
        if self.config['require_weekly_ma_cross']:
            is_bullish, ma_s, ma_l = check_weekly_ma_cross(weekly_df, idx=-1)
            details['weekly_ma_cross_bullish'] = is_bullish
            
            if not is_bullish:
                confirmed = False
                reasons.append(f"Weekly MA cross bearish (MA10 {ma_s:.2f} < MA30 {ma_l:.2f})")
        
        # 3. Weekly RSI check
        if self.config['weekly_rsi_filter']:
            weekly_rsi = get_weekly_rsi(weekly_df, idx=-1)
            details['weekly_rsi'] = weekly_rsi
            
            if weekly_rsi > self.config['weekly_rsi_max']:
                confirmed = False
                reasons.append(f"Weekly RSI overbought ({weekly_rsi:.1f} > {self.config['weekly_rsi_max']})")
        
        return confirmed, reasons, details
    
    def confirm_sell_signal(self, daily_df, idx=-1):
        """
        Confirm a daily sell signal with weekly timeframe.
        
        Sell signals are less strict - we want to exit when conditions deteriorate.
        Weekly confirmation is optional for sells.
        
        Args:
            daily_df: Daily OHLCV DataFrame
            idx: Index of the signal
        
        Returns:
            tuple: (confirmed: bool, reasons: list, details: dict)
        """
        reasons = []
        details = {}
        
        # Get weekly data
        weekly_df = self.get_weekly_data(daily_df)
        
        if weekly_df.empty:
            return True, [], {'weekly_available': False}
        
        details['weekly_available'] = True
        
        # For sells, just check if weekly trend is bearish (adds confidence)
        is_bullish, _, _ = check_weekly_trend(weekly_df, idx=-1)
        details['weekly_trend_bullish'] = is_bullish
        
        if not is_bullish:
            reasons.append("Weekly trend confirms bearish")
        
        # Always allow sell signals
        return True, reasons, details

    def add_weekly_trend_column(self, daily_df):
        """
        Pre-compute weekly trend and map it back to daily DataFrame.
        
        This is much more efficient than calling confirm_buy_signal per bar,
        as weekly data is computed once and mapped globally.
        
        Args:
            daily_df: Daily OHLCV DataFrame with DatetimeIndex
        
        Returns:
            DataFrame: daily_df with 'Weekly_Trend' column added
        """
        df = daily_df.copy()
        df['Weekly_Trend'] = None
        
        weekly_df = self.get_weekly_data(df)
        if weekly_df.empty or len(weekly_df) < self.config.get('weekly_trend_ma', 20):
            return df
        
        # Compute weekly MA
        ma_period = self.config.get('weekly_trend_ma', 20)
        weekly_close = weekly_df['Close']
        weekly_ma = weekly_close.rolling(ma_period).mean()
        
        # Determine regime for each week
        weekly_regime = pd.Series(index=weekly_df.index, dtype=object)
        for i in range(len(weekly_df)):
            c = weekly_close.iloc[i]
            m = weekly_ma.iloc[i]
            if pd.isna(m):
                weekly_regime.iloc[i] = None
            elif c > m:
                weekly_regime.iloc[i] = 'BULL'
            else:
                weekly_regime.iloc[i] = 'BEAR'
        
        # Map weekly regime to daily bars using forward-fill
        # Each daily bar gets the regime of its corresponding week
        weekly_regime.index = weekly_df.index
        # Reindex to daily and forward-fill
        daily_regime = weekly_regime.reindex(df.index, method='ffill')
        df['Weekly_Trend'] = daily_regime
        
        return df


# Default instance
DEFAULT_MTF_CONFIRM = MultiTimeframeConfirm()

# Strict instance (requires all conditions)
STRICT_MTF_CONFIRM = MultiTimeframeConfirm(config={
    'require_weekly_trend': True,
    'require_weekly_ma_cross': True,
    'weekly_trend_ma': 20,
    'weekly_rsi_filter': True,
    'weekly_rsi_max': 65,
})

# Relaxed instance (only basic trend check)
RELAXED_MTF_CONFIRM = MultiTimeframeConfirm(config={
    'require_weekly_trend': True,
    'require_weekly_ma_cross': False,
    'weekly_trend_ma': 20,
    'weekly_rsi_filter': False,
    'weekly_rsi_max': 75,
})
