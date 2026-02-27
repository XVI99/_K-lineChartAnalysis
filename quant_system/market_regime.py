"""
Market Regime Detection Module

Determines if the overall market (e.g., CSI 300) is in BULL / NEUTRAL / BEAR state.
Used to filter buy signals in bearish market environments.
"""
import os
import sys
import pandas as pd
import numpy as np
from enum import Enum

# Add parent for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from custom_data import get_price
except ImportError:
    get_price = None


class MarketRegime(Enum):
    BULL = "BULL"
    NEUTRAL = "NEUTRAL"
    BEAR = "BEAR"


def get_market_index_data(index_code="sh000300", days=120):
    """
    Fetch market index (CSI 300) daily data.

    Args:
        index_code (str): Index code in format 'shXXXXXX' or 'szXXXXXX'.
            Default: 'sh000300' (沪深300).
        days (int): Number of trading days to fetch.

    Returns:
        pd.DataFrame: OHLCV data with DatetimeIndex, or empty DataFrame on failure.
    """
    # Try using the same custom_data provider
    if get_price is not None:
        try:
            df = get_price(index_code, count=days, frequency='1d')
            if df is not None and not df.empty:
                # Standardize columns
                df = df.rename(columns={
                    'open': 'Open', 'high': 'High', 'low': 'Low',
                    'close': 'Close', 'volume': 'Volume'
                })
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col not in df.columns:
                        return pd.DataFrame()
                return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            print(f"[MarketRegime] Failed to fetch index data: {e}")

    # Try akshare fallback
    try:
        import akshare as ak
        from datetime import datetime, timedelta

        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=int(days * 1.5))

        # CSI 300 = 000300
        code = index_code.replace("sh", "").replace("sz", "")
        df = ak.stock_zh_index_daily(symbol=index_code)
        if df is not None and not df.empty:
            df = df.rename(columns={
                'date': 'Date', 'open': 'Open', 'high': 'High',
                'low': 'Low', 'close': 'Close', 'volume': 'Volume'
            })
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            return df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(days)
    except Exception as e:
        print(f"[MarketRegime] Akshare fallback failed: {e}")

    return pd.DataFrame()


def check_market_regime(index_df, ma_short=20, ma_long=60):
    """
    Determine market regime based on moving average structure.

    Rules:
        - BULL:    Close > MA_short > MA_long (strong uptrend)
        - NEUTRAL: Close > MA_long (above long-term trend)
        - BEAR:    Close < MA_long (below long-term trend)

    Args:
        index_df (pd.DataFrame): OHLCV DataFrame for market index.
        ma_short (int): Short MA period (default 20).
        ma_long (int): Long MA period (default 60).

    Returns:
        MarketRegime: BULL, NEUTRAL, or BEAR.
    """
    if index_df.empty or len(index_df) < ma_long:
        return MarketRegime.NEUTRAL  # Not enough data, assume neutral

    close = index_df['Close']
    ma_s = close.rolling(window=ma_short).mean()
    ma_l = close.rolling(window=ma_long).mean()

    current_close = close.iloc[-1]
    current_ma_s = ma_s.iloc[-1]
    current_ma_l = ma_l.iloc[-1]

    if pd.isna(current_ma_s) or pd.isna(current_ma_l):
        return MarketRegime.NEUTRAL

    if current_close > current_ma_s > current_ma_l:
        return MarketRegime.BULL
    elif current_close > current_ma_l:
        return MarketRegime.NEUTRAL
    else:
        return MarketRegime.BEAR


def get_market_regime_filter(index_code="sh000300", days=120, ma_short=20, ma_long=60):
    """
    Convenience function: fetch index data and determine regime + trading permission.

    Returns:
        tuple: (regime: MarketRegime, allow_long: bool, position_scale: float)
            - BULL: (True, 1.0) — full position
            - NEUTRAL: (True, 0.5) — half position
            - BEAR: (False, 0.0) — no new longs
    """
    index_df = get_market_index_data(index_code, days)
    regime = check_market_regime(index_df, ma_short, ma_long)

    if regime == MarketRegime.BULL:
        return regime, True, 1.0
    elif regime == MarketRegime.NEUTRAL:
        return regime, True, 0.5
    else:  # BEAR
        return regime, False, 0.0
