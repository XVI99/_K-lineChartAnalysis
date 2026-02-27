"""
Risk Management Utilities for Quantitative Trading System
Contains: ATR, Position Sizing, Volatility Stop, Liquidity Check,
          Kelly Criterion, Chandelier Exit, VaR/CVaR, EMA Trailing Stop
"""
import pandas as pd
import numpy as np

def calculate_atr(df, period=14):
    """
    Calculate Average True Range (ATR) for volatility measurement.
    
    ATR = Average of True Range over 'period' days
    True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    
    Args:
        df: DataFrame with 'High', 'Low', 'Close' columns
        period: Lookback period (default 14)
    
    Returns:
        Series of ATR values
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


def calculate_position_size(budget, current_price, atr, risk_pct=0.02, atr_multiplier=1.5):
    """
    Calculate optimal position size based on ATR and risk tolerance.
    
    Formula:
        Risk Per Trade = Budget * risk_pct
        Stop Distance = ATR * atr_multiplier
        Shares = Risk Per Trade / Stop Distance
    
    Args:
        budget: Total capital (e.g., 5000 CNY)
        current_price: Current stock price
        atr: Current ATR value
        risk_pct: Max risk per trade as fraction (default 2%)
        atr_multiplier: Stop loss distance in ATR units (default 1.5)
    
    Returns:
        int: Number of shares to buy (rounded to lot of 100 for A-shares)
    """
    if atr <= 0 or current_price <= 0:
        return 0
    
    risk_per_trade = budget * risk_pct
    stop_distance = atr * atr_multiplier
    
    # Calculate raw shares based on risk
    risk_based_shares = risk_per_trade / stop_distance
    
    # Also check max affordable shares
    max_affordable = budget / current_price
    
    # Take the minimum to stay within both constraints
    shares = min(risk_based_shares, max_affordable)
    
    # Round down to nearest 100 (A-share lot size)
    shares = int(shares // 100) * 100
    
    return max(shares, 0)


def calculate_volatility_stop(entry_price, atr, multiplier=2.0, direction='long'):
    """
    Calculate stop loss based on volatility (ATR).
    
    Args:
        entry_price: Entry price
        atr: Current ATR value
        multiplier: ATR multiplier for stop distance (default 2.0)
        direction: 'long' or 'short'
    
    Returns:
        float: Stop loss price
    """
    stop_distance = atr * multiplier
    
    if direction == 'long':
        stop_loss = entry_price - stop_distance
    else:
        stop_loss = entry_price + stop_distance
    
    return round(stop_loss, 2)


def calculate_take_profit(entry_price, atr, risk_reward_ratio=2.0, direction='long'):
    """
    Calculate take profit based on volatility and risk/reward.
    
    Args:
        entry_price: Entry price
        atr: Current ATR value
        risk_reward_ratio: Target R:R (default 2.0 meaning 1:2)
        direction: 'long' or 'short'
    
    Returns:
        float: Take profit price
    """
    # Risk is 2*ATR (same as stop), reward is risk * ratio
    risk_distance = atr * 2.0
    reward_distance = risk_distance * risk_reward_ratio
    
    if direction == 'long':
        take_profit = entry_price + reward_distance
    else:
        take_profit = entry_price - reward_distance
    
    return round(take_profit, 2)


def check_liquidity(df, min_turnover=50_000_000, lookback=20):
    """
    Check if stock has sufficient liquidity.
    
    Args:
        df: DataFrame with 'Volume' and 'Close' columns
        min_turnover: Minimum average daily turnover in CNY (default 50M)
        lookback: Days to average (default 20)
    
    Returns:
        tuple: (is_liquid: bool, avg_turnover: float)
    """
    if len(df) < lookback:
        return False, 0
    
    # Calculate turnover (Volume * Close approximation)
    # Note: For A-shares, Volume is usually in shares, not CNY
    turnover = df['Volume'].iloc[-lookback:] * df['Close'].iloc[-lookback:]
    avg_turnover = turnover.mean()
    
    return avg_turnover >= min_turnover, avg_turnover


def check_trend_filter(df, ma_period=200):
    """
    Check if stock is in uptrend (Close > MA200).
    
    Args:
        df: DataFrame with 'Close' column
        ma_period: Moving average period (default 200)
    
    Returns:
        tuple: (is_uptrend: bool, ma_value: float, close_value: float)
    """
    if len(df) < ma_period:
        # Not enough data, assume neutral
        return True, 0, df['Close'].iloc[-1]
    
    ma = df['Close'].rolling(window=ma_period).mean()
    current_close = df['Close'].iloc[-1]
    current_ma = ma.iloc[-1]
    
    is_uptrend = current_close > current_ma
    
    return is_uptrend, current_ma, current_close


def check_signal_persistence(df, signal_col, lookback=3):
    """
    Check if a signal occurred in recent days (not just today).
    
    Args:
        df: DataFrame with signal column
        signal_col: Name of the signal column
        lookback: Days to check (default 3)
    
    Returns:
        tuple: (has_recent_signal: bool, days_since_signal: int)
    """
    if signal_col not in df.columns:
        return False, -1
    
    recent = df[signal_col].iloc[-lookback:]
    
    # Find if any signal in recent period
    has_signal = recent.sum() >= 1
    
    # Find how many days ago
    if has_signal:
        # Find the most recent True
        for i, val in enumerate(reversed(recent.values)):
            if val == 1 or val == True:
                return True, i
    
    return False, -1


# --- Convenience Function for Budget Monitor ---
def generate_trade_plan_v2(entry_price, atr, budget, risk_pct=0.02):
    """
    Generate a complete trade plan with ATR-based risk management.
    
    Returns:
        dict with Entry, StopLoss, TakeProfit, Shares, RiskAmount
    """
    stop_loss = calculate_volatility_stop(entry_price, atr, multiplier=2.0)
    take_profit = calculate_take_profit(entry_price, atr, risk_reward_ratio=2.0)
    shares = calculate_position_size(budget, entry_price, atr, risk_pct=risk_pct)
    
    risk_amount = (entry_price - stop_loss) * shares if shares > 0 else 0
    
    return {
        "Entry": entry_price,
        "StopLoss": stop_loss,
        "TakeProfit": take_profit,
        "Shares": shares,
        "RiskAmount": round(risk_amount, 2),
        "RiskReward": "1:2",
        "ATR": round(atr, 2)
    }


# =============================================================
# Phase 2: Advanced Risk Management Functions
# =============================================================

def calculate_kelly_position(win_rate, avg_win, avg_loss, budget, current_price,
                             kelly_fraction=0.5):
    """
    Calculate position size using the Kelly Criterion (half-Kelly for safety).

    Kelly formula: f* = (p * b - q) / b
    where p = win_rate, q = 1 - p, b = avg_win / |avg_loss|

    Args:
        win_rate (float): Historical win rate (0-1), e.g. 0.55.
        avg_win (float): Average winning trade return (e.g. 0.08 for 8%).
        avg_loss (float): Average losing trade return (negative, e.g. -0.04).
        budget (float): Total capital.
        current_price (float): Current stock price.
        kelly_fraction (float): Fraction of Kelly to use (0.5 = half Kelly).

    Returns:
        int: Number of shares (rounded to A-share 100-lot).
    """
    if win_rate <= 0 or avg_win <= 0 or avg_loss >= 0 or current_price <= 0:
        return 0

    b = avg_win / abs(avg_loss)  # payoff ratio
    q = 1 - win_rate

    # Kelly fraction
    kelly_f = (win_rate * b - q) / b

    # Clamp to [0, 1] — negative Kelly means don't bet
    kelly_f = max(0.0, min(kelly_f, 1.0))

    # Apply fractional Kelly
    position_fraction = kelly_f * kelly_fraction

    # Calculate shares
    invest_amount = budget * position_fraction
    shares = int(invest_amount / current_price)
    shares = (shares // 100) * 100  # A-share lot

    return max(shares, 0)


def calculate_chandelier_exit(high_since_entry, atr, multiplier=3.0):
    """
    Calculate Chandelier Exit (trailing stop based on highest high).

    Chandelier Exit (Long) = Highest High - multiplier * ATR

    This is superior to fixed trailing stop in trending markets.

    Args:
        high_since_entry (float): Highest price since entry.
        atr (float): Current ATR value.
        multiplier (float): ATR multiplier (default 3.0).

    Returns:
        float: Chandelier exit price (stop level).
    """
    if atr <= 0:
        return high_since_entry * 0.9  # fallback: 10% trailing
    return round(high_since_entry - multiplier * atr, 2)


def calculate_var(returns, confidence=0.95):
    """
    Calculate Value at Risk (VaR) using historical simulation.

    VaR answers: "What is the maximum loss at X% confidence?"

    Args:
        returns (array-like): Series of daily returns.
        confidence (float): Confidence level (default 0.95 = 95%).

    Returns:
        float: VaR as a negative number (e.g., -0.03 means 3% max loss).
    """
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]
    if len(returns) == 0:
        return 0.0
    return float(np.percentile(returns, (1 - confidence) * 100))


def calculate_cvar(returns, confidence=0.95):
    """
    Calculate Conditional VaR / Expected Shortfall (CVaR).

    CVaR answers: "If we exceed VaR, what is the expected average loss?"
    Always worse (more negative) than VaR.

    Args:
        returns (array-like): Series of daily returns.
        confidence (float): Confidence level (default 0.95).

    Returns:
        float: CVaR as a negative number.
    """
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]
    if len(returns) == 0:
        return 0.0
    var = calculate_var(returns, confidence)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) > 0 else var


def calculate_ema_trailing_stop(df, period=20):
    """
    Calculate EMA-based trailing stop levels for the entire DataFrame.

    In trending markets, using EMA as a trailing stop allows profits to run
    while locking in gains when trend weakens.

    Args:
        df: DataFrame with 'Close' column.
        period (int): EMA period (default 20).

    Returns:
        pd.Series: EMA trailing stop levels (same index as df).
    """
    return df['Close'].ewm(span=period, adjust=False).mean()
