"""
Pattern Quality Scoring Module

Replaces hard-coded linear scoring (e.g., Morning Star +3, Engulfing +2) with a
confidence-based scoring system that considers:
- Pattern standard-ness (body/shadow ratios)
- Position relative to support/resistance
- Volume confirmation
- Trend alignment

Output: 0-100 confidence score per pattern, plus a weighted aggregate score.
"""
import pandas as pd
import numpy as np


def score_pattern_quality(df, pattern_col, idx=-1):
    """
    Score the quality of a detected pattern on a single bar.

    Components (total 100):
        - Form Quality   (30 pts): Body vs range ratio, shadow proportions
        - Position        (25 pts): Distance from MA, support/resistance proximity
        - Volume          (25 pts): Volume relative to MA(20)
        - Trend Alignment (20 pts): Agreement with MA20 / MA60 trend

    Args:
        df (pd.DataFrame): DataFrame with OHLCV and pattern columns.
        pattern_col (str): Name of the pattern column (e.g., 'Bull_Engulf').
        idx (int): Row index to score (default -1 = latest).

    Returns:
        float: Quality score (0-100). Returns 0 if pattern is not triggered.
    """
    if pattern_col not in df.columns:
        return 0.0

    row = df.iloc[idx]

    # Check if pattern is actually triggered
    if row.get(pattern_col, 0) != 1:
        return 0.0

    score = 0.0
    close = row['Close']
    open_ = row['Open']
    high = row['High']
    low = row['Low']
    volume = row.get('Volume', 0)

    body = abs(close - open_)
    full_range = high - low if high > low else 0.001
    upper_shadow = high - max(close, open_)
    lower_shadow = min(close, open_) - low

    # === 1. Form Quality (0-30) ===
    body_ratio = body / full_range
    # Strong candle body = high score
    if body_ratio > 0.7:
        score += 30
    elif body_ratio > 0.5:
        score += 22
    elif body_ratio > 0.3:
        score += 15
    else:
        score += 8

    # === 2. Position (0-25) ===
    # Check distance from MA20
    if len(df) >= 20:
        ma20 = df['Close'].rolling(20).mean().iloc[idx]
        if not pd.isna(ma20) and ma20 > 0:
            distance = (close - ma20) / ma20
            # Bullish patterns at/below MA20 get bonus
            if 'Bull' in pattern_col or 'Buy' in pattern_col or 'Hammer' in pattern_col:
                if distance < -0.02:  # Below MA20 → good buy area
                    score += 25
                elif distance < 0.02:  # Near MA20
                    score += 18
                else:
                    score += 8
            # Bearish patterns above MA20
            elif 'Bear' in pattern_col or 'Sell' in pattern_col or 'Shooting' in pattern_col:
                if distance > 0.05:  # Well above MA20 → overbought
                    score += 25
                elif distance > 0.0:
                    score += 15
                else:
                    score += 5
            else:
                score += 12  # Unknown type → neutral
        else:
            score += 12
    else:
        score += 12

    # === 3. Volume Confirmation (0-25) ===
    if len(df) >= 20 and volume > 0:
        vol_ma = df['Volume'].rolling(20).mean().iloc[idx]
        if not pd.isna(vol_ma) and vol_ma > 0:
            vol_ratio = volume / vol_ma
            if vol_ratio > 2.0:
                score += 25
            elif vol_ratio > 1.5:
                score += 20
            elif vol_ratio > 1.0:
                score += 12
            else:
                score += 5
        else:
            score += 10
    else:
        score += 10

    # === 4. Trend Alignment (0-20) ===
    if len(df) >= 60:
        ma60 = df['Close'].rolling(60).mean().iloc[idx]
        ma20 = df['Close'].rolling(20).mean().iloc[idx]
        if not pd.isna(ma60) and not pd.isna(ma20):
            bullish_trend = close > ma20 > ma60
            bearish_trend = close < ma20 < ma60

            is_bull_pattern = any(k in pattern_col for k in ['Bull', 'Buy', 'Hammer', 'Morning', 'Soldier'])
            is_bear_pattern = any(k in pattern_col for k in ['Bear', 'Sell', 'Shooting', 'Hanging', 'Dead'])

            if (is_bull_pattern and bullish_trend) or (is_bear_pattern and bearish_trend):
                score += 20  # Pattern aligns with trend
            elif (is_bull_pattern and not bearish_trend) or (is_bear_pattern and not bullish_trend):
                score += 10  # Neutral alignment
            else:
                score += 3   # Counter-trend
        else:
            score += 10
    else:
        score += 10

    return round(score, 1)


def get_weighted_score(df, pattern_weights, idx=-1):
    """
    Calculate a weighted aggregate score across multiple patterns.

    Replaces the hard-coded Score += 3/2/1 logic in budget_monitor.py.

    Args:
        df (pd.DataFrame): DataFrame with OHLCV + pattern columns.
        pattern_weights (dict): {pattern_col: base_weight} e.g.:
            {'Bull_Engulf': 2, 'Morning_Star': 3, 'Hammer': 1, ...}
        idx (int): Row index to evaluate.

    Returns:
        tuple: (total_score: float, details: list[tuple(pattern, quality, weighted)])
    """
    total_score = 0.0
    details = []

    for pattern_col, weight in pattern_weights.items():
        if pattern_col not in df.columns:
            continue

        row_val = df[pattern_col].iloc[idx] if idx < len(df) else 0
        if row_val != 1:
            continue

        quality = score_pattern_quality(df, pattern_col, idx)
        # Normalize quality to 0-1 and multiply by weight
        weighted = weight * (quality / 100.0)
        total_score += weighted
        details.append((pattern_col, quality, round(weighted, 2)))

    return round(total_score, 2), details


# Default pattern weights (same hierarchy as budget_monitor.py)
DEFAULT_BUY_WEIGHTS = {
    'Morning_Star': 3,
    'White_Three_Soldiers': 3,
    'Bull_Engulf': 2,
    'BottomSignal': 2,
    'Hammer': 1,
    'InvertedHammer': 1,
    'Bull_BeltHold': 1,
    'BuySignal': 1,
    'RSI_OverSold': 1,
}

DEFAULT_SELL_WEIGHTS = {
    'Bear_Engulf': 2,
    'TopSignal': 2,
    'Bear_BeltHold': 1,
    'ShootingStar': 1,
    'Hanging_Man': 1,
    'Advance_Block': 1,
    'SellSignal': 1,
}


def filter_signals_by_quality(df, pattern_weights=None, min_quality=70, min_score=5, idx=-1):
    """
    Filter signals based on pattern quality and aggregate score.
    
    Phase 1 optimization: Only accept signals that meet quality thresholds.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV + pattern columns.
        pattern_weights (dict): Pattern weights for scoring. Uses DEFAULT_BUY_WEIGHTS if None.
        min_quality (float): Minimum quality score (0-100) for individual patterns. Default 70.
        min_score (float): Minimum aggregate score to pass filter. Default 5.
        idx (int): Row index to evaluate.
        
    Returns:
        tuple: (passed: bool, total_score: float, quality_details: dict)
            - passed: True if signal passes quality filter
            - total_score: Aggregate weighted score
            - quality_details: Dict with individual pattern scores
    """
    if pattern_weights is None:
        pattern_weights = DEFAULT_BUY_WEIGHTS
    
    total_score = 0.0
    quality_details = {}
    best_quality = 0.0
    
    for pattern_col, weight in pattern_weights.items():
        if pattern_col not in df.columns:
            continue
            
        row_val = df[pattern_col].iloc[idx] if idx < len(df) else 0
        if row_val != 1:
            continue
        
        quality = score_pattern_quality(df, pattern_col, idx)
        weighted = weight * (quality / 100.0)
        total_score += weighted
        
        quality_details[pattern_col] = {
            'quality': quality,
            'weight': weight,
            'weighted_score': round(weighted, 2)
        }
        
        if quality > best_quality:
            best_quality = quality
    
    # Signal passes if:
    # 1. Total score meets minimum threshold
    # 2. At least one pattern has quality >= min_quality
    passed = total_score >= min_score and best_quality >= min_quality
    
    return passed, round(total_score, 2), quality_details


def get_pattern_quality_score(df, pattern_weights=None, idx=-1):
    """
    Get the best pattern quality score for the given bar.
    
    This is a simplified version that returns just the quality score
    for integration with the signal filter.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV + pattern columns.
        pattern_weights (dict): Pattern weights. Uses DEFAULT_BUY_WEIGHTS if None.
        idx (int): Row index to evaluate.
        
    Returns:
        float: Best quality score among triggered patterns (0-100).
    """
    if pattern_weights is None:
        pattern_weights = DEFAULT_BUY_WEIGHTS
    
    best_quality = 0.0
    
    for pattern_col in pattern_weights.keys():
        if pattern_col not in df.columns:
            continue
            
        row_val = df[pattern_col].iloc[idx] if idx < len(df) else 0
        if row_val != 1:
            continue
        
        quality = score_pattern_quality(df, pattern_col, idx)
        if quality > best_quality:
            best_quality = quality
    
    return best_quality


class PatternQualityFilter:
    """
    Pattern Quality Filter for use in backtesting.
    
    Integrates with SignalFilter to provide quality-based signal filtering.
    """
    
    def __init__(self, min_quality=70, min_score=5, pattern_weights=None):
        """
        Initialize the pattern quality filter.
        
        Args:
            min_quality (float): Minimum quality score threshold (0-100).
            min_score (float): Minimum aggregate score threshold.
            pattern_weights (dict): Custom pattern weights.
        """
        self.min_quality = min_quality
        self.min_score = min_score
        self.pattern_weights = pattern_weights or DEFAULT_BUY_WEIGHTS.copy()
    
    def check_buy_signal(self, df, idx=-1):
        """
        Check if a buy signal passes quality filter.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV + pattern columns.
            idx (int): Row index to check.
            
        Returns:
            tuple: (passed: bool, score: float, details: dict)
        """
        return filter_signals_by_quality(
            df,
            self.pattern_weights,
            self.min_quality,
            self.min_score,
            idx
        )
    
    def get_quality_score(self, df, idx=-1):
        """
        Get the best quality score for the bar.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV + pattern columns.
            idx (int): Row index to check.
            
        Returns:
            float: Quality score (0-100).
        """
        return get_pattern_quality_score(df, self.pattern_weights, idx)
