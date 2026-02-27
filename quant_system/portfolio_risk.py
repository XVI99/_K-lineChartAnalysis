"""
Portfolio-Level Risk Management Module

Controls:
- Total portfolio risk heat (max aggregate risk exposure)
- Industry concentration limits
- Drawdown-triggered position reduction
"""
import pandas as pd
import numpy as np


def check_total_portfolio_risk(holdings, max_heat=0.10):
    """
    Check if total portfolio risk exceeds the maximum allowed heat.

    Portfolio Heat = Sum of (risk_per_trade) for all positions / total_capital.
    If heat > max_heat, reject new positions.

    Args:
        holdings (list[dict]): List of position dicts, each containing:
            - 'RiskAmount': dollar risk for this position
            - 'TotalCapital': total portfolio capital (same for all)
        max_heat (float): Maximum aggregate risk (default 10%).

    Returns:
        tuple: (allow_new_position: bool, current_heat: float)
    """
    if not holdings:
        return True, 0.0

    total_capital = holdings[0].get('TotalCapital', 100000)
    if total_capital <= 0:
        return False, 1.0

    total_risk = sum(h.get('RiskAmount', 0) for h in holdings)
    current_heat = total_risk / total_capital

    return current_heat < max_heat, round(current_heat, 4)


def check_industry_concentration(holdings, max_pct=0.30):
    """
    Check if any single industry exceeds the concentration limit.

    Args:
        holdings (list[dict]): List of position dicts, each containing:
            - 'Industry': industry/sector name (str)
            - 'MarketValue': current market value of position
        max_pct (float): Maximum allocation per industry (default 30%).

    Returns:
        tuple: (is_ok: bool, violations: dict[str, float])
            violations maps industry -> actual percentage for exceeded industries.
    """
    if not holdings:
        return True, {}

    total_value = sum(h.get('MarketValue', 0) for h in holdings)
    if total_value <= 0:
        return True, {}

    # Group by industry
    industry_values = {}
    for h in holdings:
        ind = h.get('Industry', 'Unknown')
        industry_values[ind] = industry_values.get(ind, 0) + h.get('MarketValue', 0)

    violations = {}
    for ind, val in industry_values.items():
        pct = val / total_value
        if pct > max_pct:
            violations[ind] = round(pct, 4)

    return len(violations) == 0, violations


def check_drawdown_reduction(equity_curve, threshold=0.15):
    """
    Check if portfolio drawdown has exceeded threshold, and return
    a position scaling factor.

    When drawdown > threshold, reduce new position sizes by 50%.
    When drawdown > 2*threshold, reduce by 80% (near-halt).

    Args:
        equity_curve (list[float] or pd.Series): Historical equity values.
        threshold (float): Drawdown threshold (default 15%).

    Returns:
        tuple: (current_drawdown: float, position_scale: float)
            position_scale: 1.0 = normal, 0.5 = reduced, 0.2 = near-halt
    """
    if len(equity_curve) < 2:
        return 0.0, 1.0

    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    current_dd = drawdown[-1]  # most recent drawdown (negative number)

    if current_dd < -2 * threshold:
        return round(current_dd, 4), 0.2
    elif current_dd < -threshold:
        return round(current_dd, 4), 0.5
    else:
        return round(current_dd, 4), 1.0


def get_industry_by_code(symbol):
    """
    Get industry/sector for a stock by its code prefix.

    Simplified mapping for A-shares based on code prefix.
    For production, use akshare's stock_individual_info_em or a local CSV.

    Args:
        symbol (str): Stock code, e.g., '600519'.

    Returns:
        str: Industry name (simplified).
    """
    # Simplified prefix-based mapping
    prefix = symbol[:3] if len(symbol) >= 3 else symbol

    mapping = {
        '600': '大盘蓝筹',
        '601': '大盘蓝筹',
        '603': '中小盘',
        '605': '中小盘',
        '000': '深圳主板',
        '001': '深圳主板',
        '002': '中小板',
        '003': '中小板',
        '300': '创业板',
        '301': '创业板',
        '688': '科创板',
    }

    return mapping.get(prefix, '其他')
