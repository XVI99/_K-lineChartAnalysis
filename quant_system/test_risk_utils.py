"""
Unit tests for risk_utils.py new functions.
Tests: Kelly position, Chandelier exit, VaR/CVaR, EMA trailing stop.
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from risk_utils import (
    calculate_kelly_position, calculate_chandelier_exit,
    calculate_var, calculate_cvar, calculate_ema_trailing_stop
)
from portfolio_risk import (
    check_total_portfolio_risk, check_industry_concentration,
    check_drawdown_reduction
)


class TestKellyPosition:
    def test_positive_expectancy(self):
        # 55% win rate, avg win 8%, avg loss -4% → should get some shares
        shares = calculate_kelly_position(
            win_rate=0.55, avg_win=0.08, avg_loss=-0.04,
            budget=100000, current_price=10.0, kelly_fraction=0.5
        )
        assert shares > 0
        assert shares % 100 == 0  # A-share lot

    def test_negative_expectancy_returns_zero(self):
        # 30% win rate with bad ratio → Kelly < 0 → 0 shares
        shares = calculate_kelly_position(
            win_rate=0.30, avg_win=0.03, avg_loss=-0.06,
            budget=100000, current_price=10.0, kelly_fraction=0.5
        )
        assert shares == 0

    def test_invalid_inputs(self):
        assert calculate_kelly_position(0, 0.08, -0.04, 100000, 10.0) == 0
        assert calculate_kelly_position(0.55, 0.08, 0.04, 100000, 10.0) == 0  # positive avg_loss


class TestChandelierExit:
    def test_basic(self):
        stop = calculate_chandelier_exit(high_since_entry=100.0, atr=2.0, multiplier=3.0)
        assert stop == 94.0  # 100 - 3*2

    def test_zero_atr_fallback(self):
        stop = calculate_chandelier_exit(high_since_entry=100.0, atr=0, multiplier=3.0)
        assert stop == 90.0  # fallback: 10% trailing


class TestVaRCVaR:
    def test_var(self):
        np.random.seed(42)
        returns = np.random.normal(-0.001, 0.02, 250)
        var95 = calculate_var(returns, confidence=0.95)
        assert var95 < 0  # Should be negative

    def test_cvar_worse_than_var(self):
        np.random.seed(42)
        returns = np.random.normal(-0.001, 0.02, 250)
        var = calculate_var(returns, 0.95)
        cvar = calculate_cvar(returns, 0.95)
        assert cvar <= var  # CVaR is always worse (more negative)

    def test_empty_returns(self):
        assert calculate_var([], 0.95) == 0.0
        assert calculate_cvar([], 0.95) == 0.0


class TestEMATrailingStop:
    def test_ema_shape(self):
        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        df = pd.DataFrame({"Close": np.linspace(10, 15, 50)}, index=dates)
        ema = calculate_ema_trailing_stop(df, period=20)
        assert len(ema) == 50
        # EMA should lag below the linearly rising close
        assert ema.iloc[-1] < df["Close"].iloc[-1]


class TestPortfolioRisk:
    def test_heat_under_limit(self):
        holdings = [
            {"RiskAmount": 2000, "TotalCapital": 100000},
            {"RiskAmount": 3000, "TotalCapital": 100000},
        ]
        allowed, heat = check_total_portfolio_risk(holdings, max_heat=0.10)
        assert allowed  # 5000/100000 = 5% < 10%

    def test_heat_over_limit(self):
        holdings = [
            {"RiskAmount": 6000, "TotalCapital": 100000},
            {"RiskAmount": 6000, "TotalCapital": 100000},
        ]
        allowed, heat = check_total_portfolio_risk(holdings, max_heat=0.10)
        assert not allowed  # 12000/100000 = 12% > 10%

    def test_industry_concentration(self):
        holdings = [
            {"Industry": "科技", "MarketValue": 40000},
            {"Industry": "科技", "MarketValue": 30000},
            {"Industry": "消费", "MarketValue": 30000},
        ]
        ok, violations = check_industry_concentration(holdings, max_pct=0.30)
        assert not ok
        assert "科技" in violations  # 70% > 30%

    def test_drawdown_reduction(self):
        equity = [100000, 105000, 110000, 100000, 90000, 85000]
        dd, scale = check_drawdown_reduction(equity, threshold=0.15)
        assert dd < 0
        # Max was 110000, current 85000 → dd = -22.7% → between 15% and 30% → scale = 0.5
        assert scale == 0.5

    def test_severe_drawdown_near_halt(self):
        # 50% drawdown → exceeds 2*15% = 30% → scale = 0.2
        equity = [100000, 120000, 60000]
        dd, scale = check_drawdown_reduction(equity, threshold=0.15)
        assert scale == 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
