"""
Unit tests for the upgraded BacktestEngine.
Tests: next_open execution, stamp tax, limit-up/down skip, time stop, 100-lot rounding.
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_engine import BacktestEngine


def _make_df(closes, opens=None, signals=None, highs=None, lows=None, volumes=None):
    """Helper to create a test DataFrame with OHLCV + Signal."""
    n = len(closes)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    if opens is None:
        opens = closes  # default: open = close
    if highs is None:
        highs = [max(o, c) + 0.5 for o, c in zip(opens, closes)]
    if lows is None:
        lows = [min(o, c) - 0.5 for o, c in zip(opens, closes)]
    if volumes is None:
        volumes = [1000000] * n
    if signals is None:
        signals = [0] * n
    df = pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes,
        "Volume": volumes, "Signal": signals,
    }, index=dates)
    return df


class TestNextOpenExecution:
    """Test that signals execute at next bar's open price, not current close."""

    def test_buy_executes_at_next_open(self):
        # Signal on bar 2 (index 2), should execute at bar 3's open
        closes = [10.0, 10.0, 10.0, 11.0, 12.0]
        opens =  [10.0, 10.0, 10.0, 10.5, 11.5]
        signals = [0, 0, 1, 0, 0]  # buy signal on bar 2

        df = _make_df(closes, opens=opens, signals=signals)
        engine = BacktestEngine(
            initial_capital=100000, commission=0, slippage=0, stamp_tax=0,
            execution_mode="next_open"
        )
        engine.run(df)

        # Should have one BUY trade
        buys = [t for t in engine.trade_log if t["Type"] == "BUY"]
        assert len(buys) == 1
        # Execution price should be bar 3's open = 10.5, not bar 2's close = 10.0
        assert buys[0]["Price"] == 10.5

    def test_sell_executes_at_next_open(self):
        closes = [10.0, 10.0, 10.0, 11.0, 12.0, 11.0, 10.0]
        opens =  [10.0, 10.0, 10.0, 10.5, 11.5, 11.2, 10.5]
        signals = [0, 0, 1, 0, 0, -1, 0]  # buy on bar2, sell on bar5

        df = _make_df(closes, opens=opens, signals=signals)
        engine = BacktestEngine(
            initial_capital=100000, commission=0, slippage=0, stamp_tax=0,
            execution_mode="next_open"
        )
        engine.run(df)

        sells = [t for t in engine.trade_log if "SELL" in t["Type"]]
        assert len(sells) == 1
        # Sell signal on bar 5 → executes at bar 6's open = 10.5
        assert sells[0]["Price"] == 10.5

    def test_close_mode_executes_at_close(self):
        closes = [10.0, 10.0, 10.0, 11.0, 12.0]
        opens  = [10.0, 10.0, 10.0, 10.5, 11.5]
        signals = [0, 0, 1, 0, 0]

        df = _make_df(closes, opens=opens, signals=signals)
        engine = BacktestEngine(
            initial_capital=100000, commission=0, slippage=0, stamp_tax=0,
            execution_mode="close"  # legacy mode
        )
        engine.run(df)

        buys = [t for t in engine.trade_log if t["Type"] == "BUY"]
        assert len(buys) == 1
        # Should execute at bar 2's close = 10.0
        assert buys[0]["Price"] == 10.0


class TestStampTax:
    """Test that stamp tax is only applied on sells."""

    def test_stamp_tax_on_sell(self):
        closes = [10.0, 10.0, 10.0, 10.0, 10.0]
        signals = [0, 1, 0, -1, 0]

        df = _make_df(closes, signals=signals)
        engine = BacktestEngine(
            initial_capital=10000, commission=0, slippage=0,
            stamp_tax=0.001,  # 0.1% on sell
            execution_mode="close"
        )
        engine.run(df)

        sells = [t for t in engine.trade_log if "SELL" in t["Type"]]
        assert len(sells) == 1

        # Revenue should be: shares * price * (1 - 0.001)
        shares = sells[0]["Shares"]
        expected_revenue = shares * 10.0 * (1 - 0.001)
        assert abs(sells[0]["Revenue"] - expected_revenue) < 0.01


class TestLimitUpDown:
    """Test that trades are skipped on limit-up/limit-down bars."""

    def test_skip_buy_on_limit_up(self):
        # Bar 2 is limit up from bar 1 (10 -> 11, +10%)
        closes = [10.0, 10.0, 11.0, 11.5, 12.0]
        signals = [0, 0, 1, 0, 0]  # buy signal on bar 2 (limit up)

        df = _make_df(closes, signals=signals)
        engine = BacktestEngine(
            initial_capital=100000, commission=0, slippage=0, stamp_tax=0,
            execution_mode="close"
        )
        engine.run(df)

        buys = [t for t in engine.trade_log if t["Type"] == "BUY"]
        # Buy should be skipped because bar 2 is limit up
        assert len(buys) == 0

    def test_skip_sell_on_limit_down(self):
        # Buy on bar 1, then bar 3 is limit down (11 -> 9.9, -10%)
        closes = [10.0, 10.0, 11.0, 9.9, 10.0]
        signals = [0, 1, 0, -1, 0]

        df = _make_df(closes, signals=signals)
        engine = BacktestEngine(
            initial_capital=100000, commission=0, slippage=0, stamp_tax=0,
            execution_mode="close"
        )
        engine.run(df)

        sells = [t for t in engine.trade_log if "SELL" in t["Type"]]
        # Sell should be skipped on limit-down bar
        assert len(sells) == 0


class TestTimeStop:
    """Test that positions are force-exited after N days with insufficient profit."""

    def test_time_stop_triggers(self):
        # Buy on bar 1, price stays flat for 5 bars (0% profit)
        closes = [10.0] * 10
        signals = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

        df = _make_df(closes, signals=signals)
        engine = BacktestEngine(
            initial_capital=100000, commission=0, slippage=0, stamp_tax=0,
            time_stop_days=5, time_stop_min_profit=0.02,
            execution_mode="close"
        )
        engine.run(df)

        # Should have a time-stop sell
        sells = [t for t in engine.trade_log if "SELL" in t["Type"]]
        assert len(sells) == 1
        assert "Time Stop" in sells[0]["Type"]

    def test_time_stop_does_not_trigger_if_profitable(self):
        # Buy on bar 1, price rises 5% over time
        closes = [10.0, 10.0, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7]
        signals = [0, 1, 0, 0, 0, 0, 0, 0]

        df = _make_df(closes, signals=signals)
        engine = BacktestEngine(
            initial_capital=100000, commission=0, slippage=0, stamp_tax=0,
            time_stop_days=5, time_stop_min_profit=0.02,
            execution_mode="close"
        )
        engine.run(df)

        # Profit is ~7% after 6 bars → time stop should NOT trigger
        sells = [t for t in engine.trade_log if "SELL" in t["Type"]]
        assert len(sells) == 0


class TestLotRounding:
    """Test A-share 100-lot rounding."""

    def test_round_to_100(self):
        closes = [10.0, 10.0, 10.0, 10.0]
        signals = [0, 1, 0, 0]

        df = _make_df(closes, signals=signals)
        engine = BacktestEngine(
            initial_capital=1050,  # Can afford 105 shares but should round to 100
            commission=0, slippage=0, stamp_tax=0,
            execution_mode="close"
        )
        engine.run(df)

        buys = [t for t in engine.trade_log if t["Type"] == "BUY"]
        assert len(buys) == 1
        assert buys[0]["Shares"] == 100  # Rounded down to 100


class TestPerformanceMetrics:
    """Test that performance metrics include new fields."""

    def test_metrics_keys(self):
        closes = [10.0, 10.0, 10.0, 11.0, 10.0]
        signals = [0, 1, 0, -1, 0]

        df = _make_df(closes, signals=signals)
        engine = BacktestEngine(
            initial_capital=100000, commission=0, slippage=0, stamp_tax=0,
            execution_mode="close"
        )
        engine.run(df)
        perf = engine.calculate_performance()

        assert "Profit Factor" in perf
        assert "Avg Win" in perf
        assert "Avg Loss" in perf
        assert "Win Rate" in perf
        assert "Sharpe Ratio" in perf


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
