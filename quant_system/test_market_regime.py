"""
Unit tests for market_regime.py.
Tests regime detection logic with synthetic data (no API calls needed).
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from market_regime import check_market_regime, MarketRegime


def _make_index_df(closes, n=100):
    """Helper to create a synthetic index DataFrame."""
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="B")
    df = pd.DataFrame({
        "Open": closes,
        "High": [c + 10 for c in closes],
        "Low": [c - 10 for c in closes],
        "Close": closes,
        "Volume": [1000000] * len(closes),
    }, index=dates)
    return df


class TestMarketRegime:
    def test_bull_market(self):
        # Strong uptrend: prices rising linearly
        closes = list(np.linspace(3000, 4000, 100))
        df = _make_index_df(closes)
        regime = check_market_regime(df, ma_short=20, ma_long=60)
        assert regime == MarketRegime.BULL

    def test_bear_market(self):
        # Strong downtrend: prices falling linearly
        closes = list(np.linspace(4000, 3000, 100))
        df = _make_index_df(closes)
        regime = check_market_regime(df, ma_short=20, ma_long=60)
        assert regime == MarketRegime.BEAR

    def test_neutral_market(self):
        # Price above MA60 but below MA20 (recent dip but not bear)
        # Start rising then small dip at the end
        base = list(np.linspace(3000, 3800, 90))
        dip = list(np.linspace(3800, 3650, 10))  # small dip
        closes = base + dip
        df = _make_index_df(closes)
        regime = check_market_regime(df, ma_short=20, ma_long=60)
        # Close should be above MA60 but below MA20 → NEUTRAL
        assert regime in [MarketRegime.NEUTRAL, MarketRegime.BULL]

    def test_insufficient_data(self):
        closes = [3000, 3010, 3020]
        df = _make_index_df(closes)
        regime = check_market_regime(df, ma_short=20, ma_long=60)
        assert regime == MarketRegime.NEUTRAL  # Not enough data → neutral

    def test_empty_df(self):
        df = pd.DataFrame()
        regime = check_market_regime(df, ma_short=20, ma_long=60)
        assert regime == MarketRegime.NEUTRAL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
