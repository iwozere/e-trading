import pytest
import pandas as pd
import numpy as np
from src.vectorbt.indicators.signals import StrategyInd
from src.vectorbt.data.loader import DataLoader
from src.shared.indicators.adapters import RSI, BBANDS

def test_signal_consistency():
    # 1. Create dummy data
    dates = pd.date_range("2024-01-01", periods=100, freq="1H")
    close = pd.Series(np.random.randn(100).cumsum() + 100, index=dates, name="Close")

    # 2. Generate signals via StrategyInd
    # StrategyInd.run returns a wrapper with outputs as attributes
    res = StrategyInd.run(
        close,
        rsi_window=14,
        rsi_lower=30,
        rsi_upper=70,
        bb_window=20,
        bb_std=2.0
    )

    # 3. Manually calculate with adapters to verify consistency
    rsi_vals = RSI.compute(close, window=14)
    bb_vals = BBANDS.compute(close, window=20, nbdevup=2.0, nbdevdn=2.0)

    expected_entries = (rsi_vals < 30) & (close < bb_vals['lowerband'])
    expected_short_entries = (rsi_vals > 70) & (close > bb_vals['upperband'])

    # 4. Compare
    pd.testing.assert_series_equal(res.entries, expected_entries, check_names=False)
    pd.testing.assert_series_equal(res.short_entries, expected_short_entries, check_names=False)

    print("✅ Signal consistency test passed")

if __name__ == "__main__":
    test_signal_consistency()
