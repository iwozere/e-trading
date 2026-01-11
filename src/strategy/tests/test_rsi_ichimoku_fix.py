import pytest
import backtrader as bt
from unittest.mock import MagicMock
from src.strategy.entry.rsi_ichimoku_entry_mixin import RSIIchimokuEntryMixin
from src.strategy.indicator_factory import IndicatorFactory

class MockStrategy(bt.Strategy):
    """Minimal strategy for testing mixins."""
    def __init__(self):
        self.indicators = {}

def test_rsi_ichimoku_new_architecture():
    """
    Test that RSIIchimokuEntryMixin works correctly with the new architecture
    when indicators are manually registered in the strategy.
    """
    # 1. Setup mock data and strategy
    data = MagicMock(spec=bt.feeds.DataBase)
    # Mock close price with some values to avoid IndexError
    data.close = MagicMock()
    data.close.__getitem__.side_effect = lambda i: 100.0 if i == 0 else 99.0
    data.__len__.return_value = 100

    strategy = MockStrategy()
    strategy.data = data

    # 2. Setup Ichimoku indicators manually (as the Factory would)
    # In a real scenario, IndicatorFactory would create these
    strategy.indicators = {
        'entry_rsi': [50.0] * 100,
        'entry_ichimoku_tenkan': [100.0] * 100,
        'entry_ichimoku_senkou_a': [101.0] * 100,
        'entry_ichimoku_senkou_b': [102.0] * 100
    }

    # 3. Initialize Mixin
    params = {"rsi_oversold": 30}
    mixin = RSIIchimokuEntryMixin(params=params)
    mixin.strategy = strategy

    # 4. Verify
    assert mixin.are_indicators_ready() is True

    # should_enter should not raise KeyError
    try:
        signal = mixin.should_enter()
        assert isinstance(signal, bool)
    except KeyError as e:
        pytest.fail(f"should_enter raised KeyError: {e}")

def test_rsi_ichimoku_missing_indicators():
    """Test that indicators_ready returns False if indicators are missing."""
    strategy = MockStrategy()
    strategy.indicators = {} # Missing indicators

    mixin = RSIIchimokuEntryMixin()
    mixin.strategy = strategy

    assert mixin.are_indicators_ready() is False
    assert mixin.should_enter() is False
