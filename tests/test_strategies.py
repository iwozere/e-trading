from datetime import datetime, timedelta

import backtrader as bt
import numpy as np
import pandas as pd
import pytest
from src.data.data_loader import DataLoader
from src.strategy.base_strategy import BaseStrategy
from src.strategy.ichimoku_rsi_volume_strategy import IchimokuRsiVolumeStrategy
from src.strategy.rsi_bb_strategy import MeanReversionRsiBbStrategy
from src.strategy.rsi_bb_volume_strategy import RsiBollVolumeStrategy
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

"""
Unit tests for all trading strategies in src/strategy.

These tests ensure that each strategy can be instantiated, run on dummy data, and that it logs trades (or at least has a 'trades' attribute).
The tests use pytest and Backtrader with randomly generated dummy data.
"""


def make_dummy_data(length=100):
    """
    Generate a dummy pandas DataFrame with OHLCV columns for testing strategies.
    """
    idx = pd.date_range("2023-01-01", periods=length, freq="H")
    df = pd.DataFrame(
        {
            "open": np.random.rand(length) * 10 + 100,
            "high": np.random.rand(length) * 12 + 102,
            "low": np.random.rand(length) * 8 + 98,
            "close": np.random.rand(length) * 10 + 100,
            "volume": np.random.rand(length) * 1000 + 100,
        },
        index=idx,
    )
    df["high"] = np.maximum(df["high"], df["open"])
    df["high"] = np.maximum(df["high"], df["close"])
    df["low"] = np.minimum(df["low"], df["open"])
    df["low"] = np.minimum(df["low"], df["close"])
    return df


@pytest.mark.parametrize(
    "strategy_cls",
    [
        IchimokuRsiVolumeStrategy,
        RsiBollVolumeStrategy,
        MeanReversionRsiBbStrategy,
    ],
)
def test_strategy_runs_and_logs_trades(strategy_cls):
    """
    Test that a strategy can run on dummy data and has a 'trades' attribute (list).
    The test passes if no error is raised and the attribute exists.
    """
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_cls)
    df = make_dummy_data(120)
    data_feed = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.001)
    results = cerebro.run()
    strategy = results[0]
    assert hasattr(strategy, "trades")
    assert isinstance(strategy.trades, list)
    # Should not raise, and trades can be empty if no signals, but type must be list


# Test data setup
@pytest.fixture
def sample_data():
    dates = pd.date_range(start="2024-01-01", end="2024-01-10", freq="1H")
    data = pd.DataFrame(
        {
            "open": np.random.randn(len(dates)).cumsum() + 100,
            "high": np.random.randn(len(dates)).cumsum() + 101,
            "low": np.random.randn(len(dates)).cumsum() + 99,
            "close": np.random.randn(len(dates)).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, len(dates)),
        },
        index=dates,
    )
    return data


@pytest.fixture
def strategy_classes():
    return [
        IchimokuRsiVolumeStrategy,
        RsiBollVolumeStrategy,
        MeanReversionRsiBbStrategy,
    ]
