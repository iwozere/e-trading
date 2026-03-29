"""
Smoke test: factory mock broker + ``wrap_broker_for_cerebro`` + ``cerebro.run()``.

Closes the plan gap: automated end-to-end Cerebro attach using ``get_broker``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.trading.broker.backtrader_availability import BACKTRADER_AVAILABLE
from src.trading.broker.backtrader_broker_bridge import wrap_broker_for_cerebro
from src.trading.broker.broker_factory import get_broker

pytestmark = pytest.mark.skipif(not BACKTRADER_AVAILABLE, reason="backtrader not installed")


def test_get_broker_mock_wrap_and_cerebro_run():
    import backtrader as bt

    df = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000, 1000],
        },
        index=pd.date_range("2020-01-01", periods=2, freq="D"),
    )

    core = get_broker(
        {
            "type": "mock",
            "trading_mode": "paper",
            "cash": 50_000.0,
        }
    )
    bridge = wrap_broker_for_cerebro(core)
    assert isinstance(bridge, bt.broker.BrokerBase)

    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.setbroker(bridge)

    class EmptyStrategy(bt.Strategy):
        def next(self):
            pass

    cerebro.addstrategy(EmptyStrategy)
    results = cerebro.run()
    assert results is not None
