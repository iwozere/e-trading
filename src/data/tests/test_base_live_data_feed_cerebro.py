"""
Regression tests for BaseLiveDataFeed's Backtrader integration.

Covers a bug where BaseLiveDataFeed never actually delivered bars to
Cerebro: islive() was never overridden (so Cerebro used preload()+runonce(),
the vectorized backtest path), and _load() unconditionally returned None,
which preload() treats as "feed exhausted" on the very first call. Net
effect: cerebro.run() finished instantly having processed zero bars - not
the 1000+ cached historical bars, and never anything live - regardless of
how long the surrounding bot process stayed up.
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Any, List

import backtrader as bt
import pandas as pd
import pytest

from src.data.feed.base_live_data_feed import BaseLiveDataFeed


class _StubDataManager:
    """Stand-in for DataManager returning a fixed, pre-built OHLCV DataFrame."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def get_ohlcv(self, **_kwargs: object) -> pd.DataFrame:
        return self._df


class _CountingStrategy(bt.Strategy):
    """Records one entry per next() call so tests can verify bar delivery."""

    params = (("counter", None),)

    def next(self):
        self.p.counter.append(self.data.close[0])


def _make_ohlcv_df(n: int, start: datetime | None = None) -> pd.DataFrame:
    start = start or datetime(2026, 1, 1)
    index = pd.date_range(start, periods=n, freq="h", name="datetime")
    return pd.DataFrame(
        {
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [1000.0 for _ in range(n)],
        },
        index=index,
    )


def _make_feed(n_bars: int) -> BaseLiveDataFeed:
    historical_df = _make_ohlcv_df(n_bars)
    # BaseLiveDataFeed's _connect_realtime/_disconnect_realtime/_get_latest_data
    # are @abstractmethod but the class isn't ABCMeta-enforced (see the
    # pre-existing, unrelated test_abstract_methods failure) - their no-op
    # bodies are enough here; the background update thread just logs a failed
    # connect attempt and sleeps, which doesn't affect _load()/Cerebro.
    return BaseLiveDataFeed(
        symbol="TEST",
        interval="1h",
        lookback_bars=n_bars,
        data_manager=_StubDataManager(historical_df),
    )


def test_islive_returns_true():
    """Without this, Cerebro silently switches to the backtest (preload/runonce) path."""
    feed = _make_feed(1)
    feed.should_stop = True
    assert feed.islive() is True


def test_load_replays_historical_backlog_into_cerebro():
    """Every cached historical bar must actually reach the strategy's next()."""
    n_bars = 5
    feed = _make_feed(n_bars)
    # Skip the live-polling phase: once the backlog is drained, should_stop=True
    # makes _load() return False (feed exhausted) instead of blocking on the
    # live queue, so cerebro.run() completes without needing a live bar pushed.
    feed.should_stop = True

    cerebro = bt.Cerebro()
    cerebro.adddata(feed)
    calls: List[Any] = []
    cerebro.addstrategy(_CountingStrategy, counter=calls)
    cerebro.run()

    assert len(calls) == n_bars


def test_load_delivers_queued_live_bar():
    """A bar queued via _process_new_data() after the backlog drains must reach next()."""
    n_bars = 2
    feed = _make_feed(n_bars)

    cerebro = bt.Cerebro()
    cerebro.adddata(feed)
    calls: List[Any] = []
    cerebro.addstrategy(_CountingStrategy, counter=calls)

    runner = threading.Thread(target=cerebro.run, daemon=True)
    runner.start()

    deadline = time.monotonic() + 10
    while len(calls) < n_bars and time.monotonic() < deadline:
        time.sleep(0.02)
    assert len(calls) == n_bars, "historical backlog was not fully replayed"

    new_bar = _make_ohlcv_df(1, start=datetime(2026, 1, 1) + timedelta(hours=n_bars))
    feed._process_new_data(new_bar)

    deadline = time.monotonic() + 10
    while len(calls) < n_bars + 1 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert len(calls) == n_bars + 1, "live bar queued via _process_new_data() was not delivered"

    feed.should_stop = True
    runner.join(timeout=15)

    assert not runner.is_alive(), "cerebro.run() did not exit after should_stop was set"
    assert len(calls) == n_bars + 1
    assert calls[-1] == pytest.approx(100.5)
