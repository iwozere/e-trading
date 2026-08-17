"""
Tests for LiveTradingBot's start/stop lifecycle.

Regression coverage for a bug where LiveTradingBot.start() (the CLI/systemd
entry point) returned - and tore its asyncio.run() event loop down - the
instant StrategyManager.start_instance() scheduled the bot's background work
(Backtrader engine, real-time data feed, trading loop) as fire-and-forget
tasks, without ever awaiting them. In production this meant the bot process
exited within milliseconds of "started successfully" and was perpetually
relaunched by systemd (Restart=always), never actually running.
"""

import threading
import time
from typing import cast

from src.trading.live_trading_bot import LiveTradingBot
from src.trading.strategy_manager import StrategyManager


class _FakeManager:
    """Stand-in for StrategyManager exposing only the async methods LiveTradingBot uses."""

    def __init__(self, start_result: bool = True):
        self.start_result = start_result
        self.start_calls = 0
        self.stop_calls = 0

    async def start_instance(self, _instance_id: str) -> bool:
        self.start_calls += 1
        return self.start_result

    async def stop_instance(self, _instance_id: str) -> bool:
        self.stop_calls += 1
        return True


def _make_bot(manager: _FakeManager) -> LiveTradingBot:
    """Build a LiveTradingBot without going through __init__'s config loading."""
    bot = LiveTradingBot.__new__(LiveTradingBot)
    bot.config_file = "test.json"
    bot.manager = cast(StrategyManager, manager)
    bot.instance_id = "test-instance"
    bot._event_loop = None
    bot._stop_event = None
    return bot


def test_start_blocks_until_stop_is_called():
    """start() must not return the instant start_instance() schedules background
    work - it should block until stop() signals shutdown, otherwise asyncio.run()
    tears the loop (and those background tasks) down before they ever run."""
    manager = _FakeManager(start_result=True)
    bot = _make_bot(manager)

    def stop_after_delay():
        time.sleep(0.2)
        bot.stop()

    stopper = threading.Thread(target=stop_after_delay)
    start_time = time.monotonic()
    stopper.start()
    bot.start()  # blocks (via asyncio.run) until stop() sets _stop_event
    elapsed = time.monotonic() - start_time
    stopper.join()

    assert elapsed >= 0.2, "start() returned before stop() was called"
    assert manager.start_calls == 1
    assert manager.stop_calls == 1


def test_start_returns_immediately_when_start_instance_fails():
    """If start_instance() fails, start() must return promptly rather than
    blocking forever, preserving the existing fast-fail/restart cadence."""
    manager = _FakeManager(start_result=False)
    bot = _make_bot(manager)

    start_time = time.monotonic()
    bot.start()
    elapsed = time.monotonic() - start_time

    assert elapsed < 1.0, "start() blocked despite start_instance() failing"
    assert manager.start_calls == 1
    assert manager.stop_calls == 0
