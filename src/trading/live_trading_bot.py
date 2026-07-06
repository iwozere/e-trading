"""
Live Trading Bot Wrapper
------------------------

Refactored to be a thin wrapper around StrategyManager (Option B).
This maintains backward compatibility with existing CLI tools and the Web UI
while unifying the execution engine.
"""

import asyncio
import json
import signal
import sys
from typing import Any, Dict

from src.notification.logger import setup_logger
from src.trading.constants import TRADING_CONFIG_DIR
from src.trading.strategy_manager import StrategyManager

_logger = setup_logger(__name__)


class LiveTradingBot:
    """
    Thin wrapper that delegates bot lifecycle to StrategyManager.
    """

    def __init__(self, config_file: str):
        """
        Initialize the bot wrapper by loading config and registering with StrategyManager.
        """
        self.config_file = config_file
        self.manager = StrategyManager()

        # Load and hydrate configuration
        hydrated_config = self._load_and_hydrate_config()

        # Register instance with manager
        self.instance_id = self.manager.add_instance(hydrated_config)
        # Event loop captured in _start_and_capture_loop() so stop() can safely
        # schedule the shutdown coroutine from a signal handler via
        # asyncio.run_coroutine_threadsafe().
        self._event_loop: asyncio.AbstractEventLoop | None = None
        _logger.info("Registered bot %s with StrategyManager. Instance ID: %s", config_file, self.instance_id)

    def _load_and_hydrate_config(self) -> Dict[str, Any]:
        """Load, validate and hydrate the configuration manifest."""
        config_path = TRADING_CONFIG_DIR / self.config_file
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            raw_config = json.load(f)

        # Ensure bot_id is present
        if "bot_id" not in raw_config:
            raw_config["bot_id"] = self.config_file.split(".")[0]

        # Use Factory to resolve manifest integrations
        from src.config.configuration_factory import config_factory

        return config_factory.load_manifest(raw_config)

    def start(self):
        """Start the bot instance via the manager."""
        _logger.info("Starting bot instance %s...", self.instance_id)
        try:
            # get_running_loop() raises RuntimeError when there is no running event
            # loop, which lets us distinguish a CLI entry-point (create a fresh loop)
            # from an async caller such as the Web UI (schedule on the existing loop).
            # This replaces the deprecated asyncio.get_event_loop() pattern which
            # raises DeprecationWarning in Python 3.10 and RuntimeError in 3.12+.
            asyncio.get_running_loop()
            # Already inside an async context — schedule without blocking.
            asyncio.ensure_future(self.manager.start_instance(self.instance_id))
        except RuntimeError:
            # No running event loop — CLI entry point.
            # asyncio.run() creates a fresh loop and keeps it alive while
            # background tasks (Backtrader / bot loops) are pending.
            asyncio.run(self._start_and_capture_loop())
        except Exception:
            _logger.exception("Failed to start bot via manager:")
            raise

    async def _start_and_capture_loop(self) -> None:
        """Coroutine for asyncio.run() CLI startup; stores the event loop reference."""
        self._event_loop = asyncio.get_running_loop()
        await self.manager.start_instance(self.instance_id)

    def stop(self):
        """Stop the bot instance via the manager."""
        _logger.info("Stopping bot instance %s...", self.instance_id)
        try:
            asyncio.get_running_loop()
            # Inside an async context — schedule without blocking.
            asyncio.ensure_future(self.manager.stop_instance(self.instance_id))
        except RuntimeError:
            # Not in an async context (e.g. called from a SIGINT/SIGTERM signal
            # handler that interrupted asyncio.run() in the main thread).
            # run_coroutine_threadsafe() is the correct way to schedule a
            # coroutine from a non-async context onto a running event loop.
            if self._event_loop is not None and self._event_loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self.manager.stop_instance(self.instance_id),
                    self._event_loop,
                )
            else:
                asyncio.run(self.manager.stop_instance(self.instance_id))
        except Exception:
            _logger.exception("Failed to stop bot via manager:")

    def get_status(self) -> Dict[str, Any]:
        """Get status from the manager."""
        return self.manager.get_instance_status(self.instance_id) or {}

    def restart(self):
        """Restart via the manager."""
        _logger.info("Restarting bot instance %s...", self.instance_id)
        try:
            asyncio.get_running_loop()
            asyncio.ensure_future(self.manager.restart_instance(self.instance_id))
        except RuntimeError:
            if self._event_loop is not None and self._event_loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self.manager.restart_instance(self.instance_id),
                    self._event_loop,
                )
            else:
                asyncio.run(self.manager.restart_instance(self.instance_id))


def main():
    """CLI entry point."""
    if len(sys.argv) != 2:
        print("Usage: python live_trading_bot.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    bot = LiveTradingBot(config_file)

    def signal_handler(signum, frame):
        _logger.info("Received signal %s, shutting down...", signum)
        bot.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    bot.start()


if __name__ == "__main__":
    main()
