"""
Live Trading Bot Wrapper
------------------------

Refactored to be a thin wrapper around StrategyManager (Option B).
This maintains backward compatibility with existing CLI tools and the Web UI
while unifying the execution engine.
"""

import asyncio
import json
import sys
import signal
from typing import Any, Dict, Optional

from src.trading.strategy_manager import StrategyManager
from src.trading.constants import TRADING_CONFIG_DIR
from src.notification.logger import setup_logger

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
        _logger.info("Registered bot %s with StrategyManager. Instance ID: %s", 
                     config_file, self.instance_id)

    def _load_and_hydrate_config(self) -> Dict[str, Any]:
        """Load, validate and hydrate the configuration manifest."""
        config_path = TRADING_CONFIG_DIR / self.config_file
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            raw_config = json.load(f)

        # Ensure bot_id is present
        if 'bot_id' not in raw_config:
            raw_config['bot_id'] = self.config_file.split('.')[0]

        # Use Factory to resolve manifest integrations
        from src.config.configuration_factory import config_factory
        return config_factory.load_manifest(raw_config)

    def start(self):
        """Start the bot instance via the manager."""
        _logger.info("Starting bot instance %s...", self.instance_id)
        try:
            # We use a new event loop for CLI compatibility if one isn't running
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in a loop (e.g. called from another async func)
                asyncio.ensure_future(self.manager.start_instance(self.instance_id))
            else:
                loop.run_until_complete(self.manager.start_instance(self.instance_id))
                # Loop must stay running for the async tasks to proceed if this is the main entry
                if not loop.is_running() and str(sys.argv[0]).endswith("live_trading_bot.py"):
                     loop.run_forever()
        except Exception:
            _logger.exception("Failed to start bot via manager:")
            raise

    def stop(self):
        """Stop the bot instance via the manager."""
        _logger.info("Stopping bot instance %s...", self.instance_id)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.manager.stop_instance(self.instance_id))
            else:
                loop.run_until_complete(self.manager.stop_instance(self.instance_id))
        except Exception:
            _logger.exception("Failed to stop bot via manager:")

    def get_status(self) -> Dict[str, Any]:
        """Get status from the manager."""
        return self.manager.get_instance_status(self.instance_id)

    def restart(self):
        """Restart via the manager."""
        _logger.info("Restarting bot instance %s...", self.instance_id)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(self.manager.restart_instance(self.instance_id))
        else:
            loop.run_until_complete(self.manager.restart_instance(self.instance_id))


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
