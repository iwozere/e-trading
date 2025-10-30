"""This module runs the live trading bot."""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import signal
import time
from typing import Optional

from src.trading.live_trading_bot import LiveTradingBot
from src.trading.config_validator import validate_config_file, print_validation_results
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def main(config_name: Optional[str] = None):
    """
    Main function to run the live trading bot.

    Args:
        config_name: Configuration file name (e.g., '0001.json')
    """
    _logger.info("Starting live trading bot runner.")

    # Get config file name
    if not config_name:
        if len(sys.argv) != 2:
            _logger.error("Usage: python trading_bot.py <config.json>")
            _logger.error("Example: python trading_bot.py 0001.json")
            sys.exit(1)
        config_name = sys.argv[1]

    # Validate configuration file
    config_path = f"config/trading/{config_name}"
    _logger.info("Validating configuration: %s", config_path)

    is_valid, errors, warnings = validate_config_file(config_path)
    print_validation_results(is_valid, errors, warnings)

    if not is_valid:
        _logger.error("Configuration validation failed. Please fix the errors above.")
        sys.exit(1)

    if warnings:
        _logger.warning("Configuration has warnings. Please review them above.")

    # Create and start the live trading bot
    try:
        _logger.info("Creating live trading bot with config: %s", config_name)
        bot = LiveTradingBot(config_name)

        # Setup signal handlers for graceful shutdown
        def signal_handler(signum):
            _logger.info("Received signal %s, shutting down...", signum)
            bot.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Initialize heartbeat manager
        _logger.info("Initializing heartbeat manager...")
        heartbeat_manager = None
        try:
            from src.common.heartbeat_manager import HeartbeatManager

            def trading_bot_health_check():
                """Health check function for trading bot."""
                try:
                    # Check if bot is running and healthy
                    if bot and hasattr(bot, 'is_running') and bot.is_running:
                        # You can add more specific health checks here
                        # For example, check last trade time, connection status, etc.
                        return {
                            'status': 'HEALTHY',
                            'metadata': {
                                'config_name': config_name,
                                'bot_running': True,
                                'last_check': time.time()
                            }
                        }
                    elif bot:
                        return {
                            'status': 'DOWN',
                            'error_message': 'Trading bot not running',
                            'metadata': {
                                'config_name': config_name,
                                'bot_running': False
                            }
                        }
                    else:
                        return {
                            'status': 'DOWN',
                            'error_message': 'Trading bot not initialized',
                            'metadata': {
                                'config_name': config_name,
                                'bot_running': False
                            }
                        }
                except Exception as e:
                    return {
                        'status': 'DOWN',
                        'error_message': f'Health check failed: {str(e)}',
                        'metadata': {
                            'config_name': config_name
                        }
                    }

            # Create and start heartbeat manager
            heartbeat_manager = HeartbeatManager(
                system='trading_bot',
                interval_seconds=30
            )
            heartbeat_manager.set_health_check_function(trading_bot_health_check)
            heartbeat_manager.start_heartbeat()

            _logger.info("Heartbeat manager started for trading bot")

        except Exception as e:
            _logger.exception("Failed to initialize heartbeat manager:")

        # Update signal handler to stop heartbeat
        def signal_handler_with_heartbeat(signum):
            _logger.info("Received signal %s, shutting down...", signum)
            if heartbeat_manager:
                heartbeat_manager.stop_heartbeat()
                _logger.info("Stopped trading bot heartbeat")
            bot.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler_with_heartbeat)
        signal.signal(signal.SIGTERM, signal_handler_with_heartbeat)

        # Start the bot
        _logger.info("Starting live trading bot...")
        bot.start()

    except KeyboardInterrupt:
        _logger.info("Received keyboard interrupt, shutting down...")
        if heartbeat_manager:
            heartbeat_manager.stop_heartbeat()
        if 'bot' in locals():
            bot.stop()
    except Exception as e:
        _logger.exception("Error running live trading bot:")
        if heartbeat_manager:
            heartbeat_manager.stop_heartbeat()
        if 'bot' in locals():
            bot.stop()
        sys.exit(1)

    _logger.info("Live trading bot stopped.")


if __name__ == "__main__":
    main("0001.json")
