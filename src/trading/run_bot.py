import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import signal
from typing import Optional

from src.notification.logger import _logger
from src.trading.live_trading_bot import LiveTradingBot
from src.trading.config_validator import validate_config_file, print_validation_results


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
            _logger.error("Usage: python run_bot.py <config.json>")
            _logger.error("Example: python run_bot.py 0001.json")
            sys.exit(1)
        config_name = sys.argv[1]
    
    # Validate configuration file
    config_path = f"config/trading/{config_name}"
    _logger.info(f"Validating configuration: {config_path}")
    
    is_valid, errors, warnings = validate_config_file(config_path)
    print_validation_results(is_valid, errors, warnings)
    
    if not is_valid:
        _logger.error("Configuration validation failed. Please fix the errors above.")
        sys.exit(1)
    
    if warnings:
        _logger.warning("Configuration has warnings. Please review them above.")
    
    # Create and start the live trading bot
    try:
        _logger.info(f"Creating live trading bot with config: {config_name}")
        bot = LiveTradingBot(config_name)
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            _logger.info(f"Received signal {signum}, shutting down...")
            bot.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start the bot
        _logger.info("Starting live trading bot...")
        bot.start()
        
    except KeyboardInterrupt:
        _logger.info("Received keyboard interrupt, shutting down...")
        if 'bot' in locals():
            bot.stop()
    except Exception as e:
        _logger.error(f"Error running live trading bot: {e}", exc_info=e)
        if 'bot' in locals():
            bot.stop()
        sys.exit(1)
    
    _logger.info("Live trading bot stopped.")


if __name__ == "__main__":
    main("0001.json")
