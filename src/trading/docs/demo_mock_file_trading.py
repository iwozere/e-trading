#!/usr/bin/env python3
"""
Demo: Mock File Trading Bot
--------------------------

Demonstrates how to create and run a test trading bot using:
- Mock broker for simulated trading
- File data feed for CSV data
- Realistic paper trading simulation

This demo shows how to configure a bot that behaves like a real trading bot
but uses historical CSV data and simulated order execution.
"""

import json
import asyncio
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.trading.broker.broker_factory import get_broker
from src.data.feed.file_data_feed import FileDataFeed
from src.trading.services.bot_config_validator import (
    BotConfigValidator,
    validate_database_bot_record,
    print_validation_results
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class MockFileTradingDemo:
    """Demonstrates mock file trading bot setup and execution."""

    def __init__(self):
        self.config_dir = PROJECT_ROOT / "config" / "test_bot_configurations"
        self.data_dir = PROJECT_ROOT / "data"

    def load_test_configuration(self, config_name: str) -> dict:
        """Load a test configuration from JSON file."""
        config_path = self.config_dir / f"{config_name}.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        _logger.info(f"Loaded configuration: {config_name}")
        return config

    def validate_configuration(self, config: dict) -> bool:
        """Validate the bot configuration."""
        print(f"\n{'='*60}")
        print("CONFIGURATION VALIDATION")
        print(f"{'='*60}")

        # Create a mock database record for validation
        bot_record = {
            "id": 1,
            "user_id": 1,
            "type": "paper",
            "status": "stopped",
            "config": config,
            "description": config.get("testing", {}).get("description", "Test bot"),
            "started_at": None,
            "last_heartbeat": None,
            "error_count": 0,
            "current_balance": None,
            "total_pnl": None,
            "extra_metadata": None,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": None
        }

        is_valid, errors, warnings = validate_database_bot_record(bot_record)
        print_validation_results(config["id"], is_valid, errors, warnings)

        return is_valid

    def check_data_file(self, config: dict) -> bool:
        """Check if the data file exists and is accessible."""
        data_config = config.get("data", {})
        file_path = data_config.get("file_path")

        if not file_path:
            _logger.error("No file_path specified in data configuration")
            return False

        full_path = PROJECT_ROOT / file_path

        if not full_path.exists():
            _logger.error(f"Data file not found: {full_path}")
            return False

        # Check file size and basic format
        try:
            import pandas as pd
            df = pd.read_csv(full_path, nrows=5)  # Read first 5 rows
            _logger.info(f"Data file check passed: {full_path}")
            _logger.info(f"Columns: {list(df.columns)}")
            _logger.info(f"Sample data shape: {df.shape}")
            return True
        except Exception as e:
            _logger.error(f"Error reading data file: {e}")
            return False

    def create_broker(self, config: dict):
        """Create a broker instance from configuration."""
        broker_config = config["broker"]

        print(f"\n{'='*60}")
        print("BROKER CREATION")
        print(f"{'='*60}")

        try:
            broker = get_broker(broker_config)
            _logger.info(f"Created broker: {broker.__class__.__name__}")
            _logger.info(f"Trading mode: {broker_config['trading_mode']}")
            _logger.info(f"Initial cash: ${broker_config['cash']:,.2f}")
            return broker
        except Exception as e:
            _logger.error(f"Failed to create broker: {e}")
            raise

    def create_data_feed(self, config: dict):
        """Create a data feed instance from configuration."""
        data_config = config["data"]

        print(f"\n{'='*60}")
        print("DATA FEED CREATION")
        print(f"{'='*60}")

        try:
            file_path = PROJECT_ROOT / data_config["file_path"]

            data_feed = FileDataFeed(
                dataname=str(file_path),
                symbol=config["symbol"],
                datetime_col=data_config.get("datetime_col", "datetime"),
                open_col=data_config.get("open_col", "open"),
                high_col=data_config.get("high_col", "high"),
                low_col=data_config.get("low_col", "low"),
                close_col=data_config.get("close_col", "close"),
                volume_col=data_config.get("volume_col", "volume"),
                separator=data_config.get("separator", ","),
                simulate_realtime=data_config.get("simulate_realtime", False),
                realtime_interval=data_config.get("realtime_interval", 60),
                fromdate=data_config.get("fromdate"),
                todate=data_config.get("todate"),
                on_new_bar=self._on_new_bar_callback
            )

            _logger.info(f"Created data feed for {config['symbol']}")
            _logger.info(f"Data source: {data_config['file_path']}")
            _logger.info(f"Real-time simulation: {data_config.get('simulate_realtime', False)}")

            return data_feed
        except Exception as e:
            _logger.error(f"Failed to create data feed: {e}")
            raise

    def _on_new_bar_callback(self, symbol: str, timestamp, bar_data: dict):
        """Callback function for new bar events."""
        _logger.debug(f"New bar for {symbol} at {timestamp}: "
                     f"O={bar_data['open']:.4f}, H={bar_data['high']:.4f}, "
                     f"L={bar_data['low']:.4f}, C={bar_data['close']:.4f}, "
                     f"V={bar_data['volume']:.0f}")

    async def run_trading_simulation(self, config: dict, duration_seconds: int = 30):
        """Run a trading simulation for a specified duration."""
        print(f"\n{'='*60}")
        print("TRADING SIMULATION")
        print(f"{'='*60}")

        try:
            # Create broker and data feed
            broker = self.create_broker(config)
            data_feed = self.create_data_feed(config)

            _logger.info(f"Starting trading simulation for {duration_seconds} seconds...")

            # Get initial status
            initial_status = data_feed.get_status()
            _logger.info(f"Data feed status: {initial_status}")

            # Simulate trading for the specified duration
            import time
            start_time = time.time()

            while time.time() - start_time < duration_seconds:
                # In a real implementation, this would be where the strategy
                # analyzes the data and makes trading decisions

                # For demo purposes, just log status periodically
                if int(time.time() - start_time) % 10 == 0:
                    status = data_feed.get_status()
                    _logger.info(f"Simulation running... Data points: {status['current_index']}/{status['total_rows']}")

                await asyncio.sleep(1)

            # Stop data feed
            data_feed.stop()

            # Get final status
            final_status = data_feed.get_status()
            _logger.info(f"Final data feed status: {final_status}")

            _logger.info("Trading simulation completed successfully!")

        except Exception as e:
            _logger.exception(f"Error in trading simulation: {e}")
            raise

    def demonstrate_configuration(self, config_name: str):
        """Demonstrate a complete configuration setup."""
        print(f"\n{'='*80}")
        print(f"MOCK FILE TRADING DEMO - {config_name.upper()}")
        print(f"{'='*80}")

        try:
            # Load configuration
            config = self.load_test_configuration(config_name)

            # Validate configuration
            if not self.validate_configuration(config):
                _logger.error("Configuration validation failed!")
                return False

            # Check data file
            if not self.check_data_file(config):
                _logger.error("Data file check failed!")
                return False

            # Create components (without running simulation)
            broker = self.create_broker(config)
            data_feed = self.create_data_feed(config)

            print(f"\n{'='*60}")
            print("SETUP SUMMARY")
            print(f"{'='*60}")

            print(f"✅ Configuration: {config['name']}")
            print(f"✅ Symbol: {config['symbol']}")
            print(f"✅ Broker: {broker.__class__.__name__} ({config['broker']['trading_mode']} mode)")
            print(f"✅ Data Feed: File-based ({config['data']['file_path']})")
            print(f"✅ Strategy: {config['strategy']['type']}")
            print(f"✅ Initial Balance: ${config['broker']['cash']:,.2f}")

            # Clean up
            data_feed.stop()

            return True

        except Exception as e:
            _logger.exception(f"Error demonstrating configuration: {e}")
            return False

    async def run_full_demo(self):
        """Run the complete demo with all configurations."""
        print("🚀 Mock File Trading Bot Demo")
        print("This demo shows how to set up test trading bots using CSV data and mock brokers.")

        configurations = [
            "mock_file_test_btc_5m",
            "mock_file_test_eth_15m",
            "mock_file_test_ltc_simple"
        ]

        successful_demos = 0

        for config_name in configurations:
            try:
                success = self.demonstrate_configuration(config_name)
                if success:
                    successful_demos += 1
                    _logger.info(f"✅ Demo completed successfully: {config_name}")
                else:
                    _logger.error(f"❌ Demo failed: {config_name}")
            except Exception as e:
                _logger.exception(f"❌ Demo error for {config_name}: {e}")

        print(f"\n{'='*80}")
        print("DEMO SUMMARY")
        print(f"{'='*80}")
        print(f"Successful demos: {successful_demos}/{len(configurations)}")

        if successful_demos == len(configurations):
            print("🎉 All demos completed successfully!")
            print("\nYou can now:")
            print("• Use these configurations in your trading bot database")
            print("• Modify the configurations for your specific needs")
            print("• Run actual trading simulations with the enhanced trading service")
        else:
            print("⚠️  Some demos failed. Check the logs for details.")

        return successful_demos == len(configurations)


async def main():
    """Main demo function."""
    demo = MockFileTradingDemo()

    try:
        # Run the full demo
        success = await demo.run_full_demo()

        if success:
            print("\n" + "="*80)
            print("NEXT STEPS")
            print("="*80)
            print("1. Insert these configurations into your trading_bots database table")
            print("2. Use the enhanced trading service to run the bots")
            print("3. Monitor the bots through the notification system")
            print("4. Analyze the results using the performance metrics")

            # Show sample database insert
            print("\nSample database insert:")
            print("INSERT INTO trading_bots (user_id, type, status, config, description)")
            print("VALUES (1, 'paper', 'stopped', '<config_json>', 'Mock file test bot');")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        _logger.exception(f"Demo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())