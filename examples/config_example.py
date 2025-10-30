#!/usr/bin/env python3
"""
Configuration Management Example
================================

This example demonstrates how to use the new simplified Pydantic-based
configuration system for the crypto trading platform.

Features demonstrated:
- Creating configurations using Pydantic models
- Configuration validation
- Loading and saving configurations
- Converting old configurations to new format
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.config_models import TradingBotConfig, OptimizerConfig, DataConfig, BrokerType, DataSourceType, StrategyType, Environment
from src.config.config_loader import load_config, save_config, validate_config_file, create_sample_config, convert_old_config

from src.notification.logger import setup_logger
logger = setup_logger(__name__)

def example_basic_usage():
    """Demonstrate basic configuration management usage"""
    print("\n=== Basic Configuration Management ===")

    # Create a trading bot configuration using Pydantic models
    trading_config = TradingBotConfig(
        bot_id="example_bot_001",
        symbol="BTCUSDT",
        broker_type=BrokerType.BINANCE_PAPER,
        data_source=DataSourceType.BINANCE,
        strategy_type=StrategyType.CUSTOM,
        description="Example trading bot for demonstration",
        initial_balance=10000.0,
        risk_per_trade=0.02,
        max_open_trades=5
    )

    print(f"Created trading config: {trading_config.bot_id}")
    print(f"Symbol: {trading_config.symbol}")
    print(f"Initial balance: ${trading_config.initial_balance}")
    print(f"Risk per trade: {trading_config.risk_per_trade * 100}%")

    # Save the configuration
    save_config(trading_config, "config/trading/example_bot_001.json")
    print(f"Saved to: config/trading/example_bot_001.json")

    # Load the configuration back
    loaded_config = load_config("config/trading/example_bot_001.json")
    print(f"Loaded config: {loaded_config.bot_id}")

    return trading_config


def example_environment_management():
    """Demonstrate environment-specific configuration management"""
    print("\n=== Environment Management ===")

    # Create configurations for different environments
    environments = [
        (Environment.DEVELOPMENT, "BTCUSDT", 5000.0, 0.03),
        (Environment.STAGING, "ETHUSDT", 10000.0, 0.02),
        (Environment.PRODUCTION, "BTCUSDT", 50000.0, 0.01)
    ]

    for env, symbol, balance, risk in environments:
        config = TradingBotConfig(
            bot_id=f"env_bot_{env.value}",
            environment=env,
            symbol=symbol,
            broker_type=BrokerType.BINANCE_PAPER,
            data_source=DataSourceType.BINANCE,
            strategy_type=StrategyType.CUSTOM,
            description=f"Environment-specific bot for {env.value}",
            initial_balance=balance,
            risk_per_trade=risk,
            max_open_trades=3 if env == Environment.PRODUCTION else 5
        )

        # Adjust settings based on environment
        if env == Environment.DEVELOPMENT:
            config.log_level = "DEBUG"
            config.notifications_enabled = False
        elif env == Environment.STAGING:
            config.log_level = "INFO"
            config.telegram_enabled = True
        elif env == Environment.PRODUCTION:
            config.log_level = "WARNING"
            config.telegram_enabled = True
            config.email_enabled = True
            config.stop_loss_pct = 3.0  # Tighter for production

        # Save environment-specific config
        save_config(config, f"config/trading/env_bot_{env.value}.json")
        print(f"Created {env.value} config:")
        print(f"  Symbol: {config.symbol}")
        print(f"  Balance: ${config.initial_balance}")
        print(f"  Risk: {config.risk_per_trade * 100}%")
        print(f"  Log level: {config.log_level}")
        print(f"  Notifications: {config.notifications_enabled}")


def example_optimizer_configuration():
    """Demonstrate optimizer configuration management"""
    print("\n=== Optimizer Configuration ===")

    # Create basic optimizer config
    basic_optimizer = OptimizerConfig(
        optimizer_type="optuna",
        initial_capital=10000.0,
        n_trials=100,
        n_jobs=1,
        description="Basic optimization example"
    )

    print(f"Basic optimizer: {basic_optimizer.n_trials} trials")
    print(f"Initial capital: ${basic_optimizer.initial_capital}")

    # Create advanced optimizer config
    advanced_optimizer = OptimizerConfig(
        optimizer_type="optuna",
        initial_capital=50000.0,
        n_trials=500,
        n_jobs=-1,  # Use all cores
        position_size=0.2,
        plot=True,
        save_trades=True,
        output_dir="results/advanced",
        description="Advanced optimization example"
    )

    print(f"Advanced optimizer: {advanced_optimizer.n_trials} trials")
    print(f"Parallel jobs: {advanced_optimizer.n_jobs}")
    print(f"Position size: {advanced_optimizer.position_size * 100}%")

    # Save both configurations
    save_config(basic_optimizer, "config/optimizer/basic_optimizer.json")
    save_config(advanced_optimizer, "config/optimizer/advanced_optimizer.json")


def example_data_configuration():
    """Demonstrate data feed configuration management"""
    print("\n=== Data Configuration ===")

    # Create different data feed configurations
    data_configs = [
        DataConfig(
            data_source=DataSourceType.BINANCE,
            symbol="BTCUSDT",
            interval="1h",
            testnet=True,
            description="Binance testnet data feed"
        ),
        DataConfig(
            data_source=DataSourceType.YAHOO,
            symbol="AAPL",
            interval="5m",
            description="Yahoo Finance data feed"
        ),
        DataConfig(
            data_source=DataSourceType.IBKR,
            symbol="SPY",
            interval="1m",
            host="127.0.0.1",
            port=7497,
            client_id=1,
            description="IBKR data feed"
        )
    ]

    for i, config in enumerate(data_configs):
        print(f"{config.data_source.value.upper()} data feed:")
        print(f"  Symbol: {config.symbol}")
        print(f"  Interval: {config.interval}")
        print(f"  Lookback: {config.lookback_bars} bars")

        # Save configuration
        save_config(config, f"config/data/{config.data_source.value}_{config.symbol.lower()}.json")


def example_configuration_validation():
    """Demonstrate configuration validation"""
    print("\n=== Configuration Validation ===")

    # Create a valid configuration
    valid_config = TradingBotConfig(
        bot_id="valid_bot",
        symbol="BTCUSDT",
        broker_type=BrokerType.BINANCE_PAPER,
        data_source=DataSourceType.BINANCE,
        strategy_type=StrategyType.CUSTOM
    )

    print("✅ Valid configuration created successfully")

    # Try to create an invalid configuration
    try:
        invalid_config = TradingBotConfig(
            bot_id="",  # Empty bot_id should fail
            symbol="BTCUSDT",
            broker_type=BrokerType.BINANCE_PAPER,
            data_source=DataSourceType.BINANCE,
            strategy_type=StrategyType.CUSTOM
        )
    except Exception as e:
        print(f"❌ Invalid configuration caught: {e}")

    # Validate a configuration file
    save_config(valid_config, "config/trading/valid_bot.json")
    is_valid, errors, warnings = validate_config_file("config/trading/valid_bot.json")

    print(f"File validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
    if errors:
        print(f"Errors: {errors}")
    if warnings:
        print(f"Warnings: {warnings}")


def example_sample_configs():
    """Demonstrate creating sample configurations"""
    print("\n=== Sample Configurations ===")

    # Create sample configurations
    create_sample_config("config/trading/sample_trading.json", "trading")
    create_sample_config("config/optimizer/sample_optimizer.json", "optimizer")
    create_sample_config("config/data/sample_data.json", "data")

    print("✅ Sample configurations created")


def example_config_conversion():
    """Demonstrate converting old configurations to new format"""
    print("\n=== Configuration Conversion ===")

    # Create an old-style configuration file
    old_config = {
        "bot_id": "old_bot_001",
        "broker": {
            "type": "binance_paper",
            "initial_balance": 10000.0,
            "commission": 0.001
        },
        "trading": {
            "symbol": "BTCUSDT",
            "position_size": 0.1,
            "max_positions": 5
        },
        "data": {
            "data_source": "binance",
            "symbol": "BTCUSDT",
            "interval": "1h"
        },
        "strategy": {
            "type": "custom",
            "entry_logic": {"name": "RSIBBVolumeEntryMixin", "params": {}},
            "exit_logic": {"name": "RSIBBExitMixin", "params": {}}
        }
    }

    import json
    with open("config/trading/old_config.json", "w") as f:
        json.dump(old_config, f, indent=2)

    # Convert to new format
    convert_old_config("config/trading/old_config.json", "config/trading/converted_config.json")

    # Load and validate the converted config
    converted_config = load_config("config/trading/converted_config.json")
    print(f"✅ Converted config: {converted_config.bot_id}")
    print(f"Symbol: {converted_config.symbol}")
    print(f"Broker type: {converted_config.broker_type}")


def main():
    """Main function to run all examples"""
    print("🚀 Configuration Management Examples")
    print("=" * 50)

    try:
        # Create config directories if they don't exist
        Path("config/trading").mkdir(parents=True, exist_ok=True)
        Path("config/optimizer").mkdir(parents=True, exist_ok=True)
        Path("config/data").mkdir(parents=True, exist_ok=True)

        # Run examples
        example_basic_usage()
        example_environment_management()
        example_optimizer_configuration()
        example_data_configuration()
        example_configuration_validation()
        example_sample_configs()
        example_config_conversion()

        print("\n✅ All examples completed successfully!")
        print("\n📁 Check the 'config/' directory for generated configuration files.")

    except Exception as e:
        logger.exception("❌ Error running examples: %s", str(e))


if __name__ == "__main__":
    main()