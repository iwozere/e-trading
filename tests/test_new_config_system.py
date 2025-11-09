#!/usr/bin/env python3
"""
Test New Configuration System
============================

This script tests the new simplified Pydantic-based configuration system
to ensure it works correctly after the refactoring.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.model.config_models import TradingBotConfig, OptimizerConfig, DataConfig, BrokerType, DataSourceType
from src.config.config_loader import load_config, save_config, validate_config_file, create_sample_config


def test_trading_bot_config():
    """Test TradingBotConfig creation and validation"""
    print("Testing TradingBotConfig...")

    try:
        # Create a valid config
        config = TradingBotConfig(
            bot_id="test_bot_001",
            name="Test Bot",
            symbol="BTCUSDT",
            broker_type=BrokerType.BINANCE_PAPER,
            data_source=DataSourceType.BINANCE,
            strategy_name="RSIStrategy",
            description="Test configuration"
        )

        print(f"✅ Created config: {config.bot_id}")
        print(f"   Symbol: {config.symbol}")
        print(f"   Broker: {config.broker_type}")
        print(f"   Data source: {config.data_source}")

        # Test validation
        assert config.bot_id == "test_bot_001"
        assert config.symbol == "BTCUSDT"
        assert config.broker_type == BrokerType.BINANCE_PAPER
        assert config.data_source == DataSourceType.BINANCE

        # Test invalid config (should raise exception)
        try:
            invalid_config = TradingBotConfig(
                bot_id="",  # Empty bot_id should fail
                name="Test Bot",
                symbol="BTCUSDT",
                broker_type=BrokerType.BINANCE_PAPER,
                data_source=DataSourceType.BINANCE,
                strategy_name="RSIStrategy"
            )
            print("❌ Invalid config should have failed")
            return False
        except Exception as e:
            print(f"✅ Invalid config correctly rejected: {e}")

        return True

    except Exception as e:
        print(f"❌ Error testing TradingBotConfig: {e}")
        return False


def test_config_loading_saving():
    """Test loading and saving configurations"""
    print("\nTesting config loading and saving...")

    try:
        # Create a config
        config = TradingBotConfig(
            bot_id="test_save_load",
            name="Test Save Load Bot",
            symbol="ETHUSDT",
            broker_type=BrokerType.BINANCE_PAPER,
            data_source=DataSourceType.BINANCE,
            strategy_name="MACDStrategy",
            description="Test save/load"
        )

        # Save config
        save_config(config, "test_config.json")
        print("✅ Config saved")

        # Load config
        loaded_config = load_config("test_config.json")
        print("✅ Config loaded")

        # Verify loaded config
        assert loaded_config.bot_id == config.bot_id
        assert loaded_config.symbol == config.symbol
        assert loaded_config.broker_type == config.broker_type

        print(f"   Loaded: {loaded_config.bot_id}")
        print(f"   Symbol: {loaded_config.symbol}")

        # Clean up
        os.remove("test_config.json")
        print("✅ Test file cleaned up")

        return True

    except Exception as e:
        print(f"❌ Error testing config loading/saving: {e}")
        return False


def test_config_validation():
    """Test configuration validation"""
    print("\nTesting config validation...")

    try:
        # Create a valid config and save it
        config = TradingBotConfig(
            bot_id="test_validation",
            name="Test Validation Bot",
            symbol="BTCUSDT",
            broker_type=BrokerType.BINANCE_PAPER,
            data_source=DataSourceType.BINANCE,
            strategy_name="BollingerStrategy"
        )

        save_config(config, "test_validation.json")

        # Validate the file
        is_valid, errors, warnings = validate_config_file("test_validation.json")

        if is_valid:
            print("✅ Config validation passed")
            if warnings:
                print(f"   Warnings: {warnings}")
        else:
            print(f"❌ Config validation failed: {errors}")
            return False

        # Clean up
        os.remove("test_validation.json")

        return True

    except Exception as e:
        print(f"❌ Error testing config validation: {e}")
        return False


def test_optimizer_config():
    """Test OptimizerConfig"""
    print("\nTesting OptimizerConfig...")

    try:
        config = OptimizerConfig(
            optimizer_id="test_optimizer",
            name="Test Optimizer",
            strategy_name="RSIStrategy",
            param_ranges={"rsi_period": [14, 21], "overbought": [70, 80]},
            symbol="BTCUSDT",
            start_date="2023-01-01",
            end_date="2023-12-31"
        )

        print(f"✅ Created optimizer config: {config.optimizer_type}")
        print(f"   Trials: {config.n_trials}")
        print(f"   Capital: ${config.initial_capital}")

        return True

    except Exception as e:
        print(f"❌ Error testing OptimizerConfig: {e}")
        return False


def test_data_config():
    """Test DataConfig"""
    print("\nTesting DataConfig...")

    try:
        config = DataConfig(
            data_id="test_data",
            name="Test Data Config",
            data_source=DataSourceType.BINANCE,
            symbols=["BTCUSDT", "ETHUSDT"],
            interval="1h"
        )

        print(f"✅ Created data config: {config.data_source}")
        print(f"   Symbols: {config.symbols}")
        print(f"   Interval: {config.interval}")

        return True

    except Exception as e:
        print(f"❌ Error testing DataConfig: {e}")
        return False


def test_sample_configs():
    """Test creating sample configurations"""
    print("\nTesting sample config creation...")

    try:
        # Create sample configs
        create_sample_config("test_sample_trading.json", "trading")
        create_sample_config("test_sample_optimizer.json", "optimizer")
        create_sample_config("test_sample_data.json", "data")

        print("Sample trading configuration created: test_sample_trading.json")
        print("Sample optimizer configuration created: test_sample_optimizer.json")
        print("Sample data configuration created: test_sample_data.json")

        # Verify files exist
        assert os.path.exists("test_sample_trading.json")
        assert os.path.exists("test_sample_optimizer.json")
        assert os.path.exists("test_sample_data.json")

        # Load and verify trading config
        trading_config = load_config("test_sample_trading.json")
        print("✅ Sample configs created")
        print(f"   Trading sample: {trading_config.bot_id}")

        # Clean up
        os.remove("test_sample_trading.json")
        os.remove("test_sample_optimizer.json")
        os.remove("test_sample_data.json")
        print("✅ Sample files cleaned up")

        return True

    except Exception as e:
        print(f"❌ Error testing sample configs: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Testing New Configuration System")
    print("=" * 50)

    tests = [
        test_trading_bot_config,
        test_config_loading_saving,
        test_config_validation,
        test_optimizer_config,
        test_data_config,
        test_sample_configs
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! The new config system is working correctly.")
    else:
        print("❌ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()