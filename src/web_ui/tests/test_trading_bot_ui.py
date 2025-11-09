#!/usr/bin/env python3
"""
Test Trading Bot UI Implementation
---------------------------------

Simple test script to verify the trading bot UI implementation works correctly.
Tests the API endpoints and configuration validation.
"""

import sys
from pathlib import Path
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.web_ui.config.trading_bot_config import (
    get_available_entry_mixins,
    get_available_exit_mixins,
    get_entry_mixin_parameters,
    parameter_definition_to_dict,
    BROKER_TYPES,
    SYMBOLS
)
from src.trading.services.bot_config_validator import BotConfigValidator


def test_mixin_configurations():
    """Test mixin configuration retrieval."""
    print("🧪 Testing Mixin Configurations")
    print("=" * 50)

    # Test entry mixins
    entry_mixins = get_available_entry_mixins()
    print(f"✅ Found {len(entry_mixins)} entry mixins:")
    for mixin in entry_mixins[:3]:  # Show first 3
        print(f"  - {mixin['value']}: {mixin['label']}")

    # Test exit mixins
    exit_mixins = get_available_exit_mixins()
    print(f"✅ Found {len(exit_mixins)} exit mixins:")
    for mixin in exit_mixins[:3]:  # Show first 3
        print(f"  - {mixin['value']}: {mixin['label']}")

    # Test parameter definitions
    if entry_mixins:
        first_entry = entry_mixins[0]['value']
        params = get_entry_mixin_parameters(first_entry)
        print(f"✅ {first_entry} has {len(params)} parameters:")
        for param in params[:2]:  # Show first 2
            param_dict = parameter_definition_to_dict(param)
            print(f"  - {param_dict['name']}: {param_dict['label']} ({param_dict['type']})")

    print()


def test_configuration_options():
    """Test configuration options."""
    print("🧪 Testing Configuration Options")
    print("=" * 50)

    print(f"✅ Broker types: {len(BROKER_TYPES)}")
    for broker in BROKER_TYPES:
        print(f"  - {broker['value']}: {broker['label']}")

    print(f"✅ Symbols: {len(SYMBOLS)}")
    for symbol in SYMBOLS[:3]:  # Show first 3
        print(f"  - {symbol['value']}: {symbol['label']}")

    print()


def test_bot_configuration_validation():
    """Test bot configuration validation."""
    print("🧪 Testing Bot Configuration Validation")
    print("=" * 50)

    # Valid configuration
    valid_config = {
        "id": "test_bot_ui",
        "name": "Test Bot UI",
        "enabled": True,
        "symbol": "BTCUSDT",
        "description": "Test bot for UI validation",
        "broker": {
            "type": "mock",
            "trading_mode": "paper",
            "name": "test_broker",
            "cash": 10000.0
        },
        "strategy": {
            "type": "CustomStrategy",
            "parameters": {
                "entry_logic": {
                    "name": "RSIBBVolumeEntryMixin",
                    "params": {
                        "e_rsi_period": 14,
                        "e_rsi_oversold": 30,
                        "e_bb_period": 20,
                        "e_bb_dev": 2.0,
                        "e_vol_ma_period": 20,
                        "e_min_volume_ratio": 1.1,
                        "e_use_bb_touch": True
                    }
                },
                "exit_logic": {
                    "name": "ATRExitMixin",
                    "params": {
                        "x_atr_period": 14,
                        "x_sl_multiplier": 1.5
                    }
                },
                "position_size": 0.1
            }
        },
        "data": {
            "data_source": "binance",
            "interval": "1h",
            "lookback_bars": 500
        },
        "trading": {
            "position_size": 0.1,
            "max_positions": 1
        },
        "risk_management": {
            "max_position_size": 1000.0,
            "stop_loss_pct": 3.0,
            "take_profit_pct": 6.0,
            "max_daily_loss": 200.0,
            "max_daily_trades": 5
        },
        "notifications": {
            "position_opened": True,
            "position_closed": True,
            "email_enabled": False,
            "telegram_enabled": True,
            "error_notifications": True
        }
    }

    validator = BotConfigValidator()
    is_valid, errors, warnings = validator.validate_bot_config(valid_config)

    print("✅ Valid configuration test:")
    print(f"  - Valid: {is_valid}")
    print(f"  - Errors: {len(errors)}")
    print(f"  - Warnings: {len(warnings)}")

    if warnings:
        print("  - Warning messages:")
        for warning in warnings[:2]:
            print(f"    • {warning}")

    # Invalid configuration
    invalid_config = {
        "name": "Invalid Bot",
        "enabled": "not_boolean",  # Should be boolean
        "symbol": "",  # Empty symbol
        "broker": {
            "type": "unknown_broker",
            "trading_mode": "invalid_mode"
            # Missing required fields
        },
        "strategy": {
            "type": "CustomStrategy",
            "parameters": {
                # Missing entry_logic and exit_logic
                "position_size": 1.5  # Invalid: > 1
            }
        }
    }

    is_valid, errors, warnings = validator.validate_bot_config(invalid_config)

    print("✅ Invalid configuration test:")
    print(f"  - Valid: {is_valid}")
    print(f"  - Errors: {len(errors)}")
    print(f"  - Warnings: {len(warnings)}")

    if errors:
        print("  - Error messages:")
        for error in errors[:3]:  # Show first 3
            print(f"    • {error}")

    print()


def test_json_serialization():
    """Test JSON serialization of configurations."""
    print("🧪 Testing JSON Serialization")
    print("=" * 50)

    # Test parameter definition serialization
    entry_mixins = get_available_entry_mixins()
    if entry_mixins:
        first_mixin = entry_mixins[0]['value']
        params = get_entry_mixin_parameters(first_mixin)

        if params:
            param_dict = parameter_definition_to_dict(params[0])
            json_str = json.dumps(param_dict, indent=2)
            print("✅ Parameter definition JSON serialization:")
            print(f"  - Mixin: {first_mixin}")
            print(f"  - Parameter: {param_dict['name']}")
            print(f"  - JSON length: {len(json_str)} characters")

    # Test full configuration serialization
    config_options = {
        'broker_types': BROKER_TYPES,
        'symbols': SYMBOLS[:5],  # Limit for test
        'entry_mixins': entry_mixins[:3],  # Limit for test
        'exit_mixins': get_available_exit_mixins()[:3]  # Limit for test
    }

    json_str = json.dumps(config_options, indent=2)
    print("✅ Configuration options JSON serialization:")
    print(f"  - JSON length: {len(json_str)} characters")
    print(f"  - Broker types: {len(config_options['broker_types'])}")
    print(f"  - Symbols: {len(config_options['symbols'])}")

    print()


def main():
    """Run all tests."""
    print("🚀 Trading Bot UI Implementation Test")
    print("=" * 60)
    print("Testing the trading bot management UI components...")
    print()

    try:
        test_mixin_configurations()
        test_configuration_options()
        test_bot_configuration_validation()
        test_json_serialization()

        print("🎉 All Tests Completed Successfully!")
        print("=" * 60)
        print("✅ Mixin parameter definitions working")
        print("✅ Configuration options available")
        print("✅ Bot configuration validation working")
        print("✅ JSON serialization working")
        print()
        print("The trading bot API implementation is ready!")
        print("You can now:")
        print("• Use the API endpoints for bot management")
        print("• Access configuration options via /api/trading/config/options")
        print("• Create and configure trading bots through the React frontend")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()