#!/usr/bin/env python3
"""
Demo: Bot Configuration Integration
----------------------------------

Demonstrates the database integration layer for bot configuration management.
Shows validation, schema retrieval, and configuration management functions.
"""

import json
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.trading.services.bot_config_validator import (
    BotConfigValidator,
    validate_bot_config_json,
    validate_database_bot_record,
    print_validation_results
)
from src.data.db.services import trading_service


def demo_configuration_validation():
    """Demonstrate configuration validation functionality."""
    print("=" * 60)
    print("BOT CONFIGURATION VALIDATION DEMO")
    print("=" * 60)

    # Example valid configuration
    valid_config = {
        "id": "rsi_atr_btc_paper",
        "name": "RSI+ATR BTC Paper Trading",
        "enabled": True,
        "symbol": "BTCUSDT",
        "broker": {
            "type": "binance",
            "trading_mode": "paper",
            "name": "rsi_atr_btc_paper_broker",
            "cash": 100.0
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

    print("\n1. Testing VALID configuration:")
    validator = BotConfigValidator()
    is_valid, errors, warnings = validator.validate_bot_config(valid_config)
    print_validation_results("rsi_atr_btc_paper", is_valid, errors, warnings)

    # Example invalid configuration
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

    print("\n2. Testing INVALID configuration:")
    is_valid, errors, warnings = validator.validate_bot_config(invalid_config)
    print_validation_results("invalid_bot", is_valid, errors, warnings)


def demo_database_record_validation():
    """Demonstrate database record validation."""
    print("\n" + "=" * 60)
    print("DATABASE RECORD VALIDATION DEMO")
    print("=" * 60)

    # Example database record
    bot_record = {
        "id": 1,
        "user_id": 1,
        "type": "paper",
        "status": "stopped",
        "config": {
            "id": "test_bot_1",
            "name": "Test Bot",
            "enabled": True,
            "symbol": "BTCUSDT",
            "broker": {
                "type": "binance",
                "trading_mode": "paper",
                "name": "test_broker",
                "cash": 1000.0
            },
            "strategy": {
                "type": "CustomStrategy",
                "parameters": {
                    "entry_logic": {
                        "name": "RSIEntryMixin",
                        "params": {"rsi_period": 14}
                    },
                    "exit_logic": {
                        "name": "ATRExitMixin",
                        "params": {"atr_period": 14}
                    }
                }
            }
        },
        "description": "Test bot description",
        "started_at": None,
        "last_heartbeat": None,
        "error_count": 0,
        "current_balance": None,
        "total_pnl": None,
        "extra_metadata": None,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": None
    }

    print("\n1. Testing valid database record:")
    is_valid, errors, warnings = validate_database_bot_record(bot_record)
    print_validation_results("database_bot_1", is_valid, errors, warnings)

    # Invalid database record
    invalid_record = {
        "id": 2,
        "user_id": None,  # Invalid: required field is None
        "type": "invalid_type",  # Invalid: not 'paper' or 'live'
        "status": "unknown_status",  # Warning: unknown status
        "config": "invalid_json"  # Invalid: not valid JSON
    }

    print("\n2. Testing invalid database record:")
    is_valid, errors, warnings = validate_database_bot_record(invalid_record)
    print_validation_results("invalid_database_bot", is_valid, errors, warnings)


def demo_json_validation():
    """Demonstrate JSON string validation."""
    print("\n" + "=" * 60)
    print("JSON STRING VALIDATION DEMO")
    print("=" * 60)

    # Valid JSON configuration
    valid_json_config = {
        "id": "json_test_bot",
        "name": "JSON Test Bot",
        "enabled": True,
        "symbol": "ETHUSDT",
        "broker": {
            "type": "binance",
            "trading_mode": "paper",
            "name": "json_test_broker",
            "cash": 500.0
        },
        "strategy": {
            "type": "CustomStrategy",
            "parameters": {
                "entry_logic": {
                    "name": "MACDEntryMixin",
                    "params": {"fast_period": 12, "slow_period": 26}
                },
                "exit_logic": {
                    "name": "FixedExitMixin",
                    "params": {"take_profit_pct": 5.0, "stop_loss_pct": 2.0}
                }
            }
        }
    }

    json_string = json.dumps(valid_json_config, indent=2)
    print("\n1. Testing valid JSON configuration:")
    print("JSON Config:")
    print(json_string[:200] + "..." if len(json_string) > 200 else json_string)

    is_valid, errors, warnings = validate_bot_config_json(json_string)
    print_validation_results("json_test_bot", is_valid, errors, warnings)

    # Invalid JSON
    invalid_json = '{"id": "invalid", "name": "Invalid", "enabled": true, "invalid_json": }'

    print("\n2. Testing invalid JSON:")
    print("Invalid JSON:", invalid_json)
    is_valid, errors, warnings = validate_bot_config_json(invalid_json)
    print_validation_results("invalid_json_bot", is_valid, errors, warnings)


def demo_configuration_schema():
    """Demonstrate configuration schema retrieval."""
    print("\n" + "=" * 60)
    print("CONFIGURATION SCHEMA DEMO")
    print("=" * 60)

    schema = trading_service.get_bot_configuration_schema()

    print("\nBot Configuration Schema:")
    print(f"Type: {schema['type']}")
    print(f"Required fields: {', '.join(schema['required'])}")

    print("\nAvailable properties:")
    for prop_name, prop_def in schema['properties'].items():
        prop_type = prop_def.get('type', 'unknown')
        description = prop_def.get('description', 'No description')
        print(f"  - {prop_name} ({prop_type}): {description}")

        # Show nested properties for objects
        if prop_type == 'object' and 'properties' in prop_def:
            for nested_name, nested_def in prop_def['properties'].items():
                nested_type = nested_def.get('type', 'unknown')
                nested_desc = nested_def.get('description', 'No description')
                print(f"    - {nested_name} ({nested_type}): {nested_desc}")


def demo_trading_service_functions():
    """Demonstrate trading service integration functions."""
    print("\n" + "=" * 60)
    print("TRADING SERVICE INTEGRATION DEMO")
    print("=" * 60)

    print("\nAvailable trading service functions for bot configuration:")

    functions = [
        'upsert_bot',
        'get_bot_by_id',
        'get_enabled_bots',
        'get_bots_by_status',
        'update_bot_status',
        'update_bot_performance',
        'validate_bot_configuration',
        'validate_all_bot_configurations',
        'get_bot_configuration_schema'
    ]

    for func_name in functions:
        if hasattr(trading_service, func_name):
            func = getattr(trading_service, func_name)
            print(f"  ✅ {func_name}: {func.__doc__.split('.')[0] if func.__doc__ else 'Available'}")
        else:
            print(f"  ❌ {func_name}: Not available")

    print("\nNote: Database operations require a valid database connection.")
    print("The validation and schema functions work without database access.")


def main():
    """Run all demonstrations."""
    print("🚀 Bot Configuration Integration Demo")
    print("This demo shows the database integration layer for trading bot configuration management.")

    try:
        demo_configuration_validation()
        demo_database_record_validation()
        demo_json_validation()
        demo_configuration_schema()
        demo_trading_service_functions()

        print("\n" + "=" * 60)
        print("✅ DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nThe database integration layer provides:")
        print("• Comprehensive configuration validation")
        print("• Database schema validation for bot records")
        print("• JSON configuration parsing and validation")
        print("• Configuration schema documentation")
        print("• Enhanced trading service with bot management functions")
        print("\nAll functions are ready for use in the enhanced trading service!")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()