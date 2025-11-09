"""
Test Bot Configuration Integration
---------------------------------

Tests for the database integration layer for bot configuration management.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import json
from src.trading.services.bot_config_validator import (
    BotConfigValidator,
    validate_bot_config_json,
    validate_database_bot_record
)
from src.data.db.services.trading_service import trading_service, get_bot_configuration_schema


class TestBotConfigValidator:
    """Test bot configuration validation functionality."""

    def test_valid_bot_config(self):
        """Test validation of a valid bot configuration."""
        config = {
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
                        "params": {"rsi_period": 14, "rsi_oversold": 30}
                    },
                    "exit_logic": {
                        "name": "ATRExitMixin",
                        "params": {"atr_period": 14, "sl_multiplier": 1.5}
                    },
                    "position_size": 0.1
                }
            },
            "data": {
                "data_source": "binance",
                "interval": "1h",
                "lookback_bars": 500
            },
            "risk_management": {
                "stop_loss_pct": 3.0,
                "take_profit_pct": 6.0,
                "max_daily_trades": 5
            },
            "notifications": {
                "position_opened": True,
                "position_closed": True,
                "telegram_enabled": True
            }
        }

        validator = BotConfigValidator()
        is_valid, errors, warnings = validator.validate_bot_config(config)

        assert is_valid, f"Configuration should be valid. Errors: {errors}"
        assert len(errors) == 0, f"Should have no errors: {errors}"

    def test_missing_required_fields(self):
        """Test validation with missing required fields."""
        config = {
            "name": "Test Bot",
            "enabled": True
            # Missing: symbol, broker, strategy (id, name, enabled are database fields, not config fields)
        }

        validator = BotConfigValidator()
        is_valid, errors, warnings = validator.validate_bot_config(config)

        assert not is_valid, "Configuration should be invalid"
        assert len(errors) >= 3, f"Should have at least 3 errors for missing fields: {errors}"

        # Check for specific missing fields (config fields only, not database fields)
        error_text = " ".join(errors)
        assert "symbol" in error_text
        assert "broker" in error_text
        assert "strategy" in error_text

    def test_invalid_broker_config(self):
        """Test validation with invalid broker configuration."""
        config = {
            "id": "test_bot_1",
            "name": "Test Bot",
            "enabled": True,
            "symbol": "BTCUSDT",
            "broker": {
                "type": "invalid_broker",
                "trading_mode": "invalid_mode"
                # Note: 'name' is optional in schema, not required
            },
            "strategy": {
                "type": "CustomStrategy",
                "parameters": {}
            }
        }

        validator = BotConfigValidator()
        is_valid, errors, warnings = validator.validate_bot_config(config)

        assert not is_valid, "Configuration should be invalid"
        assert any("trading_mode" in error for error in errors), f"Should have trading_mode error: {errors}"
        assert any("type" in error or "broker" in error for error in errors), f"Should have broker type error: {errors}"

    def test_invalid_strategy_config(self):
        """Test validation with invalid strategy configuration."""
        config = {
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
                    # Missing entry_logic and exit_logic (reported as warnings in semantic validation)
                    "position_size": 1.5  # Invalid: > 1 (reported as error)
                }
            }
        }

        validator = BotConfigValidator()
        is_valid, errors, warnings = validator.validate_bot_config(config)

        assert not is_valid, "Configuration should be invalid"

        # position_size violation should be an error
        error_text = " ".join(errors)
        assert "position_size" in error_text

        # entry_logic and exit_logic missing are reported as warnings
        warning_text = " ".join(warnings)
        assert "entry_logic" in warning_text
        assert "exit_logic" in warning_text

    def test_validate_json_string(self):
        """Test validation from JSON string."""
        config_dict = {
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
        }

        config_json = json.dumps(config_dict)
        is_valid, errors, warnings = validate_bot_config_json(config_json)

        assert is_valid, f"Configuration should be valid. Errors: {errors}"

    def test_validate_database_record(self):
        """Test validation of complete database record."""
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

        is_valid, errors, warnings = validate_database_bot_record(bot_record)

        assert is_valid, f"Database record should be valid. Errors: {errors}"

    def test_invalid_database_record(self):
        """Test validation of invalid database record."""
        bot_record = {
            "id": 1,
            "user_id": None,  # Invalid: required field is None
            "type": "invalid_type",  # Invalid: not 'paper' or 'live'
            "status": "unknown_status",  # Warning: unknown status
            "config": "invalid_json"  # Invalid: not valid JSON
        }

        is_valid, errors, warnings = validate_database_bot_record(bot_record)

        assert not is_valid, "Database record should be invalid"
        assert len(errors) > 0, f"Should have errors: {errors}"


class TestTradingServiceIntegration:
    """Test trading service integration with configuration validation."""

    def test_get_bot_configuration_schema(self):
        """Test getting the bot configuration schema."""
        schema = get_bot_configuration_schema()

        assert isinstance(schema, dict)
        assert "type" in schema
        assert "required" in schema
        assert "properties" in schema

        # Check required fields are present (config fields only, not database fields)
        required_fields = schema["required"]
        assert "symbol" in required_fields
        assert "broker" in required_fields
        assert "strategy" in required_fields

        # Database fields (id, name, enabled) should NOT be in config schema
        assert "id" not in required_fields
        assert "name" not in required_fields
        assert "enabled" not in required_fields

    def test_configuration_validation_functions_exist(self):
        """Test that configuration validation functions are available."""
        # Test trading_service instance methods
        assert hasattr(trading_service, 'validate_bot_configuration')
        assert hasattr(trading_service, 'validate_all_bot_configurations')
        assert callable(trading_service.validate_bot_configuration)
        assert callable(trading_service.validate_all_bot_configurations)

        # Test module-level function (not an instance method)
        assert callable(get_bot_configuration_schema)


if __name__ == "__main__":
    # Run basic tests
    test_validator = TestBotConfigValidator()
    test_validator.test_valid_bot_config()
    test_validator.test_missing_required_fields()
    test_validator.test_invalid_broker_config()
    test_validator.test_invalid_strategy_config()
    test_validator.test_validate_json_string()
    test_validator.test_validate_database_record()
    test_validator.test_invalid_database_record()

    test_integration = TestTradingServiceIntegration()
    test_integration.test_get_bot_configuration_schema()
    test_integration.test_configuration_validation_functions_exist()

    print("✅ All tests passed!")