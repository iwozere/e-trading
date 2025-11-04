"""
Bot Configuration Validator
---------------------------

Validates trading bot configurations stored in the database.
Ensures all required fields are present and valid before bot startup.

This module now delegates to schema_validator.py for JSON Schema-based validation.
"""

from typing import Dict, Any, List, Tuple, Optional
import json
from decimal import Decimal
from src.notification.logger import setup_logger
from src.trading.services.schema_validator import validate_bot_configuration

_logger = setup_logger(__name__)


class BotConfigValidator:
    """
    Validates trading bot configurations from database records.

    Provides comprehensive validation for bot configurations including
    broker settings, strategy parameters, risk management, and notifications.
    """

    def __init__(self):
        """Initialize the bot configuration validator."""
        self.errors = []
        self.warnings = []

    def validate_bot_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a complete bot configuration using JSON Schema.

        Args:
            config: Bot configuration dictionary from database

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        # Delegate to schema-based validator
        return validate_bot_configuration(config)

    def _validate_required_fields(self, config: Dict[str, Any]) -> None:
        """Validate required top-level configuration fields."""
        required_fields = ["id", "name", "enabled", "symbol", "broker", "strategy"]

        for field in required_fields:
            if field not in config:
                self.errors.append(f"Missing required field: {field}")
            elif config[field] is None:
                self.errors.append(f"Required field '{field}' cannot be null")

        # Validate field types
        if "enabled" in config and not isinstance(config["enabled"], bool):
            self.errors.append("Field 'enabled' must be a boolean")

        if "symbol" in config and not isinstance(config["symbol"], str):
            self.errors.append("Field 'symbol' must be a string")
        elif "symbol" in config and not config["symbol"].strip():
            self.errors.append("Field 'symbol' cannot be empty")

    def _validate_broker_config(self, broker_config: Dict[str, Any]) -> None:
        """Validate broker configuration."""
        required_fields = ["type", "trading_mode", "name"]

        for field in required_fields:
            if field not in broker_config:
                self.errors.append(f"Missing required broker field: {field}")

        # Validate broker type
        if "type" in broker_config:
            valid_types = ["binance", "paper", "alpaca", "interactive_brokers"]
            if broker_config["type"] not in valid_types:
                self.warnings.append(f"Unknown broker type: {broker_config['type']}")

        # Validate trading mode
        if "trading_mode" in broker_config:
            valid_modes = ["paper", "live"]
            if broker_config["trading_mode"] not in valid_modes:
                self.errors.append(f"Invalid trading_mode: {broker_config['trading_mode']}. Must be 'paper' or 'live'")

        # Validate cash amount for paper trading
        if broker_config.get("trading_mode") == "paper":
            if "cash" not in broker_config:
                self.errors.append("Paper trading requires 'cash' field in broker config")
            elif not isinstance(broker_config["cash"], (int, float)) or broker_config["cash"] <= 0:
                self.errors.append("Broker 'cash' must be a positive number")

    def _validate_strategy_config(self, strategy_config: Dict[str, Any]) -> None:
        """Validate strategy configuration."""
        if "type" not in strategy_config:
            self.errors.append("Missing required strategy field: type")
            return

        strategy_type = strategy_config["type"]

        # Validate CustomStrategy configuration
        if strategy_type == "CustomStrategy":
            self._validate_custom_strategy_config(strategy_config)
        else:
            self.warnings.append(f"Unknown strategy type: {strategy_type}. Will attempt to load dynamically.")

    def _validate_custom_strategy_config(self, strategy_config: Dict[str, Any]) -> None:
        """Validate CustomStrategy specific configuration."""
        if "parameters" not in strategy_config:
            self.errors.append("CustomStrategy requires 'parameters' field")
            return

        params = strategy_config["parameters"]

        # Validate entry logic
        if "entry_logic" not in params:
            self.errors.append("CustomStrategy parameters require 'entry_logic' field")
        else:
            self._validate_mixin_config(params["entry_logic"], "entry_logic")

        # Validate exit logic
        if "exit_logic" not in params:
            self.errors.append("CustomStrategy parameters require 'exit_logic' field")
        else:
            self._validate_mixin_config(params["exit_logic"], "exit_logic")

        # Validate position size
        if "position_size" in params:
            pos_size = params["position_size"]
            if not isinstance(pos_size, (int, float)) or pos_size <= 0 or pos_size > 1:
                self.errors.append("position_size must be a number between 0 and 1")

    def _validate_mixin_config(self, mixin_config: Dict[str, Any], mixin_type: str) -> None:
        """Validate entry/exit logic mixin configuration."""
        if "name" not in mixin_config:
            self.errors.append(f"{mixin_type} requires 'name' field")

        if "params" not in mixin_config:
            self.warnings.append(f"{mixin_type} has no parameters defined")
        elif not isinstance(mixin_config["params"], dict):
            self.errors.append(f"{mixin_type} 'params' must be a dictionary")

        # Validate common mixin parameters
        if "params" in mixin_config:
            params = mixin_config["params"]

            # Check for negative periods
            for key, value in params.items():
                if key.endswith("_period") and isinstance(value, (int, float)) and value <= 0:
                    self.errors.append(f"{mixin_type} parameter '{key}' must be positive")

    def _validate_data_config(self, data_config: Dict[str, Any]) -> None:
        """Validate data source configuration."""
        if "data_source" not in data_config:
            self.errors.append("Missing required data field: data_source")

        if "interval" not in data_config:
            self.errors.append("Missing required data field: interval")
        elif data_config["interval"] not in ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]:
            self.warnings.append(f"Unusual interval: {data_config['interval']}")

        if "lookback_bars" in data_config:
            lookback = data_config["lookback_bars"]
            if not isinstance(lookback, int) or lookback <= 0:
                self.errors.append("lookback_bars must be a positive integer")
            elif lookback > 5000:
                self.warnings.append("lookback_bars is very high, may impact performance")

    def _validate_trading_config(self, trading_config: Dict[str, Any]) -> None:
        """Validate trading configuration."""
        if "position_size" in trading_config:
            pos_size = trading_config["position_size"]
            if not isinstance(pos_size, (int, float)) or pos_size <= 0:
                self.errors.append("trading position_size must be a positive number")

        if "max_positions" in trading_config:
            max_pos = trading_config["max_positions"]
            if not isinstance(max_pos, int) or max_pos <= 0:
                self.errors.append("max_positions must be a positive integer")
            elif max_pos > 10:
                self.warnings.append("max_positions is quite high, consider risk implications")

    def _validate_risk_management(self, risk_config: Dict[str, Any]) -> None:
        """Validate risk management configuration."""
        # Validate percentage fields
        percentage_fields = ["stop_loss_pct", "take_profit_pct"]
        for field in percentage_fields:
            if field in risk_config:
                value = risk_config[field]
                if not isinstance(value, (int, float)) or value <= 0:
                    self.errors.append(f"{field} must be a positive number")

        # Validate stop loss vs take profit
        if "stop_loss_pct" in risk_config and "take_profit_pct" in risk_config:
            if risk_config["take_profit_pct"] <= risk_config["stop_loss_pct"]:
                self.warnings.append("take_profit_pct should be greater than stop_loss_pct")

        # Validate monetary fields
        monetary_fields = ["max_position_size", "max_daily_loss"]
        for field in monetary_fields:
            if field in risk_config:
                value = risk_config[field]
                if not isinstance(value, (int, float)) or value <= 0:
                    self.errors.append(f"{field} must be a positive number")

        # Validate trade limits
        if "max_daily_trades" in risk_config:
            max_trades = risk_config["max_daily_trades"]
            if not isinstance(max_trades, int) or max_trades <= 0:
                self.errors.append("max_daily_trades must be a positive integer")
            elif max_trades > 100:
                self.warnings.append("max_daily_trades is very high, consider risk implications")

    def _validate_notifications_config(self, notif_config: Dict[str, Any]) -> None:
        """Validate notifications configuration."""
        boolean_fields = [
            "position_opened", "position_closed", "email_enabled",
            "telegram_enabled", "error_notifications", "risk_alerts"
        ]

        for field in boolean_fields:
            if field in notif_config and not isinstance(notif_config[field], bool):
                self.errors.append(f"Notification field '{field}' must be a boolean")

        # Validate performance summary frequency
        if "performance_summaries" in notif_config:
            valid_frequencies = ["none", "daily", "weekly", "monthly"]
            if notif_config["performance_summaries"] not in valid_frequencies:
                self.errors.append(f"Invalid performance_summaries frequency: {notif_config['performance_summaries']}")


def validate_bot_config_json(config_json: str) -> Tuple[bool, List[str], List[str]]:
    """
    Validate bot configuration from JSON string using JSON Schema.

    Args:
        config_json: JSON string containing bot configuration

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    try:
        config = json.loads(config_json)
        return validate_bot_configuration(config)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {str(e)}"], []
    except Exception as e:
        return False, [f"Validation error: {str(e)}"], []


def validate_database_bot_record(bot_record: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """
    Validate a complete bot record from the database.

    Args:
        bot_record: Complete bot record from trading_bots table

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    errors = []
    warnings = []

    # Validate database record fields (excluding 'type' as per user directive)
    required_db_fields = ["id", "user_id", "status", "config"]
    for field in required_db_fields:
        if field not in bot_record or bot_record[field] is None:
            errors.append(f"Missing required database field: {field}")

    # Validate status
    if "status" in bot_record:
        valid_statuses = ["stopped", "starting", "running", "error", "stopping", "disabled"]
        if bot_record["status"] not in valid_statuses:
            warnings.append(f"Unknown bot status: {bot_record['status']}")

    # If we have errors at the database level, don't validate config
    if errors:
        return False, errors, warnings

    # Validate the configuration JSON using schema validator
    try:
        config = bot_record["config"]
        if isinstance(config, str):
            config = json.loads(config)

        # Use schema-based validator directly
        config_valid, config_errors, config_warnings = validate_bot_configuration(config)
        errors.extend(config_errors)
        warnings.extend(config_warnings)

        return len(errors) == 0, errors, warnings

    except json.JSONDecodeError as e:
        errors.append(f"Invalid configuration JSON: {str(e)}")
        return False, errors, warnings
    except Exception as e:
        errors.append(f"Configuration validation error: {str(e)}")
        return False, errors, warnings


def print_validation_results(bot_id: str, is_valid: bool, errors: List[str], warnings: List[str]) -> None:
    """
    Print validation results in a formatted way.

    Args:
        bot_id: Bot identifier for context
        is_valid: Whether the configuration is valid
        errors: List of validation errors
        warnings: List of validation warnings
    """
    print(f"\n{'='*60}")
    print(f"BOT CONFIGURATION VALIDATION - {bot_id}")
    print(f"{'='*60}")

    if is_valid:
        print("✅ Configuration is VALID")
    else:
        print("❌ Configuration is INVALID")

    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")

    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")

    if not errors and not warnings:
        print("\n✅ No issues found!")

    print(f"{'='*60}\n")