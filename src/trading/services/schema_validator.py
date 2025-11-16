"""
JSON Schema-based Configuration Validator
------------------------------------------

Validates trading bot configurations using YAML-defined JSON schemas.
Provides structured error reporting and human-readable messages.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple
from functools import lru_cache
from jsonschema import Draft7Validator, ValidationError

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class SchemaValidator:
    """
    JSON Schema-based validator for bot configurations.

    Features:
    - YAML-defined schemas for maintainability
    - JSON Schema Draft 7 standard validation
    - Structured error reporting with field paths
    - Human-readable error messages for logs
    - Schema caching for performance
    """

    SCHEMA_DIR = Path(__file__).parent.parent.parent.parent / "config" / "schemas"

    def __init__(self):
        """Initialize the schema validator."""
        self.schema_cache: Dict[str, Dict[str, Any]] = {}
        _logger.debug("Initialized SchemaValidator with schema dir: %s", self.SCHEMA_DIR)

    @lru_cache(maxsize=10)
    def load_schema(self, schema_name: str) -> Dict[str, Any]:
        """
        Load and cache a YAML schema file.

        Args:
            schema_name: Name of schema file (e.g., 'bot_config.yaml')

        Returns:
            Parsed schema dictionary

        Raises:
            FileNotFoundError: If schema file doesn't exist
            yaml.YAMLError: If schema file is invalid YAML
        """
        schema_path = self.SCHEMA_DIR / schema_name

        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = yaml.safe_load(f)

            _logger.debug("Loaded schema: %s", schema_name)
            return schema

        except yaml.YAMLError as e:
            _logger.error("Invalid YAML in schema %s: %s", schema_name, e)
            raise

    def validate_bot_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a bot configuration against the schema.

        Args:
            config: Bot configuration dictionary

        Returns:
            Tuple of (is_valid, errors, warnings)
            - is_valid: True if validation passed
            - errors: List of error messages
            - warnings: List of warning messages
        """
        errors = []
        warnings = []

        try:
            # Load schema
            schema = self.load_schema("bot_config.yaml")

            # Create validator
            validator = Draft7Validator(schema)

            # Validate
            validation_errors = sorted(validator.iter_errors(config), key=lambda e: e.path)

            if validation_errors:
                for error in validation_errors:
                    error_msg = self._format_validation_error(error)
                    errors.append(error_msg)

            # Additional semantic validations
            semantic_warnings = self._semantic_validations(config)
            warnings.extend(semantic_warnings)

            is_valid = len(errors) == 0

            if is_valid:
                _logger.debug("Configuration validation passed")
            else:
                _logger.warning("Configuration validation failed with %d error(s)", len(errors))

            return is_valid, errors, warnings

        except Exception as e:
            _logger.exception("Error during schema validation:")
            errors.append(f"Schema validation error: {str(e)}")
            return False, errors, warnings

    def _format_validation_error(self, error: ValidationError) -> str:
        """
        Format a validation error into a human-readable message.

        Args:
            error: ValidationError from jsonschema

        Returns:
            Formatted error message with field path
        """
        # Build field path
        path_parts = []
        for part in error.absolute_path:
            if isinstance(part, int):
                path_parts.append(f"[{part}]")
            else:
                if path_parts:
                    path_parts.append(f".{part}")
                else:
                    path_parts.append(str(part))

        field_path = "".join(path_parts) if path_parts else "root"

        # Format error message based on error type
        if error.validator == "required":
            missing_field = error.message.split("'")[1]
            return f"Missing required field: {field_path}.{missing_field}"

        elif error.validator == "type":
            expected_type = error.validator_value
            return f"Invalid type for '{field_path}': expected {expected_type}"

        elif error.validator == "enum":
            valid_values = ", ".join(f"'{v}'" for v in error.validator_value)
            actual_value = error.instance
            return f"Invalid value '{actual_value}' for '{field_path}': must be one of [{valid_values}]"

        elif error.validator == "pattern":
            return f"Invalid format for '{field_path}': {error.message}"

        elif error.validator in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            return f"Value constraint violation for '{field_path}': {error.message}"

        else:
            # Generic error message
            return f"Validation error for '{field_path}': {error.message}"

    def _semantic_validations(self, config: Dict[str, Any]) -> List[str]:
        """
        Perform semantic validations beyond schema structure.

        Args:
            config: Bot configuration

        Returns:
            List of warning messages
        """
        warnings = []

        # Check broker cash for paper trading
        broker = config.get("broker", {})
        if broker.get("trading_mode") == "paper":
            if "cash" not in broker:
                warnings.append("Paper trading mode requires 'broker.cash' field")
            elif broker.get("cash", 0) <= 0:
                warnings.append("Paper trading cash should be greater than 0")

        # Check live trading has credentials
        if broker.get("trading_mode") == "live":
            broker_type = broker.get("type")
            if broker_type in ("binance", "alpaca", "ibkr"):
                if not broker.get("api_key") or not broker.get("api_secret"):
                    warnings.append(f"Live trading with {broker_type} requires 'api_key' and 'api_secret'")

        # Check file data source has file_path
        data = config.get("data", {})
        if data.get("data_source") == "file":
            if not data.get("file_path"):
                warnings.append("File data source requires 'data.file_path'")
            else:
                file_path = Path(data["file_path"])
                if not file_path.exists():
                    warnings.append(f"Data file does not exist: {data['file_path']}")

        # Check CustomStrategy has entry/exit logic
        strategy = config.get("strategy", {})
        if strategy.get("type") == "CustomStrategy":
            params = strategy.get("parameters", {})
            if not params.get("entry_logic"):
                warnings.append("CustomStrategy missing 'strategy.parameters.entry_logic'")
            if not params.get("exit_logic"):
                warnings.append("CustomStrategy missing 'strategy.parameters.exit_logic'")

        # Check notification channels are enabled if notifications requested
        notifications = config.get("notifications", {})
        has_notification_types = any([
            notifications.get("position_opened"),
            notifications.get("position_closed"),
            notifications.get("error_notifications"),
            notifications.get("daily_summary")
        ])

        has_channels = any([
            notifications.get("email_enabled"),
            notifications.get("telegram_enabled")
        ])

        if has_notification_types and not has_channels:
            warnings.append("Notifications enabled but no delivery channels (email/telegram) are enabled")

        return warnings

    def validate_and_format(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration and return formatted result.

        Args:
            config: Bot configuration

        Returns:
            Dictionary with validation results:
            {
                'valid': bool,
                'errors': List[str],
                'warnings': List[str],
                'summary': str  # Human-readable summary
            }
        """
        is_valid, errors, warnings = self.validate_bot_config(config)

        # Build summary
        if is_valid:
            if warnings:
                summary = f"Configuration valid with {len(warnings)} warning(s)"
            else:
                summary = "Configuration valid"
        else:
            summary = f"Configuration invalid: {len(errors)} error(s), {len(warnings)} warning(s)"

        return {
            'valid': is_valid,
            'errors': errors,
            'warnings': warnings,
            'summary': summary
        }


# Global schema validator instance
_schema_validator = SchemaValidator()


def validate_bot_configuration(config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """
    Convenience function to validate a bot configuration.

    Args:
        config: Bot configuration dictionary

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    return _schema_validator.validate_bot_config(config)


def validate_bot_configuration_detailed(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to validate and get detailed results.

    Args:
        config: Bot configuration dictionary

    Returns:
        Dictionary with validation results including summary
    """
    return _schema_validator.validate_and_format(config)


# Example usage and testing
if __name__ == "__main__":
    # Test configuration
    test_config = {
        "symbol": "BTCUSDT",
        "broker": {
            "type": "backtrader",
            "trading_mode": "paper",
            "cash": 10000.0,
            "commission": 0.001
        },
        "data": {
            "data_source": "file",
            "file_path": "c:/dev/cursor/e-trading/data/_full/BTCUSDT_1h_20220101_20250707.csv",
            "symbol": "BTCUSDT",
            "interval": "1h",
            "simulate_realtime": False,
            "datetime_col": "datetime",
            "open_col": "open",
            "high_col": "high",
            "low_col": "low",
            "close_col": "close",
            "volume_col": "volume"
        },
        "strategy": {
            "type": "CustomStrategy",
            "parameters": {
                "entry_logic": {
                    "name": "RSIBBVolumeEntryMixin",
                    "params": {
                        "entry_e_bb_dev": 2.66,
                        "entry_e_bb_period": 27,
                        "entry_e_rsi_period": 22,
                        "entry_e_vol_ma_period": 32,
                        "entry_e_min_volume_ratio": 1.17,
                        "entry_e_rsi_oversold": 38
                    }
                },
                "exit_logic": {
                    "name": "ATRExitMixin",
                    "params": {
                        "exit_x_atr_period": 9,
                        "exit_x_tp_multiplier": 5,
                        "exit_x_sl_multiplier": 2.27
                    }
                },
                "position_size": 0.1
            }
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

    # Validate
    result = validate_bot_configuration_detailed(test_config)

    print(result['summary'])
    print()

    if result['errors']:
        print("Errors:")
        for error in result['errors']:
            print(f"  - {error}")
        print()

    if result['warnings']:
        print("Warnings:")
        for warning in result['warnings']:
            print(f"  - {warning}")
