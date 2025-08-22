"""
Configuration Validator Module
-----------------------------

This module provides validation for live trading bot configurations.
It ensures all required parameters are present and valid before starting the bot.

Classes:
- ConfigValidator: Validates trading bot configurations using Pydantic models
"""

from typing import Dict, Any, List, Tuple
from pathlib import Path
from src.notification.logger import setup_logger
from src.config.config_loader import validate_config_file as pydantic_validate_config_file
from src.model.config_models import TradingBotConfig

_logger = setup_logger(__name__)


class ConfigValidator:
    """
    Validates live trading bot configurations using Pydantic models.

    This class leverages Pydantic validation to ensure that all required
    parameters are present and valid before the bot starts.
    """

    def __init__(self):
        """Initialize the configuration validator."""
        self.errors = []
        self.warnings = []

    def validate_config(self, config: TradingBotConfig) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a TradingBotConfig instance.

        Args:
            config: TradingBotConfig instance to validate

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []

        try:
            # Pydantic validation is already done when creating the config
            # Add additional business logic validation here

            # Validate take profit vs stop loss
            if config.take_profit_pct <= config.stop_loss_pct:
                self.warnings.append("Take profit should be greater than stop loss")

            # Validate risk per trade
            if config.risk_per_trade > 0.05:
                self.warnings.append("Risk per trade is quite high (>5%)")

            # Validate max open trades
            if config.max_open_trades > 10:
                self.warnings.append("Maximum open trades is quite high (>10)")

            # Validate position size
            if config.position_size > 0.5:
                self.warnings.append("Position size is quite large (>50%)")

            # Validate symbol format
            if not config.symbol.isalnum():
                self.warnings.append("Symbol contains non-alphanumeric characters")

            return True, self.errors, self.warnings

        except Exception as e:
            self.errors.append(f"Configuration validation error: {str(e)}")
            return False, self.errors, self.warnings

    def validate_config_dict(self, config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a configuration dictionary by converting it to TradingBotConfig.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        try:
            # Convert dict to TradingBotConfig (this will trigger Pydantic validation)
            trading_config = TradingBotConfig(**config)
            return self.validate_config(trading_config)
        except Exception as e:
            return False, [str(e)], []

    def validate_config_file(self, config_path: str) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a configuration file.

        Args:
            config_path: Path to configuration file

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        try:
            # Use the new config loader's validation function
            return pydantic_validate_config_file(config_path)
        except Exception as e:
            return False, [str(e)], []


def validate_config_file(config_file: str) -> Tuple[bool, List[str], List[str]]:
    """
    Validate a configuration file using the new Pydantic-based validation.

    Args:
        config_file: Path to configuration file

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    try:
        config_path = Path(config_file)
        if not config_path.exists():
            return False, [f"Configuration file not found: {config_file}"], []

        # Use the new config loader's validation
        return pydantic_validate_config_file(config_path)

    except Exception as e:
        return False, [f"Error validating configuration: {str(e)}"], []


def print_validation_results(is_valid: bool, errors: List[str], warnings: List[str]):
    """
    Print validation results in a formatted way.

    Args:
        is_valid: Whether the configuration is valid
        errors: List of validation errors
        warnings: List of validation warnings
    """
    print("\n" + "="*50)
    print("CONFIGURATION VALIDATION RESULTS")
    print("="*50)

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

    print("="*50 + "\n")


def create_sample_config(output_path: str = "config/trading/sample_config.json"):
    """
    Create a sample configuration file for reference.

    Args:
        output_path: Path where to save the sample configuration
    """
    try:
        from src.config.config_loader import create_sample_config as create_sample

        create_sample(output_path, "trading")
        print(f"✅ Sample configuration created: {output_path}")

    except Exception as e:
        print(f"❌ Error creating sample configuration: {e}")


def convert_old_config(old_config_path: str, new_config_path: str):
    """
    Convert an old configuration format to the new simplified format.

    Args:
        old_config_path: Path to old configuration file
        new_config_path: Path for new configuration file
    """
    try:
        from src.config.config_loader import convert_old_config as convert_config

        convert_config(old_config_path, new_config_path)
        print(f"✅ Configuration converted: {new_config_path}")

    except Exception as e:
        print(f"❌ Error converting configuration: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python config_validator.py <config_file>")
        print("Example: python config_validator.py config/trading/0001.json")
        sys.exit(1)

    config_file = sys.argv[1]
    is_valid, errors, warnings = validate_config_file(config_file)
    print_validation_results(is_valid, errors, warnings)

    if not is_valid:
        sys.exit(1)
