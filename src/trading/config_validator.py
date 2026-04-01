"""
Configuration Validator Module
-----------------------------

Unified validation for trading bot configurations.
This module enforces the single flat config format and uses the same
schema-validation path as the DB-driven runner.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

from src.notification.logger import setup_logger
from src.config.configuration_factory import config_factory
from src.trading.services.schema_validator import validate_bot_configuration

_logger = setup_logger(__name__)


class ConfigValidator:
    """Validate trading configurations via the unified schema path."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        self.errors = []
        self.warnings = []
        try:
            is_valid, errors, warnings = validate_bot_configuration(config)
            self.errors.extend(errors)
            self.warnings.extend(warnings)
            return is_valid, self.errors, self.warnings
        except Exception as e:
            self.errors.append(f"Configuration validation error: {str(e)}")
            return False, self.errors, self.warnings

    def validate_config_dict(self, config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        try:
            # Enforce unified shape and references through the same loader.
            hydrated = config_factory.load_manifest(config)
            return self.validate_config(hydrated)
        except Exception as e:
            return False, [str(e)], []

    def validate_config_file(self, config_path: str) -> Tuple[bool, List[str], List[str]]:
        return validate_config_file(config_path)


def validate_config_file(config_file: str) -> Tuple[bool, List[str], List[str]]:
    """Validate a file using the same path as DB/runtime loading."""
    try:
        config_path = Path(config_file)
        if not config_path.exists():
            return False, [f"Configuration file not found: {config_file}"], []

        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        hydrated = config_factory.load_manifest(raw)
        is_valid, errors, warnings = validate_bot_configuration(hydrated)
        return is_valid, errors, warnings
    except Exception as e:
        return False, [f"Error validating configuration: {str(e)}"], []


def print_validation_results(is_valid: bool, errors: List[str], warnings: List[str]):
    print("\n" + "=" * 50)
    print("CONFIGURATION VALIDATION RESULTS")
    print("=" * 50)

    if is_valid:
        print("Configuration is VALID")
    else:
        print("Configuration is INVALID")

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")

    if not errors and not warnings:
        print("\nNo issues found")

    print("=" * 50 + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python config_validator.py <config_file>")
        print("Example: python config_validator.py config/trading/0001.json")
        sys.exit(1)

    config_file = sys.argv[1]
    ok, errs, warns = validate_config_file(config_file)
    print_validation_results(ok, errs, warns)
    if not ok:
        sys.exit(1)
