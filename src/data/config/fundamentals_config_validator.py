"""
Fundamentals Configuration Validator

This module provides validation for the fundamentals configuration file to ensure
it contains valid provider sequences, TTL settings, and field priorities.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

_logger = logging.getLogger(__name__)

class FundamentalsConfigValidator:
    """
    Validates fundamentals configuration file structure and content.
    """

    def __init__(self):
        """Initialize the configuration validator."""
        self.required_sections = [
            'provider_sequences',
            'refresh_intervals',
            'field_priorities',
            'provider_settings',
            'combination_strategies',
            'cache_settings',
            'data_validation'
        ]

        self.valid_data_types = [
            'statements', 'ratios', 'profile', 'calendar', 'dividends',
            'splits', 'insider_trading', 'analyst_estimates'
        ]

        self.valid_providers = [
            'fmp', 'yfinance', 'alpha_vantage', 'twelvedata', 'ibkr',
            'polygon', 'finnhub', 'tiingo', 'binance', 'coingecko'
        ]

        self.valid_combination_strategies = [
            'priority_based', 'quality_based', 'consensus'
        ]

    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate the fundamentals configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check required sections
        for section in self.required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")

        if errors:
            return False, errors

        # Validate each section
        errors.extend(self._validate_provider_sequences(config.get('provider_sequences', {})))
        errors.extend(self._validate_refresh_intervals(config.get('refresh_intervals', {})))
        errors.extend(self._validate_field_priorities(config.get('field_priorities', {})))
        errors.extend(self._validate_provider_settings(config.get('provider_settings', {})))
        errors.extend(self._validate_combination_strategies(config.get('combination_strategies', {})))
        errors.extend(self._validate_cache_settings(config.get('cache_settings', {})))
        errors.extend(self._validate_data_validation(config.get('data_validation', {})))

        return len(errors) == 0, errors

    def _validate_provider_sequences(self, sequences: Dict[str, Any]) -> List[str]:
        """Validate provider sequences section."""
        errors = []

        if not isinstance(sequences, dict):
            errors.append("provider_sequences must be a dictionary")
            return errors

        for data_type, provider_list in sequences.items():
            if data_type not in self.valid_data_types:
                errors.append(f"Invalid data type in provider_sequences: {data_type}")

            if not isinstance(provider_list, list):
                errors.append(f"Provider sequence for {data_type} must be a list")
                continue

            for provider in provider_list:
                if provider not in self.valid_providers:
                    errors.append(f"Invalid provider in {data_type} sequence: {provider}")

        return errors

    def _validate_refresh_intervals(self, intervals: Dict[str, Any]) -> List[str]:
        """Validate refresh intervals section."""
        errors = []

        if not isinstance(intervals, dict):
            errors.append("refresh_intervals must be a dictionary")
            return errors

        for data_type, interval_str in intervals.items():
            if data_type not in self.valid_data_types:
                errors.append(f"Invalid data type in refresh_intervals: {data_type}")

            if not isinstance(interval_str, str):
                errors.append(f"Refresh interval for {data_type} must be a string")
                continue

            # Validate interval format (e.g., "7d", "14d", "90d")
            if not self._is_valid_interval_format(interval_str):
                errors.append(f"Invalid interval format for {data_type}: {interval_str}")

        return errors

    def _validate_field_priorities(self, priorities: Dict[str, Any]) -> List[str]:
        """Validate field priorities section."""
        errors = []

        if not isinstance(priorities, dict):
            errors.append("field_priorities must be a dictionary")
            return errors

        # Recursively validate nested structure
        self._validate_nested_field_priorities(priorities, "", errors)

        return errors

    def _validate_nested_field_priorities(self, obj: Any, path: str, errors: List[str]):
        """Recursively validate nested field priorities structure."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                if isinstance(value, list):
                    # This is a provider list
                    for provider in value:
                        if provider not in self.valid_providers:
                            errors.append(f"Invalid provider in field priority {current_path}: {provider}")
                else:
                    # Continue recursion
                    self._validate_nested_field_priorities(value, current_path, errors)
        elif isinstance(obj, list):
            # This should be a provider list
            for provider in obj:
                if provider not in self.valid_providers:
                    errors.append(f"Invalid provider in field priority {path}: {provider}")

    def _validate_provider_settings(self, settings: Dict[str, Any]) -> List[str]:
        """Validate provider settings section."""
        errors = []

        if not isinstance(settings, dict):
            errors.append("provider_settings must be a dictionary")
            return errors

        for provider, config in settings.items():
            if provider not in self.valid_providers:
                errors.append(f"Invalid provider in provider_settings: {provider}")
                continue

            if not isinstance(config, dict):
                errors.append(f"Provider config for {provider} must be a dictionary")
                continue

            # Check required fields
            required_fields = ['name', 'priority', 'data_quality', 'reliability']
            for field in required_fields:
                if field not in config:
                    errors.append(f"Missing required field {field} for provider {provider}")

            # Validate priority is integer
            if 'priority' in config and not isinstance(config['priority'], int):
                errors.append(f"Priority for {provider} must be an integer")

            # Validate quality scores are between 1-5
            for score_field in ['data_quality', 'reliability']:
                if score_field in config:
                    score = config[score_field]
                    if not isinstance(score, (int, float)) or not (1 <= score <= 5):
                        errors.append(f"{score_field} for {provider} must be between 1 and 5")

        return errors

    def _validate_combination_strategies(self, strategies: Dict[str, Any]) -> List[str]:
        """Validate combination strategies section."""
        errors = []

        if not isinstance(strategies, dict):
            errors.append("combination_strategies must be a dictionary")
            return errors

        for strategy_name, config in strategies.items():
            if strategy_name not in self.valid_combination_strategies:
                errors.append(f"Invalid combination strategy: {strategy_name}")
                continue

            if not isinstance(config, dict):
                errors.append(f"Strategy config for {strategy_name} must be a dictionary")
                continue

        return errors

    def _validate_cache_settings(self, settings: Dict[str, Any]) -> List[str]:
        """Validate cache settings section."""
        errors = []

        if not isinstance(settings, dict):
            errors.append("cache_settings must be a dictionary")
            return errors

        # Validate default_ttl_days
        if 'default_ttl_days' in settings:
            ttl = settings['default_ttl_days']
            if not isinstance(ttl, (int, float)) or ttl <= 0:
                errors.append("default_ttl_days must be a positive number")

        # Validate max_cache_age_days
        if 'max_cache_age_days' in settings:
            max_age = settings['max_cache_age_days']
            if not isinstance(max_age, (int, float)) or max_age <= 0:
                errors.append("max_cache_age_days must be a positive number")

        return errors

    def _validate_data_validation(self, validation: Dict[str, Any]) -> List[str]:
        """Validate data validation section."""
        errors = []

        if not isinstance(validation, dict):
            errors.append("data_validation must be a dictionary")
            return errors

        # Validate min_quality_score
        if 'min_quality_score' in validation:
            score = validation['min_quality_score']
            if not isinstance(score, (int, float)) or not (0 <= score <= 1):
                errors.append("min_quality_score must be between 0 and 1")

        return errors

    def _is_valid_interval_format(self, interval_str: str) -> bool:
        """Check if interval string has valid format (e.g., '7d', '14d', '90d')."""
        if not interval_str:
            return False

        # Must end with 'd' for days
        if not interval_str.endswith('d'):
            return False

        # Must have numeric prefix
        try:
            days = int(interval_str[:-1])
            return days > 0
        except ValueError:
            return False

    def validate_config_file(self, config_path: str) -> Tuple[bool, List[str]]:
        """
        Validate configuration file.

        Args:
            config_path: Path to configuration file

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            return self.validate_config(config)

        except FileNotFoundError:
            return False, [f"Configuration file not found: {config_path}"]
        except json.JSONDecodeError as e:
            return False, [f"Invalid JSON in configuration file: {e}"]
        except Exception as e:
            return False, [f"Error reading configuration file: {e}"]


def validate_fundamentals_config(config_path: Optional[str] = None) -> Tuple[bool, List[str]]:
    """
    Validate fundamentals configuration file.

    Args:
        config_path: Path to configuration file. If None, uses default path.

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    if config_path is None:
        # Default path relative to project root
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent.parent
        config_path = project_root / 'config' / 'data' / 'fundamentals.json'

    validator = FundamentalsConfigValidator()
    return validator.validate_config_file(str(config_path))
