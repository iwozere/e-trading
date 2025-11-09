"""
Fundamentals Data Combiner

This module provides strategies for combining fundamentals data from multiple providers.
It implements priority-based field selection and data validation using configuration-based
provider sequences and field-specific priorities.

Configuration is loaded from config/data/fundamentals.json and provides:
- Provider sequences for different data types
- Field-specific provider priorities
- TTL settings for different data categories
- Combination strategies and validation rules
"""

import logging
import json
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

_logger = logging.getLogger(__name__)

@dataclass
class ProviderData:
    """Container for provider-specific fundamentals data."""
    provider: str
    data: Dict[str, Any]
    quality_score: float
    timestamp: datetime
    priority: int

class FundamentalsCombiner:
    """
    Combines fundamentals data from multiple providers using configurable strategies.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the fundamentals combiner with configuration-based settings.

        Args:
            config_path: Path to fundamentals configuration file. If None, uses default path.
        """
        self.config = self._load_configuration(config_path)
        self.provider_priorities = self._build_provider_priorities()
        self.provider_sequences = self.config.get('provider_sequences', {})
        self.field_priorities = self.config.get('field_priorities', {})
        self.combination_strategies = self.config.get('combination_strategies', {})
        self.cache_settings = self.config.get('cache_settings', {})
        self.data_validation = self.config.get('data_validation', {})

        # Field-specific validation rules
        self.field_validators = {
            'market_cap': self._validate_positive_number,
            'pe_ratio': self._validate_pe_ratio,
            'pb_ratio': self._validate_positive_number,
            'dividend_yield': self._validate_percentage,
            'revenue': self._validate_positive_number,
            'net_income': self._validate_number,
            'total_debt': self._validate_positive_number,
            'cash': self._validate_positive_number,
            'shares_outstanding': self._validate_positive_number,
            'book_value': self._validate_positive_number
        }

    def _load_configuration(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load fundamentals configuration from JSON file with validation.

        Args:
            config_path: Path to configuration file. If None, uses default path.

        Returns:
            Configuration dictionary
        """
        if config_path is None:
            # Default path relative to project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            config_path = os.path.join(project_root, 'config', 'data', 'fundamentals.json')

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Validate configuration
            from src.data.config.fundamentals_config_validator import validate_fundamentals_config
            is_valid, errors = validate_fundamentals_config(config_path)

            if not is_valid:
                _logger.error("Fundamentals configuration validation failed:")
                for error in errors:
                    _logger.error("  - %s", error)
                _logger.warning("Using default configuration due to validation errors")
                return self._get_default_config()

            _logger.debug("Loaded and validated fundamentals configuration from %s", config_path)
            return config

        except FileNotFoundError:
            _logger.warning("Fundamentals configuration file not found at %s, using defaults", config_path)
            return self._get_default_config()
        except json.JSONDecodeError:
            _logger.exception("Invalid JSON in fundamentals configuration file %s:", config_path)
            return self._get_default_config()
        except Exception:
            _logger.exception("Error loading fundamentals configuration from %s", config_path)
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration when file loading fails."""
        return {
            'provider_sequences': {
                'statements': ['fmp', 'alphavantage', 'yfinance', 'twelvedata'],
                'ratios': ['yfinance', 'fmp', 'alphavantage', 'twelvedata'],
                'profile': ['fmp', 'yfinance', 'alphavantage', 'twelvedata']
            },
            'field_priorities': {},
            'combination_strategies': {
                'priority_based': {'default': True}
            },
            'cache_settings': {
                'default_ttl_days': 7
            },
            'data_validation': {
                'enabled': True,
                'min_quality_score': 0.8
            }
        }

    def _build_provider_priorities(self) -> Dict[str, int]:
        """
        Build provider priorities from configuration.

        Returns:
            Dictionary mapping provider names to priority numbers (lower = higher priority)
        """
        priorities = {}
        provider_settings = self.config.get('provider_settings', {})

        for provider, settings in provider_settings.items():
            priorities[provider] = settings.get('priority', 999)

        # Fallback to default priorities if not configured
        if not priorities:
            priorities = {
                'fmp': 1,
                'yfinance': 2,
                'alpha_vantage': 3,
                'ibkr': 4,
                'polygon': 5,
                'twelvedata': 6,
                'finnhub': 7,
                'tiingo': 8,
                'binance': 9,
                'coingecko': 10
            }

        return priorities

    def get_provider_sequence(self, data_type: str) -> List[str]:
        """
        Get provider sequence for a specific data type.

        Args:
            data_type: Type of data (statements, ratios, profile, etc.)

        Returns:
            List of providers in priority order
        """
        return self.provider_sequences.get(data_type, ['fmp', 'yfinance', 'alpha_vantage'])

    def get_field_provider_priority(self, field_path: str) -> List[str]:
        """
        Get provider priority for a specific field.

        Args:
            field_path: Dot-separated field path (e.g., 'ttm_metrics.pe_ratio')

        Returns:
            List of providers in priority order for this field
        """
        # Navigate through nested field priorities
        current = self.field_priorities
        for part in field_path.split('.'):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                # Fall back to general provider sequence
                return self.get_provider_sequence('ratios' if 'ratio' in field_path.lower() else 'profile')

        if isinstance(current, list):
            return current
        else:
            # Fall back to general provider sequence
            return self.get_provider_sequence('ratios' if 'ratio' in field_path.lower() else 'profile')

    def get_ttl_for_data_type(self, data_type: str) -> int:
        """
        Get TTL in days for a specific data type.

        Args:
            data_type: Type of data (profiles, ratios, statements, etc.)

        Returns:
            TTL in days
        """
        refresh_intervals = self.config.get('refresh_intervals', {})
        interval_str = refresh_intervals.get(data_type, '7d')

        # Parse interval string (e.g., '7d', '14d', '90d')
        if interval_str.endswith('d'):
            return int(interval_str[:-1])
        elif interval_str.endswith('h'):
            return int(interval_str[:-1]) / 24
        else:
            return 7  # Default to 7 days

    def combine_snapshots(self, provider_data: Dict[str, Dict[str, Any]],
                         strategy: str = "priority_based", data_type: str = "general") -> Dict[str, Any]:
        """
        Combine fundamentals data from multiple providers.

        Args:
            provider_data: Dictionary mapping provider names to their data
            strategy: Combination strategy ('priority_based', 'quality_based', 'consensus')

        Returns:
            Combined fundamentals data dictionary
        """
        if not provider_data:
            return {}

        # Convert to ProviderData objects
        providers = []
        for provider, data in provider_data.items():
            if data:  # Skip empty data
                providers.append(ProviderData(
                    provider=provider,
                    data=data,
                    quality_score=self._calculate_data_quality(data),
                    timestamp=datetime.now(),  # Could be extracted from data if available
                    priority=self.provider_priorities.get(provider, 999)
                ))

        if not providers:
            return {}

        # Sort by priority
        providers.sort(key=lambda x: x.priority)

        # Apply combination strategy
        if strategy == "priority_based":
            return self._priority_based_combination(providers)
        elif strategy == "quality_based":
            return self._quality_based_combination(providers)
        elif strategy == "consensus":
            return self._consensus_combination(providers)
        else:
            _logger.warning("Unknown combination strategy: %s, using priority_based", strategy)
            return self._priority_based_combination(providers)

    def _priority_based_combination(self, providers: List[ProviderData]) -> Dict[str, Any]:
        """
        Combine data using priority-based field selection with field-specific priorities.
        For each field, use the provider with the highest priority for that specific field.
        """
        combined = {}
        field_sources = {}

        # Get all unique fields across all providers
        all_fields = set()
        for provider in providers:
            all_fields.update(provider.data.keys())

        # For each field, find the best provider based on field-specific priority
        for field in all_fields:
            best_value = None
            best_provider = None
            best_priority = float('inf')

            # Get field-specific provider priority
            field_priority_list = self.get_field_provider_priority(field)

            for provider in providers:
                if field in provider.data:
                    value = provider.data[field]
                    if value is not None and self._is_valid_field_value(field, value):
                        # Get priority for this provider for this field
                        try:
                            field_priority = field_priority_list.index(provider.provider)
                        except ValueError:
                            # Provider not in field priority list, use general priority
                            field_priority = self.provider_priorities.get(provider.provider, 999)

                        if field_priority < best_priority:
                            best_value = value
                            best_provider = provider.provider
                            best_priority = field_priority

            if best_value is not None:
                combined[field] = best_value
                field_sources[field] = best_provider

        # Add metadata about data sources
        combined['_metadata'] = {
            'combination_strategy': 'priority_based',
            'field_sources': field_sources,
            'providers_used': [p.provider for p in providers],
            'combination_timestamp': datetime.now().isoformat()
        }

        _logger.debug("Combined fundamentals using priority-based strategy: %d fields from %d providers",
                     len(combined), len(providers))

        return combined

    def _quality_based_combination(self, providers: List[ProviderData]) -> Dict[str, Any]:
        """
        Combine data using quality-based field selection.
        For each field, select the value from the provider with the highest quality score.
        """
        combined = {}
        field_sources = {}

        # Group fields by name across all providers
        all_fields = set()
        for provider in providers:
            all_fields.update(provider.data.keys())

        # For each field, select the best value
        for field in all_fields:
            best_value = None
            best_provider = None
            best_quality = -1

            for provider in providers:
                if field in provider.data:
                    value = provider.data[field]
                    if value is not None and self._is_valid_field_value(field, value):
                        # Use quality score as tiebreaker, then priority
                        quality_score = provider.quality_score
                        if quality_score > best_quality or (quality_score == best_quality and
                                                          provider.priority < self.provider_priorities.get(best_provider, 999)):
                            best_value = value
                            best_provider = provider.provider
                            best_quality = quality_score

            if best_value is not None:
                combined[field] = best_value
                field_sources[field] = best_provider

        # Add metadata
        combined['_metadata'] = {
            'combination_strategy': 'quality_based',
            'field_sources': field_sources,
            'providers_used': [p.provider for p in providers],
            'combination_timestamp': datetime.now().isoformat()
        }

        _logger.debug("Combined fundamentals using quality-based strategy: %d fields from %d providers",
                     len(combined), len(providers))

        return combined

    def _consensus_combination(self, providers: List[ProviderData]) -> Dict[str, Any]:
        """
        Combine data using consensus-based selection.
        For numeric fields, use average if values are close, otherwise use highest priority.
        """
        combined = {}
        field_sources = {}

        # Group fields by name across all providers
        all_fields = set()
        for provider in providers:
            all_fields.update(provider.data.keys())

        for field in all_fields:
            values = []
            providers_with_field = []

            # Collect valid values for this field
            for provider in providers:
                if field in provider.data:
                    value = provider.data[field]
                    if value is not None and self._is_valid_field_value(field, value):
                        values.append(value)
                        providers_with_field.append(provider)

            if not values:
                continue

            # For numeric fields, check for consensus
            if self._is_numeric_field(field) and len(values) > 1:
                consensus_value = self._calculate_consensus_value(values, providers_with_field)
                if consensus_value is not None:
                    combined[field] = consensus_value
                    field_sources[field] = 'consensus'
                    continue

            # Fall back to highest priority provider
            best_provider = min(providers_with_field, key=lambda x: x.priority)
            combined[field] = best_provider.data[field]
            field_sources[field] = best_provider.provider

        # Add metadata
        combined['_metadata'] = {
            'combination_strategy': 'consensus',
            'field_sources': field_sources,
            'providers_used': [p.provider for p in providers],
            'combination_timestamp': datetime.now().isoformat()
        }

        _logger.debug("Combined fundamentals using consensus strategy: %d fields from %d providers",
                     len(combined), len(providers))

        return combined

    def _calculate_data_quality(self, data: Dict[str, Any]) -> float:
        """
        Calculate data quality score based on available fields and their values.

        Args:
            data: Fundamentals data dictionary

        Returns:
            Quality score between 0.0 and 1.0
        """
        if not data:
            return 0.0

        # Define important fields and their weights
        important_fields = {
            'market_cap': 0.15,
            'pe_ratio': 0.12,
            'pb_ratio': 0.12,
            'dividend_yield': 0.08,
            'revenue': 0.10,
            'net_income': 0.10,
            'total_debt': 0.08,
            'cash': 0.08,
            'shares_outstanding': 0.08,
            'book_value': 0.09
        }

        score = 0.0
        for field, weight in important_fields.items():
            if field in data and data[field] is not None:
                if self._is_valid_field_value(field, data[field]):
                    score += weight

        return min(score, 1.0)

    def _is_valid_field_value(self, field: str, value: Any) -> bool:
        """
        Check if a field value is valid using field-specific validators.

        Args:
            field: Field name
            value: Field value

        Returns:
            True if value is valid, False otherwise
        """
        if value is None:
            return False

        # Use field-specific validator if available
        if field in self.field_validators:
            try:
                return self.field_validators[field](value)
            except Exception as e:
                _logger.debug("Validation failed for field %s: %s", field, e)
                return False

        # Default validation
        return True

    def _is_numeric_field(self, field: str) -> bool:
        """Check if a field is numeric."""
        numeric_fields = {
            'market_cap', 'pe_ratio', 'pb_ratio', 'dividend_yield',
            'revenue', 'net_income', 'total_debt', 'cash',
            'shares_outstanding', 'book_value', 'eps', 'roe', 'roa'
        }
        return field in numeric_fields

    def _calculate_consensus_value(self, values: List[Any], providers: List[ProviderData]) -> Optional[Any]:
        """
        Calculate consensus value for numeric fields.

        Args:
            values: List of values from different providers
            providers: List of providers that provided the values

        Returns:
            Consensus value or None if no consensus
        """
        if len(values) < 2:
            return values[0] if values else None

        try:
            # Convert to float for comparison
            float_values = [float(v) for v in values]

            # Check if values are close (within 10% of each other)
            min_val = min(float_values)
            max_val = max(float_values)

            if min_val == 0:
                # Avoid division by zero
                return float_values[0] if max_val == 0 else None

            # Check if values are within 10% of each other
            if (max_val - min_val) / min_val <= 0.1:
                # Use average of all values
                return sum(float_values) / len(float_values)
            else:
                # No consensus, return None to fall back to priority-based selection
                return None

        except (ValueError, TypeError):
            # Non-numeric values, no consensus possible
            return None

    # Field-specific validators
    def _validate_positive_number(self, value: Any) -> bool:
        """Validate that value is a positive number."""
        try:
            num = float(value)
            return num > 0
        except (ValueError, TypeError):
            return False

    def _validate_number(self, value: Any) -> bool:
        """Validate that value is a number."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def _validate_pe_ratio(self, value: Any) -> bool:
        """Validate PE ratio (should be positive or reasonable negative)."""
        try:
            num = float(value)
            # PE ratio can be negative for companies with losses
            return -1000 < num < 1000
        except (ValueError, TypeError):
            return False

    def _validate_percentage(self, value: Any) -> bool:
        """Validate percentage value (0-100)."""
        try:
            num = float(value)
            return 0 <= num <= 100
        except (ValueError, TypeError):
            return False


# Global combiner instance
_fundamentals_combiner = None

def get_fundamentals_combiner(config_path: Optional[str] = None) -> FundamentalsCombiner:
    """
    Get the global fundamentals combiner instance.

    Args:
        config_path: Path to fundamentals configuration file. If None, uses default path.

    Returns:
        FundamentalsCombiner instance
    """
    global _fundamentals_combiner
    if _fundamentals_combiner is None:
        _fundamentals_combiner = FundamentalsCombiner(config_path)
    return _fundamentals_combiner
