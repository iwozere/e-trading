"""
Unit tests for the configuration manager.

Tests YAML loading, validation, environment variable substitution,
and type-safe configuration object creation.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.ml.pipeline.p04_short_squeeze.config.config_manager import (
    ConfigManager, ConfigValidationError
)
from src.ml.pipeline.p04_short_squeeze.config.data_classes import (
    PipelineConfig, SchedulingConfig, ScreenerConfig
)


class TestConfigManager(unittest.TestCase):
    """Test cases for ConfigManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config_manager = ConfigManager()

        # Sample valid configuration
        self.valid_config = {
            'scheduling': {
                'screener': {'frequency': 'weekly', 'day': 'monday', 'time': '08:00'},
                'deep_scan': {'frequency': 'daily', 'time': '10:00'},
                'timezone': 'Europe/Zurich'
            },
            'screener': {
                'universe': {
                    'min_market_cap': 100_000_000,
                    'max_market_cap': 10_000_000_000,
                    'min_avg_volume': 200_000,
                    'exchanges': ['NYSE', 'NASDAQ']
                },
                'filters': {
                    'si_percent_min': 0.15,
                    'days_to_cover_min': 5.0,
                    'float_max': 100_000_000,
                    'top_k_candidates': 50
                },
                'scoring': {
                    'weights': {
                        'short_interest_pct': 0.4,
                        'days_to_cover': 0.3,
                        'float_ratio': 0.2,
                        'volume_consistency': 0.1
                    }
                }
            },
            'deep_scan': {
                'batch_size': 10,
                'api_delay_seconds': 0.2,
                'metrics': {
                    'volume_lookback_days': 14,
                    'sentiment_lookback_hours': 24,
                    'options_min_volume': 100
                },
                'scoring': {
                    'weights': {
                        'volume_spike': 0.35,
                        'sentiment_24h': 0.25,
                        'call_put_ratio': 0.20,
                        'borrow_fee': 0.20
                    }
                }
            },
            'alerting': {
                'thresholds': {
                    'high': {'squeeze_score': 0.8, 'min_si_percent': 0.25, 'min_volume_spike': 4.0, 'min_sentiment': 0.6},
                    'medium': {'squeeze_score': 0.6, 'min_si_percent': 0.20, 'min_volume_spike': 3.0, 'min_sentiment': 0.5},
                    'low': {'squeeze_score': 0.4, 'min_si_percent': 0.15, 'min_volume_spike': 2.0, 'min_sentiment': 0.4}
                },
                'cooldown': {'high_alert_days': 7, 'medium_alert_days': 5, 'low_alert_days': 3},
                'channels': {
                    'telegram': {'enabled': True, 'chat_ids': ['@trading_alerts']},
                    'email': {'enabled': True, 'recipients': ['trader@example.com']}
                }
            }
        }

    def test_build_config_from_dict_valid(self):
        """Test building configuration from valid dictionary."""
        config = self.config_manager._build_config_from_dict(self.valid_config)

        self.assertIsInstance(config, PipelineConfig)
        self.assertIsInstance(config.scheduling, SchedulingConfig)
        self.assertIsInstance(config.screener, ScreenerConfig)

        # Test specific values
        self.assertEqual(config.scheduling.timezone, 'Europe/Zurich')
        self.assertEqual(config.screener.universe.min_market_cap, 100_000_000)
        self.assertEqual(config.screener.filters.si_percent_min, 0.15)

    def test_validate_config_valid(self):
        """Test validation of valid configuration."""
        config = self.config_manager._build_config_from_dict(self.valid_config)

        # Should not raise any exception
        self.config_manager._validate_config(config)

    def test_validate_config_invalid_weights(self):
        """Test validation fails for invalid scoring weights."""
        invalid_config = self.valid_config.copy()
        invalid_config['screener']['scoring']['weights']['short_interest_pct'] = 0.8  # Makes total > 1.0

        config = self.config_manager._build_config_from_dict(invalid_config)

        with self.assertRaises(ConfigValidationError) as context:
            self.config_manager._validate_config(config)

        self.assertIn("weights sum to", str(context.exception))

    def test_validate_config_invalid_market_cap(self):
        """Test validation fails for invalid market cap."""
        invalid_config = self.valid_config.copy()
        invalid_config['screener']['universe']['min_market_cap'] = -1000

        config = self.config_manager._build_config_from_dict(invalid_config)

        with self.assertRaises(ConfigValidationError) as context:
            self.config_manager._validate_config(config)

        self.assertIn("market cap must be positive", str(context.exception))

    def test_validate_config_invalid_alert_thresholds(self):
        """Test validation fails for invalid alert threshold order."""
        invalid_config = self.valid_config.copy()
        invalid_config['alerting']['thresholds']['low']['squeeze_score'] = 0.9  # Higher than high threshold

        config = self.config_manager._build_config_from_dict(invalid_config)

        with self.assertRaises(ConfigValidationError) as context:
            self.config_manager._validate_config(config)

        self.assertIn("ascending order", str(context.exception))

    def test_substitute_env_vars_simple(self):
        """Test simple environment variable substitution."""
        test_config = {
            'test_value': '${TEST_VAR}',
            'nested': {
                'another_value': '${ANOTHER_VAR}'
            }
        }

        with patch.dict(os.environ, {'TEST_VAR': 'test_result', 'ANOTHER_VAR': 'another_result'}):
            result = self.config_manager._substitute_env_vars(test_config)

        self.assertEqual(result['test_value'], 'test_result')
        self.assertEqual(result['nested']['another_value'], 'another_result')

    def test_substitute_env_vars_with_default(self):
        """Test environment variable substitution with default values."""
        test_config = {
            'with_default': '${MISSING_VAR:default_value}',
            'without_default': '${EXISTING_VAR}'
        }

        with patch.dict(os.environ, {'EXISTING_VAR': 'existing_value'}, clear=True):
            result = self.config_manager._substitute_env_vars(test_config)

        self.assertEqual(result['with_default'], 'default_value')
        self.assertEqual(result['without_default'], 'existing_value')

    def test_substitute_env_vars_missing_required(self):
        """Test environment variable substitution fails for missing required vars."""
        test_config = {'required_var': '${MISSING_REQUIRED_VAR}'}

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ConfigValidationError) as context:
                self.config_manager._substitute_env_vars(test_config)

            self.assertIn("Environment variable MISSING_REQUIRED_VAR not found", str(context.exception))

    def test_load_config_from_file(self):
        """Test loading configuration from YAML file."""
        import yaml

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.valid_config, f)
            temp_path = f.name

        try:
            config = self.config_manager.load_config(temp_path)

            self.assertIsInstance(config, PipelineConfig)
            self.assertEqual(config.scheduling.timezone, 'Europe/Zurich')
            self.assertIsNotNone(config.run_id)
            self.assertIsNotNone(config.created_at)

        finally:
            os.unlink(temp_path)

    def test_load_config_file_not_found(self):
        """Test loading configuration from non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.config_manager.load_config('/non/existent/path.yaml')

    def test_get_config_before_loading(self):
        """Test getting configuration before loading raises error."""
        with self.assertRaises(RuntimeError) as context:
            self.config_manager.get_config()

        self.assertIn("Configuration not loaded", str(context.exception))

    def test_get_specific_configs(self):
        """Test getting specific configuration sections."""
        config = self.config_manager._build_config_from_dict(self.valid_config)
        self.config_manager._config = config

        screener_config = self.config_manager.get_screener_config()
        deep_scan_config = self.config_manager.get_deep_scan_config()
        alert_config = self.config_manager.get_alert_config()
        scheduling_config = self.config_manager.get_scheduling_config()

        self.assertIsInstance(screener_config, ScreenerConfig)
        self.assertEqual(screener_config.universe.min_market_cap, 100_000_000)
        self.assertEqual(deep_scan_config.batch_size, 10)
        self.assertEqual(alert_config.cooldown.high_alert_days, 7)
        self.assertEqual(scheduling_config.timezone, 'Europe/Zurich')

    def test_export_config_to_dict(self):
        """Test exporting configuration to dictionary."""
        config = self.config_manager._build_config_from_dict(self.valid_config)
        self.config_manager._config = config

        exported = self.config_manager.export_config_to_dict()

        self.assertIsInstance(exported, dict)
        self.assertIn('scheduling', exported)
        self.assertIn('screener', exported)
        self.assertEqual(exported['scheduling']['timezone'], 'Europe/Zurich')


if __name__ == '__main__':
    unittest.main()