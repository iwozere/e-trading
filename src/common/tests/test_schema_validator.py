"""
Unit tests for AlertSchemaValidator service.

Tests JSON schema validation for alert and schedule configurations
with various valid and invalid configurations.
"""

import unittest
import json
import tempfile
import os
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.common.alerts.schema_validator import AlertSchemaValidator, ValidationResult


class TestAlertSchemaValidator(unittest.TestCase):
    """Test cases for AlertSchemaValidator functionality."""

    def setUp(self):
        """Set up test fixtures with temporary schema directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.schema_dir = Path(self.temp_dir)

        # Create test schemas
        self._create_test_schemas()

        self.validator = AlertSchemaValidator(str(self.schema_dir))

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def _create_test_schemas(self):
        """Create minimal test schemas for testing."""
        alert_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["ticker", "timeframe", "rule"],
            "properties": {
                "ticker": {
                    "type": "string",
                    "pattern": "^[A-Z0-9]+$"
                },
                "timeframe": {
                    "type": "string",
                    "enum": ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
                },
                "rule": {
                    "type": "object",
                    "required": ["indicator", "comparison", "value"],
                    "properties": {
                        "indicator": {"type": "string"},
                        "comparison": {
                            "type": "string",
                            "enum": ["gt", "gte", "lt", "lte", "eq", "ne"]
                        },
                        "value": {"type": "number"}
                    }
                },
                "rearm": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["hysteresis", "cooldown", "persistence"]
                        },
                        "value": {"type": "number"}
                    }
                },
                "options": {
                    "type": "object",
                    "properties": {
                        "lookback": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 1000
                        }
                    }
                }
            }
        }

        schedule_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["action"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["data_sync", "cleanup", "backup", "report"]
                },
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "days_back": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 365
                        }
                    }
                },
                "options": {
                    "type": "object",
                    "properties": {
                        "timeout": {
                            "type": "integer",
                            "minimum": 60,
                            "maximum": 3600
                        },
                        "retry_count": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 5
                        }
                    }
                }
            }
        }

        # Write schemas to files
        with open(self.schema_dir / "alert.json", 'w') as f:
            json.dump(alert_schema, f, indent=2)

        with open(self.schema_dir / "schedule.json", 'w') as f:
            json.dump(schedule_schema, f, indent=2)

    def test_validate_valid_alert_config(self):
        """Test validation of valid alert configurations."""
        valid_configs = [
            {
                "ticker": "BTCUSDT",
                "timeframe": "1h",
                "rule": {
                    "indicator": "rsi",
                    "comparison": "gt",
                    "value": 70
                }
            },
            {
                "ticker": "ETHUSDT",
                "timeframe": "15m",
                "rule": {
                    "indicator": "sma",
                    "comparison": "lt",
                    "value": 50000
                },
                "rearm": {
                    "type": "hysteresis",
                    "value": 5
                },
                "options": {
                    "lookback": 100
                }
            }
        ]

        for config in valid_configs:
            with self.subTest(config=config):
                result = self.validator.validate_alert_config(config)

                self.assertIsInstance(result, ValidationResult)
                self.assertTrue(result.is_valid)
                self.assertEqual(len(result.errors), 0)

    def test_validate_invalid_alert_config(self):
        """Test validation of invalid alert configurations."""
        invalid_configs = [
            # Missing required fields
            {
                "ticker": "BTCUSDT",
                "timeframe": "1h"
                # Missing rule
            },
            # Invalid ticker format
            {
                "ticker": "btc-usdt",  # Should be uppercase, no hyphens
                "timeframe": "1h",
                "rule": {
                    "indicator": "rsi",
                    "comparison": "gt",
                    "value": 70
                }
            },
            # Invalid timeframe
            {
                "ticker": "BTCUSDT",
                "timeframe": "2h",  # Not in enum
                "rule": {
                    "indicator": "rsi",
                    "comparison": "gt",
                    "value": 70
                }
            },
            # Invalid comparison operator
            {
                "ticker": "BTCUSDT",
                "timeframe": "1h",
                "rule": {
                    "indicator": "rsi",
                    "comparison": "greater_than",  # Should be "gt"
                    "value": 70
                }
            },
            # Invalid lookback value
            {
                "ticker": "BTCUSDT",
                "timeframe": "1h",
                "rule": {
                    "indicator": "rsi",
                    "comparison": "gt",
                    "value": 70
                },
                "options": {
                    "lookback": 2000  # Exceeds maximum
                }
            }
        ]

        for config in invalid_configs:
            with self.subTest(config=config):
                result = self.validator.validate_alert_config(config)

                self.assertIsInstance(result, ValidationResult)
                self.assertFalse(result.is_valid)
                self.assertGreater(len(result.errors), 0)

    def test_validate_valid_schedule_config(self):
        """Test validation of valid schedule configurations."""
        valid_configs = [
            {
                "action": "data_sync"
            },
            {
                "action": "cleanup",
                "parameters": {
                    "symbols": ["BTCUSDT", "ETHUSDT"],
                    "days_back": 30
                },
                "options": {
                    "timeout": 300,
                    "retry_count": 2
                }
            }
        ]

        for config in valid_configs:
            with self.subTest(config=config):
                result = self.validator.validate_schedule_config(config)

                self.assertIsInstance(result, ValidationResult)
                self.assertTrue(result.is_valid)
                self.assertEqual(len(result.errors), 0)

    def test_validate_invalid_schedule_config(self):
        """Test validation of invalid schedule configurations."""
        invalid_configs = [
            # Missing required action
            {
                "parameters": {
                    "symbols": ["BTCUSDT"]
                }
            },
            # Invalid action
            {
                "action": "invalid_action"
            },
            # Invalid days_back value
            {
                "action": "cleanup",
                "parameters": {
                    "days_back": 500  # Exceeds maximum
                }
            },
            # Invalid timeout value
            {
                "action": "data_sync",
                "options": {
                    "timeout": 30  # Below minimum
                }
            }
        ]

        for config in invalid_configs:
            with self.subTest(config=config):
                result = self.validator.validate_schedule_config(config)

                self.assertIsInstance(result, ValidationResult)
                self.assertFalse(result.is_valid)
                self.assertGreater(len(result.errors), 0)

    def test_validate_config_with_job_type(self):
        """Test generic validate_config method with job type parameter."""
        alert_config = {
            "ticker": "BTCUSDT",
            "timeframe": "1h",
            "rule": {
                "indicator": "rsi",
                "comparison": "gt",
                "value": 70
            }
        }

        schedule_config = {
            "action": "data_sync"
        }

        # Test alert validation
        result = self.validator.validate_config(alert_config, "alert")
        self.assertTrue(result.is_valid)

        # Test schedule validation
        result = self.validator.validate_config(schedule_config, "schedule")
        self.assertTrue(result.is_valid)

        # Test invalid job type
        result = self.validator.validate_config(alert_config, "invalid_type")
        self.assertFalse(result.is_valid)
        self.assertIn("Unknown job type", result.errors[0])

    def test_load_schema(self):
        """Test schema loading and caching."""
        # Load alert schema
        alert_schema = self.validator.load_schema("alert")
        self.assertIsNotNone(alert_schema)
        self.assertIsInstance(alert_schema, dict)
        self.assertIn("properties", alert_schema)

        # Load schedule schema
        schedule_schema = self.validator.load_schema("schedule")
        self.assertIsNotNone(schedule_schema)
        self.assertIsInstance(schedule_schema, dict)

        # Test caching - should return same object
        cached_schema = self.validator.load_schema("alert")
        self.assertIs(alert_schema, cached_schema)

        # Test non-existent schema
        missing_schema = self.validator.load_schema("nonexistent")
        self.assertIsNone(missing_schema)

    def test_schema_caching(self):
        """Test schema caching functionality."""
        # Initially no schemas cached
        self.assertEqual(len(self.validator.get_cached_schemas()), 0)

        # Load a schema
        self.validator.load_schema("alert")
        self.assertEqual(len(self.validator.get_cached_schemas()), 1)
        self.assertIn("alert", self.validator.get_cached_schemas())

        # Load another schema
        self.validator.load_schema("schedule")
        self.assertEqual(len(self.validator.get_cached_schemas()), 2)

        # Clear cache
        self.validator.clear_cache()
        self.assertEqual(len(self.validator.get_cached_schemas()), 0)

    def test_validation_error_formatting(self):
        """Test that validation errors are properly formatted."""
        invalid_config = {
            "ticker": "btc",  # Invalid format
            "timeframe": "invalid",  # Invalid enum value
            "rule": {
                "indicator": "rsi",
                "comparison": "gt"
                # Missing required 'value' field
            }
        }

        result = self.validator.validate_alert_config(invalid_config)

        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)

        # Check that error messages are descriptive
        error_text = " ".join(result.errors)
        self.assertIn("ticker", error_text.lower())
        self.assertIn("timeframe", error_text.lower())
        self.assertIn("value", error_text.lower())

    def test_warnings_generation(self):
        """Test that warnings are generated for potentially problematic configs."""
        # Alert config without rearm (should generate warning)
        config_no_rearm = {
            "ticker": "BTCUSDT",
            "timeframe": "1h",
            "rule": {
                "indicator": "rsi",
                "comparison": "gt",
                "value": 70
            }
        }

        result = self.validator.validate_alert_config(config_no_rearm)
        self.assertTrue(result.is_valid)
        self.assertGreater(len(result.warnings), 0)
        self.assertIn("rearm", result.warnings[0].lower())

        # Alert config with large lookback (should generate warning)
        config_large_lookback = {
            "ticker": "BTCUSDT",
            "timeframe": "1h",
            "rule": {
                "indicator": "rsi",
                "comparison": "gt",
                "value": 70
            },
            "options": {
                "lookback": 800  # Large but valid value
            }
        }

        result = self.validator.validate_alert_config(config_large_lookback)
        self.assertTrue(result.is_valid)
        self.assertGreater(len(result.warnings), 0)
        # Check if any warning contains "lookback"
        warning_text = " ".join(result.warnings).lower()
        self.assertIn("lookback", warning_text)

    def test_invalid_schema_directory(self):
        """Test behavior with invalid schema directory."""
        invalid_validator = AlertSchemaValidator("/nonexistent/path")

        # Should handle gracefully
        result = invalid_validator.validate_alert_config({
            "ticker": "BTCUSDT",
            "timeframe": "1h",
            "rule": {
                "indicator": "rsi",
                "comparison": "gt",
                "value": 70
            }
        })

        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)

    def test_malformed_schema_file(self):
        """Test behavior with malformed schema files."""
        # Create a malformed JSON file
        malformed_path = self.schema_dir / "malformed.json"
        with open(malformed_path, 'w') as f:
            f.write("{ invalid json }")

        # Should handle gracefully
        schema = self.validator.load_schema("malformed")
        self.assertIsNone(schema)


if __name__ == '__main__':
    unittest.main()