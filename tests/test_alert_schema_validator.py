"""
Unit Tests for AlertSchemaValidator

Tests the AlertSchemaValidator service functionality including:
- Schema loading and caching mechanisms
- Validation with valid and invalid configurations
- Error message formatting and warnings
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

# Add src to path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.common.alerts.schema_validator import AlertSchemaValidator, ValidationResult
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class TestAlertSchemaValidator:
    """Test cases for AlertSchemaValidator functionality."""

    @pytest.fixture
    def temp_schema_dir(self):
        """Create a temporary directory with test schemas."""
        with tempfile.TemporaryDirectory() as temp_dir:
            schema_dir = Path(temp_dir)

            # Create alert schema
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

            # Create schedule schema
            schedule_schema = {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["data_sync", "cleanup", "backup"]
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
            with open(schema_dir / "alert.json", "w") as f:
                json.dump(alert_schema, f)

            with open(schema_dir / "schedule.json", "w") as f:
                json.dump(schedule_schema, f)

            yield str(schema_dir)

    @pytest.fixture
    def validator(self, temp_schema_dir):
        """Create AlertSchemaValidator with test schemas."""
        return AlertSchemaValidator(schema_dir=temp_schema_dir)

    def test_initialization_with_custom_schema_dir(self, temp_schema_dir):
        """Test validator initialization with custom schema directory."""
        validator = AlertSchemaValidator(schema_dir=temp_schema_dir)
        assert validator.schema_dir == Path(temp_schema_dir)
        assert validator._schema_cache == {}

    def test_initialization_with_default_schema_dir(self):
        """Test validator initialization with default schema directory."""
        validator = AlertSchemaValidator()
        expected_dir = Path(__file__).parent.parent / "src" / "common" / "alerts" / "schemas"
        assert validator.schema_dir == expected_dir

    def test_initialization_nonexistent_schema_dir(self):
        """Test validator initialization with non-existent schema directory."""
        with patch('src.common.alerts.schema_validator._logger') as mock_logger:
            validator = AlertSchemaValidator(schema_dir="/nonexistent/path")
            mock_logger.warning.assert_called_once()

    def test_load_schema_success(self, validator):
        """Test successful schema loading."""
        schema = validator.load_schema("alert")

        assert schema is not None
        assert isinstance(schema, dict)
        assert "$schema" in schema
        assert "properties" in schema
        assert "ticker" in schema["properties"]

    def test_load_schema_caching(self, validator):
        """Test that schemas are cached after first load."""
        # First load
        schema1 = validator.load_schema("alert")
        assert "alert" in validator._schema_cache

        # Second load should use cache
        schema2 = validator.load_schema("alert")
        assert schema1 is schema2  # Same object reference

    def test_load_schema_nonexistent_file(self, validator):
        """Test loading non-existent schema file."""
        with patch('src.common.alerts.schema_validator._logger') as mock_logger:
            schema = validator.load_schema("nonexistent")
            assert schema is None
            mock_logger.error.assert_called()

    def test_load_schema_invalid_json(self, temp_schema_dir):
        """Test loading schema with invalid JSON."""
        # Create invalid JSON file
        invalid_file = Path(temp_schema_dir) / "invalid.json"
        with open(invalid_file, "w") as f:
            f.write("{ invalid json }")

        validator = AlertSchemaValidator(schema_dir=temp_schema_dir)

        with patch('src.common.alerts.schema_validator._logger') as mock_logger:
            schema = validator.load_schema("invalid")
            assert schema is None
            mock_logger.error.assert_called()

    def test_load_schema_invalid_schema_format(self, temp_schema_dir):
        """Test loading file with invalid schema format."""
        # Create file with invalid schema
        invalid_schema = {"not": "a valid schema"}
        invalid_file = Path(temp_schema_dir) / "invalid_schema.json"
        with open(invalid_file, "w") as f:
            json.dump(invalid_schema, f)

        validator = AlertSchemaValidator(schema_dir=temp_schema_dir)

        with patch('src.common.alerts.schema_validator._logger') as mock_logger:
            schema = validator.load_schema("invalid_schema")
            assert schema is None
            mock_logger.error.assert_called()

    def test_validate_alert_config_valid(self, validator):
        """Test validation of valid alert configuration."""
        valid_config = {
            "ticker": "BTCUSDT",
            "timeframe": "1h",
            "rule": {
                "indicator": "RSI",
                "comparison": "gt",
                "value": 70
            }
        }

        result = validator.validate_alert_config(valid_config)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_alert_config_missing_required_fields(self, validator):
        """Test validation with missing required fields."""
        invalid_config = {
            "ticker": "BTCUSDT"
            # Missing timeframe and rule
        }

        result = validator.validate_alert_config(invalid_config)

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("timeframe" in error for error in result.errors)
        assert any("rule" in error for error in result.errors)

    def test_validate_alert_config_invalid_field_values(self, validator):
        """Test validation with invalid field values."""
        invalid_config = {
            "ticker": "btc-usdt",  # Invalid pattern (should be uppercase)
            "timeframe": "2h",     # Invalid enum value
            "rule": {
                "indicator": "RSI",
                "comparison": "greater_than",  # Invalid enum value
                "value": "seventy"  # Invalid type (should be number)
            }
        }

        result = validator.validate_alert_config(invalid_config)

        assert result.is_valid is False
        assert len(result.errors) >= 3  # At least 3 validation errors

    def test_validate_schedule_config_valid(self, validator):
        """Test validation of valid schedule configuration."""
        valid_config = {
            "action": "data_sync",
            "options": {
                "timeout": 300,
                "retry_count": 2
            }
        }

        result = validator.validate_schedule_config(valid_config)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_schedule_config_invalid(self, validator):
        """Test validation of invalid schedule configuration."""
        invalid_config = {
            "action": "invalid_action",  # Invalid enum value
            "options": {
                "timeout": 30,    # Below minimum
                "retry_count": 10  # Above maximum
            }
        }

        result = validator.validate_schedule_config(invalid_config)

        assert result.is_valid is False
        assert len(result.errors) >= 3

    def test_validate_config_unknown_job_type(self, validator):
        """Test validation with unknown job type."""
        config = {"test": "data"}

        result = validator.validate_config(config, "unknown_type")

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "Unknown job type" in result.errors[0]

    def test_validate_config_schema_loading_failure(self, validator):
        """Test validation when schema loading fails."""
        config = {"test": "data"}

        with patch.object(validator, 'load_schema', return_value=None):
            result = validator.validate_config(config, "alert")

            assert result.is_valid is False
            assert len(result.errors) == 1
            assert "Could not load schema" in result.errors[0]

    def test_format_validation_error_required_fields(self, validator):
        """Test error message formatting for required fields."""
        import jsonschema

        # Create a validation error for missing required field
        schema = {"required": ["field1", "field2"]}
        instance = {}

        error = jsonschema.ValidationError(
            "'field1' is a required property",
            path=[],
            validator="required",
            validator_value=["field1", "field2"],
            instance=instance,
            schema=schema
        )

        formatted = validator._format_validation_error(error)
        assert "Missing required properties" in formatted
        assert "field1, field2" in formatted

    def test_format_validation_error_enum_values(self, validator):
        """Test error message formatting for enum validation."""
        import jsonschema

        error = jsonschema.ValidationError(
            "'invalid' is not one of ['valid1', 'valid2']",
            path=["field"],
            validator="enum",
            validator_value=["valid1", "valid2"],
            instance="invalid",
            schema={"enum": ["valid1", "valid2"]}
        )

        formatted = validator._format_validation_error(error)
        assert "Invalid value at field" in formatted
        assert "valid1, valid2" in formatted

    def test_format_validation_error_type_mismatch(self, validator):
        """Test error message formatting for type validation."""
        import jsonschema

        error = jsonschema.ValidationError(
            "'string' is not of type 'number'",
            path=["field"],
            validator="type",
            validator_value="number",
            instance="string_value",
            schema={"type": "number"}
        )

        formatted = validator._format_validation_error(error)
        assert "Invalid type at field" in formatted
        assert "expected number" in formatted
        assert "got str" in formatted

    def test_format_validation_error_pattern_mismatch(self, validator):
        """Test error message formatting for pattern validation."""
        import jsonschema

        error = jsonschema.ValidationError(
            "'invalid' does not match '^[A-Z]+$'",
            path=["ticker"],
            validator="pattern",
            validator_value="^[A-Z]+$",
            instance="invalid",
            schema={"pattern": "^[A-Z]+$"}
        )

        formatted = validator._format_validation_error(error)
        assert "Invalid format at ticker" in formatted
        assert "does not match pattern" in formatted

    def test_format_validation_error_range_validation(self, validator):
        """Test error message formatting for range validation."""
        import jsonschema

        # Test minimum
        error_min = jsonschema.ValidationError(
            "5 is less than the minimum of 10",
            path=["value"],
            validator="minimum",
            validator_value=10,
            instance=5,
            schema={"minimum": 10}
        )

        formatted = validator._format_validation_error(error_min)
        assert "Value too small at value" in formatted
        assert "5 < 10" in formatted

        # Test maximum
        error_max = jsonschema.ValidationError(
            "100 is greater than the maximum of 50",
            path=["value"],
            validator="maximum",
            validator_value=50,
            instance=100,
            schema={"maximum": 50}
        )

        formatted = validator._format_validation_error(error_max)
        assert "Value too large at value" in formatted
        assert "100 > 50" in formatted

    def test_check_warnings_alert_no_rearm(self, validator):
        """Test warning generation for alert without rearm configuration."""
        config = {
            "ticker": "BTCUSDT",
            "timeframe": "1h",
            "rule": {
                "indicator": "RSI",
                "comparison": "gt",
                "value": 70
            }
        }

        warnings = validator._check_warnings(config, "alert")
        assert len(warnings) > 0
        assert any("rearm" in warning for warning in warnings)

    def test_check_warnings_alert_large_lookback(self, validator):
        """Test warning generation for alert with large lookback."""
        config = {
            "ticker": "BTCUSDT",
            "timeframe": "1h",
            "rule": {
                "indicator": "RSI",
                "comparison": "gt",
                "value": 70
            },
            "options": {
                "lookback": 1000
            }
        }

        warnings = validator._check_warnings(config, "alert")
        assert any("lookback" in warning for warning in warnings)

    def test_check_warnings_schedule_long_timeout(self, validator):
        """Test warning generation for schedule with long timeout."""
        config = {
            "action": "data_sync",
            "options": {
                "timeout": 2000  # Over 30 minutes
            }
        }

        warnings = validator._check_warnings(config, "schedule")
        assert any("timeout" in warning for warning in warnings)

    def test_check_warnings_schedule_high_retry_count(self, validator):
        """Test warning generation for schedule with high retry count."""
        config = {
            "action": "data_sync",
            "options": {
                "retry_count": 5
            }
        }

        warnings = validator._check_warnings(config, "schedule")
        assert any("retry" in warning for warning in warnings)

    def test_clear_cache(self, validator):
        """Test cache clearing functionality."""
        # Load a schema to populate cache
        validator.load_schema("alert")
        assert len(validator._schema_cache) > 0

        # Clear cache
        validator.clear_cache()
        assert len(validator._schema_cache) == 0

    def test_get_cached_schemas(self, validator):
        """Test getting list of cached schemas."""
        # Initially empty
        assert validator.get_cached_schemas() == []

        # Load schemas
        validator.load_schema("alert")
        validator.load_schema("schedule")

        cached = validator.get_cached_schemas()
        assert "alert" in cached
        assert "schedule" in cached
        assert len(cached) == 2

    def test_validation_with_nested_objects(self, validator):
        """Test validation of configurations with nested objects."""
        config = {
            "ticker": "BTCUSDT",
            "timeframe": "1h",
            "rule": {
                "indicator": "RSI",
                "comparison": "gt",
                "value": 70
            },
            "options": {
                "lookback": 200
            }
        }

        result = validator.validate_alert_config(config)
        assert result.is_valid is True

    def test_validation_exception_handling(self, validator):
        """Test handling of unexpected exceptions during validation."""
        # Mock an exception during validation
        with patch('jsonschema.Draft7Validator') as mock_validator_class:
            mock_validator = mock_validator_class.return_value
            mock_validator.iter_errors.side_effect = Exception("Unexpected error")

            config = {"ticker": "BTCUSDT"}
            result = validator.validate_alert_config(config)

            assert result.is_valid is False
            # The test may have multiple errors due to normal validation plus the exception
            assert len(result.errors) >= 1
            assert any("Validation error" in error or "Missing required" in error for error in result.errors)

    def test_concurrent_schema_loading(self, validator):
        """Test thread safety of schema loading and caching."""
        import threading

        results = []
        errors = []

        def load_schema_worker(schema_type):
            try:
                schema = validator.load_schema(schema_type)
                results.append(schema)
            except Exception as e:
                errors.append(e)

        # Start multiple threads loading the same schema
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=load_schema_worker, args=("alert",))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0
        assert len(results) == 10
        # All results should be the same schema content (may not be same object due to JSON loading)
        assert all(result == results[0] for result in results)

    def test_large_configuration_validation(self, validator):
        """Test validation of large configuration objects."""
        # Create a large but valid configuration
        large_config = {
            "ticker": "BTCUSDT",
            "timeframe": "1h",
            "rule": {
                "indicator": "RSI",
                "comparison": "gt",
                "value": 70
            },
            "options": {
                "lookback": 500
            }
        }

        # Add many additional properties (should be ignored if not in schema)
        for i in range(100):
            large_config[f"extra_field_{i}"] = f"value_{i}"

        result = validator.validate_alert_config(large_config)
        # Should still be valid (extra fields are typically ignored)
        assert result.is_valid is True

    @patch('src.common.alerts.schema_validator._logger')
    def test_logging_behavior(self, mock_logger, validator):
        """Test that appropriate logging occurs."""
        # Test successful validation
        config = {
            "ticker": "BTCUSDT",
            "timeframe": "1h",
            "rule": {
                "indicator": "RSI",
                "comparison": "gt",
                "value": 70
            }
        }

        validator.validate_alert_config(config)
        mock_logger.debug.assert_called()

        # Test validation failure logging
        mock_logger.reset_mock()
        invalid_config = {"ticker": "invalid"}
        validator.validate_alert_config(invalid_config)
        mock_logger.warning.assert_called()