"""
JSON schema validator for alert and schedule configurations.

This module provides validation services for task_params in job_schedules table,
ensuring configurations conform to expected schemas before execution.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import jsonschema
from jsonschema import Draft7Validator

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


@dataclass
class ValidationResult:
    """Result of schema validation with error details."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class AlertSchemaValidator:
    """
    Validator for alert and schedule configurations using JSON Schema.

    Provides caching of schemas and detailed error reporting for invalid configurations.
    """

    def __init__(self, schema_dir: Optional[str] = None):
        """
        Initialize the schema validator.

        Args:
            schema_dir: Directory containing JSON schema files (defaults to schemas/ subdirectory)
        """
        if schema_dir is None:
            # Default to schemas directory relative to this file
            current_dir = Path(__file__).parent
            schema_dir = current_dir / "schemas"

        self.schema_dir = Path(schema_dir)
        self._schema_cache: Dict[str, Dict[str, Any]] = {}

        if not self.schema_dir.exists():
            _logger.warning("Schema directory does not exist: %s", self.schema_dir)

        _logger.debug("Initialized AlertSchemaValidator with schema_dir: %s", self.schema_dir)

    def validate_alert_config(self, task_params: Dict[str, Any]) -> ValidationResult:
        """
        Validate alert configuration against alert schema.

        Args:
            task_params: Alert configuration dictionary

        Returns:
            ValidationResult with validation status and error details
        """
        return self._validate_config(task_params, "alert")

    def validate_schedule_config(self, task_params: Dict[str, Any]) -> ValidationResult:
        """
        Validate schedule configuration against schedule schema.

        Args:
            task_params: Schedule configuration dictionary

        Returns:
            ValidationResult with validation status and error details
        """
        return self._validate_config(task_params, "schedule")

    def validate_config(self, task_params: Dict[str, Any], job_type: str) -> ValidationResult:
        """
        Validate configuration against the appropriate schema based on job type.

        Args:
            task_params: Configuration dictionary
            job_type: Type of job ('alert' or 'schedule')

        Returns:
            ValidationResult with validation status and error details
        """
        if job_type not in ["alert", "schedule"]:
            return ValidationResult(
                is_valid=False,
                errors=[f"Unknown job type: {job_type}. Must be 'alert' or 'schedule'"],
                warnings=[]
            )

        return self._validate_config(task_params, job_type)

    def _validate_config(self, task_params: Dict[str, Any], schema_type: str) -> ValidationResult:
        """
        Internal method to validate configuration against a specific schema.

        Args:
            task_params: Configuration dictionary
            schema_type: Type of schema to validate against

        Returns:
            ValidationResult with validation status and error details
        """
        try:
            schema = self.load_schema(schema_type)
            if not schema:
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Could not load schema for type: {schema_type}"],
                    warnings=[]
                )

            # Create validator instance
            validator = Draft7Validator(schema)

            # Collect all validation errors
            errors = []
            warnings = []

            for error in validator.iter_errors(task_params):
                error_msg = self._format_validation_error(error)
                errors.append(error_msg)
                _logger.debug("Validation error for %s: %s", schema_type, error_msg)

            # Check for warnings (optional fields, deprecated usage, etc.)
            warnings.extend(self._check_warnings(task_params, schema_type))

            is_valid = len(errors) == 0

            if is_valid:
                _logger.debug("Configuration validated successfully for %s", schema_type)
            else:
                _logger.warning("Configuration validation failed for %s: %d errors",
                              schema_type, len(errors))

            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings
            )

        except Exception as e:
            _logger.exception("Unexpected error during validation:")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=[]
            )

    def load_schema(self, job_type: str) -> Optional[Dict[str, Any]]:
        """
        Load and cache JSON schema for the specified job type.

        Args:
            job_type: Type of job schema to load ('alert' or 'schedule')

        Returns:
            Schema dictionary or None if loading fails
        """
        if job_type in self._schema_cache:
            return self._schema_cache[job_type]

        schema_file = self.schema_dir / f"{job_type}.json"

        if not schema_file.exists():
            _logger.error("Schema file not found: %s", schema_file)
            return None

        try:
            with open(schema_file, 'r', encoding='utf-8') as f:
                schema = json.load(f)

            # Validate the schema itself
            jsonschema.Draft7Validator.check_schema(schema)

            # Cache the schema
            self._schema_cache[job_type] = schema
            _logger.debug("Loaded and cached schema for %s", job_type)

            return schema

        except json.JSONDecodeError:
            _logger.exception("Invalid JSON in schema file %s:", schema_file)
            return None
        except jsonschema.SchemaError:
            _logger.exception("Invalid schema in file %s:", schema_file)
            return None
        except Exception:
            _logger.exception("Error loading schema from %s:", schema_file)
            return None

    def _format_validation_error(self, error: jsonschema.ValidationError) -> str:
        """
        Format a validation error into a human-readable message.

        Args:
            error: ValidationError from jsonschema

        Returns:
            Formatted error message
        """
        path = " -> ".join(str(p) for p in error.absolute_path) if error.absolute_path else "root"

        if error.validator == "required":
            missing_props = ", ".join(error.validator_value)
            return f"Missing required properties at {path}: {missing_props}"
        elif error.validator == "enum":
            allowed_values = ", ".join(str(v) for v in error.validator_value)
            return f"Invalid value at {path}: '{error.instance}'. Allowed values: {allowed_values}"
        elif error.validator == "type":
            return f"Invalid type at {path}: expected {error.validator_value}, got {type(error.instance).__name__}"
        elif error.validator == "pattern":
            return f"Invalid format at {path}: '{error.instance}' does not match pattern '{error.validator_value}'"
        elif error.validator == "minimum":
            return f"Value too small at {path}: {error.instance} < {error.validator_value}"
        elif error.validator == "maximum":
            return f"Value too large at {path}: {error.instance} > {error.validator_value}"
        else:
            return f"Validation error at {path}: {error.message}"

    def _check_warnings(self, task_params: Dict[str, Any], schema_type: str) -> List[str]:
        """
        Check for potential warnings in the configuration.

        Args:
            task_params: Configuration dictionary
            schema_type: Type of schema being validated

        Returns:
            List of warning messages
        """
        warnings = []

        if schema_type == "alert":
            # Check for potentially problematic alert configurations
            if "rearm" not in task_params:
                warnings.append("No rearm configuration specified - alert may trigger repeatedly")

            if "options" in task_params and "lookback" in task_params["options"]:
                lookback = task_params["options"]["lookback"]
                if lookback > 500:
                    warnings.append(f"Large lookback value ({lookback}) may impact performance")

        elif schema_type == "schedule":
            # Check for potentially problematic schedule configurations
            if "options" in task_params:
                options = task_params["options"]
                if "timeout" in options and options["timeout"] > 1800:  # 30 minutes
                    warnings.append("Long timeout may cause resource issues")

                if "retry_count" in options and options["retry_count"] > 3:
                    warnings.append("High retry count may cause excessive resource usage")

        return warnings

    def clear_cache(self) -> None:
        """Clear the schema cache, forcing reload on next access."""
        self._schema_cache.clear()
        _logger.debug("Schema cache cleared")

    def get_cached_schemas(self) -> List[str]:
        """
        Get list of currently cached schema types.

        Returns:
            List of cached schema type names
        """
        return list(self._schema_cache.keys())


def get_schedule_summary(config_json: Any) -> Dict[str, Any]:
    """
    Get a summary of schedule configuration.

    Args:
        config_json: Schedule configuration (JSON string or dict)

    Returns:
        Dictionary with summary fields (type, scheduled_time, ticker, etc.)
    """
    try:
        if isinstance(config_json, str):
            config = json.loads(config_json)
        else:
            config = config_json

        if not isinstance(config, dict):
            return {"error": "Invalid configuration format"}

        return {
            "type": config.get("schedule_type"),
            "scheduled_time": config.get("scheduled_time", "09:00"),
            "ticker": config.get("ticker", ""),
            "list_type": config.get("list_type", ""),
            "period": config.get("period", ""),
            "email": config.get("email", False)
        }
    except Exception as e:
        return {"error": f"Error parsing schedule config: {str(e)}"}