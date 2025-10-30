"""
Channel Configuration Validation

Validation utilities and schemas for channel plugin configurations.
Provides type-safe configuration handling and validation.
"""

from typing import Dict, Any, List, Optional, Union, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import re

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class ConfigValidationError(Exception):
    """Exception raised when configuration validation fails."""

    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        """
        Initialize configuration validation error.

        Args:
            message: Error message
            field: Field name that failed validation
            value: Value that failed validation
        """
        self.field = field
        self.value = value
        super().__init__(message)


@dataclass
class ValidationRule:
    """Configuration validation rule."""

    field_name: str
    required: bool = True
    field_type: Optional[Type] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[callable] = None
    description: str = ""

    def validate(self, value: Any) -> None:
        """
        Validate a value against this rule.

        Args:
            value: Value to validate

        Raises:
            ConfigValidationError: If validation fails
        """
        # Check if field is required
        if value is None:
            if self.required:
                raise ConfigValidationError(
                    f"Required field '{self.field_name}' is missing",
                    field=self.field_name,
                    value=value
                )
            return  # Optional field with None value is valid

        # Check type
        if self.field_type is not None and not isinstance(value, self.field_type):
            raise ConfigValidationError(
                f"Field '{self.field_name}' must be of type {self.field_type.__name__}, got {type(value).__name__}",
                field=self.field_name,
                value=value
            )

        # Check string length
        if isinstance(value, str):
            if self.min_length is not None and len(value) < self.min_length:
                raise ConfigValidationError(
                    f"Field '{self.field_name}' must be at least {self.min_length} characters long",
                    field=self.field_name,
                    value=value
                )

            if self.max_length is not None and len(value) > self.max_length:
                raise ConfigValidationError(
                    f"Field '{self.field_name}' must be at most {self.max_length} characters long",
                    field=self.field_name,
                    value=value
                )

            # Check pattern
            if self.pattern is not None and not re.match(self.pattern, value):
                raise ConfigValidationError(
                    f"Field '{self.field_name}' does not match required pattern: {self.pattern}",
                    field=self.field_name,
                    value=value
                )

        # Check numeric ranges
        if isinstance(value, (int, float)):
            if self.min_value is not None and value < self.min_value:
                raise ConfigValidationError(
                    f"Field '{self.field_name}' must be at least {self.min_value}",
                    field=self.field_name,
                    value=value
                )

            if self.max_value is not None and value > self.max_value:
                raise ConfigValidationError(
                    f"Field '{self.field_name}' must be at most {self.max_value}",
                    field=self.field_name,
                    value=value
                )

        # Check allowed values
        if self.allowed_values is not None and value not in self.allowed_values:
            raise ConfigValidationError(
                f"Field '{self.field_name}' must be one of {self.allowed_values}, got {value}",
                field=self.field_name,
                value=value
            )

        # Custom validation
        if self.custom_validator is not None:
            try:
                self.custom_validator(value)
            except Exception as e:
                raise ConfigValidationError(
                    f"Custom validation failed for field '{self.field_name}': {str(e)}",
                    field=self.field_name,
                    value=value
                )


class ConfigValidator:
    """Configuration validator for channel plugins."""

    def __init__(self, channel_name: str):
        """
        Initialize configuration validator.

        Args:
            channel_name: Name of the channel this validator is for
        """
        self.channel_name = channel_name
        self.rules: List[ValidationRule] = []
        self._logger = setup_logger(f"{__name__}.{channel_name}")

    def add_rule(self, rule: ValidationRule) -> None:
        """
        Add a validation rule.

        Args:
            rule: Validation rule to add
        """
        self.rules.append(rule)

    def require_field(
        self,
        field_name: str,
        field_type: Type,
        description: str = "",
        **kwargs
    ) -> None:
        """
        Add a required field validation rule.

        Args:
            field_name: Name of the field
            field_type: Expected type of the field
            description: Description of the field
            **kwargs: Additional validation parameters
        """
        rule = ValidationRule(
            field_name=field_name,
            required=True,
            field_type=field_type,
            description=description,
            **kwargs
        )
        self.add_rule(rule)

    def optional_field(
        self,
        field_name: str,
        field_type: Type,
        default_value: Any = None,
        description: str = "",
        **kwargs
    ) -> None:
        """
        Add an optional field validation rule.

        Args:
            field_name: Name of the field
            field_type: Expected type of the field
            default_value: Default value if field is missing
            description: Description of the field
            **kwargs: Additional validation parameters
        """
        rule = ValidationRule(
            field_name=field_name,
            required=False,
            field_type=field_type,
            description=description,
            **kwargs
        )
        self.add_rule(rule)

    def validate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration against all rules.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Validated and normalized configuration

        Raises:
            ConfigValidationError: If validation fails
        """
        validated_config = config.copy()
        errors = []

        # Validate each rule
        for rule in self.rules:
            try:
                value = config.get(rule.field_name)
                rule.validate(value)

                # Set default values for optional fields
                if value is None and not rule.required:
                    # Look for default in rule or use None
                    default_value = getattr(rule, 'default_value', None)
                    if default_value is not None:
                        validated_config[rule.field_name] = default_value

            except ConfigValidationError as e:
                errors.append(str(e))

        # Check for unknown fields
        known_fields = {rule.field_name for rule in self.rules}
        unknown_fields = set(config.keys()) - known_fields

        if unknown_fields:
            self._logger.warning(
                "Unknown configuration fields for channel %s: %s",
                self.channel_name, list(unknown_fields)
            )

        # Raise combined error if any validation failed
        if errors:
            error_message = f"Configuration validation failed for channel '{self.channel_name}':\n" + "\n".join(f"  - {error}" for error in errors)
            raise ConfigValidationError(error_message)

        self._logger.info("Configuration validated successfully for channel: %s", self.channel_name)
        return validated_config

    def get_schema(self) -> Dict[str, Any]:
        """
        Get configuration schema for documentation.

        Returns:
            Schema dictionary describing all fields
        """
        schema = {
            "channel": self.channel_name,
            "fields": {}
        }

        for rule in self.rules:
            field_info = {
                "type": rule.field_type.__name__ if rule.field_type else "any",
                "required": rule.required,
                "description": rule.description
            }

            # Add constraints
            if rule.min_length is not None:
                field_info["min_length"] = rule.min_length
            if rule.max_length is not None:
                field_info["max_length"] = rule.max_length
            if rule.min_value is not None:
                field_info["min_value"] = rule.min_value
            if rule.max_value is not None:
                field_info["max_value"] = rule.max_value
            if rule.pattern is not None:
                field_info["pattern"] = rule.pattern
            if rule.allowed_values is not None:
                field_info["allowed_values"] = rule.allowed_values

            schema["fields"][rule.field_name] = field_info

        return schema


# Common validation functions
def validate_url(url: str) -> None:
    """Validate URL format."""
    url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    if not re.match(url_pattern, url):
        raise ValueError("Invalid URL format")


def validate_email(email: str) -> None:
    """Validate email format."""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        raise ValueError("Invalid email format")


def validate_phone(phone: str) -> None:
    """Validate phone number format."""
    # Simple international phone number validation
    phone_pattern = r'^\+?[1-9]\d{1,14}$'
    if not re.match(phone_pattern, phone.replace(' ', '').replace('-', '')):
        raise ValueError("Invalid phone number format")


def validate_positive_integer(value: int) -> None:
    """Validate positive integer."""
    if not isinstance(value, int) or value <= 0:
        raise ValueError("Value must be a positive integer")


def validate_non_negative_integer(value: int) -> None:
    """Validate non-negative integer."""
    if not isinstance(value, int) or value < 0:
        raise ValueError("Value must be a non-negative integer")


# Pre-defined validation rules for common configuration patterns
class CommonValidationRules:
    """Common validation rules for channel configurations."""

    @staticmethod
    def api_token(field_name: str = "api_token", description: str = "API authentication token") -> ValidationRule:
        """API token validation rule."""
        return ValidationRule(
            field_name=field_name,
            required=True,
            field_type=str,
            min_length=10,
            description=description
        )

    @staticmethod
    def api_url(field_name: str = "api_url", description: str = "API base URL") -> ValidationRule:
        """API URL validation rule."""
        return ValidationRule(
            field_name=field_name,
            required=True,
            field_type=str,
            custom_validator=validate_url,
            description=description
        )

    @staticmethod
    def timeout_seconds(field_name: str = "timeout_seconds", default: int = 30) -> ValidationRule:
        """Timeout validation rule."""
        return ValidationRule(
            field_name=field_name,
            required=False,
            field_type=int,
            min_value=1,
            max_value=300,
            description=f"Request timeout in seconds (default: {default})"
        )

    @staticmethod
    def rate_limit(field_name: str = "rate_limit_per_minute", default: int = 60) -> ValidationRule:
        """Rate limit validation rule."""
        return ValidationRule(
            field_name=field_name,
            required=False,
            field_type=int,
            min_value=1,
            max_value=10000,
            description=f"Rate limit in messages per minute (default: {default})"
        )

    @staticmethod
    def max_retries(field_name: str = "max_retries", default: int = 3) -> ValidationRule:
        """Max retries validation rule."""
        return ValidationRule(
            field_name=field_name,
            required=False,
            field_type=int,
            min_value=0,
            max_value=10,
            description=f"Maximum retry attempts (default: {default})"
        )

    @staticmethod
    def email_address(field_name: str, description: str = "Email address") -> ValidationRule:
        """Email address validation rule."""
        return ValidationRule(
            field_name=field_name,
            required=True,
            field_type=str,
            custom_validator=validate_email,
            description=description
        )

    @staticmethod
    def phone_number(field_name: str, description: str = "Phone number") -> ValidationRule:
        """Phone number validation rule."""
        return ValidationRule(
            field_name=field_name,
            required=True,
            field_type=str,
            custom_validator=validate_phone,
            description=description
        )