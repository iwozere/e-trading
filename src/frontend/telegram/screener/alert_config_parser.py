#!/usr/bin/env python3
"""
Alert Configuration Parser
Parses and validates JSON configurations for indicator-based alerts.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


@dataclass
class IndicatorConfig:
    """Configuration for a single indicator."""
    name: str
    parameters: Dict[str, Any]
    condition: Dict[str, Any]


@dataclass
class AlertConfig:
    """Complete alert configuration."""
    alert_type: str  # 'price' or 'indicator'
    logic: str  # 'AND', 'OR', or None for single condition
    conditions: List[IndicatorConfig]
    timeframe: str = "15m"
    alert_action: str = "notify"


class AlertConfigParser:
    """
    Parser for alert JSON configurations.
    Supports both simple and complex indicator-based alerts.
    """

    # Supported indicators and their required parameters
    SUPPORTED_INDICATORS = {
        "RSI": {
            "required": ["period"],
            "optional": [],
            "defaults": {"period": 14}
        },
        "BollingerBands": {
            "required": ["period"],
            "optional": ["deviation"],
            "defaults": {"period": 20, "deviation": 2}
        },
        "MACD": {
            "required": [],
            "optional": ["fast_period", "slow_period", "signal_period"],
            "defaults": {"fast_period": 12, "slow_period": 26, "signal_period": 9}
        },
        "SMA": {
            "required": ["period"],
            "optional": [],
            "defaults": {"period": 20}
        }
    }

    # Supported conditions for each indicator
    SUPPORTED_CONDITIONS = {
        "RSI": ["<", ">", "<=", ">=", "==", "!="],
        "BollingerBands": ["above_upper_band", "below_lower_band", "between_bands"],
        "MACD": ["crossover", "crossunder", "above_signal", "below_signal"],
        "SMA": ["<", ">", "<=", ">=", "==", "!=", "crossover", "crossunder"]
    }

    # Supported timeframes
    SUPPORTED_TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]

    # Supported alert actions
    SUPPORTED_ACTIONS = ["BUY", "SELL", "HOLD", "notify"]

    def parse_config(self, config_json: str) -> AlertConfig:
        """
        Parse JSON configuration into AlertConfig object.

        Args:
            config_json: JSON string containing alert configuration

        Returns:
            AlertConfig object

        Raises:
            ValueError: If configuration is invalid
        """
        try:
            config = json.loads(config_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON configuration: {e}")

        return self._parse_config_dict(config)

    def _parse_config_dict(self, config: Dict[str, Any]) -> AlertConfig:
        """Parse configuration dictionary into AlertConfig."""

        # Validate basic structure
        if "type" not in config:
            raise ValueError("Configuration must include 'type' field")

        alert_type = config["type"]
        if alert_type not in ["price", "indicator"]:
            raise ValueError(f"Unsupported alert type: {alert_type}")

        if alert_type == "price":
            return self._parse_price_config(config)
        else:
            return self._parse_indicator_config(config)

    def _parse_price_config(self, config: Dict[str, Any]) -> AlertConfig:
        """Parse price-based alert configuration."""
        required_fields = ["threshold", "condition"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Price alert must include '{field}' field")

        # Create a simple indicator config for price
        indicator_config = IndicatorConfig(
            name="PRICE",
            parameters={"threshold": config["threshold"]},
            condition={"operator": config["condition"], "value": config["threshold"]}
        )

        return AlertConfig(
            alert_type="price",
            logic=None,
            conditions=[indicator_config],
            timeframe=config.get("timeframe", "15m"),
            alert_action=config.get("alert_action", "notify")
        )

    def _parse_indicator_config(self, config: Dict[str, Any]) -> AlertConfig:
        """Parse indicator-based alert configuration."""

        # Check if it's a single indicator or multiple indicators
        if "indicator" in config:
            # Single indicator
            conditions = [self._parse_single_indicator(config)]
            logic = None
        elif "conditions" in config:
            # Multiple indicators
            conditions = []
            for condition_config in config["conditions"]:
                conditions.append(self._parse_single_indicator(condition_config))
            logic = config.get("logic", "AND")
            if logic not in ["AND", "OR"]:
                raise ValueError(f"Unsupported logic operator: {logic}")
        else:
            raise ValueError("Indicator configuration must include 'indicator' or 'conditions' field")

        return AlertConfig(
            alert_type="indicator",
            logic=logic,
            conditions=conditions,
            timeframe=config.get("timeframe", "15m"),
            alert_action=config.get("alert_action", "notify")
        )

    def _parse_single_indicator(self, config: Dict[str, Any]) -> IndicatorConfig:
        """Parse single indicator configuration."""

        if "indicator" not in config:
            raise ValueError("Indicator configuration must include 'indicator' field")

        indicator_name = config["indicator"]
        if indicator_name not in self.SUPPORTED_INDICATORS:
            raise ValueError(f"Unsupported indicator: {indicator_name}")

        # Parse parameters
        parameters = self._parse_parameters(config, indicator_name)

        # Parse condition
        condition = self._parse_condition(config, indicator_name)

        return IndicatorConfig(
            name=indicator_name,
            parameters=parameters,
            condition=condition
        )

    def _parse_parameters(self, config: Dict[str, Any], indicator_name: str) -> Dict[str, Any]:
        """Parse indicator parameters."""
        indicator_config = self.SUPPORTED_INDICATORS[indicator_name]
        parameters = indicator_config["defaults"].copy()

        if "parameters" in config:
            user_params = config["parameters"]
            for param, value in user_params.items():
                if param in indicator_config["required"] or param in indicator_config["optional"]:
                    parameters[param] = value
                else:
                    _logger.warning("Unknown parameter '%s' for indicator '%s'", param, indicator_name)

        # Validate required parameters
        for required_param in indicator_config["required"]:
            if required_param not in parameters:
                raise ValueError(f"Required parameter '{required_param}' missing for indicator '{indicator_name}'")

        return parameters

    def _parse_condition(self, config: Dict[str, Any], indicator_name: str) -> Dict[str, Any]:
        """Parse indicator condition."""
        if "condition" not in config:
            raise ValueError(f"Condition missing for indicator '{indicator_name}'")

        condition = config["condition"]
        supported_conditions = self.SUPPORTED_CONDITIONS[indicator_name]

        if isinstance(condition, dict):
            # Complex condition (e.g., {"operator": "<", "value": 30})
            if "operator" not in condition:
                raise ValueError(f"Condition must include 'operator' field for indicator '{indicator_name}'")

            operator = condition["operator"]
            if operator not in supported_conditions:
                raise ValueError(f"Unsupported condition '{operator}' for indicator '{indicator_name}'")

            return condition
        else:
            # Simple condition (e.g., "< 30")
            raise ValueError(f"Condition must be a dictionary for indicator '{indicator_name}'")

    def validate_config(self, config_json: str) -> Tuple[bool, List[str]]:
        """
        Validate JSON configuration.

        Args:
            config_json: JSON string to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        try:
            config = self.parse_config(config_json)

            # Validate timeframe
            if config.timeframe not in self.SUPPORTED_TIMEFRAMES:
                errors.append(f"Unsupported timeframe: {config.timeframe}")

            # Validate alert action
            if config.alert_action not in self.SUPPORTED_ACTIONS:
                errors.append(f"Unsupported alert action: {config.alert_action}")

            # Validate indicators
            for condition in config.conditions:
                if condition.name not in self.SUPPORTED_INDICATORS:
                    errors.append(f"Unsupported indicator: {condition.name}")

                # Validate parameters
                indicator_config = self.SUPPORTED_INDICATORS[condition.name]
                for param in condition.parameters:
                    if param not in indicator_config["required"] + indicator_config["optional"]:
                        errors.append(f"Unknown parameter '{param}' for indicator '{condition.name}'")

                # Validate condition
                supported_conditions = self.SUPPORTED_CONDITIONS[condition.name]
                operator = condition.condition.get("operator")
                if operator and operator not in supported_conditions:
                    errors.append(f"Unsupported condition '{operator}' for indicator '{condition.name}'")

        except ValueError as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(f"Unexpected error: {e}")

        return len(errors) == 0, errors

    def get_required_data_points(self, config_json: str) -> int:
        """
        Determine how many data points are needed for the alert configuration.

        Args:
            config_json: JSON configuration string

        Returns:
            Number of data points required
        """
        try:
            config = self.parse_config(config_json)
            max_period = 50  # Default minimum

            for condition in config.conditions:
                if condition.name == "RSI":
                    max_period = max(max_period, condition.parameters.get("period", 14) + 10)
                elif condition.name == "BollingerBands":
                    max_period = max(max_period, condition.parameters.get("period", 20) + 10)
                elif condition.name == "MACD":
                    slow_period = condition.parameters.get("slow_period", 26)
                    max_period = max(max_period, slow_period + 20)
                elif condition.name == "SMA":
                    max_period = max(max_period, condition.parameters.get("period", 20) + 10)

            return max_period
        except Exception:
            return 100  # Safe default

    def create_sample_configs(self) -> Dict[str, str]:
        """Create sample configurations for documentation."""
        return {
            "simple_rsi": json.dumps({
                "type": "indicator",
                "indicator": "RSI",
                "parameters": {"period": 14},
                "condition": {"operator": "<", "value": 30},
                "alert_action": "BUY",
                "timeframe": "15m"
            }, indent=2),

            "bollinger_bands": json.dumps({
                "type": "indicator",
                "indicator": "BollingerBands",
                "parameters": {"period": 20, "deviation": 2},
                "condition": {"operator": "below_lower_band"},
                "alert_action": "BUY",
                "timeframe": "1h"
            }, indent=2),

            "macd_crossover": json.dumps({
                "type": "indicator",
                "indicator": "MACD",
                "parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
                "condition": {"operator": "crossover"},
                "alert_action": "BUY",
                "timeframe": "4h"
            }, indent=2),

            "complex_and_logic": json.dumps({
                "type": "indicator",
                "logic": "AND",
                "conditions": [
                    {
                        "indicator": "RSI",
                        "parameters": {"period": 14},
                        "condition": {"operator": "<", "value": 30}
                    },
                    {
                        "indicator": "BollingerBands",
                        "parameters": {"period": 20, "deviation": 2},
                        "condition": {"operator": "below_lower_band"}
                    }
                ],
                "alert_action": "BUY",
                "timeframe": "15m"
            }, indent=2),

            "price_alert": json.dumps({
                "type": "price",
                "threshold": 150.00,
                "condition": "below",
                "alert_action": "notify",
                "timeframe": "15m"
            }, indent=2)
        }


# Convenience functions
def parse_alert_config(config_json: str) -> AlertConfig:
    """Parse alert configuration from JSON string."""
    parser = AlertConfigParser()
    return parser.parse_config(config_json)


def validate_alert_config(config_json: str) -> Tuple[bool, List[str]]:
    """Validate alert configuration."""
    parser = AlertConfigParser()
    return parser.validate_config(config_json)


def get_required_data_points(config_json: str) -> int:
    """Get required data points for alert configuration."""
    parser = AlertConfigParser()
    return parser.get_required_data_points(config_json)


if __name__ == "__main__":
    # Test the parser
    parser = AlertConfigParser()
    samples = parser.create_sample_configs()

    print("Sample Alert Configurations:")
    print("=" * 50)

    for name, config in samples.items():
        print(f"\n{name.upper()}:")
        print(config)

        # Validate the config
        is_valid, errors = parser.validate_config(config)
        if is_valid:
            print("✅ Valid configuration")
        else:
            print(f"❌ Invalid configuration: {errors}")

        # Get required data points
        data_points = parser.get_required_data_points(config)
        print(f"📊 Required data points: {data_points}")
