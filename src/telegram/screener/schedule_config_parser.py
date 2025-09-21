#!/usr/bin/env python3
"""
Schedule Configuration Parser
Parses and validates JSON configurations for scheduled reports and screeners.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# Supported schedule types
SUPPORTED_SCHEDULE_TYPES = ["report", "screener", "custom"]
SUPPORTED_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
SUPPORTED_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
SUPPORTED_PROVIDERS = ["yf", "alpha_vantage", "polygon"]
SUPPORTED_LIST_TYPES = ["us_small_cap", "us_medium_cap", "us_large_cap", "swiss_shares", "custom_list"]


@dataclass
class ScheduleConfig:
    """Configuration for a scheduled task."""
    schedule_type: str
    ticker: Optional[str] = None
    list_type: Optional[str] = None
    scheduled_time: str = "09:00"
    period: str = "1y"
    interval: str = "1d"
    provider: str = "yf"
    indicators: Optional[str] = None
    email: bool = False
    config_json: Optional[str] = None
    schedule_config: str = "simple"


class ScheduleConfigParser:
    """Parser for schedule configurations."""

    def __init__(self):
        """Initialize the schedule configuration parser."""
        pass

    def parse_config(self, config_json: str) -> ScheduleConfig:
        """
        Parse a JSON configuration string into a ScheduleConfig object.

        Args:
            config_json: JSON string containing schedule configuration

        Returns:
            ScheduleConfig object

        Raises:
            ValueError: If configuration is invalid
        """
        try:
            config_dict = json.loads(config_json)
            return self._parse_config_dict(config_dict)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")

    def _parse_config_dict(self, config_dict: Dict[str, Any]) -> ScheduleConfig:
        """Parse configuration dictionary."""
        schedule_type = config_dict.get("type", "report")

        if schedule_type == "report":
            return self._parse_report_config(config_dict)
        elif schedule_type == "screener":
            return self._parse_screener_config(config_dict)
        elif schedule_type == "custom":
            return self._parse_custom_config(config_dict)
        else:
            raise ValueError(f"Unsupported schedule type: {schedule_type}")

    def _parse_report_config(self, config_dict: Dict[str, Any]) -> ScheduleConfig:
        """Parse report schedule configuration."""
        return ScheduleConfig(
            schedule_type="report",
            ticker=config_dict.get("ticker"),
            scheduled_time=config_dict.get("scheduled_time", "09:00"),
            period=config_dict.get("period", "1y"),
            interval=config_dict.get("interval", "1d"),
            provider=config_dict.get("provider", "yf"),
            indicators=config_dict.get("indicators"),
            email=config_dict.get("email", False),
            config_json=json.dumps(config_dict),
            schedule_config="advanced"
        )

    def _parse_screener_config(self, config_dict: Dict[str, Any]) -> ScheduleConfig:
        """Parse screener schedule configuration."""
        return ScheduleConfig(
            schedule_type="screener",
            list_type=config_dict.get("list_type"),
            scheduled_time=config_dict.get("scheduled_time", "09:00"),
            period=config_dict.get("period", "1y"),
            interval=config_dict.get("interval", "1d"),
            provider=config_dict.get("provider", "yf"),
            indicators=config_dict.get("indicators"),
            email=config_dict.get("email", False),
            config_json=json.dumps(config_dict),
            schedule_config="advanced"
        )

    def _parse_custom_config(self, config_dict: Dict[str, Any]) -> ScheduleConfig:
        """Parse custom schedule configuration."""
        return ScheduleConfig(
            schedule_type="custom",
            scheduled_time=config_dict.get("scheduled_time", "09:00"),
            config_json=json.dumps(config_dict),
            schedule_config="advanced"
        )

    def validate_config(self, config_json: str) -> Tuple[bool, List[str]]:
        """
        Validate a schedule configuration.

        Args:
            config_json: JSON string containing schedule configuration

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        try:
            config_dict = json.loads(config_json)
        except json.JSONDecodeError as e:
            return False, [f"Invalid JSON format: {e}"]

        # Validate required fields
        schedule_type = config_dict.get("type")
        if not schedule_type:
            errors.append("Missing required field: 'type'")
        elif schedule_type not in SUPPORTED_SCHEDULE_TYPES:
            errors.append(f"Unsupported schedule type: {schedule_type}")

        # Validate schedule type specific fields
        if schedule_type == "report":
            if not config_dict.get("ticker"):
                errors.append("Report schedule requires 'ticker' field")
        elif schedule_type == "screener":
            if not config_dict.get("list_type"):
                errors.append("Screener schedule requires 'list_type' field")
            elif config_dict.get("list_type") not in SUPPORTED_LIST_TYPES:
                errors.append(f"Unsupported list type: {config_dict.get('list_type')}")

        # Validate optional fields
        if "period" in config_dict and config_dict["period"] not in SUPPORTED_PERIODS:
            errors.append(f"Unsupported period: {config_dict['period']}")

        if "interval" in config_dict and config_dict["interval"] not in SUPPORTED_INTERVALS:
            errors.append(f"Unsupported interval: {config_dict['interval']}")

        if "provider" in config_dict and config_dict["provider"] not in SUPPORTED_PROVIDERS:
            errors.append(f"Unsupported provider: {config_dict['provider']}")

        # Validate scheduled_time format
        scheduled_time = config_dict.get("scheduled_time", "09:00")
        if not self._is_valid_time_format(scheduled_time):
            errors.append(f"Invalid time format: {scheduled_time}. Use HH:MM format")

        return len(errors) == 0, errors

    def _is_valid_time_format(self, time_str: str) -> bool:
        """Validate time format (HH:MM)."""
        try:
            if not isinstance(time_str, str):
                return False
            parts = time_str.split(":")
            if len(parts) != 2:
                return False
            hour, minute = int(parts[0]), int(parts[1])
            return 0 <= hour <= 23 and 0 <= minute <= 59
        except (ValueError, TypeError):
            return False

    def create_sample_configs(self) -> Dict[str, str]:
        """Create sample configurations for different schedule types."""
        samples = {
            "simple_report": json.dumps({
                "type": "report",
                "ticker": "AAPL",
                "scheduled_time": "09:00",
                "period": "1y",
                "interval": "1d",
                "email": True
            }),
            "advanced_report": json.dumps({
                "type": "report",
                "ticker": "TSLA",
                "scheduled_time": "16:30",
                "period": "6mo",
                "interval": "1h",
                "indicators": "RSI,MACD,BollingerBands",
                "provider": "yf",
                "email": True
            }),
            "screener_small_cap": json.dumps({
                "type": "screener",
                "list_type": "us_small_cap",
                "scheduled_time": "08:00",
                "period": "1y",
                "interval": "1d",
                "indicators": "PE,PB,ROE",
                "email": True
            }),
            "screener_large_cap": json.dumps({
                "type": "screener",
                "list_type": "us_large_cap",
                "scheduled_time": "17:00",
                "period": "2y",
                "interval": "1d",
                "indicators": "PE,PB,ROE,ROA",
                "email": False
            }),
            "custom_schedule": json.dumps({
                "type": "custom",
                "scheduled_time": "12:00",
                "description": "Custom market analysis"
            })
        }
        return samples

    def get_required_fields(self, schedule_type: str) -> List[str]:
        """Get required fields for a schedule type."""
        if schedule_type == "report":
            return ["type", "ticker"]
        elif schedule_type == "screener":
            return ["type", "list_type"]
        elif schedule_type == "custom":
            return ["type"]
        else:
            return ["type"]

    def get_optional_fields(self, schedule_type: str) -> List[str]:
        """Get optional fields for a schedule type."""
        common_fields = ["scheduled_time", "period", "interval", "provider", "indicators", "email"]

        if schedule_type == "report":
            return common_fields
        elif schedule_type == "screener":
            return common_fields
        elif schedule_type == "custom":
            return ["scheduled_time", "description"]
        else:
            return common_fields


# Convenience functions
def parse_schedule_config(config_json: str) -> ScheduleConfig:
    """Parse schedule configuration from JSON string."""
    parser = ScheduleConfigParser()
    return parser.parse_config(config_json)

def validate_schedule_config(config_json: str) -> Tuple[bool, List[str]]:
    """Validate schedule configuration."""
    parser = ScheduleConfigParser()
    return parser.validate_config(config_json)

def get_schedule_summary(config_json: str) -> Dict[str, Any]:
    """Get a summary of the schedule configuration."""
    try:
        config = parse_schedule_config(config_json)
        return {
            "type": config.schedule_type,
            "ticker": config.ticker,
            "list_type": config.list_type,
            "scheduled_time": config.scheduled_time,
            "period": config.period,
            "interval": config.interval,
            "indicators": config.indicators,
            "email": config.email
        }
    except Exception as e:
        return {"error": str(e)}
