#!/usr/bin/env python3
"""
Report Configuration Parser
Parses and validates JSON configurations for report commands.
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

# Supported report types
SUPPORTED_REPORT_TYPES = ["analysis", "screener", "custom"]
SUPPORTED_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
SUPPORTED_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
SUPPORTED_PROVIDERS = ["yf", "alpha_vantage", "polygon"]
SUPPORTED_INDICATORS = ["RSI", "MACD", "BollingerBands", "SMA", "EMA", "ADX", "ATR", "Stochastic", "WilliamsR"]
SUPPORTED_FUNDAMENTAL_INDICATORS = ["PE", "PB", "ROE", "ROA", "DebtEquity", "CurrentRatio", "EPS", "Revenue", "ProfitMargin"]


@dataclass
class ReportConfig:
    """Configuration for a report."""
    report_type: str
    tickers: List[str]
    period: str = "2y"
    interval: str = "1d"
    provider: str = "yf"
    indicators: Optional[List[str]] = None
    fundamental_indicators: Optional[List[str]] = None
    email: bool = False
    include_chart: bool = True
    include_fundamentals: bool = True
    include_technicals: bool = True
    custom_analysis: Optional[Dict[str, Any]] = None


class ReportConfigParser:
    """Parser for report configurations."""

    @staticmethod
    def validate_report_config(config_json: str) -> Tuple[bool, List[str]]:
        """
        Validate a report configuration JSON string.

        Args:
            config_json: JSON string containing report configuration

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        try:
            config = json.loads(config_json)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON format: {str(e)}")
            return False, errors

        # Validate required fields
        if "report_type" not in config:
            errors.append("Missing required field: report_type")
        elif config["report_type"] not in SUPPORTED_REPORT_TYPES:
            errors.append(f"Unsupported report_type: {config['report_type']}. Must be one of: {', '.join(SUPPORTED_REPORT_TYPES)}")

        if "tickers" not in config:
            errors.append("Missing required field: tickers")
        elif not isinstance(config["tickers"], list) or len(config["tickers"]) == 0:
            errors.append("tickers must be a non-empty list")

        # Validate optional fields
        if "period" in config and config["period"] not in SUPPORTED_PERIODS:
            errors.append(f"Unsupported period: {config['period']}. Must be one of: {', '.join(SUPPORTED_PERIODS)}")

        if "interval" in config and config["interval"] not in SUPPORTED_INTERVALS:
            errors.append(f"Unsupported interval: {config['interval']}. Must be one of: {', '.join(SUPPORTED_INTERVALS)}")

        if "provider" in config and config["provider"] not in SUPPORTED_PROVIDERS:
            errors.append(f"Unsupported provider: {config['provider']}. Must be one of: {', '.join(SUPPORTED_PROVIDERS)}")

        if "indicators" in config:
            if not isinstance(config["indicators"], list):
                errors.append("indicators must be a list")
            else:
                for indicator in config["indicators"]:
                    # Convert to uppercase for case-insensitive comparison
                    indicator_upper = indicator.upper()
                    if indicator_upper not in [ind.upper() for ind in SUPPORTED_INDICATORS]:
                        errors.append(f"Unsupported indicator: {indicator}. Must be one of: {', '.join(SUPPORTED_INDICATORS)}")

        if "fundamental_indicators" in config:
            if not isinstance(config["fundamental_indicators"], list):
                errors.append("fundamental_indicators must be a list")
            else:
                for indicator in config["fundamental_indicators"]:
                    # Convert to uppercase for case-insensitive comparison
                    indicator_upper = indicator.upper()
                    if indicator_upper not in [ind.upper() for ind in SUPPORTED_FUNDAMENTAL_INDICATORS]:
                        errors.append(f"Unsupported fundamental indicator: {indicator}. Must be one of: {', '.join(SUPPORTED_FUNDAMENTAL_INDICATORS)}")

        if "email" in config and not isinstance(config["email"], bool):
            errors.append("email must be a boolean")

        if "include_chart" in config and not isinstance(config["include_chart"], bool):
            errors.append("include_chart must be a boolean")

        if "include_fundamentals" in config and not isinstance(config["include_fundamentals"], bool):
            errors.append("include_fundamentals must be a boolean")

        if "include_technicals" in config and not isinstance(config["include_technicals"], bool):
            errors.append("include_technicals must be a boolean")

        return len(errors) == 0, errors

    @staticmethod
    def parse_report_config(config_json: str) -> Optional[ReportConfig]:
        """
        Parse a report configuration JSON string into a ReportConfig object.

        Args:
            config_json: JSON string containing report configuration

        Returns:
            ReportConfig object or None if parsing fails
        """
        try:
            config = json.loads(config_json)

            return ReportConfig(
                report_type=config.get("report_type", "analysis"),
                tickers=config.get("tickers", []),
                period=config.get("period", "2y"),
                interval=config.get("interval", "1d"),
                provider=config.get("provider", "yf"),
                indicators=config.get("indicators"),
                fundamental_indicators=config.get("fundamental_indicators"),
                email=config.get("email", False),
                include_chart=config.get("include_chart", True),
                include_fundamentals=config.get("include_fundamentals", True),
                include_technicals=config.get("include_technicals", True),
                custom_analysis=config.get("custom_analysis")
            )
        except Exception as e:
            _logger.exception("Error parsing report config: %s", e)
            return None

    @staticmethod
    def get_report_summary(config_json: str) -> Dict[str, Any]:
        """
        Get a summary of the report configuration for display purposes.

        Args:
            config_json: JSON string containing report configuration

        Returns:
            Dictionary with summary information
        """
        try:
            config = json.loads(config_json)

            summary = {
                "type": config.get("report_type", "Unknown"),
                "tickers": config.get("tickers", []),
                "period": config.get("period", "2y"),
                "interval": config.get("interval", "1d"),
                "provider": config.get("provider", "yf"),
                "email": config.get("email", False)
            }

            # Add indicators summary
            if "indicators" in config and config["indicators"]:
                summary["indicators"] = ", ".join(config["indicators"])

            if "fundamental_indicators" in config and config["fundamental_indicators"]:
                summary["fundamental_indicators"] = ", ".join(config["fundamental_indicators"])

            # Add features summary
            features = []
            if config.get("include_chart", True):
                features.append("Chart")
            if config.get("include_fundamentals", True):
                features.append("Fundamentals")
            if config.get("include_technicals", True):
                features.append("Technicals")

            if features:
                summary["features"] = ", ".join(features)

            return summary

        except Exception as e:
            _logger.exception("Error getting report summary: %s", e)
            return {"error": f"Error parsing configuration: {str(e)}"}

    @staticmethod
    def create_default_config(tickers: List[str], **kwargs) -> str:
        """
        Create a default report configuration JSON string.

        Args:
            tickers: List of ticker symbols
            **kwargs: Additional configuration options

        Returns:
            JSON string with default configuration
        """
        config = {
            "report_type": "analysis",
            "tickers": tickers,
            "period": kwargs.get("period", "2y"),
            "interval": kwargs.get("interval", "1d"),
            "provider": kwargs.get("provider", "yf"),
            "indicators": kwargs.get("indicators", ["RSI", "MACD"]),
            "fundamental_indicators": kwargs.get("fundamental_indicators", ["PE", "PB", "ROE"]),
            "email": kwargs.get("email", False),
            "include_chart": kwargs.get("include_chart", True),
            "include_fundamentals": kwargs.get("include_fundamentals", True),
            "include_technicals": kwargs.get("include_technicals", True)
        }

        return json.dumps(config, indent=2)

    @staticmethod
    def get_example_configs() -> Dict[str, str]:
        """
        Get example report configurations for documentation.

        Returns:
            Dictionary of example configurations
        """
        return {
            "basic_analysis": json.dumps({
                "report_type": "analysis",
                "tickers": ["AAPL", "MSFT"],
                "period": "1y",
                "interval": "1d",
                "provider": "yf",
                "indicators": ["RSI", "MACD"],
                "email": True
            }, indent=2),

            "technical_analysis": json.dumps({
                "report_type": "analysis",
                "tickers": ["TSLA"],
                "period": "6mo",
                "interval": "1h",
                "provider": "yf",
                "indicators": ["RSI", "MACD", "BollingerBands", "SMA"],
                "include_fundamentals": False,
                "email": False
            }, indent=2),

            "fundamental_analysis": json.dumps({
                "report_type": "analysis",
                "tickers": ["AAPL", "GOOGL", "MSFT"],
                "period": "2y",
                "interval": "1d",
                "provider": "yf",
                "fundamental_indicators": ["PE", "PB", "ROE", "ROA", "DebtEquity"],
                "include_technicals": False,
                "email": True
            }, indent=2),

            "crypto_analysis": json.dumps({
                "report_type": "analysis",
                "tickers": ["BTCUSDT", "ETHUSDT"],
                "period": "3mo",
                "interval": "4h",
                "provider": "yf",
                "indicators": ["RSI", "MACD", "BollingerBands"],
                "include_fundamentals": False,
                "email": True
            }, indent=2)
        }
