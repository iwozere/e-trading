#!/usr/bin/env python3
"""
Enhanced Screener Configuration Parser
Parses and validates JSON configurations for advanced screeners that combine
fundamental and technical analysis.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import json
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# Supported screener types
SUPPORTED_SCREENER_TYPES = ["fundamental", "technical", "hybrid"]
SUPPORTED_LIST_TYPES = ["us_small_cap", "us_medium_cap", "us_large_cap", "swiss_shares", "custom_list"]

# Supported fundamental indicators
SUPPORTED_FUNDAMENTAL_INDICATORS = [
    "PE", "Forward_PE", "PB", "PS", "PEG", "Debt_Equity", "Current_Ratio",
    "Quick_Ratio", "ROE", "ROA", "Operating_Margin", "Profit_Margin",
    "Revenue_Growth", "Net_Income_Growth", "Free_Cash_Flow", "Dividend_Yield",
    "Payout_Ratio", "DCF"
]

# Supported technical indicators
SUPPORTED_TECHNICAL_INDICATORS = [
    "RSI", "MACD", "BollingerBands", "SMA", "EMA", "ADX", "ATR",
    "Stochastic", "WilliamsR", "CCI", "ROC", "MFI"
]

# Supported periods and intervals
SUPPORTED_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
SUPPORTED_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
SUPPORTED_PROVIDERS = ["yf", "alpha_vantage", "polygon"]


@dataclass
class FundamentalCriteria:
    """Fundamental screening criteria."""
    indicator: str
    operator: str  # "min", "max", "range"
    value: Union[float, Dict[str, float]]  # Single value or range {"min": x, "max": y}
    weight: float = 1.0
    required: bool = False


@dataclass
class TechnicalCriteria:
    """Technical screening criteria."""
    indicator: str
    parameters: Dict[str, Any]  # Indicator-specific parameters
    condition: Dict[str, Any]  # Condition to check
    weight: float = 1.0
    required: bool = False


@dataclass
class ScreenerConfig:
    """Configuration for an enhanced screener."""
    screener_type: str  # "fundamental", "technical", "hybrid"
    list_type: str
    screener_name: Optional[str] = None  # Name of the screener (for email titles)
    fundamental_criteria: Optional[List[FundamentalCriteria]] = None
    technical_criteria: Optional[List[TechnicalCriteria]] = None
    fmp_criteria: Optional[Dict[str, Any]] = None  # FMP screening criteria
    fmp_strategy: Optional[str] = None  # Predefined FMP strategy name
    period: str = "1y"
    interval: str = "1d"
    provider: str = "yf"
    max_results: int = 10
    min_score: float = 7.0  # Minimum composite score (0-10)
    include_technical_analysis: bool = False
    include_fundamental_analysis: bool = True
    email: bool = False
    config_json: Optional[str] = None


class ScreenerConfigParser:
    """Parser for enhanced screener configurations."""

    def __init__(self):
        """Initialize the screener configuration parser."""
        pass

    def parse_config(self, config_json: str) -> ScreenerConfig:
        """
        Parse a JSON configuration string into a ScreenerConfig object.

        Args:
            config_json: JSON string containing screener configuration

        Returns:
            ScreenerConfig object

        Raises:
            ValueError: If configuration is invalid
        """
        try:
            config_dict = json.loads(config_json)
            return self._parse_config_dict(config_dict)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")

    def _parse_config_dict(self, config_dict: Dict[str, Any]) -> ScreenerConfig:
        """Parse configuration dictionary."""
        screener_type = config_dict.get("screener_type", "fundamental")
        list_type = config_dict.get("list_type")

        if not list_type:
            raise ValueError("Missing required field: 'list_type'")

        # Parse fundamental criteria
        fundamental_criteria = None
        if "fundamental_criteria" in config_dict:
            fundamental_criteria = self._parse_fundamental_criteria(
                config_dict["fundamental_criteria"]
            )

        # Parse technical criteria
        technical_criteria = None
        if "technical_criteria" in config_dict:
            technical_criteria = self._parse_technical_criteria(
                config_dict["technical_criteria"]
            )

        # Parse FMP criteria
        fmp_criteria = config_dict.get("fmp_criteria")
        fmp_strategy = config_dict.get("fmp_strategy")

        return ScreenerConfig(
            screener_name=config_dict.get("screener_name"),
            screener_type=screener_type,
            list_type=list_type,
            fundamental_criteria=fundamental_criteria,
            technical_criteria=technical_criteria,
            fmp_criteria=fmp_criteria,
            fmp_strategy=fmp_strategy,
            period=config_dict.get("period", "1y"),
            interval=config_dict.get("interval", "1d"),
            provider=config_dict.get("provider", "yf"),
            max_results=config_dict.get("max_results", 10),
            min_score=config_dict.get("min_score", 7.0),
            include_technical_analysis=config_dict.get("include_technical_analysis", False),
            include_fundamental_analysis=config_dict.get("include_fundamental_analysis", True),
            email=config_dict.get("email", False),
            config_json=json.dumps(config_dict)
        )

    def _parse_fundamental_criteria(self, criteria_list: List[Dict[str, Any]]) -> List[FundamentalCriteria]:
        """Parse fundamental criteria list."""
        parsed_criteria = []

        for criteria in criteria_list:
            indicator = criteria.get("indicator")
            operator = criteria.get("operator")
            value = criteria.get("value")
            weight = criteria.get("weight", 1.0)
            required = criteria.get("required", False)

            if not all([indicator, operator, value is not None]):
                raise ValueError("Fundamental criteria must have indicator, operator, and value")

            parsed_criteria.append(FundamentalCriteria(
                indicator=indicator,
                operator=operator,
                value=value,
                weight=weight,
                required=required
            ))

        return parsed_criteria

    def _parse_technical_criteria(self, criteria_list: List[Dict[str, Any]]) -> List[TechnicalCriteria]:
        """Parse technical criteria list."""
        parsed_criteria = []

        for criteria in criteria_list:
            indicator = criteria.get("indicator")
            parameters = criteria.get("parameters", {})
            condition = criteria.get("condition")
            weight = criteria.get("weight", 1.0)
            required = criteria.get("required", False)

            if not all([indicator, condition]):
                raise ValueError("Technical criteria must have indicator and condition")

            parsed_criteria.append(TechnicalCriteria(
                indicator=indicator,
                parameters=parameters,
                condition=condition,
                weight=weight,
                required=required
            ))

        return parsed_criteria

    def validate_config(self, config_json: str) -> Tuple[bool, List[str]]:
        """
        Validate a screener configuration.

        Args:
            config_json: JSON string containing screener configuration

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        try:
            config_dict = json.loads(config_json)
        except json.JSONDecodeError as e:
            return False, [f"Invalid JSON format: {e}"]

        # Validate required fields
        if not config_dict.get("list_type"):
            errors.append("Missing required field: 'list_type'")
        elif config_dict.get("list_type") not in SUPPORTED_LIST_TYPES:
            errors.append(f"Unsupported list type: {config_dict.get('list_type')}")

        # Validate screener type
        screener_type = config_dict.get("screener_type", "fundamental")
        if screener_type not in SUPPORTED_SCREENER_TYPES:
            errors.append(f"Unsupported screener type: {screener_type}")

        # Validate fundamental criteria
        if "fundamental_criteria" in config_dict:
            fundamental_errors = self._validate_fundamental_criteria(
                config_dict["fundamental_criteria"]
            )
            errors.extend(fundamental_errors)

        # Validate technical criteria
        if "technical_criteria" in config_dict:
            technical_errors = self._validate_technical_criteria(
                config_dict["technical_criteria"]
            )
            errors.extend(technical_errors)

        # Validate FMP criteria
        if "fmp_criteria" in config_dict:
            fmp_errors = self._validate_fmp_criteria(config_dict["fmp_criteria"])
            errors.extend(fmp_errors)

        # Validate FMP strategy
        if "fmp_strategy" in config_dict:
            fmp_strategy = config_dict["fmp_strategy"]
            if not isinstance(fmp_strategy, str):
                errors.append("fmp_strategy must be a string")

        # Validate optional fields
        if "period" in config_dict and config_dict["period"] not in SUPPORTED_PERIODS:
            errors.append(f"Unsupported period: {config_dict['period']}")

        if "interval" in config_dict and config_dict["interval"] not in SUPPORTED_INTERVALS:
            errors.append(f"Unsupported interval: {config_dict['interval']}")

        if "provider" in config_dict and config_dict["provider"] not in SUPPORTED_PROVIDERS:
            errors.append(f"Unsupported provider: {config_dict['provider']}")

        # Validate numeric fields
        if "max_results" in config_dict:
            try:
                max_results = int(config_dict["max_results"])
                if max_results <= 0 or max_results > 50:
                    errors.append("max_results must be between 1 and 50")
            except (ValueError, TypeError):
                errors.append("max_results must be a positive integer")

        if "min_score" in config_dict:
            try:
                min_score = float(config_dict["min_score"])
                if min_score < 0 or min_score > 10:
                    errors.append("min_score must be between 0 and 10")
            except (ValueError, TypeError):
                errors.append("min_score must be a number between 0 and 10")

        return len(errors) == 0, errors

    def _validate_fundamental_criteria(self, criteria_list: List[Dict[str, Any]]) -> List[str]:
        """Validate fundamental criteria."""
        errors = []

        if not isinstance(criteria_list, list):
            return ["fundamental_criteria must be a list"]

        for i, criteria in enumerate(criteria_list):
            if not isinstance(criteria, dict):
                errors.append(f"Fundamental criteria {i} must be an object")
                continue

            indicator = criteria.get("indicator")
            if not indicator:
                errors.append(f"Fundamental criteria {i} missing 'indicator' field")
            elif indicator not in SUPPORTED_FUNDAMENTAL_INDICATORS:
                errors.append(f"Unsupported fundamental indicator: {indicator}")

            operator = criteria.get("operator")
            if operator not in ["min", "max", "range"]:
                errors.append(f"Invalid operator '{operator}' in fundamental criteria {i}")

            if "value" not in criteria:
                errors.append(f"Fundamental criteria {i} missing 'value' field")

            weight = criteria.get("weight", 1.0)
            try:
                weight = float(weight)
                if weight < 0:
                    errors.append(f"Weight must be non-negative in fundamental criteria {i}")
            except (ValueError, TypeError):
                errors.append(f"Weight must be a number in fundamental criteria {i}")

        return errors

    def _validate_technical_criteria(self, criteria_list: List[Dict[str, Any]]) -> List[str]:
        """Validate technical criteria."""
        errors = []

        if not isinstance(criteria_list, list):
            return ["technical_criteria must be a list"]

        for i, criteria in enumerate(criteria_list):
            if not isinstance(criteria, dict):
                errors.append(f"Technical criteria {i} must be an object")
                continue

            indicator = criteria.get("indicator")
            if not indicator:
                errors.append(f"Technical criteria {i} missing 'indicator' field")
            elif indicator not in SUPPORTED_TECHNICAL_INDICATORS:
                errors.append(f"Unsupported technical indicator: {indicator}")

            if "condition" not in criteria:
                errors.append(f"Technical criteria {i} missing 'condition' field")

            weight = criteria.get("weight", 1.0)
            try:
                weight = float(weight)
                if weight < 0:
                    errors.append(f"Weight must be non-negative in technical criteria {i}")
            except (ValueError, TypeError):
                errors.append(f"Weight must be a number in technical criteria {i}")

        return errors

    def _validate_fmp_criteria(self, fmp_criteria: Dict[str, Any]) -> List[str]:
        """Validate FMP criteria."""
        errors = []

        if not isinstance(fmp_criteria, dict):
            return ["fmp_criteria must be a dictionary"]

        # Import FMP integration for validation
        try:
            from src.telegram.screener.fmp_integration import validate_fmp_criteria
            is_valid, fmp_errors = validate_fmp_criteria(fmp_criteria)
            if not is_valid:
                errors.extend(fmp_errors)
        except ImportError:
            # Fallback validation if FMP integration is not available
            supported_criteria = {
                "marketCapMoreThan", "marketCapLowerThan",
                "peRatioLessThan", "peRatioMoreThan",
                "priceToBookRatioLessThan", "priceToBookRatioMoreThan",
                "priceToSalesRatioLessThan", "priceToSalesRatioMoreThan",
                "debtToEquityLessThan", "debtToEquityMoreThan",
                "currentRatioMoreThan", "currentRatioLessThan",
                "quickRatioMoreThan", "quickRatioLessThan",
                "returnOnEquityMoreThan", "returnOnEquityLessThan",
                "returnOnAssetsMoreThan", "returnOnAssetsLessThan",
                "returnOnCapitalEmployedMoreThan", "returnOnCapitalEmployedLessThan",
                "dividendYieldMoreThan", "dividendYieldLessThan",
                "payoutRatioLessThan", "payoutRatioMoreThan",
                "betaLessThan", "betaMoreThan",
                "exchange", "Country", "isETF", "isFund", "isActivelyTrading", "limit"
            }

            invalid_criteria = set(fmp_criteria.keys()) - supported_criteria
            if invalid_criteria:
                errors.append(f"Unsupported FMP criteria: {', '.join(invalid_criteria)}")

        return errors

    def create_sample_configs(self) -> Dict[str, str]:
        """Create sample configurations for different screener types."""
        samples = {
            "fundamental_only": json.dumps({
                "screener_type": "fundamental",
                "list_type": "us_medium_cap",
                "fundamental_criteria": [
                    {
                        "indicator": "PE",
                        "operator": "max",
                        "value": 15,
                        "weight": 1.0,
                        "required": True
                    },
                    {
                        "indicator": "PB",
                        "operator": "max",
                        "value": 1.5,
                        "weight": 0.8,
                        "required": True
                    },
                    {
                        "indicator": "ROE",
                        "operator": "min",
                        "value": 15,
                        "weight": 0.9,
                        "required": False
                    }
                ],
                "max_results": 10,
                "min_score": 7.0,
                "email": True
            }, indent=2),

            "hybrid_fundamental_technical": json.dumps({
                "screener_type": "hybrid",
                "list_type": "us_medium_cap",
                "fundamental_criteria": [
                    {
                        "indicator": "PE",
                        "operator": "max",
                        "value": 15,
                        "weight": 1.0,
                        "required": True
                    },
                    {
                        "indicator": "ROE",
                        "operator": "min",
                        "value": 12,
                        "weight": 0.8,
                        "required": False
                    }
                ],
                "technical_criteria": [
                    {
                        "indicator": "RSI",
                        "parameters": {"period": 14},
                        "condition": {"operator": "<", "value": 70},
                        "weight": 0.6,
                        "required": False
                    },
                    {
                        "indicator": "BollingerBands",
                        "parameters": {"period": 20, "deviation": 2},
                        "condition": {"operator": "not_above_upper_band"},
                        "weight": 0.5,
                        "required": False
                    }
                ],
                "period": "6mo",
                "interval": "1d",
                "max_results": 15,
                "min_score": 6.5,
                "include_technical_analysis": True,
                "include_fundamental_analysis": True,
                "email": True
            }, indent=2),

            "technical_only": json.dumps({
                "screener_type": "technical",
                "list_type": "us_large_cap",
                "technical_criteria": [
                    {
                        "indicator": "RSI",
                        "parameters": {"period": 14},
                        "condition": {"operator": "<", "value": 30},
                        "weight": 1.0,
                        "required": True
                    },
                    {
                        "indicator": "MACD",
                        "parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
                        "condition": {"operator": "above_signal"},
                        "weight": 0.8,
                        "required": False
                    }
                ],
                "period": "3mo",
                "interval": "1d",
                "max_results": 8,
                "min_score": 7.5,
                "include_technical_analysis": True,
                "include_fundamental_analysis": False,
                "email": False
            }, indent=2),

            "advanced_hybrid": json.dumps({
                "screener_type": "hybrid",
                "list_type": "us_small_cap",
                "fundamental_criteria": [
                    {
                        "indicator": "PE",
                        "operator": "max",
                        "value": 12,
                        "weight": 1.0,
                        "required": True
                    },
                    {
                        "indicator": "PB",
                        "operator": "max",
                        "value": 1.2,
                        "weight": 0.9,
                        "required": True
                    },
                    {
                        "indicator": "ROE",
                        "operator": "min",
                        "value": 18,
                        "weight": 0.8,
                        "required": False
                    },
                    {
                        "indicator": "Debt_Equity",
                        "operator": "max",
                        "value": 0.4,
                        "weight": 0.7,
                        "required": False
                    }
                ],
                "technical_criteria": [
                    {
                        "indicator": "RSI",
                        "parameters": {"period": 14},
                        "condition": {"operator": "range", "min": 30, "max": 70},
                        "weight": 0.6,
                        "required": False
                    },
                    {
                        "indicator": "BollingerBands",
                        "parameters": {"period": 20, "deviation": 2},
                        "condition": {"operator": "between_bands"},
                        "weight": 0.5,
                        "required": False
                    },
                    {
                        "indicator": "SMA",
                        "parameters": {"period": 50},
                        "condition": {"operator": "above", "value": "close"},
                        "weight": 0.4,
                        "required": False
                    }
                ],
                "period": "1y",
                "interval": "1d",
                "provider": "yf",
                "max_results": 12,
                "min_score": 7.0,
                "include_technical_analysis": True,
                "include_fundamental_analysis": True,
                "email": True
            }, indent=2),

            "fmp_enhanced_screener": json.dumps({
                "screener_type": "hybrid",
                "list_type": "us_medium_cap",
                "fmp_criteria": {
                    "marketCapMoreThan": 2000000000,
                    "peRatioLessThan": 20,
                    "returnOnEquityMoreThan": 0.12,
                    "debtToEquityLessThan": 0.5,
                    "limit": 50
                },
                "fundamental_criteria": [
                    {
                        "indicator": "PE",
                        "operator": "max",
                        "value": 15,
                        "weight": 1.0,
                        "required": True
                    },
                    {
                        "indicator": "ROE",
                        "operator": "min",
                        "value": 15,
                        "weight": 0.9,
                        "required": False
                    }
                ],
                "technical_criteria": [
                    {
                        "indicator": "RSI",
                        "parameters": {"period": 14},
                        "condition": {"operator": "<", "value": 70},
                        "weight": 0.6,
                        "required": False
                    }
                ],
                "period": "6mo",
                "interval": "1d",
                "max_results": 15,
                "min_score": 7.0,
                "include_technical_analysis": True,
                "include_fundamental_analysis": True,
                "email": True
            }, indent=2),

            "fmp_strategy_screener": json.dumps({
                "screener_type": "hybrid",
                "list_type": "us_large_cap",
                "fmp_strategy": "conservative_value",
                "fundamental_criteria": [
                    {
                        "indicator": "PE",
                        "operator": "max",
                        "value": 12,
                        "weight": 1.0,
                        "required": True
                    }
                ],
                "technical_criteria": [
                    {
                        "indicator": "BollingerBands",
                        "parameters": {"period": 20, "deviation": 2},
                        "condition": {"operator": "between_bands"},
                        "weight": 0.5,
                        "required": False
                    }
                ],
                "period": "1y",
                "interval": "1d",
                "max_results": 10,
                "min_score": 7.5,
                "include_technical_analysis": True,
                "include_fundamental_analysis": True,
                "email": True
            }, indent=2)
        }
        return samples

    def get_required_fields(self, screener_type: str) -> List[str]:
        """Get required fields for a screener type."""
        base_fields = ["list_type"]

        if screener_type == "fundamental":
            return base_fields + ["fundamental_criteria"]
        elif screener_type == "technical":
            return base_fields + ["technical_criteria"]
        elif screener_type == "hybrid":
            return base_fields + ["fundamental_criteria", "technical_criteria"]
        else:
            return base_fields

    def get_optional_fields(self, screener_type: str) -> List[str]:
        """Get optional fields for a screener type."""
        common_fields = [
            "screener_type", "period", "interval", "provider", "max_results",
            "min_score", "include_technical_analysis", "include_fundamental_analysis", "email"
        ]

        if screener_type == "fundamental":
            return common_fields + ["technical_criteria"]
        elif screener_type == "technical":
            return common_fields + ["fundamental_criteria"]
        elif screener_type == "hybrid":
            return common_fields
        else:
            return common_fields


# Convenience functions
def parse_screener_config(config_json: str) -> ScreenerConfig:
    """Parse screener configuration from JSON string."""
    parser = ScreenerConfigParser()
    return parser.parse_config(config_json)

def validate_screener_config(config_json: str) -> Tuple[bool, List[str]]:
    """Validate screener configuration."""
    parser = ScreenerConfigParser()
    return parser.validate_config(config_json)

def get_screener_summary(config_json: str) -> Dict[str, Any]:
    """Get a summary of the screener configuration."""
    try:
        config = parse_screener_config(config_json)
        summary = {
            "screener_type": config.screener_type,
            "list_type": config.list_type,
            "fundamental_criteria_count": len(config.fundamental_criteria) if config.fundamental_criteria else 0,
            "technical_criteria_count": len(config.technical_criteria) if config.technical_criteria else 0,
            "period": config.period,
            "interval": config.interval,
            "max_results": config.max_results,
            "min_score": config.min_score,
            "include_technical_analysis": config.include_technical_analysis,
            "include_fundamental_analysis": config.include_fundamental_analysis,
            "email": config.email
        }

        # Add FMP information if available
        if config.fmp_criteria:
            summary["fmp_criteria_count"] = len(config.fmp_criteria)
            summary["fmp_criteria_keys"] = list(config.fmp_criteria.keys())

        if config.fmp_strategy:
            summary["fmp_strategy"] = config.fmp_strategy

        return summary
    except Exception as e:
        return {"error": str(e)}
