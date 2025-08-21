#!/usr/bin/env python3
"""
Alert Logic Evaluator
Evaluates alert conditions using parsed configurations and calculated indicators.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from src.notification.logger import setup_logger
from src.frontend.telegram.screener.alert_config_parser import AlertConfig, IndicatorConfig, parse_alert_config
from src.frontend.telegram.screener.indicator_calculator import IndicatorCalculator
from src.common import get_ohlcv

_logger = setup_logger(__name__)


class AlertLogicEvaluator:
    """
    Evaluates alert conditions using parsed configurations and calculated indicators.
    """

    def __init__(self):
        """Initialize the alert logic evaluator."""
        self.calculator = IndicatorCalculator()

    def evaluate_alert(self, alert: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate if an alert should be triggered.

        Args:
            alert: Alert dictionary from database

        Returns:
            Tuple of (should_trigger, evaluation_details)
        """
        try:
            alert_type = alert.get("alert_type", "price")

            if alert_type == "price":
                return self._evaluate_price_alert(alert)
            elif alert_type == "indicator":
                return self._evaluate_indicator_alert(alert)
            else:
                _logger.warning("Unknown alert type: %s", alert_type)
                return False, {"error": f"Unknown alert type: {alert_type}"}

        except Exception as e:
            _logger.exception("Error evaluating alert %s: ", alert.get('id'))
            return False, {"error": str(e)}

    def _evaluate_price_alert(self, alert: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate price-based alert."""
        try:
            ticker = alert["ticker"]
            target_price = alert["price"]
            condition = alert["condition"]
            timeframe = alert.get("timeframe", "15m")

            # Get current price
            current_price = self._get_current_price(ticker, timeframe)
            if current_price is None:
                return False, {"error": f"Could not get current price for {ticker}"}

            # Evaluate condition
            triggered = False
            if condition == "above" and current_price > target_price:
                triggered = True
            elif condition == "below" and current_price < target_price:
                triggered = True

            details = {
                "ticker": ticker,
                "current_price": current_price,
                "target_price": target_price,
                "condition": condition,
                "triggered": triggered
            }

            return triggered, details

        except Exception as e:
            _logger.exception("Error evaluating price alert: ")
            return False, {"error": str(e)}

    def _evaluate_indicator_alert(self, alert: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate indicator-based alert."""
        try:
            ticker = alert["ticker"]
            config_json = alert.get("config_json")
            timeframe = alert.get("timeframe", "15m")

            if not config_json:
                return False, {"error": "No configuration JSON found"}

            # Parse configuration
            config = parse_alert_config(config_json)

            # Get market data
            data = self._get_market_data(ticker, timeframe, config)
            if data is None or data.empty:
                return False, {"error": f"Could not get market data for {ticker}"}

            # Calculate indicators
            indicators = self._calculate_indicators(data, config)

            # Evaluate conditions
            triggered, condition_results = self._evaluate_conditions(config, indicators, data)

            details = {
                "ticker": ticker,
                "timeframe": timeframe,
                "config": config,
                "current_price": data['close'].iloc[-1] if not data.empty else None,
                "indicators": indicators,
                "condition_results": condition_results,
                "triggered": triggered
            }

            return triggered, details

        except Exception as e:
            _logger.exception("Error evaluating indicator alert: ")
            return False, {"error": str(e)}

    def _get_current_price(self, ticker: str, timeframe: str) -> Optional[float]:
        """Get current price for a ticker."""
        try:
            # Determine provider based on ticker length
            provider = "yf" if len(ticker) < 5 else "bnc"

            # Get recent data
            data = get_ohlcv(ticker, timeframe, "1d", provider)
            if data is None or data.empty:
                return None

            return data['close'].iloc[-1]

        except Exception as e:
            _logger.error("Error getting current price for %s: %s", ticker, e)
            return None

    def _get_market_data(self, ticker: str, timeframe: str, config: AlertConfig) -> Optional[pd.DataFrame]:
        """Get market data for indicator calculation."""
        try:
            # Determine provider based on ticker length
            provider = "yf" if len(ticker) < 5 else "bnc"

            # Calculate required data points
            required_points = self._calculate_required_data_points(config)

            # Get data (more than required to ensure we have enough)
            data = get_ohlcv(ticker, timeframe, f"{required_points + 20}d", provider)

            if data is None or data.empty:
                _logger.warning("No data available for %s", ticker)
                return None

            return data

        except Exception as e:
            _logger.error("Error getting market data for %s: %s", ticker, e)
            return None

    def _calculate_required_data_points(self, config: AlertConfig) -> int:
        """Calculate required data points for the alert configuration."""
        max_period = 50  # Default minimum

        for condition in config.conditions:
            if condition.name == "RSI":
                max_period = max(max_period, condition.parameters.get("period", 14) + 10)
            elif condition.name == "BollingerBands":
                max_period = max(max_period, condition.parameters.get("period", 20) + 10)
            elif condition.name == "MACD":
                slow_period = condition.parameters.get("slow_period", 26)
                signal_period = condition.parameters.get("signal_period", 9)
                max_period = max(max_period, slow_period + signal_period + 10)
            elif condition.name == "SMA":
                max_period = max(max_period, condition.parameters.get("period", 20) + 10)

        return max_period

    def _calculate_indicators(self, data: pd.DataFrame, config: AlertConfig) -> Dict[str, Any]:
        """Calculate all required indicators for the alert configuration."""
        indicators = {}

        for condition in config.conditions:
            indicator_name = condition.name
            parameters = condition.parameters

            try:
                if indicator_name == "RSI":
                    period = parameters.get("period", 14)
                    indicators[indicator_name] = self.calculator.calculate_rsi(data, period)
                elif indicator_name == "BollingerBands":
                    period = parameters.get("period", 20)
                    deviation = parameters.get("deviation", 2)
                    indicators[indicator_name] = self.calculator.calculate_bollinger_bands(data, period, deviation)
                elif indicator_name == "MACD":
                    fast_period = parameters.get("fast_period", 12)
                    slow_period = parameters.get("slow_period", 26)
                    signal_period = parameters.get("signal_period", 9)
                    indicators[indicator_name] = self.calculator.calculate_macd(data, fast_period, slow_period, signal_period)
                elif indicator_name == "SMA":
                    period = parameters.get("period", 20)
                    indicators[indicator_name] = self.calculator.calculate_sma(data, period)
                else:
                    _logger.warning("Unknown indicator: %s", indicator_name)

            except Exception as e:
                _logger.error("Error calculating %s: %s", indicator_name, e)
                indicators[indicator_name] = None

        return indicators

    def _evaluate_conditions(self, config: AlertConfig, indicators: Dict[str, Any], data: pd.DataFrame) -> Tuple[bool, List[Dict[str, Any]]]:
        """Evaluate all conditions in the alert configuration."""
        condition_results = []
        current_price = data['close'].iloc[-1] if not data.empty else None

        for condition in config.conditions:
            try:
                indicator_name = condition.name
                indicator_value = indicators.get(indicator_name)
                condition_config = condition.condition

                # Evaluate single condition
                result = self.calculator.evaluate_condition(
                    indicator_name=indicator_name,
                    indicator_value=indicator_value,
                    condition=condition_config,
                    current_price=current_price,
                    previous_data=data
                )

                condition_results.append({
                    "indicator": indicator_name,
                    "condition": condition_config,
                    "result": result,
                    "value": self._get_indicator_value(indicator_value, indicator_name)
                })

            except Exception as e:
                _logger.error("Error evaluating condition %s: %s", condition.name, e)
                condition_results.append({
                    "indicator": condition.name,
                    "condition": condition.condition,
                    "result": False,
                    "error": str(e)
                })

        # Apply logic (AND/OR)
        if config.logic == "AND":
            triggered = all(result["result"] for result in condition_results)
        elif config.logic == "OR":
            triggered = any(result["result"] for result in condition_results)
        else:
            # Single condition
            triggered = condition_results[0]["result"] if condition_results else False

        return triggered, condition_results

    def _get_indicator_value(self, indicator_value: Any, indicator_name: str) -> Any:
        """Get the current value of an indicator."""
        try:
            if indicator_name == "RSI":
                return indicator_value.iloc[-1] if hasattr(indicator_value, 'iloc') else indicator_value
            elif indicator_name == "BollingerBands":
                if isinstance(indicator_value, dict):
                    return {
                        "upper": indicator_value['upper'].iloc[-1] if hasattr(indicator_value['upper'], 'iloc') else indicator_value['upper'],
                        "middle": indicator_value['middle'].iloc[-1] if hasattr(indicator_value['middle'], 'iloc') else indicator_value['middle'],
                        "lower": indicator_value['lower'].iloc[-1] if hasattr(indicator_value['lower'], 'iloc') else indicator_value['lower']
                    }
                return indicator_value
            elif indicator_name == "MACD":
                if isinstance(indicator_value, dict):
                    return {
                        "macd": indicator_value['macd'].iloc[-1] if hasattr(indicator_value['macd'], 'iloc') else indicator_value['macd'],
                        "signal": indicator_value['signal'].iloc[-1] if hasattr(indicator_value['signal'], 'iloc') else indicator_value['signal'],
                        "histogram": indicator_value['histogram'].iloc[-1] if hasattr(indicator_value['histogram'], 'iloc') else indicator_value['histogram']
                    }
                return indicator_value
            elif indicator_name == "SMA":
                return indicator_value.iloc[-1] if hasattr(indicator_value, 'iloc') else indicator_value
            else:
                return indicator_value

        except Exception as e:
            _logger.error("Error getting indicator value for %s: %s", indicator_name, e)
            return None

    def validate_alert_config(self, config_json: str) -> Tuple[bool, List[str]]:
        """Validate alert configuration."""
        try:
            from src.frontend.telegram.screener.alert_config_parser import validate_alert_config
            return validate_alert_config(config_json)
        except Exception as e:
            return False, [str(e)]

    def get_alert_summary(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Get a summary of the alert for display purposes."""
        try:
            alert_type = alert.get("alert_type", "price")

            if alert_type == "price":
                return {
                    "type": "Price Alert",
                    "ticker": alert["ticker"],
                    "condition": f"{alert['condition']} ${alert['price']:.2f}",
                    "timeframe": alert.get("timeframe", "15m"),
                    "action": alert.get("alert_action", "notify")
                }
            elif alert_type == "indicator":
                config_json = alert.get("config_json")
                if config_json:
                    config = parse_alert_config(config_json)
                    return {
                        "type": "Indicator Alert",
                        "ticker": alert["ticker"],
                        "indicators": [c.name for c in config.conditions],
                        "logic": config.logic or "single",
                        "timeframe": config.timeframe,
                        "action": config.alert_action
                    }
                else:
                    return {
                        "type": "Indicator Alert",
                        "ticker": alert["ticker"],
                        "error": "Invalid configuration"
                    }
            else:
                return {
                    "type": "Unknown Alert",
                    "ticker": alert.get("ticker", "Unknown"),
                    "error": f"Unknown alert type: {alert_type}"
                }

        except Exception as e:
            _logger.error("Error getting alert summary: %s", e)
            return {
                "type": "Error",
                "ticker": alert.get("ticker", "Unknown"),
                "error": str(e)
            }


# Convenience functions
def evaluate_alert(alert: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate if an alert should be triggered."""
    evaluator = AlertLogicEvaluator()
    return evaluator.evaluate_alert(alert)


def validate_alert_config(config_json: str) -> Tuple[bool, List[str]]:
    """Validate alert configuration."""
    evaluator = AlertLogicEvaluator()
    return evaluator.validate_alert_config(config_json)


def get_alert_summary(alert: Dict[str, Any]) -> Dict[str, Any]:
    """Get alert summary for display."""
    evaluator = AlertLogicEvaluator()
    return evaluator.get_alert_summary(alert)


if __name__ == "__main__":
    # Test the evaluator
    print("Testing Alert Logic Evaluator...")

    evaluator = AlertLogicEvaluator()

    # Test price alert
    price_alert = {
        "id": 1,
        "ticker": "AAPL",
        "alert_type": "price",
        "price": 150.0,
        "condition": "below",
        "timeframe": "15m"
    }

    triggered, details = evaluator.evaluate_alert(price_alert)
    print(f"Price Alert Test: Triggered={triggered}, Details={details}")

    # Test indicator alert
    indicator_alert = {
        "id": 2,
        "ticker": "AAPL",
        "alert_type": "indicator",
        "config_json": '{"type":"indicator","indicator":"RSI","parameters":{"period":14},"condition":{"operator":"<","value":30},"alert_action":"BUY","timeframe":"15m"}',
        "timeframe": "15m"
    }

    triggered, details = evaluator.evaluate_alert(indicator_alert)
    print(f"Indicator Alert Test: Triggered={triggered}, Details={details}")

    print("🎉 Alert Logic Evaluator tests completed!")
