#!/usr/bin/env python3
"""
Alert Logic Evaluator
Evaluates alert conditions using parsed configurations and calculated indicators.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from src.notification.logger import setup_logger
from src.telegram.screener.alert_config_parser import AlertConfig, IndicatorConfig, parse_alert_config
from src.indicators.service import IndicatorService
from src.indicators.models import TickerIndicatorsRequest, IndicatorResultSet
from src.common import get_ohlcv, determine_provider, get_ticker_info

_logger = setup_logger(__name__)


class AlertLogicEvaluator:
    """
    Evaluates alert conditions using parsed configurations and calculated indicators.
    """

    def __init__(self, indicator_service: Optional[IndicatorService] = None):
        """Initialize the alert logic evaluator."""
        self.indicator_service = indicator_service or IndicatorService()

    async def evaluate_alert(self, alert: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
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
                return await self._evaluate_indicator_alert(alert)
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

    async def _evaluate_indicator_alert(self, alert: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
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

            # Calculate indicators using IndicatorService
            indicators = await self._calculate_indicators(data, config, ticker)

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
            # Determine provider based on ticker characteristics
            provider = determine_provider(ticker)

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
            # Determine provider based on ticker characteristics
            provider = determine_provider(ticker)

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

    async def _calculate_indicators(self, data: pd.DataFrame, config: AlertConfig, ticker: str) -> Dict[str, Any]:
        """Calculate all required indicators for the alert configuration using IndicatorService."""
        indicators = {}

        # Collect all unique indicators needed
        indicator_names = []
        for condition in config.conditions:
            indicator_name = condition.name
            if indicator_name not in indicator_names:
                # Map indicator names to service names
                if indicator_name == "BollingerBands":
                    indicator_names.append("bbands")
                elif indicator_name == "RSI":
                    indicator_names.append("rsi")
                elif indicator_name == "MACD":
                    indicator_names.append("macd")
                elif indicator_name == "SMA":
                    indicator_names.append("sma")
                else:
                    _logger.warning("Unknown indicator: %s", indicator_name)

        if not indicator_names:
            return indicators

        try:
            # Create request for IndicatorService
            request = TickerIndicatorsRequest(
                ticker=ticker,
                timeframe=config.timeframe or "1d",
                period="100d",  # Get enough data for calculations
                indicators=indicator_names
            )

            # Get indicators from service
            result_set = await self.indicator_service.compute_for_ticker(request)

            # Map results back to expected format
            for condition in config.conditions:
                indicator_name = condition.name
                parameters = condition.parameters

                try:
                    if indicator_name == "RSI":
                        rsi_value = result_set.technical.get("rsi")
                        if rsi_value and rsi_value.value is not None:
                            # Create a series-like object for compatibility
                            indicators[indicator_name] = pd.Series([rsi_value.value], index=[data.index[-1]])
                        else:
                            indicators[indicator_name] = pd.Series([np.nan], index=[data.index[-1]])

                    elif indicator_name == "BollingerBands":
                        bb_upper = result_set.technical.get("bbands_upper")
                        bb_middle = result_set.technical.get("bbands_middle")
                        bb_lower = result_set.technical.get("bbands_lower")

                        if bb_upper and bb_middle and bb_lower:
                            indicators[indicator_name] = {
                                'upper': pd.Series([bb_upper.value], index=[data.index[-1]]),
                                'middle': pd.Series([bb_middle.value], index=[data.index[-1]]),
                                'lower': pd.Series([bb_lower.value], index=[data.index[-1]])
                            }
                        else:
                            indicators[indicator_name] = {
                                'upper': pd.Series([np.nan], index=[data.index[-1]]),
                                'middle': pd.Series([np.nan], index=[data.index[-1]]),
                                'lower': pd.Series([np.nan], index=[data.index[-1]])
                            }

                    elif indicator_name == "MACD":
                        macd_value = result_set.technical.get("macd_macd")
                        signal_value = result_set.technical.get("macd_signal")
                        histogram_value = result_set.technical.get("macd_histogram")

                        if macd_value and signal_value and histogram_value:
                            indicators[indicator_name] = {
                                'macd': pd.Series([macd_value.value], index=[data.index[-1]]),
                                'signal': pd.Series([signal_value.value], index=[data.index[-1]]),
                                'histogram': pd.Series([histogram_value.value], index=[data.index[-1]])
                            }
                        else:
                            indicators[indicator_name] = {
                                'macd': pd.Series([np.nan], index=[data.index[-1]]),
                                'signal': pd.Series([np.nan], index=[data.index[-1]]),
                                'histogram': pd.Series([np.nan], index=[data.index[-1]])
                            }

                    elif indicator_name == "SMA":
                        sma_value = result_set.technical.get("sma")
                        if sma_value and sma_value.value is not None:
                            indicators[indicator_name] = pd.Series([sma_value.value], index=[data.index[-1]])
                        else:
                            indicators[indicator_name] = pd.Series([np.nan], index=[data.index[-1]])

                except Exception as e:
                    _logger.error("Error processing %s result: %s", indicator_name, e)
                    indicators[indicator_name] = None

        except Exception as e:
            _logger.exception("Error calculating indicators using IndicatorService: %s", e)
            # Fallback: set all indicators to None
            for condition in config.conditions:
                indicators[condition.name] = None

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
                result = self._evaluate_condition(
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

    def _evaluate_condition(self, indicator_name: str, indicator_value: Any, condition: Dict[str, Any],
                          current_price: float = None, previous_data: pd.DataFrame = None) -> bool:
        """
        Evaluate if an indicator condition is met.

        Args:
            indicator_name: Name of the indicator
            indicator_value: Current indicator value(s)
            condition: Condition dictionary with operator and value
            current_price: Current price (for price-based conditions)
            previous_data: Previous data for crossover detection

        Returns:
            True if condition is met, False otherwise
        """
        try:
            operator = condition.get("operator")
            value = condition.get("value")

            if indicator_name == "PRICE":
                return self._evaluate_price_condition(current_price, operator, value)
            elif indicator_name == "RSI":
                return self._evaluate_rsi_condition(indicator_value, operator, value)
            elif indicator_name == "BollingerBands":
                return self._evaluate_bollinger_condition(indicator_value, current_price, operator)
            elif indicator_name == "MACD":
                return self._evaluate_macd_condition(indicator_value, previous_data, operator)
            elif indicator_name == "SMA":
                return self._evaluate_sma_condition(indicator_value, current_price, operator, value)
            else:
                _logger.warning("Unknown indicator for evaluation: %s", indicator_name)
                return False

        except Exception as e:
            _logger.error("Error evaluating condition for %s: %s", indicator_name, e)
            return False

    def _evaluate_price_condition(self, current_price: float, operator: str, value: float) -> bool:
        """Evaluate price-based condition."""
        if current_price is None or value is None:
            return False

        if operator == "above":
            return current_price > value
        elif operator == "below":
            return current_price < value
        elif operator == ">":
            return current_price > value
        elif operator == "<":
            return current_price < value
        elif operator == ">=":
            return current_price >= value
        elif operator == "<=":
            return current_price <= value
        elif operator == "==":
            return current_price == value
        elif operator == "!=":
            return current_price != value
        else:
            _logger.warning("Unknown price operator: %s", operator)
            return False

    def _evaluate_rsi_condition(self, rsi_value: pd.Series, operator: str, value: float) -> bool:
        """Evaluate RSI condition."""
        if rsi_value is None or rsi_value.empty or pd.isna(rsi_value.iloc[-1]) or value is None:
            return False

        current_rsi = rsi_value.iloc[-1]

        if operator == "<":
            return current_rsi < value
        elif operator == ">":
            return current_rsi > value
        elif operator == "<=":
            return current_rsi <= value
        elif operator == ">=":
            return current_rsi >= value
        elif operator == "==":
            return current_rsi == value
        elif operator == "!=":
            return current_rsi != value
        else:
            _logger.warning("Unknown RSI operator: %s", operator)
            return False

    def _evaluate_bollinger_condition(self, bb_values: Dict[str, pd.Series], current_price: float, operator: str) -> bool:
        """Evaluate Bollinger Bands condition."""
        if current_price is None or bb_values is None:
            return False

        # Get the latest values
        upper = bb_values['upper'].iloc[-1] if not bb_values['upper'].empty else np.nan
        lower = bb_values['lower'].iloc[-1] if not bb_values['lower'].empty else np.nan

        if pd.isna(upper) or pd.isna(lower):
            return False

        if operator == "above_upper_band":
            return current_price > upper
        elif operator == "below_lower_band":
            return current_price < lower
        elif operator == "between_bands":
            return lower <= current_price <= upper
        else:
            _logger.warning("Unknown Bollinger Bands operator: %s", operator)
            return False

    def _evaluate_macd_condition(self, macd_values: Dict[str, pd.Series], previous_data: pd.DataFrame, operator: str) -> bool:
        """Evaluate MACD condition."""
        if macd_values is None:
            return False

        macd = macd_values['macd']
        signal = macd_values['signal']

        if len(macd) < 2 or len(signal) < 2:
            return False

        current_macd = macd.iloc[-1]
        current_signal = signal.iloc[-1]
        prev_macd = macd.iloc[-2] if len(macd) > 1 else current_macd
        prev_signal = signal.iloc[-2] if len(signal) > 1 else current_signal

        if pd.isna(current_macd) or pd.isna(current_signal) or pd.isna(prev_macd) or pd.isna(prev_signal):
            return False

        if operator == "crossover":
            # MACD crosses above signal line
            return prev_macd <= prev_signal and current_macd > current_signal
        elif operator == "crossunder":
            # MACD crosses below signal line
            return prev_macd >= prev_signal and current_macd < current_signal
        elif operator == "above_signal":
            return current_macd > current_signal
        elif operator == "below_signal":
            return current_macd < current_signal
        else:
            _logger.warning("Unknown MACD operator: %s", operator)
            return False

    def _evaluate_sma_condition(self, sma_value: pd.Series, current_price: float, operator: str, value: float = None) -> bool:
        """Evaluate SMA condition."""
        if sma_value is None or sma_value.empty or pd.isna(sma_value.iloc[-1]):
            return False

        current_sma = sma_value.iloc[-1]

        if operator in ["<", ">", "<=", ">=", "==", "!="]:
            # Compare SMA with a value
            if value is None:
                return False
            if operator == "<":
                return current_sma < value
            elif operator == ">":
                return current_sma > value
            elif operator == "<=":
                return current_sma <= value
            elif operator == ">=":
                return current_sma >= value
            elif operator == "==":
                return current_sma == value
            elif operator == "!=":
                return current_sma != value
        elif operator in ["crossover", "crossunder"]:
            # Compare SMA with current price
            if current_price is None:
                return False
            if operator == "crossover":
                return current_price > current_sma
            elif operator == "crossunder":
                return current_price < current_sma
        else:
            _logger.warning("Unknown SMA operator: %s", operator)
            return False

        return False

    def validate_alert_config(self, config_json: str) -> Tuple[bool, List[str]]:
        """Validate alert configuration."""
        try:
            from src.telegram.screener.alert_config_parser import validate_alert_config
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
async def evaluate_alert(alert: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate if an alert should be triggered."""
    evaluator = AlertLogicEvaluator()
    return await evaluator.evaluate_alert(alert)


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
    async def test_evaluator():
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

        triggered, details = await evaluator.evaluate_alert(price_alert)
        print(f"Price Alert Test: Triggered={triggered}, Details={details}")

        # Test indicator alert
        indicator_alert = {
            "id": 2,
            "ticker": "AAPL",
            "alert_type": "indicator",
            "config_json": '{"type":"indicator","indicator":"RSI","parameters":{"period":14},"condition":{"operator":"<","value":30},"alert_action":"BUY","timeframe":"15m"}',
            "timeframe": "15m"
        }

        triggered, details = await evaluator.evaluate_alert(indicator_alert)
        print(f"Indicator Alert Test: Triggered={triggered}, Details={details}")

        print("🎉 Alert Logic Evaluator tests completed!")

    asyncio.run(test_evaluator())
