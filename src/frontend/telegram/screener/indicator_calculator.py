#!/usr/bin/env python3
"""
Indicator Calculator
Calculates technical indicators for alert evaluation.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import talib
from typing import Dict, Any, Optional, Tuple
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class IndicatorCalculator:
    """
    Calculator for technical indicators used in alert evaluation.
    """

    def __init__(self):
        """Initialize the indicator calculator."""
        pass

    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI) using TALib.

        Args:
            data: DataFrame with 'close' column
            period: RSI period (default: 14)

        Returns:
            RSI values as pandas Series
        """
        try:
            if len(data) < period + 1:
                _logger.warning("Insufficient data for RSI calculation: %d < %d", len(data), period + 1)
                return pd.Series([np.nan] * len(data))

            # Use TALib for RSI calculation
            rsi = talib.RSI(data['close'].values, timeperiod=period)
            return pd.Series(rsi, index=data.index)

        except Exception as e:
            _logger.error("Error calculating RSI with TALib: %s", e)
            # Fallback to manual calculation if TALib fails
            try:
                delta = data['close'].diff()
                gains = delta.where(delta > 0, 0)
                losses = -delta.where(delta < 0, 0)
                avg_gains = gains.rolling(window=period).mean()
                avg_losses = losses.rolling(window=period).mean()
                rs = avg_gains / avg_losses
                rsi = 100 - (100 / (1 + rs))
                return rsi
            except Exception as fallback_error:
                _logger.error("Fallback RSI calculation also failed: %s", fallback_error)
                return pd.Series([np.nan] * len(data))

    def calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20, deviation: float = 2) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands using TALib.

        Args:
            data: DataFrame with 'close' column
            period: Moving average period (default: 20)
            deviation: Standard deviation multiplier (default: 2)

        Returns:
            Dictionary with 'upper', 'middle', 'lower' bands
        """
        try:
            if len(data) < period:
                _logger.warning("Insufficient data for Bollinger Bands: %d < %d", len(data), period)
                return {
                    'upper': pd.Series([np.nan] * len(data)),
                    'middle': pd.Series([np.nan] * len(data)),
                    'lower': pd.Series([np.nan] * len(data))
                }

            # Use TALib for Bollinger Bands calculation
            upper, middle, lower = talib.BBANDS(
                data['close'].values,
                timeperiod=period,
                nbdevup=deviation,
                nbdevdn=deviation,
                matype=0  # Simple Moving Average
            )

            return {
                'upper': pd.Series(upper, index=data.index),
                'middle': pd.Series(middle, index=data.index),
                'lower': pd.Series(lower, index=data.index)
            }

        except Exception as e:
            _logger.error("Error calculating Bollinger Bands with TALib: %s", e)
            # Fallback to manual calculation if TALib fails
            try:
                middle = data['close'].rolling(window=period).mean()
                std = data['close'].rolling(window=period).std()
                upper = middle + (std * deviation)
                lower = middle - (std * deviation)
                return {
                    'upper': upper,
                    'middle': middle,
                    'lower': lower
                }
            except Exception as fallback_error:
                _logger.error("Fallback Bollinger Bands calculation also failed: %s", fallback_error)
                return {
                    'upper': pd.Series([np.nan] * len(data)),
                    'middle': pd.Series([np.nan] * len(data)),
                    'lower': pd.Series([np.nan] * len(data))
                }

    def calculate_macd(self, data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence) using TALib.

        Args:
            data: DataFrame with 'close' column
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line period (default: 9)

        Returns:
            Dictionary with 'macd', 'signal', 'histogram'
        """
        try:
            if len(data) < slow_period + signal_period:
                _logger.warning("Insufficient data for MACD: %d < %d", len(data), slow_period + signal_period)
                return {
                    'macd': pd.Series([np.nan] * len(data)),
                    'signal': pd.Series([np.nan] * len(data)),
                    'histogram': pd.Series([np.nan] * len(data))
                }

            # Use TALib for MACD calculation
            macd, signal, histogram = talib.MACD(
                data['close'].values,
                fastperiod=fast_period,
                slowperiod=slow_period,
                signalperiod=signal_period
            )

            return {
                'macd': pd.Series(macd, index=data.index),
                'signal': pd.Series(signal, index=data.index),
                'histogram': pd.Series(histogram, index=data.index)
            }

        except Exception as e:
            _logger.error("Error calculating MACD with TALib: %s", e)
            # Fallback to manual calculation if TALib fails
            try:
                ema_fast = data['close'].ewm(span=fast_period).mean()
                ema_slow = data['close'].ewm(span=slow_period).mean()
                macd = ema_fast - ema_slow
                signal = macd.ewm(span=signal_period).mean()
                histogram = macd - signal
                return {
                    'macd': macd,
                    'signal': signal,
                    'histogram': histogram
                }
            except Exception as fallback_error:
                _logger.error("Fallback MACD calculation also failed: %s", fallback_error)
                return {
                    'macd': pd.Series([np.nan] * len(data)),
                    'signal': pd.Series([np.nan] * len(data)),
                    'histogram': pd.Series([np.nan] * len(data))
                }

    def calculate_sma(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA) using TALib.

        Args:
            data: DataFrame with 'close' column
            period: SMA period (default: 20)

        Returns:
            SMA values as pandas Series
        """
        try:
            if len(data) < period:
                _logger.warning("Insufficient data for SMA: %d < %d", len(data), period)
                return pd.Series([np.nan] * len(data))

            # Use TALib for SMA calculation
            sma = talib.SMA(data['close'].values, timeperiod=period)
            return pd.Series(sma, index=data.index)

        except Exception as e:
            _logger.error("Error calculating SMA with TALib: %s", e)
            # Fallback to manual calculation if TALib fails
            try:
                return data['close'].rolling(window=period).mean()
            except Exception as fallback_error:
                _logger.error("Fallback SMA calculation also failed: %s", fallback_error)
                return pd.Series([np.nan] * len(data))

    def evaluate_condition(self, indicator_name: str, indicator_value: Any, condition: Dict[str, Any],
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

    def _evaluate_rsi_condition(self, rsi_value: float, operator: str, value: float) -> bool:
        """Evaluate RSI condition."""
        if pd.isna(rsi_value) or value is None:
            return False

        if operator == "<":
            return rsi_value < value
        elif operator == ">":
            return rsi_value > value
        elif operator == "<=":
            return rsi_value <= value
        elif operator == ">=":
            return rsi_value >= value
        elif operator == "==":
            return rsi_value == value
        elif operator == "!=":
            return rsi_value != value
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
        prev_macd = macd.iloc[-2]
        prev_signal = signal.iloc[-2]

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

    def _evaluate_sma_condition(self, sma_value: float, current_price: float, operator: str, value: float = None) -> bool:
        """Evaluate SMA condition."""
        if pd.isna(sma_value):
            return False

        if operator in ["<", ">", "<=", ">=", "==", "!="]:
            # Compare SMA with a value
            if value is None:
                return False
            if operator == "<":
                return sma_value < value
            elif operator == ">":
                return sma_value > value
            elif operator == "<=":
                return sma_value <= value
            elif operator == ">=":
                return sma_value >= value
            elif operator == "==":
                return sma_value == value
            elif operator == "!=":
                return sma_value != value
        elif operator in ["crossover", "crossunder"]:
            # Compare SMA with current price
            if current_price is None:
                return False
            if operator == "crossover":
                return current_price > sma_value
            elif operator == "crossunder":
                return current_price < sma_value
        else:
            _logger.warning("Unknown SMA operator: %s", operator)
            return False

        return False

    def calculate_all_indicators(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate all required indicators for alert evaluation.

        Args:
            data: DataFrame with OHLCV data
            indicators: Dictionary of indicator configurations

        Returns:
            Dictionary with calculated indicator values
        """
        results = {}

        for indicator_name, config in indicators.items():
            try:
                if indicator_name == "RSI":
                    period = config.get("period", 14)
                    results[indicator_name] = self.calculate_rsi(data, period)
                elif indicator_name == "BollingerBands":
                    period = config.get("period", 20)
                    deviation = config.get("deviation", 2)
                    results[indicator_name] = self.calculate_bollinger_bands(data, period, deviation)
                elif indicator_name == "MACD":
                    fast_period = config.get("fast_period", 12)
                    slow_period = config.get("slow_period", 26)
                    signal_period = config.get("signal_period", 9)
                    results[indicator_name] = self.calculate_macd(data, fast_period, slow_period, signal_period)
                elif indicator_name == "SMA":
                    period = config.get("period", 20)
                    results[indicator_name] = self.calculate_sma(data, period)
                else:
                    _logger.warning("Unknown indicator: %s", indicator_name)

            except Exception as e:
                _logger.error("Error calculating %s: %s", indicator_name, e)
                results[indicator_name] = None

        return results


# Convenience functions
def calculate_indicator(data: pd.DataFrame, indicator_name: str, **kwargs) -> Any:
    """Calculate a single indicator."""
    calculator = IndicatorCalculator()

    if indicator_name == "RSI":
        return calculator.calculate_rsi(data, **kwargs)
    elif indicator_name == "BollingerBands":
        return calculator.calculate_bollinger_bands(data, **kwargs)
    elif indicator_name == "MACD":
        return calculator.calculate_macd(data, **kwargs)
    elif indicator_name == "SMA":
        return calculator.calculate_sma(data, **kwargs)
    else:
        raise ValueError(f"Unknown indicator: {indicator_name}")


if __name__ == "__main__":
    # Test the calculator
    print("Testing Indicator Calculator...")

    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    prices = np.random.randn(100).cumsum() + 100  # Random walk starting at 100

    data = pd.DataFrame({
        'open': prices,
        'high': prices + np.random.rand(100) * 2,
        'low': prices - np.random.rand(100) * 2,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

    calculator = IndicatorCalculator()

    # Test RSI
    rsi = calculator.calculate_rsi(data, 14)
    print(f"✅ RSI calculated: {rsi.iloc[-1]:.2f}")

    # Test Bollinger Bands
    bb = calculator.calculate_bollinger_bands(data, 20, 2)
    print(f"✅ Bollinger Bands calculated: Upper={bb['upper'].iloc[-1]:.2f}, Lower={bb['lower'].iloc[-1]:.2f}")

    # Test MACD
    macd = calculator.calculate_macd(data, 12, 26, 9)
    print(f"✅ MACD calculated: {macd['macd'].iloc[-1]:.2f}")

    # Test SMA
    sma = calculator.calculate_sma(data, 20)
    print(f"✅ SMA calculated: {sma.iloc[-1]:.2f}")

    print("🎉 All indicators calculated successfully!")
