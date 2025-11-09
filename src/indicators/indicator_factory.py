"""
Indicator Factory

Factory for creating technical indicators for pandas DataFrames and Backtrader strategies.
This module provides adapter integration for multiple indicator backends (TALib, pandas-ta).

Primary use case: Backtrader strategy indicator creation via BacktraderAdapter.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
from src.notification.logger import setup_logger

# Unified service integration
from src.indicators.adapters.backtrader_adapter import BacktraderAdapter, BacktraderIndicatorFactory

logger = setup_logger(__name__)


class IndicatorFactory:
    """Factory class for creating technical indicators for pandas DataFrames and Backtrader strategies."""

    def __init__(self, data: Union[pd.DataFrame, Any] = None):
        self.data = data
        self.indicators = {}

        # Initialize adapter components
        self._backtrader_adapter = BacktraderAdapter()
        self._backtrader_factory = BacktraderIndicatorFactory(self._backtrader_adapter)
        logger.info("IndicatorFactory initialized with adapter integration")

    def create_rsi(self, name: str, period: int = 14) -> pd.Series:
        """Create RSI indicator."""
        if name in self.indicators:
            return self.indicators[name]

        if self.data is None:
            raise ValueError("Data not provided for RSI calculation")

        # Calculate RSI
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        self.indicators[name] = rsi
        return rsi

    def create_bollinger_bands(self, name: str, period: int = 20, devfactor: float = 2.0) -> Dict[str, pd.Series]:
        """Create Bollinger Bands indicator."""
        if name in self.indicators:
            return self.indicators[name]

        if self.data is None:
            raise ValueError("Data not provided for Bollinger Bands calculation")

        # Calculate Bollinger Bands
        sma = self.data['close'].rolling(period).mean()
        std = self.data['close'].rolling(period).std()
        upper_band = sma + (devfactor * std)
        lower_band = sma - (devfactor * std)

        bb_data = {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }

        self.indicators[name] = bb_data
        return bb_data

    def create_atr(self, name: str, period: int = 14) -> pd.Series:
        """Create ATR (Average True Range) indicator."""
        if name in self.indicators:
            return self.indicators[name]

        if self.data is None:
            raise ValueError("Data not provided for ATR calculation")

        # Calculate True Range
        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift())
        low_close = np.abs(self.data['low'] - self.data['close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean()

        self.indicators[name] = atr
        return atr

    def create_sma(self, name: str, period: int = 20) -> pd.Series:
        """Create Simple Moving Average indicator."""
        if name in self.indicators:
            return self.indicators[name]

        if self.data is None:
            raise ValueError("Data not provided for SMA calculation")

        sma = self.data['close'].rolling(period).mean()
        self.indicators[name] = sma
        return sma

    def create_ema(self, name: str, period: int = 20) -> pd.Series:
        """Create Exponential Moving Average indicator."""
        if name in self.indicators:
            return self.indicators[name]

        if self.data is None:
            raise ValueError("Data not provided for EMA calculation")

        ema = self.data['close'].ewm(span=period).mean()
        self.indicators[name] = ema
        return ema

    def create_macd(self, name: str, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, pd.Series]:
        """Create MACD indicator."""
        if name in self.indicators:
            return self.indicators[name]

        if self.data is None:
            raise ValueError("Data not provided for MACD calculation")

        # Calculate MACD
        ema_fast = self.data['close'].ewm(span=fast_period).mean()
        ema_slow = self.data['close'].ewm(span=slow_period).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        histogram = macd_line - signal_line

        macd_data = {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

        self.indicators[name] = macd_data
        return macd_data

    def create_stochastic(self, name: str, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Create Stochastic Oscillator indicator."""
        if name in self.indicators:
            return self.indicators[name]

        if self.data is None:
            raise ValueError("Data not provided for Stochastic calculation")

        # Calculate Stochastic
        lowest_low = self.data['low'].rolling(k_period).min()
        highest_high = self.data['high'].rolling(k_period).max()

        k_percent = 100 * ((self.data['close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(d_period).mean()

        stoch_data = {
            'k': k_percent,
            'd': d_percent
        }

        self.indicators[name] = stoch_data
        return stoch_data

    def get_indicator(self, name: str) -> Optional[Union[pd.Series, Dict[str, pd.Series]]]:
        """Get an existing indicator by name."""
        return self.indicators.get(name)

    def clear_indicators(self):
        """Clear all cached indicators."""
        self.indicators.clear()

    def list_indicators(self) -> list:
        """List all available indicators."""
        return list(self.indicators.keys())

    def create_backtrader_rsi(self, data, period: int = 14, backend: str = "bt"):
        """Create Backtrader RSI indicator using unified service"""
        return self._backtrader_factory.create_rsi(
            data, period=period, backend=backend
        )

    def create_backtrader_bollinger_bands(self, data, period: int = 20, devfactor: float = 2.0, backend: str = "bt"):
        """Create Backtrader Bollinger Bands indicator using unified service"""
        return self._backtrader_factory.create_bollinger_bands(
            data, period=period, devfactor=devfactor, backend=backend
        )

    def create_backtrader_macd(self, data, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, backend: str = "bt"):
        """Create Backtrader MACD indicator using unified service"""
        return self._backtrader_factory.create_macd(
            data, fast_period=fast_period, slow_period=slow_period,
            signal_period=signal_period, backend=backend
        )
