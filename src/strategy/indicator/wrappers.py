"""
Indicator Wrappers Module

This module provides unified access to both TALib and standard Backtrader indicators,
ensuring consistent API regardless of the underlying implementation.

The wrappers abstract away the differences between TALib and standard Backtrader indicators,
providing a unified interface for all mixins.
"""

from typing import Any, Union
import backtrader as bt


class IndicatorWrapper:
    """Base class for indicator wrappers"""

    def __init__(self, indicator: Any, use_talib: bool = False):
        """
        Initialize the wrapper

        Args:
            indicator: The underlying indicator (TALib or standard Backtrader)
            use_talib: Whether the indicator is from TALib
        """
        self.indicator = indicator
        self.use_talib = use_talib

    def __getitem__(self, index: int) -> Any:
        """Allow direct indexing access like indicator[0]"""
        return self.indicator[index]

    def __len__(self) -> int:
        """Allow len() access"""
        return len(self.indicator)


class BollingerBandsWrapper(IndicatorWrapper):
    """Wrapper for Bollinger Bands indicators (TALib and standard Backtrader)"""

    @property
    def top(self):
        """Get upper band line"""
        if self.use_talib:
            return self.indicator.upperband
        return self.indicator.lines.top

    @property
    def mid(self):
        """Get middle band line"""
        if self.use_talib:
            return self.indicator.middleband
        return self.indicator.lines.mid

    @property
    def bot(self):
        """Get lower band line"""
        if self.use_talib:
            return self.indicator.lowerband
        return self.indicator.lines.bot

    @property
    def upperband(self):
        """Alias for top (TALib compatibility)"""
        return self.top

    @property
    def middleband(self):
        """Alias for mid (TALib compatibility)"""
        return self.mid

    @property
    def lowerband(self):
        """Alias for bot (TALib compatibility)"""
        return self.bot


class RSIWrapper(IndicatorWrapper):
    """Wrapper for RSI indicators (TALib and standard Backtrader)"""

    # RSI works the same for both TALib and standard Backtrader
    # No additional properties needed


class ATRWrapper(IndicatorWrapper):
    """Wrapper for ATR indicators (TALib and standard Backtrader)"""

    # ATR works the same for both TALib and standard Backtrader
    # No additional properties needed


class MAWrapper(IndicatorWrapper):
    """Wrapper for Moving Average indicators (TALib and standard Backtrader)"""

    # Moving averages work the same for both TALib and standard Backtrader
    # No additional properties needed


class MACDWrapper(IndicatorWrapper):
    """Wrapper for MACD indicators (TALib and standard Backtrader)"""

    @property
    def macd(self):
        """Get MACD line"""
        if self.use_talib:
            return self.indicator.macd
        return self.indicator.lines.macd

    @property
    def signal(self):
        """Get signal line"""
        if self.use_talib:
            return self.indicator.macdsignal
        return self.indicator.lines.signal

    @property
    def histogram(self):
        """Get histogram line"""
        if self.use_talib:
            return self.indicator.macdhist
        return self.indicator.lines.histogram


class StochasticWrapper(IndicatorWrapper):
    """Wrapper for Stochastic indicators (TALib and standard Backtrader)"""

    @property
    def k(self):
        """Get %K line"""
        if self.use_talib:
            return self.indicator.slowk
        return self.indicator.lines.percK

    @property
    def d(self):
        """Get %D line"""
        if self.use_talib:
            return self.indicator.slowd
        return self.indicator.lines.percD


class IchimokuWrapper(IndicatorWrapper):
    """Wrapper for Ichimoku indicators (TALib and standard Backtrader)"""

    @property
    def tenkan(self):
        """Get Tenkan line"""
        if self.use_talib:
            return self.indicator.tenkan
        return self.indicator.lines.tenkan

    @property
    def kijun(self):
        """Get Kijun line"""
        if self.use_talib:
            return self.indicator.kijun
        return self.indicator.lines.kijun

    @property
    def senkou_a(self):
        """Get Senkou Span A line"""
        if self.use_talib:
            return self.indicator.senkoua
        return self.indicator.lines.senkoua

    @property
    def senkou_b(self):
        """Get Senkou Span B line"""
        if self.use_talib:
            return self.indicator.senkoub
        return self.indicator.lines.senkoub

    @property
    def chikou(self):
        """Get Chikou line"""
        if self.use_talib:
            return self.indicator.chikou
        return self.indicator.lines.chikou


class SupertrendWrapper(IndicatorWrapper):
    """Wrapper for Supertrend indicators"""

    @property
    def direction(self):
        """Get direction line"""
        return self.indicator.lines.direction

    @property
    def trend(self):
        """Get trend line"""
        return self.indicator.lines.trend


def create_indicator_wrapper(indicator: Any, indicator_type: str, use_talib: bool = False) -> IndicatorWrapper:
    """
    Factory function to create appropriate wrapper for an indicator

    Args:
        indicator: The underlying indicator
        indicator_type: Type of indicator ('bb', 'rsi', 'atr', 'ma', 'macd', 'stoch', 'ichimoku', 'supertrend')
        use_talib: Whether the indicator is from TALib

    Returns:
        Appropriate wrapper instance
    """
    wrapper_map = {
        'bb': BollingerBandsWrapper,
        'bollinger': BollingerBandsWrapper,
        'rsi': RSIWrapper,
        'atr': ATRWrapper,
        'ma': MAWrapper,
        'sma': MAWrapper,
        'ema': MAWrapper,
        'macd': MACDWrapper,
        'stoch': StochasticWrapper,
        'stochastic': StochasticWrapper,
        'ichimoku': IchimokuWrapper,
        'supertrend': SupertrendWrapper,
    }

    wrapper_class = wrapper_map.get(indicator_type.lower())
    if wrapper_class is None:
        # Default to base wrapper for unknown types
        return IndicatorWrapper(indicator, use_talib)

    return wrapper_class(indicator, use_talib)
