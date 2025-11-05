"""
Backtrader Adapter for Unified Indicator Service

This adapter provides Backtrader compatibility for the unified indicator service,
maintaining the line-based interface that existing strategies expect while
leveraging the unified service's capabilities.
"""

import backtrader as bt
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, Type, List


from src.indicators.adapters.base import BaseAdapter
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class BacktraderIndicatorWrapper(bt.Indicator):
    """
    Base class for Backtrader indicator wrappers that use the unified service.

    This class provides the foundation for creating Backtrader-compatible indicators
    that delegate their calculations to the unified indicator service while maintaining
    the line-based interface expected by Backtrader strategies.
    """

    params = (
        ("backend", "bt"),  # Backend preference: bt, bt-talib, talib
        ("use_unified_service", True),  # Whether to use unified service
    )

    def __init__(self):
        super().__init__()
        self._backend = self.p.backend
        self._use_unified = self.p.use_unified_service
        self._unified_service = None
        self._fallback_impl = None
        self._data_cache = []
        self._last_computed_len = 0

        # Initialize unified service if requested
        if self._use_unified:
            try:
                from src.indicators.service import UnifiedIndicatorService
                self._unified_service = UnifiedIndicatorService()
            except ImportError as e:
                logger.warning("Failed to import UnifiedIndicatorService, falling back to native implementation: %s", e)
                self._use_unified = False

        # Initialize fallback implementation
        self._init_fallback()

    def _init_fallback(self):
        """Initialize the fallback Backtrader implementation"""
        raise NotImplementedError("Subclasses must implement _init_fallback")

    def _get_indicator_name(self) -> str:
        """Get the indicator name for the unified service"""
        raise NotImplementedError("Subclasses must implement _get_indicator_name")

    def _get_indicator_params(self) -> Dict[str, Any]:
        """Get the indicator parameters for the unified service"""
        raise NotImplementedError("Subclasses must implement _get_indicator_params")

    def _map_unified_results(self, results: Dict[str, pd.Series]):
        """Map unified service results to Backtrader lines"""
        raise NotImplementedError("Subclasses must implement _map_unified_results")

    def _build_dataframe(self) -> pd.DataFrame:
        """Build a pandas DataFrame from Backtrader data for unified service"""
        current_len = len(self.data)

        # Only rebuild if we have new data
        if current_len <= self._last_computed_len:
            return None

        # Get the data arrays
        try:
            # Get all available data up to current point
            closes = []
            opens = []
            highs = []
            lows = []
            volumes = []

            for i in range(current_len):
                closes.append(float(self.data.close[-i]))
                opens.append(float(self.data.open[-i]))
                highs.append(float(self.data.high[-i]))
                lows.append(float(self.data.low[-i]))
                if hasattr(self.data, 'volume'):
                    volumes.append(float(self.data.volume[-i]))
                else:
                    volumes.append(0.0)

            # Reverse to get chronological order
            closes.reverse()
            opens.reverse()
            highs.reverse()
            lows.reverse()
            volumes.reverse()

            # Create DataFrame
            df = pd.DataFrame({
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            })

            return df

        except Exception as e:
            logger.exception("Error building DataFrame from Backtrader data:")
            return None

    def _normalize_params(self, indicator_name: str, params: dict) -> dict:
        """
        Normalize parameter names for unified indicator service.

        Args:
            indicator_name: Name of the indicator
            params: Original parameters from strategy

        Returns:
            Normalized parameters
        """
        normalized = params.copy()
        indicator_lower = indicator_name.lower()

        # RSI parameter normalization
        if 'rsi' in indicator_lower:
            if 'period' in normalized and 'timeperiod' not in normalized:
                normalized['timeperiod'] = normalized.pop('period')
            if 'length' in normalized and 'timeperiod' not in normalized:
                normalized['timeperiod'] = normalized.pop('length')

        # Bollinger Bands normalization
        if 'bollinger' in indicator_lower or 'bb' in indicator_lower:
            if 'period' in normalized and 'timeperiod' not in normalized:
                normalized['timeperiod'] = normalized.pop('period')
            # Handle both 'dev' and 'devfactor' parameter names
            if 'dev' in normalized and 'nbdevup' not in normalized:
                dev = normalized.pop('dev')
                normalized['nbdevup'] = dev
                normalized['nbdevdn'] = dev
            if 'devfactor' in normalized and 'nbdevup' not in normalized:
                dev = normalized.pop('devfactor')
                normalized['nbdevup'] = dev
                normalized['nbdevdn'] = dev

        # ATR parameter normalization
        if 'atr' in indicator_lower:
            if 'period' in normalized and 'timeperiod' not in normalized:
                normalized['timeperiod'] = normalized.pop('period')

        # Volume normalization
        if 'volume' in indicator_lower:
            if 'period' in normalized and 'timeperiod' not in normalized:
                normalized['timeperiod'] = normalized.pop('period')

        return normalized

    def _normalize_indicator_name(self, name: str) -> str:
        """
        Normalize indicator names to match registry.

        Args:
            name: Original indicator name

        Returns:
            Normalized indicator name
        """
        name_lower = name.lower()

        # Common name mappings
        name_map = {
            'bollinger_bands': 'bbands',
            'bollinger': 'bbands',
            'moving_average': 'sma',
            'simple_moving_average': 'sma',
            'exponential_moving_average': 'ema',
        }

        return name_map.get(name_lower, name_lower)

    def next(self):
        """Called for each new bar"""
        if self._use_unified and self._unified_service:
            try:
                # Build DataFrame for unified service
                df = self._build_dataframe()
                if df is not None and len(df) > 0:
                    # Compute indicator using unified service
                    indicator_name = self._get_indicator_name()
                    params = self._get_indicator_params()

                    # Normalize indicator name and parameters
                    normalized_name = self._normalize_indicator_name(indicator_name)
                    normalized_params = self._normalize_params(normalized_name, params)

                    # Create inputs dict
                    inputs = {
                        'close': df['close'],
                        'open': df['open'],
                        'high': df['high'],
                        'low': df['low'],
                        'volume': df['volume']
                    }

                    # Compute asynchronously (we'll need to handle this synchronously in Backtrader)
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    # Get the appropriate adapter from the service
                    adapter = self._unified_service._select_provider(normalized_name)

                    results = loop.run_until_complete(
                        adapter.compute(
                            normalized_name,
                            df,
                            inputs,
                            normalized_params
                        )
                    )

                    # Map results to Backtrader lines
                    self._map_unified_results(results)
                    self._last_computed_len = len(df)
                    return

            except Exception as e:
                logger.warning("Unified service computation failed, falling back to native implementation: %s", e)
                logger.exception("Full error details:")
                self._use_unified = False

        # Use fallback implementation
        if self._fallback_impl:
            logger.debug("Using fallback Backtrader implementation for %s", self._get_indicator_name())
            self._use_fallback()
        else:
            logger.warning("No fallback implementation available for %s", self._get_indicator_name())

    def _use_fallback(self):
        """Use the fallback implementation to set line values"""
        raise NotImplementedError("Subclasses must implement _use_fallback")


class BacktraderAdapter(BaseAdapter):
    """
    Adapter that provides Backtrader compatibility for the unified indicator service.

    This adapter creates Backtrader-compatible indicator classes that can be used
    directly in Backtrader strategies while leveraging the unified service's
    calculation capabilities.
    """

    def __init__(self):
        self._indicator_registry = {}
        self._register_indicators()

    def supports(self, name: str) -> bool:
        """Check if this adapter supports the given indicator"""
        return name.lower() in self._indicator_registry

    async def compute(
        self,
        name: str,
        df: pd.DataFrame | None,
        inputs: Dict[str, pd.Series],
        params: Dict[str, Any]
    ) -> Dict[str, pd.Series]:
        """
        Compute indicator for Backtrader compatibility.

        Note: This method is primarily for interface compliance.
        Actual Backtrader integration happens through the wrapper classes.
        """
        # This adapter primarily works through wrapper classes
        # Direct computation is handled by other adapters
        raise NotImplementedError("BacktraderAdapter works through wrapper classes, not direct computation")

    def create_indicator(
        self,
        name: str,
        data: bt.feeds.DataBase,
        backend: str = "bt",
        **params
    ) -> bt.Indicator:
        """
        Create a Backtrader indicator instance.

        Args:
            name: Indicator name
            data: Backtrader data feed
            backend: Backend preference (bt, bt-talib, talib)
            **params: Indicator parameters

        Returns:
            Backtrader indicator instance
        """
        indicator_class = self._indicator_registry.get(name.lower())
        if not indicator_class:
            raise ValueError(f"Unsupported indicator: {name}")

        # Create indicator with specified backend and parameters
        return indicator_class(data, backend=backend, **params)

    def get_supported_indicators(self) -> List[str]:
        """Get list of supported indicator names"""
        return list(self._indicator_registry.keys())

    def _register_indicators(self):
        """Register all supported Backtrader indicators"""
        from src.indicators.adapters.backtrader_wrappers import (
            UnifiedRSIIndicator,
            UnifiedBollingerBandsIndicator,
            UnifiedMACDIndicator,
            UnifiedATRIndicator,
            UnifiedSMAIndicator,
            UnifiedEMAIndicator
        )

        self._indicator_registry = {
            "rsi": UnifiedRSIIndicator,
            "bollinger_bands": UnifiedBollingerBandsIndicator,
            "bollinger": UnifiedBollingerBandsIndicator,
            "bb": UnifiedBollingerBandsIndicator,
            "macd": UnifiedMACDIndicator,
            "atr": UnifiedATRIndicator,
            "sma": UnifiedSMAIndicator,
            "ema": UnifiedEMAIndicator,
        }


class BacktraderIndicatorFactory:
    """
    Factory for creating Backtrader-compatible indicators using the unified service.

    This factory provides a convenient interface for creating indicators that work
    with both the unified service and native Backtrader implementations.
    """

    def __init__(self, adapter: Optional[BacktraderAdapter] = None):
        self._adapter = adapter or BacktraderAdapter()

    def create_rsi(
        self,
        data: bt.feeds.DataBase,
        period: int = 14,
        backend: str = "bt",
        use_unified_service: bool = True
    ) -> bt.Indicator:
        """Create RSI indicator"""
        return self._adapter.create_indicator(
            "rsi",
            data,
            backend=backend,
            period=period,
            use_unified_service=use_unified_service
        )

    def create_bollinger_bands(
        self,
        data: bt.feeds.DataBase,
        period: int = 20,
        devfactor: float = 2.0,
        backend: str = "bt",
        use_unified_service: bool = True
    ) -> bt.Indicator:
        """Create Bollinger Bands indicator"""
        return self._adapter.create_indicator(
            "bollinger_bands",
            data,
            backend=backend,
            period=period,
            devfactor=devfactor,
            use_unified_service=use_unified_service
        )

    def create_macd(
        self,
        data: bt.feeds.DataBase,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        backend: str = "bt",
        use_unified_service: bool = True
    ) -> bt.Indicator:
        """Create MACD indicator"""
        return self._adapter.create_indicator(
            "macd",
            data,
            backend=backend,
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
            use_unified_service=use_unified_service
        )

    def create_atr(
        self,
        data: bt.feeds.DataBase,
        period: int = 14,
        backend: str = "bt",
        use_unified_service: bool = True
    ) -> bt.Indicator:
        """Create ATR indicator"""
        return self._adapter.create_indicator(
            "atr",
            data,
            backend=backend,
            period=period,
            use_unified_service=use_unified_service
        )

    def create_sma(
        self,
        data: bt.feeds.DataBase,
        period: int = 20,
        backend: str = "bt",
        use_unified_service: bool = True
    ) -> bt.Indicator:
        """Create Simple Moving Average indicator"""
        return self._adapter.create_indicator(
            "sma",
            data,
            backend=backend,
            period=period,
            use_unified_service=use_unified_service
        )

    def create_ema(
        self,
        data: bt.feeds.DataBase,
        period: int = 20,
        backend: str = "bt",
        use_unified_service: bool = True
    ) -> bt.Indicator:
        """Create Exponential Moving Average indicator"""
        return self._adapter.create_indicator(
            "ema",
            data,
            backend=backend,
            period=period,
            use_unified_service=use_unified_service
        )


# Backend selection utilities
class BackendSelector:
    """Utility class for backend selection and fallback logic"""

    BACKEND_PRIORITY = ["bt", "bt-talib", "talib"]

    @staticmethod
    def select_backend(
        preferred: str,
        available_backends: List[str],
        indicator_name: str
    ) -> str:
        """
        Select the best available backend for an indicator.

        Args:
            preferred: Preferred backend
            available_backends: List of available backends
            indicator_name: Name of the indicator

        Returns:
            Selected backend name
        """
        if preferred in available_backends:
            return preferred

        # Fallback to priority order
        for backend in BackendSelector.BACKEND_PRIORITY:
            if backend in available_backends:
                logger.info(
                    "Preferred backend '%s' not available for %s, using '%s'",
                    preferred, indicator_name, backend
                )
                return backend

        # Default fallback
        logger.warning(
            "No preferred backends available for %s, using 'bt'",
            indicator_name
        )
        return "bt"

    @staticmethod
    def get_available_backends() -> List[str]:
        """Get list of available backends on the system"""
        available = ["bt"]  # Backtrader is always available

        # Check for bt-talib
        try:
            import backtrader as bt
            if hasattr(bt, "talib"):
                available.append("bt-talib")
        except ImportError:
            pass

        # Check for talib
        try:
            import talib
            available.append("talib")
        except ImportError:
            pass

        return available