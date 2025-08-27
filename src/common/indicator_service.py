"""
Unified indicator service for technical and fundamental analysis.

This module provides a single, unified interface for calculating technical and fundamental
indicators using TA-Lib directly with minimal memory caching.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import talib

from src.models.indicators import (
    IndicatorResult, IndicatorSet, IndicatorCategory,
    IndicatorCalculationRequest, BatchIndicatorRequest,
    INDICATOR_DESCRIPTIONS, FUNDAMENTAL_INDICATORS, ALL_INDICATORS
)
from src.common.recommendation_engine import RecommendationEngine
from src.common import get_ohlcv, determine_provider
from src.common.indicator_config import indicator_config
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class SimpleMemoryCache:
    """Simple in-memory cache for indicator results."""

    def __init__(self, max_size: int = 1000, ttl: int = 300):
        """
        Initialize the memory cache.

        Args:
            max_size: Maximum number of cached items
            ttl: Time to live for cache items (seconds)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}
        self.access_count = {}

    def _generate_key(self, ticker: str, indicators: List[str], timeframe: str, period: str, **kwargs) -> str:
        """Generate a unique cache key based on all parameters."""
        # Sort indicators for consistent key generation
        sorted_indicators = sorted(indicators) if indicators else []

        # Include all parameters that affect the result
        key_parts = [
            ticker.upper(),
            ",".join(sorted_indicators),
            timeframe,
            period
        ]

        # Add any additional parameters
        for key, value in sorted(kwargs.items()):
            if value is not None:
                key_parts.append(f"{key}:{value}")

        return "|".join(key_parts)

    def get(self, ticker: str, indicators: List[str], timeframe: str, period: str, **kwargs) -> Optional[IndicatorSet]:
        """Get cached indicators if available and not expired."""
        key = self._generate_key(ticker, indicators, timeframe, period, **kwargs)

        if key in self.cache:
            # Check if expired
            if datetime.now().timestamp() - self.timestamps[key] > self.ttl:
                del self.cache[key]
                del self.timestamps[key]
                del self.access_count[key]
                return None

            # Update access count
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]

        return None

    def set(self, ticker: str, indicators: List[str], timeframe: str, period: str,
            indicator_set: IndicatorSet, **kwargs) -> None:
        """Cache indicator results."""
        key = self._generate_key(ticker, indicators, timeframe, period, **kwargs)

        # Evict least recently used if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_lru()

        self.cache[key] = indicator_set
        self.timestamps[key] = datetime.now().timestamp()
        self.access_count[key] = 1

    def _evict_lru(self):
        """Evict least recently used items."""
        if not self.access_count:
            return

        # Find least recently used item
        lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])

        # Remove it
        del self.cache[lru_key]
        del self.timestamps[lru_key]
        del self.access_count[lru_key]

    def clear(self):
        """Clear all cached data."""
        self.cache.clear()
        self.timestamps.clear()
        self.access_count.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "ttl": self.ttl,
            "hit_rate": "N/A"  # Would need to track hits/misses for this
        }


class IndicatorService:
    """Unified service for indicator calculations using TA-Lib directly."""

    def __init__(self, cache_size: int = 1000, cache_ttl: int = 300):
        """
        Initialize the indicator service.

        Args:
            cache_size: Maximum number of items in memory cache
            cache_ttl: Time to live for cache items (seconds)
        """
        self.cache = SimpleMemoryCache(max_size=cache_size, ttl=cache_ttl)
        self.recommendation_engine = RecommendationEngine()

        _logger.info("Indicator service initialized with TA-Lib direct calculation")

    async def get_indicators(self, request: IndicatorCalculationRequest) -> IndicatorSet:
        """
        Get indicators for a single ticker with direct TA-Lib calculation.

        Args:
            request: Indicator calculation request

        Returns:
            IndicatorSet with calculated indicators and recommendations
        """
        start_time = datetime.now()

        try:
            # Check cache first (unless force refresh)
            if not request.force_refresh:
                cached_result = self.cache.get(
                    request.ticker, request.indicators, request.timeframe, request.period
                )
                if cached_result:
                    _logger.debug("Cache hit for %s", request.ticker)
                    return cached_result

            # Calculate indicators directly using TA-Lib
            indicator_set = await self._calculate_indicators(request)

            # Cache the result
            self.cache.set(
                request.ticker, request.indicators, request.timeframe, request.period, indicator_set
            )

            duration = (datetime.now() - start_time).total_seconds()
            _logger.debug("Calculated indicators for %s in %.2fs", request.ticker, duration)

            return indicator_set

        except Exception as e:
            _logger.exception("Error calculating indicators for %s: %s", request.ticker, e)
            # Return empty indicator set on error
            return IndicatorSet(ticker=request.ticker)

    async def get_batch_indicators(self, request: BatchIndicatorRequest) -> Dict[str, IndicatorSet]:
        """
        Get indicators for multiple tickers efficiently.

        Args:
            request: Batch indicator calculation request

        Returns:
            Dictionary mapping ticker to IndicatorSet
        """
        start_time = datetime.now()

        try:
            # Calculate indicators in batches
            results = await self._calculate_batch_indicators(request)

            duration = (datetime.now() - start_time).total_seconds()
            _logger.info("Calculated indicators for %d tickers in %.2fs", len(request.tickers), duration)

            return results

        except Exception as e:
            _logger.exception("Error calculating batch indicators: %s", e)
            return {}

    async def _calculate_indicators(self, request: IndicatorCalculationRequest) -> IndicatorSet:
        """Calculate indicators for a single ticker using TA-Lib directly."""
        ticker = request.ticker.upper()
        indicator_set = IndicatorSet(ticker=ticker)

        # Determine provider if not specified
        provider = request.provider or determine_provider(ticker)

        # Get OHLCV data for technical indicators
        technical_indicators = [ind for ind in request.indicators if ind in INDICATOR_DESCRIPTIONS]
        if technical_indicators:
            try:
                df = get_ohlcv(ticker, request.timeframe, request.period, provider)
                if df is not None and not df.empty:
                    technical_results = await self._calculate_technical_indicators(
                        df, technical_indicators, ticker
                    )
                    for result in technical_results:
                        indicator_set.add_indicator(result)
            except Exception as e:
                _logger.exception("Error calculating technical indicators for %s: %s", ticker, e)

        # Get fundamental data for fundamental indicators
        fundamental_indicators = [ind for ind in request.indicators if ind in FUNDAMENTAL_INDICATORS]
        if fundamental_indicators:
            try:
                # Lazy import to avoid circular dependency
                from src.common.fundamentals import get_fundamentals
                fundamentals = get_fundamentals(ticker, provider)
                if fundamentals:
                    fundamental_results = await self._calculate_fundamental_indicators(
                        fundamentals, fundamental_indicators, ticker
                    )
                    for result in fundamental_results:
                        indicator_set.add_indicator(result)
            except Exception as e:
                _logger.exception("Error calculating fundamental indicators for %s: %s", ticker, e)

        # Calculate composite recommendation if requested
        if request.include_recommendations and indicator_set.get_all_indicators():
            composite = self.recommendation_engine.get_composite_recommendation(indicator_set)
            indicator_set.overall_recommendation = composite
            indicator_set.composite_score = composite.composite_score

        return indicator_set

    async def _calculate_batch_indicators(self, request: BatchIndicatorRequest) -> Dict[str, IndicatorSet]:
        """Calculate indicators for multiple tickers in batches."""
        results = {}

        # Process tickers in batches to avoid overwhelming the system
        batch_size = min(request.max_concurrent, 10)  # Limit concurrent requests

        for i in range(0, len(request.tickers), batch_size):
            batch_tickers = request.tickers[i:i + batch_size]

            # Create tasks for concurrent processing
            tasks = []
            for ticker in batch_tickers:
                task_request = IndicatorCalculationRequest(
                    ticker=ticker,
                    indicators=request.indicators,
                    timeframe=request.timeframe,
                    period=request.period,
                    provider=request.provider,
                    force_refresh=request.force_refresh,
                    include_recommendations=request.include_recommendations
                )
                tasks.append(self.get_indicators(task_request))

            # Execute batch concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for ticker, result in zip(batch_tickers, batch_results):
                if isinstance(result, Exception):
                    _logger.error("Error processing %s: %s", ticker, result)
                    results[ticker] = IndicatorSet(ticker=ticker)
                else:
                    results[ticker] = result

            # Small delay between batches to avoid rate limiting
            if i + batch_size < len(request.tickers):
                await asyncio.sleep(0.1)

        return results

    async def _calculate_technical_indicators(self, df: pd.DataFrame, indicators: List[str], ticker: str) -> List[IndicatorResult]:
        """Calculate technical indicators using TA-Lib directly."""
        results = []

        try:
            # Extract OHLCV data
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values
            open_price = df['open'].values

            # Calculate indicators using TA-Lib
            calculated_indicators = {}

            for indicator in indicators:
                try:
                    value = self._calculate_single_indicator(
                        indicator, high, low, close, volume, open_price
                    )
                    if value is not None:
                        calculated_indicators[indicator] = value
                except Exception as e:
                    _logger.exception("Error calculating %s for %s: %s", indicator, ticker, e)
                    continue

            # Create indicator results
            current_price = close[-1] if len(close) > 0 else None

            for indicator_name, value in calculated_indicators.items():
                try:
                    # Create context for recommendation with all available indicators
                    context = self._create_technical_context(calculated_indicators, current_price, indicator_name)

                    # Get recommendation
                    recommendation = self.recommendation_engine.get_recommendation(
                        indicator_name, value, context
                    )

                    # Create indicator result
                    result = IndicatorResult(
                        name=indicator_name,
                        value=value,
                        recommendation=recommendation,
                        category=IndicatorCategory.TECHNICAL,
                        last_updated=datetime.now(),
                        source="talib"
                    )

                    results.append(result)

                except Exception as e:
                    _logger.exception("Error creating result for %s: %s", indicator_name, e)
                    continue

        except Exception as e:
            _logger.exception("Error calculating technical indicators for %s: %s", ticker, e)

        return results

    def _calculate_single_indicator(self, indicator: str, high: np.ndarray, low: np.ndarray,
                                  close: np.ndarray, volume: np.ndarray, open_price: np.ndarray) -> Optional[float]:
        """Calculate a single technical indicator using TA-Lib with configurable parameters."""
        try:
            # Get parameters from configuration
            params = indicator_config.get_parameters(indicator)

            if indicator == "RSI":
                timeperiod = params.get("timeperiod", 14)
                result = talib.RSI(close, timeperiod=timeperiod)
                return result[-1] if len(result) > 0 and not np.isnan(result[-1]) else None

            elif indicator == "MACD":
                fastperiod = params.get("fastperiod", 12)
                slowperiod = params.get("slowperiod", 26)
                signalperiod = params.get("signalperiod", 9)
                macd, signal, hist = talib.MACD(close, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
                return macd[-1] if len(macd) > 0 and not np.isnan(macd[-1]) else None

            elif indicator == "MACD_SIGNAL":
                fastperiod = params.get("fastperiod", 12)
                slowperiod = params.get("slowperiod", 26)
                signalperiod = params.get("signalperiod", 9)
                macd, signal, hist = talib.MACD(close, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
                return signal[-1] if len(signal) > 0 and not np.isnan(signal[-1]) else None

            elif indicator == "MACD_HISTOGRAM":
                fastperiod = params.get("fastperiod", 12)
                slowperiod = params.get("slowperiod", 26)
                signalperiod = params.get("signalperiod", 9)
                macd, signal, hist = talib.MACD(close, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
                return hist[-1] if len(hist) > 0 and not np.isnan(hist[-1]) else None

            elif indicator == "BB_UPPER":
                timeperiod = params.get("timeperiod", 20)
                nbdevup = params.get("nbdevup", 2)
                nbdevdn = params.get("nbdevdn", 2)
                upper, middle, lower = talib.BBANDS(close, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn)
                return upper[-1] if len(upper) > 0 and not np.isnan(upper[-1]) else None

            elif indicator == "BB_MIDDLE":
                timeperiod = params.get("timeperiod", 20)
                nbdevup = params.get("nbdevup", 2)
                nbdevdn = params.get("nbdevdn", 2)
                upper, middle, lower = talib.BBANDS(close, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn)
                return middle[-1] if len(middle) > 0 and not np.isnan(middle[-1]) else None

            elif indicator == "BB_LOWER":
                timeperiod = params.get("timeperiod", 20)
                nbdevup = params.get("nbdevup", 2)
                nbdevdn = params.get("nbdevdn", 2)
                upper, middle, lower = talib.BBANDS(close, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn)
                return lower[-1] if len(lower) > 0 and not np.isnan(lower[-1]) else None

            elif indicator == "STOCH_K":
                fastk_period = params.get("fastk_period", 14)
                slowk_period = params.get("slowk_period", 3)
                slowd_period = params.get("slowd_period", 3)
                k, d = talib.STOCH(high, low, close, fastk_period=fastk_period, slowk_period=slowk_period, slowd_period=slowd_period)
                return k[-1] if len(k) > 0 and not np.isnan(k[-1]) else None

            elif indicator == "STOCH_D":
                fastk_period = params.get("fastk_period", 14)
                slowk_period = params.get("slowk_period", 3)
                slowd_period = params.get("slowd_period", 3)
                k, d = talib.STOCH(high, low, close, fastk_period=fastk_period, slowk_period=slowk_period, slowd_period=slowd_period)
                return d[-1] if len(d) > 0 and not np.isnan(d[-1]) else None

            elif indicator == "ADX":
                timeperiod = params.get("timeperiod", 14)
                result = talib.ADX(high, low, close, timeperiod=timeperiod)
                return result[-1] if len(result) > 0 and not np.isnan(result[-1]) else None

            elif indicator == "PLUS_DI":
                timeperiod = params.get("timeperiod", 14)
                result = talib.PLUS_DI(high, low, close, timeperiod=timeperiod)
                return result[-1] if len(result) > 0 and not np.isnan(result[-1]) else None

            elif indicator == "MINUS_DI":
                timeperiod = params.get("timeperiod", 14)
                result = talib.MINUS_DI(high, low, close, timeperiod=timeperiod)
                return result[-1] if len(result) > 0 and not np.isnan(result[-1]) else None

            elif indicator in ["SMA_FAST", "SMA_50"]:
                # Simple Moving Average Fast (50 periods)
                result = talib.SMA(close, timeperiod=50)
                return result[-1] if len(result) > 0 and not np.isnan(result[-1]) else None
            elif indicator in ["SMA_SLOW", "SMA_200"]:
                # Simple Moving Average Slow (200 periods)
                result = talib.SMA(close, timeperiod=200)
                return result[-1] if len(result) > 0 and not np.isnan(result[-1]) else None
            elif indicator in ["EMA_FAST", "EMA_12"]:
                # Exponential Moving Average Fast (12 periods)
                result = talib.EMA(close, timeperiod=12)
                return result[-1] if len(result) > 0 and not np.isnan(result[-1]) else None
            elif indicator in ["EMA_SLOW", "EMA_26"]:
                # Exponential Moving Average Slow (26 periods)
                result = talib.EMA(close, timeperiod=26)
                return result[-1] if len(result) > 0 and not np.isnan(result[-1]) else None

            elif indicator == "CCI":
                timeperiod = params.get("timeperiod", 14)
                result = talib.CCI(high, low, close, timeperiod=timeperiod)
                return result[-1] if len(result) > 0 and not np.isnan(result[-1]) else None

            elif indicator == "ROC":
                timeperiod = params.get("timeperiod", 10)
                result = talib.ROC(close, timeperiod=timeperiod)
                return result[-1] if len(result) > 0 and not np.isnan(result[-1]) else None

            elif indicator == "MFI":
                timeperiod = params.get("timeperiod", 14)
                # Ensure all inputs are double precision
                high_double = high.astype(np.float64)
                low_double = low.astype(np.float64)
                close_double = close.astype(np.float64)
                volume_double = volume.astype(np.float64)
                result = talib.MFI(high_double, low_double, close_double, volume_double, timeperiod=timeperiod)
                return result[-1] if len(result) > 0 and not np.isnan(result[-1]) else None

            elif indicator in ["WILLIAMS_R", "Williams_R"]:
                timeperiod = params.get("timeperiod", 14)
                result = talib.WILLR(high, low, close, timeperiod=timeperiod)
                return result[-1] if len(result) > 0 and not np.isnan(result[-1]) else None

            elif indicator == "ATR":
                timeperiod = params.get("timeperiod", 14)
                result = talib.ATR(high, low, close, timeperiod=timeperiod)
                return result[-1] if len(result) > 0 and not np.isnan(result[-1]) else None

            elif indicator == "OBV":
                # Ensure inputs are double precision for OBV
                close_double = close.astype(np.float64)
                volume_double = volume.astype(np.float64)
                result = talib.OBV(close_double, volume_double)
                return result[-1] if len(result) > 0 and not np.isnan(result[-1]) else None

            elif indicator == "ADR":
                timeperiod = params.get("timeperiod", 14)
                # Calculate daily range (high - low) and then take the average
                daily_range = high - low
                result = talib.SMA(daily_range, timeperiod=timeperiod)
                return result[-1] if len(result) > 0 and not np.isnan(result[-1]) else None

            else:
                _logger.warning("Unknown technical indicator: %s", indicator)
                return None

        except Exception as e:
            _logger.exception("Error calculating %s: %s", indicator, e)
            return None

    async def _calculate_fundamental_indicators(self, fundamentals: Any, indicators: List[str], ticker: str) -> List[IndicatorResult]:
        """Calculate fundamental indicators from fundamental data."""
        results = []

        try:
            for indicator_name in indicators:
                try:
                    # Get indicator value
                    value = self._extract_fundamental_value(fundamentals, indicator_name)

                    if value is not None:
                        # Get recommendation
                        recommendation = self.recommendation_engine.get_recommendation(
                            indicator_name, value
                        )

                        # Create indicator result
                        result = IndicatorResult(
                            name=indicator_name,
                            value=value,
                            recommendation=recommendation,
                            category=IndicatorCategory.FUNDAMENTAL,
                            last_updated=datetime.now(),
                            source="yfinance"
                        )

                        results.append(result)

                except Exception as e:
                    _logger.exception("Error calculating %s for %s: %s", indicator_name, ticker, e)
                    continue

        except Exception as e:
            _logger.exception("Error calculating fundamental indicators for %s: %s", ticker, e)

        return results

    def _extract_fundamental_value(self, fundamentals: Any, indicator_name: str) -> Optional[float]:
        """Extract fundamental indicator value from fundamental data."""
        try:
            if indicator_name == "PE_RATIO":
                return fundamentals.pe_ratio
            elif indicator_name == "FORWARD_PE":
                return fundamentals.forward_pe
            elif indicator_name == "PB_RATIO":
                return fundamentals.price_to_book
            elif indicator_name == "PS_RATIO":
                return fundamentals.price_to_sales
            elif indicator_name == "PEG_RATIO":
                return fundamentals.peg_ratio
            elif indicator_name == "ROE":
                return fundamentals.return_on_equity
            elif indicator_name == "ROA":
                return fundamentals.return_on_assets
            elif indicator_name == "DEBT_TO_EQUITY":
                return fundamentals.debt_to_equity
            elif indicator_name == "CURRENT_RATIO":
                return fundamentals.current_ratio
            elif indicator_name == "QUICK_RATIO":
                return fundamentals.quick_ratio
            elif indicator_name == "OPERATING_MARGIN":
                return fundamentals.operating_margin
            elif indicator_name == "PROFIT_MARGIN":
                return fundamentals.profit_margin
            elif indicator_name == "REVENUE_GROWTH":
                return fundamentals.revenue_growth
            elif indicator_name == "NET_INCOME_GROWTH":
                return fundamentals.net_income_growth
            elif indicator_name == "FREE_CASH_FLOW":
                return fundamentals.free_cash_flow
            elif indicator_name == "DIVIDEND_YIELD":
                return fundamentals.dividend_yield
            elif indicator_name == "PAYOUT_RATIO":
                return fundamentals.payout_ratio
            elif indicator_name == "BETA":
                return fundamentals.beta
            elif indicator_name == "MARKET_CAP":
                return fundamentals.market_cap
            elif indicator_name == "ENTERPRISE_VALUE":
                return fundamentals.enterprise_value
            else:
                return getattr(fundamentals, indicator_name.lower(), None)
        except Exception as e:
            _logger.exception("Error extracting %s: %s", indicator_name, e)
            return None

    def _create_technical_context(self, calculated_indicators: Dict[str, float], current_price: float, indicator_name: str) -> Dict[str, Any]:
        """Create context for technical indicator recommendations."""
        context = {}

        # Add current price
        if current_price is not None:
            context['current_price'] = current_price

        # Add specific context for different indicators
        if indicator_name in ['MACD', 'MACD_SIGNAL', 'MACD_HISTOGRAM']:
            # MACD needs signal and histogram for proper recommendations
            if 'MACD' in calculated_indicators:
                context['macd'] = calculated_indicators['MACD']
            if 'MACD_SIGNAL' in calculated_indicators:
                context['macd_signal'] = calculated_indicators['MACD_SIGNAL']
            if 'MACD_HISTOGRAM' in calculated_indicators:
                context['macd_histogram'] = calculated_indicators['MACD_HISTOGRAM']
                context['macd_hist'] = calculated_indicators['MACD_HISTOGRAM']  # Legacy support

        elif indicator_name in ['BB_UPPER', 'BB_MIDDLE', 'BB_LOWER']:
            # Bollinger Bands need all three values and current price
            if 'BB_UPPER' in calculated_indicators:
                context['bb_upper'] = calculated_indicators['BB_UPPER']
            if 'BB_MIDDLE' in calculated_indicators:
                context['bb_middle'] = calculated_indicators['BB_MIDDLE']
            if 'BB_LOWER' in calculated_indicators:
                context['bb_lower'] = calculated_indicators['BB_LOWER']

        elif indicator_name in ['STOCH_K', 'STOCH_D']:
            # Stochastic needs both K and D values
            if 'STOCH_K' in calculated_indicators:
                context['stoch_k'] = calculated_indicators['STOCH_K']
            if 'STOCH_D' in calculated_indicators:
                context['stoch_d'] = calculated_indicators['STOCH_D']

        elif indicator_name in ['ADX', 'PLUS_DI', 'MINUS_DI']:
            # ADX needs plus and minus DI values
            if 'PLUS_DI' in calculated_indicators:
                context['plus_di'] = calculated_indicators['PLUS_DI']
            if 'MINUS_DI' in calculated_indicators:
                context['minus_di'] = calculated_indicators['MINUS_DI']

        elif indicator_name in ['SMA_50', 'SMA_200']:
            # SMA needs current price for comparison
            if current_price is not None:
                context['current_price'] = current_price

        elif indicator_name == 'ADR':
            # ADR needs current price for percentage calculation
            if current_price is not None:
                context['current_price'] = current_price

        return context

    def get_available_indicators(self) -> Dict[str, List[str]]:
        """Get list of available indicators by category."""
        return {
            "technical": list(INDICATOR_DESCRIPTIONS.keys()),
            "fundamental": list(FUNDAMENTAL_INDICATORS.keys()),
            "all": list(ALL_INDICATORS.keys())
        }

    def get_indicator_description(self, indicator_name: str) -> Optional[str]:
        """Get description for an indicator."""
        return INDICATOR_DESCRIPTIONS.get(indicator_name)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()

    def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear()

    def get_service_info(self) -> Dict[str, Any]:
        """Get service information and statistics."""
        return {
            "service": "Unified Indicator Service (TA-Lib Direct)",
            "version": "1.0.0",
            "cache_stats": self.get_cache_stats(),
            "available_indicators": {
                "technical_count": len(INDICATOR_DESCRIPTIONS),
                "fundamental_count": len(FUNDAMENTAL_INDICATORS),
                "total_count": len(ALL_INDICATORS)
            }
        }


# Global instance for easy access
_indicator_service = None

def get_indicator_service() -> IndicatorService:
    """Get the global indicator service instance."""
    global _indicator_service
    if _indicator_service is None:
        _indicator_service = IndicatorService()
    return _indicator_service
