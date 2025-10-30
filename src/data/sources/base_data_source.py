"""
Base data source class for all data providers.

This module defines the common interface and functionality that all data sources
must implement, providing consistency and reducing code duplication.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timedelta
import pandas as pd
import logging
from pathlib import Path

from src.data.utils import get_data_handler, get_provider_limiter
from src.data.utils.validation import validate_ohlcv_data, get_data_quality_score

_logger = logging.getLogger(__name__)


class BaseDataSource(ABC):
    """
    Abstract base class for all data sources.

    Provides common functionality for:
    - Data fetching and caching
    - Rate limiting
    - Data validation
    - Error handling and retries
    - Configuration management
    """

    def __init__(
        self,
        provider_name: str,
        cache_enabled: bool = True,
        rate_limit_enabled: bool = True,
        validation_enabled: bool = True
    ):
        """
        Initialize base data source.

        Args:
            provider_name: Name of the data provider
            cache_enabled: Whether to enable data caching
            rate_limit_enabled: Whether to enable rate limiting
            validation_enabled: Whether to enable data validation
        """
        self.provider_name = provider_name
        self.cache_enabled = cache_enabled
        self.rate_limit_enabled = rate_limit_enabled
        self.validation_enabled = validation_enabled

        # Initialize utilities
        self.data_handler = get_data_handler(provider_name, cache_enabled)
        self.rate_limiter = get_provider_limiter(provider_name) if rate_limit_enabled else None

        # Data source state
        self._is_connected = False
        self._last_error = None
        self._error_count = 0
        self._last_successful_fetch = None

        _logger.info("Initialized %s data source", provider_name)

    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available trading symbols.

        Returns:
            List of available symbol strings
        """
        pass

    @abstractmethod
    def get_supported_intervals(self) -> List[str]:
        """
        Get list of supported data intervals.

        Returns:
            List of supported interval strings
        """
        pass

    @abstractmethod
    def fetch_historical_data(
        self,
        symbol: str,
        interval: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data.

        Args:
            symbol: Trading symbol
            interval: Data interval
            start_date: Start date for data range
            end_date: End date for data range
            limit: Maximum number of data points to fetch

        Returns:
            DataFrame with historical data or None if failed
        """
        pass

    @abstractmethod
    def start_realtime_feed(
        self,
        symbol: str,
        interval: str,
        callback: Optional[callable] = None
    ) -> bool:
        """
        Start real-time data feed.

        Args:
            symbol: Trading symbol
            interval: Data interval
            callback: Function to call with new data

        Returns:
            True if started successfully, False otherwise
        """
        pass

    @abstractmethod
    def stop_realtime_feed(self, symbol: str) -> bool:
        """
        Stop real-time data feed.

        Args:
            symbol: Trading symbol

        Returns:
            True if stopped successfully, False otherwise
        """
        pass

    def get_data_with_cache(
        self,
        symbol: str,
        interval: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True,
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Get data with caching support.

        Args:
            symbol: Trading symbol
            interval: Data interval
            start_date: Start date for data range
            end_date: End date for data range
            use_cache: Whether to use cached data
            force_refresh: Whether to force refresh from source

        Returns:
            DataFrame with data or None if failed
        """
        if not use_cache or force_refresh:
            return self._fetch_and_cache_data(symbol, interval, start_date, end_date)

        # Try to get from cache first
        cached_data = self.data_handler.get_cached_data(
            symbol, interval, start_date, end_date
        )

        if cached_data is not None:
            _logger.info("Retrieved cached data for %s %s", symbol, interval)
            return cached_data

        # Fetch from source if not in cache
        return self._fetch_and_cache_data(symbol, interval, start_date, end_date)

    def _fetch_and_cache_data(
        self,
        symbol: str,
        interval: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data from source and cache it.

        Args:
            symbol: Trading symbol
            interval: Data interval
            start_date: Start date for data range
            end_date: End date for data range

        Returns:
            DataFrame with data or None if failed
        """
        try:
            # Apply rate limiting if enabled
            if self.rate_limiter:
                if not self.rate_limiter.acquire():
                    _logger.warning("Rate limit exceeded for %s", self.provider_name)
                    return None

            # Fetch data from source
            data = self.fetch_historical_data(symbol, interval, start_date, end_date)

            if data is None or data.empty:
                _logger.warning("No data received for%s %s", symbol, interval)
                return None

            # Validate data if enabled
            if self.validation_enabled:
                validation_result = self.data_handler.validate_and_score_data(data, symbol)
                if not validation_result['is_valid']:
                    _logger.warning("Data validation failed for %s: %s", symbol, validation_result['errors'])
                    # Continue with invalid data but log warning

            # Standardize data format
            data = self.data_handler.standardize_ohlcv_data(data, symbol, interval)

            # Cache data if enabled
            if self.cache_enabled:
                self.data_handler.cache_data(data, symbol, interval, start_date, end_date)

            # Update state
            self._last_successful_fetch = datetime.now()
            self._error_count = 0
            self._last_error = None

            _logger.info("Successfully fetched %d data points for %s %s", len(data), symbol, interval)
            return data

        except Exception as e:
            self._handle_error(f"Failed to fetch data for {symbol} {interval}", e)
            return None

    def _handle_error(self, message: str, error: Exception) -> None:
        """
        Handle errors consistently across data sources.

        Args:
            message: Error message
            error: Exception that occurred
        """
        self._error_count += 1
        self._last_error = error

        _logger.error("%s: %s", message, error)

        # Log additional context for debugging
        if hasattr(error, 'response'):
            _logger.error("Response status: %s", getattr(error.response, 'status_code', 'N/A'))
            _logger.error("Response text: %s", getattr(error.response, 'text', 'N/A'))

    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get current connection and health status.

        Returns:
            Dictionary with status information
        """
        return {
            'provider': self.provider_name,
            'is_connected': self._is_connected,
            'error_count': self._error_count,
            'last_error': str(self._last_error) if self._last_error else None,
            'last_successful_fetch': self._last_successful_fetch.isoformat() if self._last_successful_fetch else None,
            'cache_enabled': self.cache_enabled,
            'rate_limit_enabled': self.rate_limit_enabled,
            'validation_enabled': self.validation_enabled
        }

    def reset_error_count(self) -> None:
        """Reset error count and clear last error."""
        self._error_count = 0
        self._last_error = None
        _logger.info("Reset error count for %s", self.provider_name)

    def is_healthy(self) -> bool:
        """
        Check if data source is healthy.

        Returns:
            True if healthy, False otherwise
        """
        # Consider unhealthy if too many recent errors
        if self._error_count > 10:
            return False

        # Consider unhealthy if no successful fetch in last hour
        if (self._last_successful_fetch and
            datetime.now() - self._last_successful_fetch > timedelta(hours=1)):
            return False

        return True

    def get_data_quality_report(
        self,
        symbol: str,
        interval: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive data quality report.

        Args:
            symbol: Trading symbol
            interval: Data interval
            start_date: Start date for data range
            end_date: End date for data range

        Returns:
            Dictionary with data quality metrics
        """
        data = self.get_data_with_cache(symbol, interval, start_date, end_date)

        if data is None or data.empty:
            return {
                'symbol': symbol,
                'interval': interval,
                'data_points': 0,
                'quality_score': 0.0,
                'errors': ['No data available'],
                'timestamp_range': None,
                'data_completeness': 0.0
            }

        # Get validation results
        validation_result = self.data_handler.validate_and_score_data(data, symbol)
        quality_score = validation_result['quality_score']

        # Calculate additional metrics
        timestamp_range = {
            'start': data['timestamp'].min().isoformat(),
            'end': data['timestamp'].max().isoformat(),
            'duration_days': (data['timestamp'].max() - data['timestamp'].min()).days
        }

        # Calculate data completeness (assuming regular intervals)
        expected_points = self._calculate_expected_points(interval, start_date, end_date)
        actual_points = len(data)
        data_completeness = actual_points / expected_points if expected_points > 0 else 0.0

        return {
            'symbol': symbol,
            'interval': interval,
            'data_points': actual_points,
            'quality_score': quality_score['quality_score'],
            'errors': validation_result['errors'],
            'timestamp_range': timestamp_range,
            'data_completeness': data_completeness,
            'validation_details': quality_score
        }

    def _calculate_expected_points(
        self,
        interval: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> int:
        """
        Calculate expected number of data points for given interval and date range.

        Args:
            interval: Data interval
            start_date: Start date
            end_date: End date

        Returns:
            Expected number of data points
        """
        if not start_date or not end_date:
            return 0

        # Convert interval to minutes
        interval_minutes = self._parse_interval_to_minutes(interval)
        if interval_minutes == 0:
            return 0

        duration = end_date - start_date
        duration_minutes = duration.total_seconds() / 60

        return int(duration_minutes / interval_minutes) + 1

    def _parse_interval_to_minutes(self, interval: str) -> int:
        """
        Parse interval string to minutes.

        Args:
            interval: Interval string (e.g., '1m', '1h', '1d')

        Returns:
            Interval in minutes
        """
        interval = interval.lower()

        if interval.endswith('m'):
            return int(interval[:-1])
        elif interval.endswith('h'):
            return int(interval[:-1]) * 60
        elif interval.endswith('d'):
            return int(interval[:-1]) * 24 * 60
        elif interval.endswith('w'):
            return int(interval[:-1]) * 7 * 24 * 60
        else:
            _logger.warning("Unknown interval format: %s", interval)
            return 0

    def cleanup(self) -> None:
        """Clean up resources and connections."""
        try:
            # Stop all real-time feeds
            symbols = getattr(self, '_active_feeds', [])
            for symbol in symbols:
                self.stop_realtime_feed(symbol)

            self._is_connected = False
            _logger.info("Cleaned up %s data source", self.provider_name)

        except Exception as e:
            _logger.exception("Error during cleanup of %s:", self.provider_name)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
