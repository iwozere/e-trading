"""
Intelligent Cached Data Downloader

This module provides a professional caching wrapper for data downloaders that:
1. Checks cache first for existing data
2. Identifies data gaps intelligently
3. Downloads only missing data from servers
4. Merges and caches the complete dataset
5. Provides seamless data access with minimal server requests

Features:
- Smart gap detection and analysis
- Intelligent data merging
- Rate limiting and error handling
- Cache validation and integrity checks
- Support for all data downloader types
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
import pandas as pd

from src.data.base_data_downloader import BaseDataDownloader
from src.data.utils.file_based_cache import FileBasedCache
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


@dataclass
class DataGap:
    """Represents a gap in cached data that needs to be filled."""
    start_date: datetime
    end_date: datetime
    gap_type: str  # 'missing_start', 'missing_end', 'internal_gap', 'no_data'
    expected_rows: int
    priority: int  # 1=high (recent data), 2=medium, 3=low (historical data)


class DataGapAnalyzer:
    """
    Analyzes cached data to identify missing periods and gaps.

    This class provides intelligent gap detection for various data scenarios:
    - Missing start/end periods
    - Internal gaps in data
    - Weekend/holiday gaps (for daily data)
    - Data quality issues
    """

    def __init__(self):
        self.interval_gap_tolerance = {
            '1m': timedelta(minutes=2),
            '5m': timedelta(minutes=10),
            '15m': timedelta(minutes=30),
            '30m': timedelta(hours=1),
            '1h': timedelta(hours=2),
            '4h': timedelta(hours=8),
            '1d': timedelta(days=7),  # Allow weekends/holidays
            '1w': timedelta(weeks=2),
            '1M': timedelta(days=35)
        }

    def analyze_gaps(self, cached_data: pd.DataFrame,
                     requested_start: datetime,
                     requested_end: datetime,
                     interval: str) -> List[DataGap]:
        """
        Identifies missing data periods that need to be downloaded.

        Args:
            cached_data: Existing cached data
            requested_start: Requested start date
            requested_end: Requested end date
            interval: Data interval (1m, 1h, 1d, etc.)

        Returns:
            List of DataGap objects representing missing periods
        """
        if cached_data.empty:
            return [DataGap(
                start_date=requested_start,
                end_date=requested_end,
                gap_type='no_data',
                expected_rows=self._estimate_rows(requested_start, requested_end, interval),
                priority=1
            )]

        gaps = []

        # Check for missing start period
        if cached_data.index.min() > requested_start:
            gap_start = requested_start
            gap_end = cached_data.index.min() - timedelta(microseconds=1)
            gaps.append(DataGap(
                start_date=gap_start,
                end_date=gap_end,
                gap_type='missing_start',
                expected_rows=self._estimate_rows(gap_start, gap_end, interval),
                priority=2  # Historical data
            ))

        # Check for missing end period
        if cached_data.index.max() < requested_end:
            gap_start = cached_data.index.max() + timedelta(microseconds=1)
            gap_end = requested_end
            gaps.append(DataGap(
                start_date=gap_start,
                end_date=gap_end,
                gap_type='missing_end',
                expected_rows=self._estimate_rows(gap_start, gap_end, interval),
                priority=1  # Recent data - high priority
            ))

        # Check for internal gaps
        internal_gaps = self._find_internal_gaps(cached_data, interval)
        gaps.extend(internal_gaps)

        # Sort gaps by priority (recent data first)
        gaps.sort(key=lambda x: x.priority)

        return gaps

    def _find_internal_gaps(self, data: pd.DataFrame, interval: str) -> List[DataGap]:
        """Find gaps within the cached data."""
        gaps = []

        if len(data) < 2:
            return gaps

        # Sort by timestamp to ensure chronological order
        sorted_data = data.sort_index()

        # Calculate expected time differences
        tolerance = self.interval_gap_tolerance.get(interval, timedelta(hours=1))

        for i in range(len(sorted_data) - 1):
            current_time = sorted_data.index[i]
            next_time = sorted_data.index[i + 1]

            expected_next = self._get_next_expected_time(current_time, interval)

            if next_time - expected_next > tolerance:
                gap_start = expected_next
                gap_end = next_time - timedelta(microseconds=1)

                gaps.append(DataGap(
                    start_date=gap_start,
                    end_date=gap_end,
                    gap_type='internal_gap',
                    expected_rows=self._estimate_rows(gap_start, gap_end, interval),
                    priority=2
                ))

        return gaps

    def _get_next_expected_time(self, current_time: datetime, interval: str) -> datetime:
        """Calculate the next expected timestamp based on interval."""
        if interval.endswith('m'):
            minutes = int(interval[:-1])
            return current_time + timedelta(minutes=minutes)
        elif interval.endswith('h'):
            hours = int(interval[:-1])
            return current_time + timedelta(hours=hours)
        elif interval.endswith('d'):
            days = int(interval[:-1])
            return current_time + timedelta(days=days)
        elif interval.endswith('w'):
            weeks = int(interval[:-1])
            return current_time + timedelta(weeks=weeks)
        elif interval.endswith('M'):
            months = int(interval[:-1])
            # Approximate month as 30 days
            return current_time + timedelta(days=months * 30)
        else:
            return current_time + timedelta(days=1)

    def _estimate_rows(self, start_date: datetime, end_date: datetime, interval: str) -> int:
        """Estimate the number of rows for a given date range and interval."""
        duration = end_date - start_date

        if interval.endswith('m'):
            minutes = int(interval[:-1])
            return int(duration.total_seconds() / (minutes * 60))
        elif interval.endswith('h'):
            hours = int(interval[:-1])
            return int(duration.total_seconds() / (hours * 3600))
        elif interval.endswith('d'):
            days = int(interval[:-1])
            return int(duration.days / days)
        elif interval.endswith('w'):
            weeks = int(interval[:-1])
            return int(duration.days / (weeks * 7))
        elif interval.endswith('M'):
            months = int(interval[:-1])
            return int(duration.days / (months * 30))
        else:
            return int(duration.days)


class CachedDataDownloader:
    """
    Professional caching wrapper for data downloaders.

    This class provides intelligent caching that:
    1. Checks cache first for existing data
    2. Identifies data gaps intelligently
    3. Downloads only missing data from servers
    4. Merges and caches the complete dataset
    5. Provides seamless data access with minimal server requests
    """

    def __init__(self, downloader: BaseDataDownloader, cache: FileBasedCache):
        """
        Initialize the cached data downloader.

        Args:
            downloader: The underlying data downloader (Binance, Yahoo, etc.)
            cache: File-based cache instance
        """
        self.downloader = downloader
        self.cache = cache
        self.gap_analyzer = DataGapAnalyzer()

        # Extract provider name from downloader class
        self.provider = self._extract_provider_name(downloader)

        _logger.info(f"Initialized CachedDataDownloader for {self.provider}")

    def _extract_provider_name(self, downloader: BaseDataDownloader) -> str:
        """Extract provider name from downloader class."""
        class_name = downloader.__class__.__name__.lower()

        if 'binance' in class_name:
            return 'binance'
        elif 'yahoo' in class_name:
            return 'yahoo'
        elif 'alpha' in class_name:
            return 'alpha_vantage'
        elif 'polygon' in class_name:
            return 'polygon'
        elif 'fmp' in class_name:
            return 'fmp'
        elif 'finnhub' in class_name:
            return 'finnhub'
        elif 'coingecko' in class_name:
            return 'coingecko'
        else:
            return 'unknown'

    def get_ohlcv(self, symbol: str, interval: str,
                   start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        """
        Get OHLCV data with intelligent caching.

        This method:
        1. Checks cache for existing data
        2. Identifies gaps in the data
        3. Downloads only missing data from the server
        4. Merges all data and caches the result
        5. Returns the complete dataset

        Args:
            symbol: Trading symbol
            interval: Data interval
            start_date: Start date
            end_date: End date
            **kwargs: Additional arguments for the downloader

        Returns:
            Complete OHLCV dataset for the requested period
        """
        try:
            _logger.info(f"Getting OHLCV data for {symbol} {interval} from {start_date} to {end_date}")

            # Step 1: Check cache for existing data
            cached_data = self._get_cached_data(symbol, interval, start_date, end_date)

            # Step 2: Analyze gaps in the data
            gaps = self.gap_analyzer.analyze_gaps(cached_data, start_date, end_date, interval)

            if not gaps:
                _logger.info(f"Complete data found in cache for {symbol} {interval}")
                return self._filter_to_requested_range(cached_data, start_date, end_date)

            # Step 3: Download missing data for each gap
            downloaded_data = []
            for gap in gaps:
                _logger.info(f"Downloading missing data for {symbol} {interval}: {gap.gap_type} "
                           f"from {gap.start_date} to {gap.end_date}")

                try:
                    gap_data = self.downloader.get_ohlcv(
                        symbol, interval, gap.start_date, gap.end_date, **kwargs
                    )

                    if not gap_data.empty:
                        downloaded_data.append(gap_data)
                        _logger.info(f"Downloaded {len(gap_data)} rows for gap {gap.gap_type}")
                    else:
                        _logger.warning(f"No data returned for gap {gap.gap_type}")

                except Exception as e:
                    _logger.error(f"Failed to download data for gap {gap.gap_type}: {e}")
                    continue

            # Step 4: Merge all data (cached + downloaded)
            if downloaded_data:
                all_data = self._merge_data(cached_data, downloaded_data)

                # Step 5: Cache the complete dataset
                self._cache_complete_dataset(symbol, interval, all_data, start_date, end_date)

                return self._filter_to_requested_range(all_data, start_date, end_date)
            else:
                _logger.warning(f"No missing data could be downloaded for {symbol} {interval}")
                return self._filter_to_requested_range(cached_data, start_date, end_date)

        except Exception as e:
            _logger.exception(f"Error in cached data download for {symbol} {interval}: {e}")
            raise

    def _get_cached_data(self, symbol: str, interval: str,
                         start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Retrieve data from cache for the specified period."""
        try:
            # Extend the search range slightly to get more context
            extended_start = start_date - timedelta(days=1)
            extended_end = end_date + timedelta(days=1)

            cached_data = self.cache.get(
                self.provider, symbol, interval,
                start_date=extended_start, end_date=extended_end,
                format='csv'
            )

            if cached_data is not None and not cached_data.empty:
                _logger.info(f"Found {len(cached_data)} cached rows for {symbol} {interval}")
                return cached_data
            else:
                _logger.info(f"No cached data found for {symbol} {interval}")
                return pd.DataFrame()

        except Exception as e:
            _logger.warning(f"Error retrieving cached data: {e}")
            return pd.DataFrame()

    def _merge_data(self, cached_data: pd.DataFrame,
                    downloaded_data: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge cached and downloaded data, handling duplicates and ordering."""
        if cached_data.empty:
            return pd.concat(downloaded_data, axis=0) if downloaded_data else pd.DataFrame()

        if not downloaded_data:
            return cached_data

        # Combine all data
        all_data = [cached_data] + downloaded_data

        # Concatenate and sort
        merged_data = pd.concat(all_data, axis=0)
        merged_data = merged_data.sort_index()

        # Remove duplicates (keep last occurrence)
        merged_data = merged_data[~merged_data.index.duplicated(keep='last')]

        _logger.info(f"Merged data: {len(cached_data)} cached + {sum(len(d) for d in downloaded_data)} "
                    f"downloaded = {len(merged_data)} total rows")

        return merged_data

    def _cache_complete_dataset(self, symbol: str, interval: str,
                               data: pd.DataFrame, start_date: datetime, end_date: datetime):
        """Cache the complete dataset."""
        try:
            success = self.cache.put(
                data, self.provider, symbol, interval,
                start_date=start_date, end_date=end_date,
                format='csv'
            )

            if success:
                _logger.info(f"Successfully cached complete dataset for {symbol} {interval}")
            else:
                _logger.warning(f"Failed to cache complete dataset for {symbol} {interval}")

        except Exception as e:
            _logger.error(f"Error caching complete dataset: {e}")

    def _filter_to_requested_range(self, data: pd.DataFrame,
                                   start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Filter data to exactly match the requested date range."""
        if data.empty:
            return data

        # Ensure we have a proper datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            _logger.warning(f"Data index is not DatetimeIndex: {type(data.index)}")
            # Try to convert the index to datetime
            try:
                if 'timestamp' in data.columns:
                    data = data.set_index('timestamp')
                elif 'datetime' in data.columns:
                    data = data.set_index('datetime')
                elif 'Date' in data.columns:
                    # Convert Date column to datetime and set as index
                    data = data.copy()
                    data['Date'] = pd.to_datetime(data['Date'])
                    data = data.set_index('Date')
                else:
                    _logger.error("Cannot find timestamp column to set as index")
                    return data
            except Exception as e:
                _logger.error(f"Error setting datetime index: {e}")
                return data

        # Now filter by date range
        try:
            filtered_data = data[
                (data.index >= start_date) &
                (data.index <= end_date)
            ]

            _logger.info(f"Filtered data to requested range: {len(filtered_data)} rows")
            return filtered_data
        except Exception as e:
            _logger.error(f"Error filtering data by date range: {e}")
            return data

    # Delegate other methods to the underlying downloader
    def get_periods(self) -> list:
        return self.downloader.get_periods()

    def get_intervals(self) -> list:
        return self.downloader.get_intervals()

    def is_valid_period_interval(self, period, interval) -> bool:
        return self.downloader.is_valid_period_interval(period, interval)

    def get_fundamentals(self, symbol: str):
        return self.downloader.get_fundamentals(symbol)

    def save_data(self, *args, **kwargs):
        return self.downloader.save_data(*args, **kwargs)

    def load_data(self, *args, **kwargs):
        return self.downloader.load_data(*args, **kwargs)

    def download_multiple_symbols(self, *args, **kwargs):
        return self.downloader.download_multiple_symbols(*args, **kwargs)


def create_cached_downloader(downloader: BaseDataDownloader,
                           cache_dir: str = "d:/data-cache") -> CachedDataDownloader:
    """
    Factory function to create a cached downloader.

    Args:
        downloader: The underlying data downloader
        cache_dir: Cache directory path

    Returns:
        CachedDataDownloader instance
    """
    from src.data.utils.file_based_cache import configure_file_cache

    cache = configure_file_cache(cache_dir=cache_dir)
    return CachedDataDownloader(downloader, cache)
