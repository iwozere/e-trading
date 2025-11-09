#!/usr/bin/env python3
"""
Cache Population Script

This script populates the unified cache with historical OHLCV data.
It can either populate specific tickers or analyze existing cache and fill missing data.

Features:
- Automatic ticker discovery from existing cache
- Missing data analysis and population
- Multiple interval support (5m, 15m, 1h, 4h, 1d)
- Year-based data storage with metadata
- Intelligent provider selection
- Mock provider detection and redownload
- Progress tracking and error handling

Mock Provider Detection:
- Automatically detects data from "mock" providers in metadata files
- Treats mock data as invalid and redownloads from real providers
- Checks both 'data_source' and 'provider_info.name' fields in metadata

Usage:
    # Populate specific tickers
    python src/data/cache/populate_cache.py --tickers AAPL,INTC --start-date 2020-01-01

    # Analyze existing cache and fill missing data
    python src/data/cache/populate_cache.py --start-date 2020-01-01

    # Custom intervals
    python src/data/cache/populate_cache.py --tickers AAPL --intervals 5m,15m,1h --start-date 2022-01-01
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import argparse
import json
from datetime import datetime, timedelta
from typing import List, Set, Dict, Optional, Tuple
import pandas as pd

from src.data.data_manager import DataManager
from src.notification.logger import setup_logger

try:
    from config.donotshare.donotshare import DATA_CACHE_DIR
except ImportError:
    DATA_CACHE_DIR = "c:/data-cache"

_logger = setup_logger(__name__)


class CachePopulator:
    """
    Cache population and analysis tool.

    This class handles:
    - Discovery of existing cached tickers
    - Analysis of missing data gaps
    - Population of missing data using DataManager
    - Progress tracking and reporting
    """

    def __init__(self, cache_dir: str = DATA_CACHE_DIR):
        """
        Initialize cache populator.

        Args:
            cache_dir: Path to cache directory
        """
        self.cache_dir = Path(cache_dir)
        self.ohlcv_dir = self.cache_dir / "ohlcv"
        self.data_manager = DataManager(cache_dir=str(cache_dir))

        # Default intervals to ensure are cached
        self.default_intervals = ["5m", "15m", "1h", "4h", "1d"]

        _logger.info("Cache populator initialized with cache dir: %s", self.cache_dir)

    def discover_existing_tickers(self) -> Set[str]:
        """
        Discover all tickers that already exist in the cache.

        Returns:
            Set of ticker symbols found in cache
        """
        tickers = set()

        if not self.ohlcv_dir.exists():
            _logger.info("OHLCV cache directory does not exist: %s", self.ohlcv_dir)
            return tickers

        for ticker_dir in self.ohlcv_dir.iterdir():
            if ticker_dir.is_dir() and not ticker_dir.name.startswith('_'):
                tickers.add(ticker_dir.name)

        _logger.info("Discovered %d existing tickers in cache: %s", len(tickers), sorted(tickers))
        return tickers

    def is_valid_cached_data(self, ticker: str, interval: str, year: int) -> bool:
        """
        Check if cached data for a specific ticker/interval/year is valid.

        Data is considered invalid if:
        - The data file doesn't exist
        - The metadata file doesn't exist
        - The provider in metadata is "mock"
        - The data doesn't cover the full year (starts too late or ends too early)

        Args:
            ticker: Ticker symbol
            interval: Data interval
            year: Year to check

        Returns:
            True if data is valid, False otherwise
        """
        interval_dir = self.ohlcv_dir / ticker / interval

        # Check if data file exists
        data_file = interval_dir / f"{year}.csv.gz"
        if not data_file.exists():
            return False

        # Check if metadata file exists
        metadata_file = interval_dir / f"{year}.metadata.json"
        if not metadata_file.exists():
            _logger.warning("Missing metadata file for %s %s %d", ticker, interval, year)
            return False

        # Check metadata for mock provider and date coverage
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Check if provider is mock
            provider = metadata.get('data_source', '').lower()
            if provider == 'mock':
                _logger.info("Found mock data for %s %s %d - marking for redownload", ticker, interval, year)
                return False

            # Also check provider_info.name if available
            provider_info = metadata.get('provider_info', {})
            provider_name = provider_info.get('name', '').lower()
            if provider_name == 'mock':
                _logger.info("Found mock provider in provider_info for %s %s %d - marking for redownload", ticker, interval, year)
                return False

            # Check date coverage - ensure data covers most of the year
            start_date_str = metadata.get('start_date')
            end_date_str = metadata.get('end_date')

            if start_date_str and end_date_str:
                try:
                    # Parse dates from metadata
                    start_date = pd.to_datetime(start_date_str).replace(tzinfo=None)
                    end_date = pd.to_datetime(end_date_str).replace(tzinfo=None)

                    # Define acceptable date ranges for the year based on interval
                    # Intraday data (1m, 5m, 15m, 30m, 1h) often has limited historical availability
                    # Daily data should have full year coverage
                    if interval in ['1m', '5m', '15m', '30m', '1h']:
                        # For intraday data, be very lenient - many providers only have 1-3 months
                        # Some providers like FMP may only provide recent months for intraday data
                        # Start can be as late as October for historical years
                        # End should be no earlier than November 1st for historical years
                        year_start_threshold = datetime(year, 10, 31)
                        year_end_threshold = datetime(year, 11, 1)
                    else:
                        # For daily/weekly/monthly data, expect full year coverage
                        # Start should be no later than January 10th (allowing for holidays/weekends)
                        # End should be no earlier than December 20th (allowing for holidays/weekends)
                        year_start_threshold = datetime(year, 1, 10)
                        year_end_threshold = datetime(year, 12, 20)

                    # For current year, adjust end threshold to today
                    current_year = datetime.now().year
                    if year == current_year:
                        today = datetime.now()
                        # If we're still early in the year, be more lenient
                        if today.month <= 6:
                            year_end_threshold = today - timedelta(days=7)  # Allow up to a week behind
                        else:
                            year_end_threshold = today - timedelta(days=3)  # Allow up to 3 days behind

                    # Check if data coverage is adequate
                    if start_date > year_start_threshold:
                        _logger.info("Data for %s %s %d starts too late (%s) - marking for redownload",
                                   ticker, interval, year, start_date.date())
                        return False

                    if end_date < year_end_threshold:
                        _logger.info("Data for %s %s %d ends too early (%s) - marking for redownload",
                                   ticker, interval, year, end_date.date())
                        return False

                    # Check minimum number of data points for the year
                    file_info = metadata.get('file_info', {})
                    rows = file_info.get('rows', 0)

                    # Define minimum expected rows based on interval
                    # Adjust expectations based on typical provider limitations
                    min_rows_by_interval = {
                        '1m': 10000,    # ~10k minutes (very limited historical intraday)
                        '5m': 2000,     # ~2k 5-minute periods (very limited historical intraday)
                        '15m': 700,     # ~700 15-minute periods (very limited historical intraday)
                        '30m': 350,     # ~350 30-minute periods (very limited historical intraday)
                        '1h': 175,      # ~175 hours (very limited historical intraday)
                        '4h': 400,      # ~400 4-hour periods in a trading year
                        '1d': 250,      # ~250 trading days in a year
                        '1w': 52,       # 52 weeks in a year
                        '1M': 12        # 12 months in a year
                    }

                    min_rows = min_rows_by_interval.get(interval, 100)

                    # For current year, adjust minimum based on how much of the year has passed
                    if year == current_year:
                        days_passed = (datetime.now() - datetime(year, 1, 1)).days
                        year_fraction = min(days_passed / 365.0, 1.0)
                        min_rows = int(min_rows * year_fraction * 0.7)  # Allow 30% tolerance
                    else:
                        min_rows = int(min_rows * 0.7)  # Allow 30% tolerance for historical years

                    if rows < min_rows:
                        _logger.info("Data for %s %s %d has too few rows (%d < %d) - marking for redownload",
                                   ticker, interval, year, rows, min_rows)
                        return False

                    _logger.debug("Data for %s %s %d is valid: %s to %s (%d rows)",
                                ticker, interval, year, start_date.date(), end_date.date(), rows)
                    return True

                except (ValueError, TypeError) as e:
                    _logger.warning("Error parsing dates for %s %s %d: %s", ticker, interval, year, e)
                    return False
            else:
                _logger.warning("Missing date information in metadata for %s %s %d", ticker, interval, year)
                return False

        except (json.JSONDecodeError, KeyError, IOError) as e:
            _logger.warning("Error reading metadata for %s %s %d: %s", ticker, interval, year, e)
            return False

    def analyze_missing_data(self, ticker: str, intervals: List[str],
                           start_date: datetime, end_date: datetime) -> Dict[str, List[int]]:
        """
        Analyze what data is missing for a ticker across intervals.

        Args:
            ticker: Ticker symbol
            intervals: List of intervals to check
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            Dictionary mapping interval to list of missing years
        """
        missing_data = {}

        ticker_dir = self.ohlcv_dir / ticker
        if not ticker_dir.exists():
            # No data exists for this ticker
            years_needed = list(range(start_date.year, end_date.year + 1))
            for interval in intervals:
                missing_data[interval] = years_needed
            return missing_data

        # Check each interval
        for interval in intervals:
            interval_dir = ticker_dir / interval
            missing_years = []

            if not interval_dir.exists():
                # No data for this interval
                missing_years = list(range(start_date.year, end_date.year + 1))
            else:
                # Check which years are missing or invalid (including mock data)
                needed_years = set(range(start_date.year, end_date.year + 1))

                for year in needed_years:
                    if not self.is_valid_cached_data(ticker, interval, year):
                        missing_years.append(year)

                missing_years = sorted(missing_years)

            if missing_years:
                missing_data[interval] = missing_years

        return missing_data

    def get_year_date_range(self, year: int, start_date: datetime, end_date: datetime) -> Tuple[datetime, datetime]:
        """
        Get the date range for a specific year, bounded by start_date and end_date.

        Args:
            year: Year to get range for
            start_date: Overall start date
            end_date: Overall end date

        Returns:
            Tuple of (year_start, year_end) dates
        """
        year_start = datetime(year, 1, 1)
        year_end = datetime(year, 12, 31, 23, 59, 59)

        # Bound by overall date range
        year_start = max(year_start, start_date)
        year_end = min(year_end, end_date)

        return year_start, year_end

    def populate_missing_data(self, ticker: str, interval: str, missing_years: List[int],
                            start_date: datetime, end_date: datetime) -> bool:
        """
        Populate missing data for a ticker/interval/years combination.

        Args:
            ticker: Ticker symbol
            interval: Data interval
            missing_years: List of years to populate
            start_date: Overall start date
            end_date: Overall end date

        Returns:
            True if successful, False otherwise
        """
        success = True

        for year in missing_years:
            try:
                year_start, year_end = self.get_year_date_range(year, start_date, end_date)

                _logger.info("Populating %s %s for year %d (%s to %s)",
                           ticker, interval, year, year_start.date(), year_end.date())

                # Use DataManager to get data (this will cache it automatically)
                data = self.data_manager.get_ohlcv(
                    symbol=ticker,
                    timeframe=interval,
                    start_date=year_start,
                    end_date=year_end,
                    force_refresh=True  # Force download even if some data exists
                )

                if data is not None and not data.empty:
                    _logger.info("✅ Successfully populated %s %s %d: %d rows",
                               ticker, interval, year, len(data))
                else:
                    _logger.warning("⚠️ No data returned for %s %s %d", ticker, interval, year)

            except Exception as e:
                _logger.error("❌ Failed to populate %s %s %d: %s", ticker, interval, year, e)
                success = False
                continue

        return success

    def populate_ticker(self, ticker: str, intervals: List[str],
                       start_date: datetime, end_date: datetime) -> Dict[str, bool]:
        """
        Populate all missing data for a single ticker.

        Args:
            ticker: Ticker symbol
            intervals: List of intervals to populate
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary mapping interval to success status
        """
        _logger.info("Analyzing missing data for %s", ticker)

        missing_data = self.analyze_missing_data(ticker, intervals, start_date, end_date)

        if not missing_data:
            _logger.info("✅ %s: All data already cached", ticker)
            return {interval: True for interval in intervals}

        results = {}

        for interval in intervals:
            if interval in missing_data:
                missing_years = missing_data[interval]
                _logger.info("%s %s: Missing %d years: %s", ticker, interval,
                           len(missing_years), missing_years)

                success = self.populate_missing_data(ticker, interval, missing_years,
                                                   start_date, end_date)
                results[interval] = success
            else:
                _logger.info("✅ %s %s: Already cached", ticker, interval)
                results[interval] = True

        return results

    def populate_cache(self, tickers: Optional[List[str]] = None,
                      intervals: Optional[List[str]] = None,
                      start_date: datetime = None,
                      end_date: datetime = None) -> Dict[str, Dict[str, bool]]:
        """
        Main method to populate cache data.

        Args:
            tickers: List of tickers to populate (None = discover from cache)
            intervals: List of intervals to populate (None = use defaults)
            start_date: Start date for data population
            end_date: End date for data population

        Returns:
            Dictionary mapping ticker to interval success status
        """
        # Set defaults
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = datetime(2020, 1, 1)
        if intervals is None:
            intervals = self.default_intervals

        # Discover tickers if not provided
        if tickers is None:
            discovered_tickers = self.discover_existing_tickers()
            if not discovered_tickers:
                _logger.warning("No existing tickers found in cache and none provided")
                return {}
            tickers = sorted(discovered_tickers)

        _logger.info("Starting cache population:")
        _logger.info("  Tickers: %s", tickers)
        _logger.info("  Intervals: %s", intervals)
        _logger.info("  Date range: %s to %s", start_date.date(), end_date.date())
        _logger.info("  Enhanced validation: Checking date coverage and data completeness")

        results = {}
        total_tickers = len(tickers)

        for i, ticker in enumerate(tickers, 1):
            _logger.info("Processing ticker %d/%d: %s", i, total_tickers, ticker)

            try:
                ticker_results = self.populate_ticker(ticker, intervals, start_date, end_date)
                results[ticker] = ticker_results

                # Summary for this ticker
                successful_intervals = sum(1 for success in ticker_results.values() if success)
                _logger.info("✅ %s: %d/%d intervals successful",
                           ticker, successful_intervals, len(intervals))

            except Exception as e:
                _logger.error("❌ Failed to process ticker %s: %s", ticker, e)
                results[ticker] = {interval: False for interval in intervals}

        # Final summary
        self.print_summary(results)
        return results

    def print_summary(self, results: Dict[str, Dict[str, bool]]):
        """Print a summary of population results."""
        _logger.info("=" * 60)
        _logger.info("CACHE POPULATION SUMMARY")
        _logger.info("=" * 60)

        total_operations = 0
        successful_operations = 0

        for ticker, ticker_results in results.items():
            successful_intervals = sum(1 for success in ticker_results.values() if success)
            total_intervals = len(ticker_results)

            total_operations += total_intervals
            successful_operations += successful_intervals

            status = "✅" if successful_intervals == total_intervals else "⚠️"
            _logger.info("%s %s: %d/%d intervals successful",
                       status, ticker, successful_intervals, total_intervals)

        success_rate = (successful_operations / total_operations * 100) if total_operations > 0 else 0
        _logger.info("-" * 60)
        _logger.info("Overall: %d/%d operations successful (%.1f%%)",
                   successful_operations, total_operations, success_rate)
        _logger.info("=" * 60)


def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Populate cache with historical OHLCV data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze existing cache and fill missing data from 2020
  python src/data/cache/populate_cache.py --start-date 2020-01-01

  # Populate specific tickers
  python src/data/cache/populate_cache.py --tickers AAPL,INTC,VT --start-date 2020-01-01

  # Custom intervals
  python src/data/cache/populate_cache.py --tickers AAPL --intervals 5m,15m,1h --start-date 2022-01-01

  # Full date range
  python src/data/cache/populate_cache.py --tickers AAPL --start-date 2020-01-01 --end-date 2023-12-31
        """
    )

    parser.add_argument(
        "--tickers",
        type=str,
        help="Comma-separated list of tickers to populate (default: discover from existing cache)"
    )

    parser.add_argument(
        "--intervals",
        type=str,
        default="5m,15m,1h,4h,1d",
        help="Comma-separated list of intervals to populate (default: 5m,15m,1h,4h,1d)"
    )

    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=datetime(2020, 1, 1),
        help="Start date for data population (YYYY-MM-DD, default: 2020-01-01)"
    )

    parser.add_argument(
        "--end-date",
        type=parse_date,
        help="End date for data population (YYYY-MM-DD, default: today)"
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=DATA_CACHE_DIR,
        help=f"Cache directory path (default: {DATA_CACHE_DIR})"
    )

    args = parser.parse_args()

    # Parse tickers and intervals
    tickers = None
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]

    intervals = [i.strip() for i in args.intervals.split(",")]

    # Initialize populator
    populator = CachePopulator(cache_dir=args.cache_dir)

    # Run population
    try:
        results = populator.populate_cache(
            tickers=tickers,
            intervals=intervals,
            start_date=args.start_date,
            end_date=args.end_date
        )

        # Exit with appropriate code
        total_operations = sum(len(ticker_results) for ticker_results in results.values())
        successful_operations = sum(
            sum(1 for success in ticker_results.values() if success)
            for ticker_results in results.values()
        )

        if successful_operations == total_operations:
            _logger.info("All operations completed successfully")
            sys.exit(0)
        else:
            _logger.warning("Some operations failed")
            sys.exit(1)

    except KeyboardInterrupt:
        _logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception:
        _logger.exception("Fatal error:")
        sys.exit(1)


if __name__ == "__main__":
    main()