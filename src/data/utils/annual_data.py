#!/usr/bin/env python3
"""
Annual Data Downloader Script

This script downloads historical OHLCV data for specified tickers and timeframes,
saving data in annual chunks to the data/annual directory.

Features:
- Downloads data year by year for better organization
- Supports multiple tickers and timeframes
- Uses appropriate data provider (Binance for crypto)
- Saves data as CSV files with naming convention: {TICKER}_{TIMEFRAME}_{STARTDATE}_{ENDDATE}.csv
- Progress tracking and error handling
- Configurable start year and end year

Default Configuration:
- Tickers: BTCUSDT, ETHUSDT, LTCUSDT
- Timeframes: 5m, 15m, 30m, 1h, 4h
- Start Year: 2020
- End Year: Current year

Usage:
    # Use defaults
    python src/data/utils/annual_data.py

    # Custom tickers
    python src/data/utils/annual_data.py --tickers BTCUSDT,ETHUSDT,BNBUSDT

    # Custom timeframes
    python src/data/utils/annual_data.py --timeframes 1h,4h,1d

    # Custom start year
    python src/data/utils/annual_data.py --start-year 2019

    # Custom end year (default is current year)
    python src/data/utils/annual_data.py --end-year 2023

    # All custom
    python src/data/utils/annual_data.py --tickers BTCUSDT --timeframes 5m,15m --start-year 2022 --end-year 2023
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
import time
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.data.data_manager import DataManager
from src.notification.logger import setup_logger

# Initialize DataManager with caching
try:
    from config.donotshare.donotshare import DATA_CACHE_DIR
except ImportError:
    DATA_CACHE_DIR = "c:/data-cache"

_logger = setup_logger(__name__)


class AnnualDataDownloader:
    """
    Downloads and saves annual OHLCV data for specified tickers and timeframes.
    """

    def __init__(self, output_dir: str = "data/annual"):
        """
        Initialize annual data downloader.

        Args:
            output_dir: Directory to save annual data files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize DataManager for direct access with explicit dates
        self.data_manager = DataManager(cache_dir=DATA_CACHE_DIR)

        _logger.info("Annual data downloader initialized")
        _logger.info("Output directory: %s", self.output_dir.absolute())

    def get_year_date_range(self, year: int) -> tuple[datetime, datetime]:
        """
        Get start and end dates for a specific year.

        Args:
            year: Year to get date range for

        Returns:
            Tuple of (start_date, end_date) for the year
        """
        start_date = datetime(year, 1, 1, 0, 0, 0)
        end_date = datetime(year, 12, 31, 23, 59, 59)

        # If current year, end at today
        current_year = datetime.now().year
        if year == current_year:
            end_date = datetime.now()

        return start_date, end_date

    def format_date_for_filename(self, date: datetime) -> str:
        """
        Format date for filename (YYYYMMDD).

        Args:
            date: Date to format

        Returns:
            Formatted date string
        """
        return date.strftime("%Y%m%d")

    def generate_filename(self, ticker: str, timeframe: str, start_date: datetime, end_date: datetime) -> str:
        """
        Generate filename for annual data.

        Format: {TICKER}_{TIMEFRAME}_{STARTDATE}_{ENDDATE}.csv

        Args:
            ticker: Ticker symbol
            timeframe: Data timeframe
            start_date: Start date
            end_date: End date

        Returns:
            Filename string
        """
        start_str = self.format_date_for_filename(start_date)
        end_str = self.format_date_for_filename(end_date)
        return f"{ticker}_{timeframe}_{start_str}_{end_str}.csv"

    def download_annual_data(self, ticker: str, timeframe: str, year: int,
                            force_refresh: bool = False) -> Dict[str, Any]:
        """
        Download data for a specific ticker/timeframe/year.

        Args:
            ticker: Ticker symbol (e.g., 'BTCUSDT')
            timeframe: Data timeframe (e.g., '5m', '1h', '1d')
            year: Year to download
            force_refresh: Force re-download even if file exists

        Returns:
            Dictionary with download results

        Note:
            Provider is automatically selected by DataManager based on symbol/timeframe
        """
        result = {
            'ticker': ticker,
            'timeframe': timeframe,
            'year': year,
            'success': False,
            'filename': None,
            'rows': 0,
            'error': None
        }

        try:
            # Get date range for the year
            start_date, end_date = self.get_year_date_range(year)

            # Generate filename
            filename = self.generate_filename(ticker, timeframe, start_date, end_date)
            filepath = self.output_dir / filename

            # Only force refresh for current year
            current_year = datetime.now().year
            use_force_refresh = force_refresh if year == current_year else False

            # Check if file already exists
            if filepath.exists() and not use_force_refresh:
                _logger.info("File already exists: %s (use --force-refresh to re-download)", filename)
                result['success'] = True
                result['filename'] = filename
                result['skipped'] = True
                return result

            _logger.info("Downloading %s %s for year %d (%s to %s) [force_refresh=%s]",
                        ticker, timeframe, year,
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"), use_force_refresh)

            # Download data using DataManager with explicit date range
            # Note: Don't use get_ohlcv() with period string, as "365d" means "365 days from NOW"
            # We need explicit start_date and end_date for historical years
            # Note: DataManager automatically selects best provider (doesn't accept provider param)
            df = self.data_manager.get_ohlcv(
                symbol=ticker,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                force_refresh=use_force_refresh
            )

            if df is None or df.empty:
                error_msg = f"No data returned for {ticker} {timeframe} {year}"
                _logger.warning(error_msg)
                result['error'] = error_msg
                return result

            # Handle timestamp in index or column
            # When loaded from UnifiedCache, timestamp is a DatetimeIndex (not column)
            if df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()  # Convert index to column

            # Filter data to exact year range (in case provider returns extra data)
            if 'timestamp' in df.columns:
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)  # Remove timezone if present
                df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

                # Add human-readable timestamp column if not present
                if 'timestamp_human' not in df.columns:
                    df['timestamp_human'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                # Place human-readable column first for convenience
                cols = ['timestamp_human'] + [col for col in df.columns if col != 'timestamp_human']
                df = df[cols]

            if df.empty:
                error_msg = f"No data within date range for {ticker} {timeframe} {year}"
                _logger.warning(error_msg)
                result['error'] = error_msg
                return result

            # Save to CSV
            df.to_csv(filepath, index=False)

            result['success'] = True
            result['filename'] = filename
            result['rows'] = len(df)
            result['filepath'] = str(filepath.absolute())

            _logger.info("✅ Saved %s: %d rows", filename, len(df))

        except Exception as e:
            error_msg = f"Error downloading {ticker} {timeframe} {year}: {e}"
            _logger.exception(error_msg)
            result['error'] = error_msg

        return result

    def download_batch(self, tickers: List[str], timeframes: List[str],
                      start_year: int, end_year: int = None,
                      force_refresh: bool = False,
                      delay: float = 0.5) -> List[Dict[str, Any]]:
        """
        Download data for multiple tickers/timeframes/years.

        Args:
            tickers: List of ticker symbols
            timeframes: List of timeframes
            start_year: First year to download
            end_year: Last year to download (defaults to current year)
            force_refresh: Force re-download even if files exist
            delay: Delay between downloads in seconds

        Returns:
            List of result dictionaries

        Note:
            Provider is automatically selected by DataManager based on symbol/timeframe
        """
        if end_year is None:
            end_year = datetime.now().year

        years = list(range(start_year, end_year + 1))

        _logger.info("Starting batch download:")
        _logger.info("  Tickers: %s", tickers)
        _logger.info("  Timeframes: %s", timeframes)
        _logger.info("  Years: %d-%d (%d years)", start_year, end_year, len(years))
        _logger.info("  Total downloads: %d", len(tickers) * len(timeframes) * len(years))

        results = []
        total_downloads = len(tickers) * len(timeframes) * len(years)
        completed = 0

        for ticker in tickers:
            for timeframe in timeframes:
                for year in years:
                    completed += 1

                    _logger.info("Progress: %d/%d - Processing %s %s %d",
                               completed, total_downloads, ticker, timeframe, year)

                    result = self.download_annual_data(
                        ticker=ticker,
                        timeframe=timeframe,
                        year=year,
                        force_refresh=force_refresh
                    )

                    results.append(result)

                    # Add delay between downloads to respect rate limits
                    if completed < total_downloads and delay > 0:
                        time.sleep(delay)

        return results

    def print_summary(self, results: List[Dict[str, Any]]):
        """
        Print summary of download results.

        Args:
            results: List of result dictionaries
        """
        _logger.info("=" * 80)
        _logger.info("DOWNLOAD SUMMARY")
        _logger.info("=" * 80)

        successful = [r for r in results if r['success']]
        skipped = [r for r in results if r.get('skipped', False)]
        failed = [r for r in results if not r['success']]

        total_rows = sum(r['rows'] for r in successful)

        _logger.info("Total downloads: %d", len(results))
        _logger.info("Successful: %d", len(successful))
        _logger.info("Skipped (already exists): %d", len(skipped))
        _logger.info("Failed: %d", len(failed))
        _logger.info("Total rows downloaded: %s", f"{total_rows:,}")

        if failed:
            _logger.warning("-" * 80)
            _logger.warning("FAILED DOWNLOADS:")
            for result in failed:
                _logger.warning("  %s %s %d: %s",
                              result['ticker'], result['timeframe'], result['year'],
                              result.get('error', 'Unknown error'))

        _logger.info("=" * 80)
        _logger.info("Output directory: %s", self.output_dir.absolute())
        _logger.info("=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download annual OHLCV data for crypto tickers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use defaults (BTCUSDT, ETHUSDT, LTCUSDT from 2020 in 5m, 15m, 30m, 1h, 4h)
  python src/data/utils/annual_data.py

  # Custom tickers
  python src/data/utils/annual_data.py --tickers BTCUSDT,ETHUSDT,BNBUSDT

  # Custom timeframes
  python src/data/utils/annual_data.py --timeframes 1h,4h,1d

  # Custom start year
  python src/data/utils/annual_data.py --start-year 2019

  # Custom end year
  python src/data/utils/annual_data.py --end-year 2023

  # Force refresh existing files
  python src/data/utils/annual_data.py --force-refresh

  # All custom
  python src/data/utils/annual_data.py --tickers BTCUSDT --timeframes 5m,15m --start-year 2022 --end-year 2023
        """
    )

    parser.add_argument(
        "--tickers",
        type=str,
        default="BTCUSDT,ETHUSDT,LTCUSDT",
        help="Comma-separated list of tickers (default: BTCUSDT,ETHUSDT,LTCUSDT)"
    )

    parser.add_argument(
        "--timeframes",
        type=str,
        default="5m,15m,30m,1h,4h",
        help="Comma-separated list of timeframes (default: 5m,15m,30m,1h,4h)"
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=2020,
        help="Start year for data download (default: 2020)"
    )

    parser.add_argument(
        "--end-year",
        type=int,
        help="End year for data download (default: current year)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/annual",
        help="Output directory for annual data files (default: data/annual)"
    )

    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-download even if files already exist"
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between downloads in seconds (default: 0.5)"
    )

    args = parser.parse_args()

    # Parse tickers and timeframes
    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    timeframes = [t.strip().lower() for t in args.timeframes.split(",")]

    # Validate start year
    current_year = datetime.now().year
    if args.start_year > current_year:
        _logger.error("Start year cannot be in the future (current year: %d)", current_year)
        sys.exit(1)

    # Validate end year
    end_year = args.end_year if args.end_year else current_year
    if end_year > current_year:
        _logger.warning("End year %d is in the future, using current year %d", end_year, current_year)
        end_year = current_year

    if end_year < args.start_year:
        _logger.error("End year cannot be before start year")
        sys.exit(1)

    # Initialize downloader
    downloader = AnnualDataDownloader(output_dir=args.output_dir)

    # Run download
    try:
        results = downloader.download_batch(
            tickers=tickers,
            timeframes=timeframes,
            start_year=args.start_year,
            end_year=end_year,
            force_refresh=args.force_refresh,
            delay=args.delay
        )

        # Print summary
        downloader.print_summary(results)

        # Exit with appropriate code
        failed = [r for r in results if not r['success']]
        if failed:
            _logger.warning("Some downloads failed")
            sys.exit(1)
        else:
            _logger.info("All downloads completed successfully")
            sys.exit(0)

    except KeyboardInterrupt:
        _logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception:
        _logger.exception("Fatal error during download:")
        sys.exit(1)


if __name__ == "__main__":
    main()
