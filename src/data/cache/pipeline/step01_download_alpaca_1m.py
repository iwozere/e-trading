#!/usr/bin/env python3
"""
Pipeline Step 1: Alpaca 1-Minute Data Downloader

This script downloads 1-minute OHLCV data from Alpaca for all tickers found in the cache directory.
Data is downloaded from 2020-01-01 to yesterday (last midnight) and stored as gzipped ticker-1m.csv.gz files.

Pipeline Features:
- Discovers tickers from existing cache directory
- Downloads 1-minute data using the new AlpacaDataDownloader
- Stores data as gzipped ticker-1m.csv.gz in each ticker folder (4x smaller files)
- Handles existing files by detecting and filling gaps
- Respects 10,000 bar limit per request
- Enhanced statistics and error reporting for pipeline
- Identifies failed tickers (likely crypto/forex not supported by Alpaca)
- Rate limiting (200 requests/minute)

Requirements:
- Alpaca API key and secret in config/donotshare/donotshare.py
- alpaca-py package (replaces deprecated alpaca-trade-api)

Usage:
    python src/data/cache/pipeline/step01_download_alpaca_1m.py
    python src/data/cache/pipeline/step01_download_alpaca_1m.py --tickers AAPL,MSFT,GOOGL
    python src/data/cache/pipeline/step01_download_alpaca_1m.py --start-date 2022-01-01
    python src/data/cache/pipeline/step01_download_alpaca_1m.py --force-refresh
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import argparse
import time
from datetime import datetime, timedelta
from typing import List, Set, Optional, Tuple, Dict, Any
import pandas as pd

from src.notification.logger import setup_logger
from src.data.downloader.alpaca_data_downloader import AlpacaDataDownloader

try:
    from config.donotshare.donotshare import DATA_CACHE_DIR
except ImportError:
    DATA_CACHE_DIR = "c:/data-cache"

_logger = setup_logger(__name__)


class AlpacaPipelineDownloader:
    """
    Pipeline Step 1: Downloads 1-minute data from Alpaca for all tickers in cache directory.

    This class handles:
    - Ticker discovery from cache directory
    - Gap detection in existing data files
    - Incremental updates (only download missing data)
    - Enhanced statistics and error reporting for pipeline
    - Rate limiting and error handling
    - Progress tracking
    """

    def __init__(self, cache_dir: str = None):
        """
        Initialize the pipeline downloader.

        Args:
            cache_dir: Cache directory path (defaults to DATA_CACHE_DIR)
        """
        self.cache_dir = Path(cache_dir or DATA_CACHE_DIR)
        self.ohlcv_dir = self.cache_dir / "ohlcv"

        # Pipeline statistics
        self.stats = {
            'total_tickers': 0,
            'successful_tickers': [],
            'failed_tickers': [],
            'skipped_tickers': [],
            'total_rows_downloaded': 0,
            'total_api_calls': 0,
            'processing_time': 0,
            'errors': {}
        }

        # Initialize Alpaca downloader
        try:
            self.downloader = AlpacaDataDownloader()
            _logger.info("Alpaca downloader initialized successfully")
        except Exception:
            _logger.exception("Failed to initialize Alpaca downloader:")
            raise

    def discover_tickers(self) -> Set[str]:
        """
        Discover all tickers that exist in the cache directory.

        Returns:
            Set of ticker symbols found in cache
        """
        tickers = set()

        if not self.ohlcv_dir.exists():
            _logger.warning("OHLCV cache directory does not exist: %s", self.ohlcv_dir)
            return tickers

        for ticker_dir in self.ohlcv_dir.iterdir():
            if ticker_dir.is_dir() and not ticker_dir.name.startswith('_'):
                # Include all tickers (let Alpaca API determine what's supported)
                ticker = ticker_dir.name.upper()
                tickers.add(ticker)

        _logger.info("Discovered %d tickers in cache: %s", len(tickers), sorted(tickers))
        return tickers

    def get_existing_data_info(self, ticker: str) -> Optional[Tuple[datetime, datetime, int]]:
        """
        Get information about existing 1m data file (supports both .csv and .csv.gz).

        Args:
            ticker: Ticker symbol

        Returns:
            Tuple of (start_date, end_date, row_count) or None if file doesn't exist
        """
        ticker_dir = self.ohlcv_dir / ticker
        csv_file = ticker_dir / f"{ticker}-1m.csv"
        csv_gz_file = ticker_dir / f"{ticker}-1m.csv.gz"

        # Check for gzipped file first, then regular CSV
        if csv_gz_file.exists():
            data_file = csv_gz_file
            is_gzipped = True
        elif csv_file.exists():
            data_file = csv_file
            is_gzipped = False
        else:
            return None

        try:
            # Read data based on file type
            if is_gzipped:
                # Read gzipped file
                df_head = pd.read_csv(data_file, compression='gzip', nrows=1)
                df_full = pd.read_csv(data_file, compression='gzip')
                df_tail = df_full.tail(1)
            else:
                # Read regular CSV file
                df_head = pd.read_csv(data_file, nrows=1)
                df_full = pd.read_csv(data_file)
                df_tail = df_full.tail(1)

            if df_head.empty or df_tail.empty:
                return None

            # Get date range
            start_date = pd.to_datetime(df_head['timestamp'].iloc[0])
            end_date = pd.to_datetime(df_tail['timestamp'].iloc[0])

            # Get total row count
            row_count = len(df_full)

            _logger.debug("Existing data for %s: %s to %s (%d rows) [%s]",
                         ticker, start_date.date(), end_date.date(), row_count,
                         "gzipped" if is_gzipped else "uncompressed")

            return start_date, end_date, row_count

        except Exception as e:
            _logger.warning("Error reading existing data for %s: %s", ticker, e)
            return None

    def calculate_download_ranges(self, ticker: str, target_start: datetime,
                                target_end: datetime) -> List[Tuple[datetime, datetime]]:
        """
        Calculate date ranges that need to be downloaded.
        Focus on filling gaps before and after existing data.

        Args:
            ticker: Ticker symbol
            target_start: Desired start date
            target_end: Desired end date

        Returns:
            List of (start_date, end_date) tuples to download
        """
        existing_info = self.get_existing_data_info(ticker)

        if existing_info is None:
            # No existing data, download everything
            return [(target_start, target_end)]

        existing_start, existing_end, row_count = existing_info
        ranges_to_download = []

        # Check if we need data before existing start (fill gap before)
        if target_start < existing_start:
            gap_end = existing_start - timedelta(minutes=1)
            ranges_to_download.append((target_start, gap_end))
            _logger.debug("Gap before existing data for %s: %s to %s",
                         ticker, target_start.date(), gap_end.date())

        # Check if we need data after existing end (fill gap after)
        if target_end > existing_end:
            gap_start = existing_end + timedelta(minutes=1)
            ranges_to_download.append((gap_start, target_end))
            _logger.debug("Gap after existing data for %s: %s to %s",
                         ticker, gap_start.date(), target_end.date())

        if not ranges_to_download:
            _logger.info("No gaps found for %s, data is up to date", ticker)
        else:
            _logger.info("Found %d gap(s) for %s", len(ranges_to_download), ticker)

        return ranges_to_download

    def download_ticker_data(self, ticker: str, start_date: datetime,
                           end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Download 1-minute data for a single ticker.

        Args:
            ticker: Ticker symbol
            start_date: Start date for data download
            end_date: End date for data download

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            _logger.info("Downloading 1m data for %s from %s to %s",
                        ticker, start_date.date(), end_date.date())

            # Track API calls
            self.stats['total_api_calls'] += 1

            # Download data using the new Alpaca downloader
            df = self.downloader.get_ohlcv(ticker, "1m", start_date, end_date)

            if df is None or df.empty:
                _logger.warning("No data returned for %s", ticker)
                return None

            # Ensure proper column names and format
            if 'timestamp' not in df.columns:
                df.reset_index(inplace=True)
                if df.index.name:
                    df.rename(columns={df.index.name: 'timestamp'}, inplace=True)

            # Ensure timestamp column exists
            if 'timestamp' not in df.columns and df.index.name is None:
                df['timestamp'] = df.index

            _logger.info("Downloaded %d rows for %s", len(df), ticker)
            return df

        except Exception as e:
            error_msg = str(e)
            _logger.error("Error downloading data for %s: %s", ticker, error_msg)

            # Categorize errors for pipeline statistics
            if "not found" in error_msg.lower() or "invalid symbol" in error_msg.lower():
                self.stats['errors'][ticker] = "Symbol not found (likely crypto/forex)"
            elif "rate limit" in error_msg.lower():
                self.stats['errors'][ticker] = "Rate limit exceeded"
            elif "authentication" in error_msg.lower():
                self.stats['errors'][ticker] = "Authentication error"
            else:
                self.stats['errors'][ticker] = f"API error: {error_msg}"

            return None

    def save_ticker_data(self, ticker: str, new_data: pd.DataFrame,
                        append_mode: bool = False) -> bool:
        """
        Save ticker data as gzipped CSV file.

        Args:
            ticker: Ticker symbol
            new_data: DataFrame with OHLCV data
            append_mode: If True, merge with existing data

        Returns:
            True if successful, False otherwise
        """
        try:
            ticker_dir = self.ohlcv_dir / ticker
            ticker_dir.mkdir(parents=True, exist_ok=True)

            csv_file = ticker_dir / f"{ticker}-1m.csv"
            csv_gz_file = ticker_dir / f"{ticker}-1m.csv.gz"

            if append_mode and (csv_gz_file.exists() or csv_file.exists()):
                # Load existing data and merge
                if csv_gz_file.exists():
                    existing_df = pd.read_csv(csv_gz_file, compression='gzip')
                    _logger.debug("Loading existing gzipped data for %s", ticker)
                else:
                    existing_df = pd.read_csv(csv_file)
                    _logger.debug("Loading existing uncompressed data for %s", ticker)

                existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])

                # Ensure new data has timestamp column
                if 'timestamp' not in new_data.columns:
                    new_data.reset_index(inplace=True)

                new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])

                # Combine and remove duplicates
                combined_df = pd.concat([existing_df, new_data], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['timestamp'])
                combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)

                # Save combined data as gzipped CSV
                combined_df.to_csv(csv_gz_file, index=False, compression='gzip')

                # Remove old uncompressed file if it exists
                if csv_file.exists():
                    csv_file.unlink()
                    _logger.debug("Removed old uncompressed file: %s", csv_file)

                _logger.info("Merged and saved %d total rows to %s (gzipped)", len(combined_df), csv_gz_file)
            else:
                # Save new data only as gzipped CSV
                if 'timestamp' not in new_data.columns:
                    new_data.reset_index(inplace=True)

                new_data.to_csv(csv_gz_file, index=False, compression='gzip')

                # Remove old uncompressed file if it exists
                if csv_file.exists():
                    csv_file.unlink()
                    _logger.debug("Removed old uncompressed file: %s", csv_file)

                _logger.info("Saved %d rows to %s (gzipped)", len(new_data), csv_gz_file)

            return True

        except Exception:
            _logger.exception("Error saving data for %s:", ticker)
            return False

    def process_ticker(self, ticker: str, start_date: datetime, end_date: datetime,
                      force_refresh: bool = False) -> Dict[str, Any]:
        """
        Process a single ticker with enhanced error handling and statistics.

        Args:
            ticker: Ticker symbol
            start_date: Start date for downloads
            end_date: End date for downloads
            force_refresh: If True, re-download all data

        Returns:
            Dictionary with processing results
        """
        result = {
            'ticker': ticker,
            'success': False,
            'rows_downloaded': 0,
            'ranges_processed': 0,
            'action': 'unknown',
            'error': None,
            'processing_time': 0
        }

        start_time = time.time()

        try:
            if force_refresh:
                # Download everything
                download_ranges = [(start_date, end_date)]
            else:
                # Calculate gaps
                download_ranges = self.calculate_download_ranges(ticker, start_date, end_date)

            if not download_ranges:
                result.update({
                    'success': True,
                    'action': 'up_to_date',
                    'processing_time': time.time() - start_time
                })
                return result

            total_new_rows = 0
            all_success = True

            for range_start, range_end in download_ranges:
                # Download data for this range
                data = self.download_ticker_data(ticker, range_start, range_end)

                if data is not None and not data.empty:
                    # Save data (append mode for gaps)
                    append_mode = not force_refresh and len(download_ranges) > 1
                    success = self.save_ticker_data(ticker, data, append_mode)

                    if success:
                        total_new_rows += len(data)
                        self.stats['total_rows_downloaded'] += len(data)
                    else:
                        all_success = False
                        break
                else:
                    _logger.warning("No data for %s in range %s to %s",
                                  ticker, range_start.date(), range_end.date())

                # Rate limiting - Alpaca allows 200 requests per minute
                time.sleep(0.3)  # ~200 requests per minute

            result.update({
                'success': all_success,
                'rows_downloaded': total_new_rows,
                'ranges_processed': len(download_ranges),
                'action': 'updated' if total_new_rows > 0 else 'no_data',
                'processing_time': time.time() - start_time
            })

        except Exception as e:
            error_msg = str(e)
            result.update({
                'success': False,
                'error': error_msg,
                'processing_time': time.time() - start_time
            })

            # Store error for statistics
            self.stats['errors'][ticker] = error_msg

        return result

    def download_all_tickers(self, tickers: List[str], start_date: datetime,
                           end_date: datetime, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Download 1-minute data for all specified tickers with enhanced pipeline statistics.

        Args:
            tickers: List of ticker symbols
            start_date: Start date for downloads
            end_date: End date for downloads
            force_refresh: If True, re-download all data

        Returns:
            Dictionary with comprehensive pipeline results
        """
        pipeline_start_time = time.time()
        self.stats['total_tickers'] = len(tickers)

        _logger.info("=" * 60)
        _logger.info("PIPELINE STEP 1: ALPACA 1M DATA DOWNLOAD")
        _logger.info("=" * 60)
        _logger.info("Tickers: %d symbols", len(tickers))
        _logger.info("Date range: %s to %s", start_date.date(), end_date.date())
        _logger.info("Force refresh: %s", force_refresh)
        _logger.info("=" * 60)

        results = {}

        for i, ticker in enumerate(tickers, 1):
            _logger.info("Processing ticker %d/%d: %s", i, len(tickers), ticker)

            result = self.process_ticker(ticker, start_date, end_date, force_refresh)
            results[ticker] = result

            # Update statistics
            if result['success']:
                if result['action'] == 'up_to_date':
                    self.stats['skipped_tickers'].append(ticker)
                    _logger.info("✅ %s: Up to date", ticker)
                elif result['action'] == 'updated':
                    self.stats['successful_tickers'].append(ticker)
                    _logger.info("✅ %s: Downloaded %d new rows", ticker, result['rows_downloaded'])
                else:
                    self.stats['successful_tickers'].append(ticker)
                    _logger.info("✅ %s: Processed successfully", ticker)
            else:
                self.stats['failed_tickers'].append(ticker)
                error = result.get('error', 'Unknown error')
                _logger.error("❌ %s: %s", ticker, error)

        # Calculate final statistics
        self.stats['processing_time'] = time.time() - pipeline_start_time

        # Print comprehensive pipeline summary
        self.print_pipeline_summary()

        return {
            'results': results,
            'statistics': self.stats
        }

    def print_pipeline_summary(self):
        """Print comprehensive pipeline summary with statistics."""
        _logger.info("=" * 80)
        _logger.info("PIPELINE STEP 1 SUMMARY")
        _logger.info("=" * 80)

        # Overall statistics
        total = self.stats['total_tickers']
        successful = len(self.stats['successful_tickers'])
        skipped = len(self.stats['skipped_tickers'])
        failed = len(self.stats['failed_tickers'])

        _logger.info("📊 PROCESSING STATISTICS:")
        _logger.info("   Total tickers processed: %d", total)
        _logger.info("   ✅ Successfully updated: %d", successful)
        _logger.info("   ⏭️  Already up to date: %d", skipped)
        _logger.info("   ❌ Failed: %d", failed)
        _logger.info("   📈 Success rate: %.1f%%", (successful + skipped) / total * 100 if total > 0 else 0)

        _logger.info("\n📈 DATA STATISTICS:")
        _logger.info("   Total rows downloaded: %d", self.stats['total_rows_downloaded'])
        _logger.info("   Total API calls made: %d", self.stats['total_api_calls'])
        _logger.info("   Processing time: %.1f seconds", self.stats['processing_time'])

        # Successful tickers
        if self.stats['successful_tickers']:
            _logger.info("\n✅ SUCCESSFULLY UPDATED TICKERS (%d):", len(self.stats['successful_tickers']))
            for ticker in sorted(self.stats['successful_tickers']):
                _logger.info("   %s", ticker)

        # Up-to-date tickers
        if self.stats['skipped_tickers']:
            _logger.info("\n⏭️  UP-TO-DATE TICKERS (%d):", len(self.stats['skipped_tickers']))
            for ticker in sorted(self.stats['skipped_tickers']):
                _logger.info("   %s", ticker)

        # Failed tickers with reasons
        if self.stats['failed_tickers']:
            _logger.info("\n❌ FAILED TICKERS (%d):", len(self.stats['failed_tickers']))
            for ticker in sorted(self.stats['failed_tickers']):
                error = self.stats['errors'].get(ticker, 'Unknown error')
                _logger.info("   %s: %s", ticker, error)

            # Categorize failures
            crypto_failures = [t for t, e in self.stats['errors'].items()
                             if "not found" in e.lower() or "crypto" in e.lower()]
            if crypto_failures:
                _logger.info("\n🔍 LIKELY CRYPTO/FOREX TICKERS (not supported by Alpaca):")
                for ticker in sorted(crypto_failures):
                    _logger.info("   %s", ticker)

        _logger.info("=" * 80)
        _logger.info("PIPELINE STEP 1 COMPLETED")
        _logger.info("=" * 80)


def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Pipeline Step 1: Download 1-minute Alpaca data for all tickers in cache directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download for all cached tickers from 2020 to yesterday
  python src/data/cache/pipeline/step01_download_alpaca_1m.py

  # Download for specific tickers
  python src/data/cache/pipeline/step01_download_alpaca_1m.py --tickers AAPL,MSFT,GOOGL

  # Custom date range
  python src/data/cache/pipeline/step01_download_alpaca_1m.py --start-date 2022-01-01 --end-date 2023-12-31

  # Force refresh (re-download all data)
  python src/data/cache/pipeline/step01_download_alpaca_1m.py --force-refresh

Environment Variables Required:
  ALPACA_API_KEY - Your Alpaca API key
  ALPACA_SECRET_KEY - Your Alpaca secret key
  ALPACA_BASE_URL - Alpaca base URL (optional, defaults to paper trading)
        """
    )

    parser.add_argument(
        "--tickers",
        type=str,
        help="Comma-separated list of tickers to download (default: discover from cache)"
    )

    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=datetime(2020, 1, 1),
        help="Start date for data download (YYYY-MM-DD, default: 2020-01-01 UTC)"
    )

    parser.add_argument(
        "--end-date",
        type=parse_date,
        default=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
        help="End date for data download (YYYY-MM-DD, default: today's midnight, includes all of yesterday)"
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=DATA_CACHE_DIR,
        help=f"Cache directory path (default: {DATA_CACHE_DIR})"
    )

    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh: re-download all data (ignore existing files)"
    )

    args = parser.parse_args()

    try:
        # Initialize pipeline downloader
        downloader = AlpacaPipelineDownloader(args.cache_dir)

        # Get tickers
        if args.tickers:
            tickers = [t.strip().upper() for t in args.tickers.split(",")]
            _logger.info("Using specified tickers: %s", tickers)
        else:
            discovered_tickers = downloader.discover_tickers()
            if not discovered_tickers:
                _logger.error("No tickers found in cache directory and none specified")
                sys.exit(1)
            tickers = sorted(discovered_tickers)

        # Run pipeline step 1
        pipeline_results = downloader.download_all_tickers(
            tickers, args.start_date, args.end_date, args.force_refresh
        )

        # Exit with appropriate code based on results
        stats = pipeline_results['statistics']
        if len(stats['failed_tickers']) == 0:
            _logger.info("Pipeline Step 1 completed successfully - all tickers processed")
            sys.exit(0)
        elif len(stats['successful_tickers']) + len(stats['skipped_tickers']) > 0:
            _logger.warning("Pipeline Step 1 completed with some failures")
            sys.exit(0)  # Continue pipeline even with some failures
        else:
            _logger.error("Pipeline Step 1 failed - no tickers processed successfully")
            sys.exit(1)

    except KeyboardInterrupt:
        _logger.info("Pipeline Step 1 cancelled by user")
        sys.exit(1)
    except Exception:
        _logger.exception("Pipeline Step 1 fatal error:")
        sys.exit(1)


if __name__ == "__main__":
    main()