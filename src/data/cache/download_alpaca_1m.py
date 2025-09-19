#!/usr/bin/env python3
"""
Alpaca 1-Minute Data Downloader

This script downloads 1-minute OHLCV data from Alpaca for all tickers found in the cache directory.
Data is downloaded from 2020-01-01 to yesterday and stored as CSV files.

Features:
- Discovers tickers from existing cache directory
- Downloads 1-minute data from Alpaca API
- Stores data as ticker-1m.csv in each ticker folder
- Handles API rate limits and pagination
- Progress tracking and error handling

Requirements:
- Alpaca API key and secret in environment variables or config
- alpaca-trade-api package

Usage:
    python src/data/cache/download_alpaca_1m.py
    python src/data/cache/download_alpaca_1m.py --tickers AAPL,MSFT,GOOGL
    python src/data/cache/download_alpaca_1m.py --start-date 2022-01-01
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import argparse
import time
from datetime import datetime, timedelta
from typing import List, Set, Optional
import pandas as pd

from src.notification.logger import setup_logger

try:
    from config.donotshare.donotshare import DATA_CACHE_DIR
except ImportError:
    DATA_CACHE_DIR = "c:/data-cache"

# Try to import Alpaca API
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("Warning: alpaca-trade-api not installed. Install with: pip install alpaca-trade-api")

_logger = setup_logger(__name__)


class AlpacaDownloader:
    """
    Alpaca 1-minute data downloader.

    This class handles downloading 1-minute OHLCV data from Alpaca
    for multiple tickers with proper rate limiting and error handling.
    """

    def __init__(self, api_key: str = None, secret_key: str = None, base_url: str = None):
        """
        Initialize Alpaca downloader.

        Args:
            api_key: Alpaca API key (if None, will try environment variables)
            secret_key: Alpaca secret key (if None, will try environment variables)
            base_url: Alpaca base URL (default: paper trading URL)
        """
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-trade-api package is required. Install with: pip install alpaca-trade-api")

        # Get API credentials
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        self.base_url = base_url or os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API credentials required. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
                "environment variables or pass them as parameters."
            )

        # Initialize Alpaca API
        self.api = tradeapi.REST(
            key_id=self.api_key,
            secret_key=self.secret_key,
            base_url=self.base_url,
            api_version='v2'
        )

        self.cache_dir = Path(DATA_CACHE_DIR)
        self.ohlcv_dir = self.cache_dir / "ohlcv"

        _logger.info("Alpaca downloader initialized with base URL: %s", self.base_url)

    def discover_tickers(self) -> Set[str]:
        """
        Discover all tickers that exist in the cache directory.

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

        _logger.info("Discovered %d tickers in cache: %s", len(tickers), sorted(tickers))
        return tickers

    def download_ticker_data(self, ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Download 1-minute data for a single ticker from Alpaca.

        Args:
            ticker: Ticker symbol
            start_date: Start date for data download
            end_date: End date for data download

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            _logger.info("Downloading 1m data for %s from %s to %s", ticker, start_date.date(), end_date.date())

            # Download data from Alpaca
            # Note: Alpaca API has limits on date ranges, so we might need to chunk large requests
            bars = self.api.get_bars(
                symbol=ticker,
                timeframe=tradeapi.TimeFrame.Minute,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                adjustment='raw',
                limit=None  # Get all available data
            )

            if not bars:
                _logger.warning("No data returned for %s", ticker)
                return None

            # Convert to DataFrame
            data = []
            for bar in bars:
                data.append({
                    'timestamp': bar.timestamp,
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': int(bar.volume)
                })

            if not data:
                _logger.warning("No bars data for %s", ticker)
                return None

            df = pd.DataFrame(data)

            # Ensure timestamp is timezone-naive
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)

            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)

            _logger.info("Downloaded %d rows for %s", len(df), ticker)
            return df

        except Exception as e:
            _logger.error("Error downloading data for %s: %s", ticker, str(e))
            return None

    def save_ticker_data(self, ticker: str, data: pd.DataFrame) -> bool:
        """
        Save ticker data as CSV file.

        Args:
            ticker: Ticker symbol
            data: DataFrame with OHLCV data

        Returns:
            True if successful, False otherwise
        """
        try:
            ticker_dir = self.ohlcv_dir / ticker
            ticker_dir.mkdir(parents=True, exist_ok=True)

            csv_file = ticker_dir / f"{ticker}-1m.csv"

            # Save as CSV
            data.to_csv(csv_file, index=False)

            _logger.info("Saved %d rows to %s", len(data), csv_file)
            return True

        except Exception as e:
            _logger.error("Error saving data for %s: %s", ticker, str(e))
            return False

    def download_all_tickers(self, tickers: List[str], start_date: datetime, end_date: datetime) -> dict:
        """
        Download 1-minute data for all specified tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date for downloads
            end_date: End date for downloads

        Returns:
            Dictionary with results for each ticker
        """
        results = {}
        total_tickers = len(tickers)

        _logger.info("Starting Alpaca 1m data download:")
        _logger.info("  Tickers: %s", tickers)
        _logger.info("  Date range: %s to %s", start_date.date(), end_date.date())

        for i, ticker in enumerate(tickers, 1):
            _logger.info("Processing ticker %d/%d: %s", i, total_tickers, ticker)

            try:
                # Download data
                data = self.download_ticker_data(ticker, start_date, end_date)

                if data is not None and not data.empty:
                    # Save data
                    success = self.save_ticker_data(ticker, data)
                    results[ticker] = {
                        'success': success,
                        'rows': len(data),
                        'start_date': data['timestamp'].min(),
                        'end_date': data['timestamp'].max()
                    }

                    if success:
                        _logger.info("✅ %s: %d rows saved", ticker, len(data))
                    else:
                        _logger.error("❌ %s: Failed to save data", ticker)
                else:
                    _logger.warning("⚠️ %s: No data available", ticker)
                    results[ticker] = {'success': False, 'rows': 0, 'error': 'No data available'}

                # Rate limiting - Alpaca allows 200 requests per minute
                # Add small delay to be safe
                time.sleep(0.5)

            except Exception as e:
                _logger.error("❌ %s: Error - %s", ticker, str(e))
                results[ticker] = {'success': False, 'rows': 0, 'error': str(e)}

        # Print summary
        self.print_summary(results)
        return results

    def print_summary(self, results: dict):
        """Print a summary of download results."""
        _logger.info("=" * 60)
        _logger.info("ALPACA 1M DOWNLOAD SUMMARY")
        _logger.info("=" * 60)

        successful = 0
        total_rows = 0

        for ticker, result in results.items():
            if result['success']:
                successful += 1
                total_rows += result['rows']
                start_date = result.get('start_date', 'Unknown')
                end_date = result.get('end_date', 'Unknown')
                if isinstance(start_date, pd.Timestamp):
                    start_date = start_date.strftime('%Y-%m-%d')
                if isinstance(end_date, pd.Timestamp):
                    end_date = end_date.strftime('%Y-%m-%d')
                _logger.info("✅ %s: %d rows (%s to %s)", ticker, result['rows'], start_date, end_date)
            else:
                error = result.get('error', 'Unknown error')
                _logger.info("❌ %s: %s", ticker, error)

        success_rate = (successful / len(results) * 100) if results else 0
        _logger.info("-" * 60)
        _logger.info("Overall: %d/%d tickers successful (%.1f%%)", successful, len(results), success_rate)
        _logger.info("Total rows downloaded: %d", total_rows)
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
        description="Download 1-minute data from Alpaca for cached tickers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download for all cached tickers from 2020
  python src/data/cache/download_alpaca_1m.py

  # Download for specific tickers
  python src/data/cache/download_alpaca_1m.py --tickers AAPL,MSFT,GOOGL

  # Custom date range
  python src/data/cache/download_alpaca_1m.py --start-date 2022-01-01 --end-date 2023-12-31

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
        help="Start date for data download (YYYY-MM-DD, default: 2020-01-01)"
    )

    parser.add_argument(
        "--end-date",
        type=parse_date,
        default=datetime.now() - timedelta(days=1),  # Yesterday
        help="End date for data download (YYYY-MM-DD, default: yesterday)"
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=DATA_CACHE_DIR,
        help=f"Cache directory path (default: {DATA_CACHE_DIR})"
    )

    args = parser.parse_args()

    try:
        # Initialize downloader
        downloader = AlpacaDownloader()

        # Get tickers
        if args.tickers:
            tickers = [t.strip().upper() for t in args.tickers.split(",")]
        else:
            discovered_tickers = downloader.discover_tickers()
            if not discovered_tickers:
                _logger.error("No tickers found in cache directory and none specified")
                sys.exit(1)
            tickers = sorted(discovered_tickers)

        # Download data
        results = downloader.download_all_tickers(tickers, args.start_date, args.end_date)

        # Exit with appropriate code
        successful = sum(1 for r in results.values() if r['success'])
        if successful == len(results):
            _logger.info("All downloads completed successfully")
            sys.exit(0)
        else:
            _logger.warning("Some downloads failed")
            sys.exit(1)

    except KeyboardInterrupt:
        _logger.info("Download cancelled by user")
        sys.exit(1)
    except Exception as e:
        _logger.error("Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()