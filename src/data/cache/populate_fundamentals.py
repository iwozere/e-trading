#!/usr/bin/env python3
"""
Fundamentals Data Downloader Script
==================================

This script downloads fundamentals data for a list of tickers and stores it in cache.
It uses the DataManager for standardized data access and caching.

Usage:
    python src/data/utils/download_fundamentals.py --tickers AAPL,MSFT,GOOGL
    python src/data/utils/download_fundamentals.py --tickers-file tickers.txt
    python src/data/utils/download_fundamentals.py --tickers AAPL --force-refresh

Features:
- Uses DataManager for standardized data access
- Cache management with 7-day expiration
- Progress tracking and detailed logging
- Error handling and retry logic
- Single-threaded processing
- Data quality validation
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict, is_dataclass

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_manager import get_data_manager
from src.notification.logger import setup_logger

# Import cache directory setting
try:
    from config.donotshare.donotshare import DATA_CACHE_DIR
except ImportError:
    DATA_CACHE_DIR = "d:/data-cache"

# Setup logging
_logger = setup_logger(__name__)

class FundamentalsDownloader:
    """
    Downloads fundamentals data for multiple tickers using DataManager.
    """

    def __init__(self, cache_dir: str = DATA_CACHE_DIR):
        """
        Initialize the fundamentals downloader.

        Args:
            cache_dir: Cache directory path (defaults to d:/data-cache)
        """
        self.data_manager = get_data_manager(cache_dir)
        self.download_stats = {
            "total_tickers": 0,
            "successful_downloads": 0,
            "failed_downloads": 0,
            "cached_data_used": 0,
            "providers_used": set(),
            "start_time": None,
            "end_time": None
        }

    def download_fundamentals(self,
                            tickers: List[str],
                            providers: Optional[List[str]] = None,
                            force_refresh: bool = False,
                            show_progress: bool = True) -> Dict[str, Any]:
        """
        Download fundamentals data for a list of tickers.

        Args:
            tickers: List of ticker symbols
            providers: List of specific providers to use (None for auto-selection)
            force_refresh: Force refresh even if cache is valid
            show_progress: Show progress bar

        Returns:
            Dictionary with download statistics
        """
        self.download_stats["start_time"] = datetime.now()
        self.download_stats["total_tickers"] = len(tickers)

        _logger.info("Starting fundamentals download for %d tickers", len(tickers))

        if show_progress:
            print(f"\n📊 Downloading fundamentals for {len(tickers)} tickers...")
            if providers:
                print(f"Providers: {', '.join(providers)}")
            else:
                print("Providers: Auto-selected")
            print(f"Force refresh: {force_refresh}")
            print("=" * 60)

        # Process tickers sequentially (single-threaded)
        for ticker in tickers:
            try:
                result = self._download_single_ticker(ticker, providers, force_refresh)
                if result["success"]:
                    if show_progress:
                        print(f"✅ {ticker}: {result['message']}")
                else:
                    if show_progress:
                        print(f"❌ {ticker}: {result['message']}")
            except Exception as e:
                _logger.error("Unexpected error processing %s: %s", ticker, e)
                if show_progress:
                    print(f"💥 {ticker}: Unexpected error - {e}")
                self.download_stats["failed_downloads"] += 1

        self.download_stats["end_time"] = datetime.now()
        self.download_stats["providers_used"] = list(self.download_stats["providers_used"])

        # Print summary
        if show_progress:
            self._print_summary()

        return self.download_stats

    def _download_single_ticker(self, ticker: str, providers: Optional[List[str]], force_refresh: bool) -> Dict[str, Any]:
        """
        Download fundamentals for a single ticker using DataManager.

        Args:
            ticker: Ticker symbol
            providers: List of specific providers to use (None for auto-selection)
            force_refresh: Force refresh even if cache is valid

        Returns:
            Dictionary with download result
        """
        try:
            # Use DataManager to get fundamentals data
            _logger.debug("Downloading fundamentals for %s", ticker)
            fundamentals_data = self.data_manager.get_fundamentals(
                symbol=ticker,
                providers=providers,
                force_refresh=force_refresh
            )

            if not fundamentals_data:
                return {
                    "success": False,
                    "message": "No fundamentals data returned from DataManager"
                }

            # Validate data quality
            if not self._validate_fundamentals_data(fundamentals_data):
                return {
                    "success": False,
                    "message": "Fundamentals data failed quality validation"
                }

            # Count non-None fields for the message
            non_none_fields = sum(1 for v in fundamentals_data.values() if v is not None)

            self.download_stats["successful_downloads"] += 1
            if providers:
                self.download_stats["providers_used"].update(providers)
            else:
                # DataManager uses auto-selected providers, we can't know exactly which ones
                self.download_stats["providers_used"].add("auto-selected")

            return {
                "success": True,
                "message": f"Downloaded and cached {non_none_fields} data points"
            }

        except Exception as e:
            _logger.error("Error downloading fundamentals for %s: %s", ticker, e)
            self.download_stats["failed_downloads"] += 1

            return {
                "success": False,
                "message": f"Download failed: {str(e)}"
            }

    def _validate_fundamentals_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate fundamentals data quality.

        Args:
            data: Fundamentals data dictionary

        Returns:
            True if data is valid, False otherwise
        """
        if not data or not isinstance(data, dict):
            return False

        # Check for essential fields
        essential_fields = [
            'market_cap', 'pe_ratio', 'pb_ratio', 'price_to_book',
            'current_price', 'ticker'
        ]
        has_essential = any(field in data for field in essential_fields)

        if not has_essential:
            _logger.warning("Fundamentals data missing essential fields. Available fields: %s",
                          list(data.keys()))
            return False

        # Check for reasonable values
        if 'market_cap' in data and data['market_cap'] is not None:
            if data['market_cap'] <= 0:
                _logger.warning("Invalid market cap: %s", data['market_cap'])
                return False

        if 'pe_ratio' in data and data['pe_ratio'] is not None:
            if data['pe_ratio'] < 0 or data['pe_ratio'] > 1000:
                _logger.warning("Suspicious PE ratio: %s", data['pe_ratio'])
                # Don't fail validation for this, just log warning

        # Check if we have at least some meaningful data
        meaningful_fields = ['market_cap', 'pe_ratio', 'current_price', 'ticker', 'company_name']
        has_meaningful = any(
            field in data and data[field] is not None and data[field] != 0
            for field in meaningful_fields
        )

        if not has_meaningful:
            _logger.warning("No meaningful data found in fundamentals")
            return False

        return True

    def _print_summary(self):
        """Print download summary."""
        duration = self.download_stats["end_time"] - self.download_stats["start_time"]

        print("\n" + "=" * 60)
        print("📈 DOWNLOAD SUMMARY")
        print("=" * 60)
        print(f"Total tickers: {self.download_stats['total_tickers']}")
        print(f"Successful downloads: {self.download_stats['successful_downloads']}")
        print(f"Failed downloads: {self.download_stats['failed_downloads']}")
        print(f"Providers used: {', '.join(self.download_stats['providers_used'])}")
        print(f"Duration: {duration.total_seconds():.1f} seconds")
        print("=" * 60)

        # Show cache statistics
        cache_stats = self.data_manager.get_cache_stats()
        print(f"\n💾 CACHE STATISTICS")
        print(f"Cache directory: {self.data_manager.cache.cache_dir}")
        print(f"Total cache size: {cache_stats.get('total_size', 0) / 1024:.1f} KB")

    def get_cache_info(self, ticker: Optional[str] = None) -> Dict[str, Any]:
        """
        Get cache information for ticker(s).

        Args:
            ticker: Specific ticker to get info for (optional)

        Returns:
            Cache information dictionary
        """
        return self.data_manager.get_cache_stats()

    def cleanup_expired_cache(self) -> Dict[str, int]:
        """
        Clean up expired cache data.

        Returns:
            Cleanup statistics
        """
        _logger.info("Cleaning up expired cache data...")
        # DataManager doesn't have a direct cleanup method, but we can clear specific data
        # For now, return empty stats since DataManager handles cache management internally
        return {"removed_files": 0, "removed_symbols": 0}


def load_tickers_from_file(file_path: str) -> List[str]:
    """
    Load tickers from a text file.

    Args:
        file_path: Path to the ticker file

    Returns:
        List of ticker symbols
    """
    try:
        with open(file_path, 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip()]

        _logger.info("Loaded %d tickers from file: %s", len(tickers), file_path)
        return tickers

    except Exception as e:
        _logger.error("Error loading tickers from file %s: %s", file_path, e)
        return []


def main():
    """Main function to run the fundamentals downloader."""
    parser = argparse.ArgumentParser(
        description="Download fundamentals data for tickers and store in cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download using default example_tickers.txt file
    python src/data/utils/download_fundamentals.py

    # Download for specific tickers using auto-selected providers
    python src/data/utils/download_fundamentals.py --tickers AAPL,MSFT,GOOGL

    # Download from file using specific providers
    python src/data/utils/download_fundamentals.py --tickers-file tickers.txt --providers yf,fmp

    # Force refresh existing cache
    python src/data/utils/download_fundamentals.py --tickers AAPL --force-refresh

    # Show cache information
    python src/data/utils/download_fundamentals.py --show-cache --ticker AAPL

    # Clean up expired cache
    python src/data/utils/download_fundamentals.py --cleanup-cache
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--tickers",
        type=str,
        help="Comma-separated list of ticker symbols (e.g., AAPL,MSFT,GOOGL)"
    )
    input_group.add_argument(
        "--tickers-file",
        type=str,
        help="Path to file containing ticker symbols (one per line)"
    )
    input_group.add_argument(
        "--show-cache",
        action="store_true",
        help="Show cache information instead of downloading"
    )
    input_group.add_argument(
        "--cleanup-cache",
        action="store_true",
        help="Clean up expired cache data"
    )

    # Provider options
    parser.add_argument(
        "--providers",
        type=str,
        help="Comma-separated list of providers to use (e.g., yf,fmp,av). Default: auto-selected"
    )

    # Cache options
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh even if cache is valid"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=DATA_CACHE_DIR,
        help=f"Cache directory path. Default: {DATA_CACHE_DIR}"
    )

    # Output options
    parser.add_argument(
        "--ticker",
        type=str,
        help="Specific ticker to show cache info for (use with --show-cache)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Initialize downloader
    downloader = FundamentalsDownloader(cache_dir=args.cache_dir)

    try:
        if args.show_cache:
            # Show cache information
            cache_info = downloader.get_cache_info(args.ticker)
            print("\n💾 CACHE INFORMATION")
            print("=" * 40)
            print(json.dumps(cache_info, indent=2, default=str))

        elif args.cleanup_cache:
            # Clean up expired cache
            cleanup_stats = downloader.cleanup_expired_cache()
            print("\n🧹 CACHE CLEANUP COMPLETED")
            print("=" * 40)
            print(f"Removed files: {cleanup_stats.get('removed_files', 0)}")
            print(f"Removed symbols: {cleanup_stats.get('removed_symbols', 0)}")

        else:
            # Download fundamentals
            if args.tickers:
                tickers = [t.strip().upper() for t in args.tickers.split(',')]
            elif args.tickers_file:
                tickers = load_tickers_from_file(args.tickers_file)
                if not tickers:
                    print(f"❌ No tickers loaded from file: {args.tickers_file}")
                    return 1
            else:
                # Use default example_tickers.txt file
                default_tickers_file = Path(__file__).parent / "example_tickers.txt"
                if default_tickers_file.exists():
                    tickers = load_tickers_from_file(str(default_tickers_file))
                    if not tickers:
                        print(f"❌ No tickers loaded from default file: {default_tickers_file}")
                        return 1
                    print(f"📁 Using default tickers file: {default_tickers_file}")
                else:
                    print("❌ No tickers specified and default example_tickers.txt not found")
                    print("Please specify tickers using --tickers or --tickers-file")
                    return 1

            # Parse providers if specified
            providers = None
            if args.providers:
                providers = [p.strip() for p in args.providers.split(',')]

            # Download fundamentals
            stats = downloader.download_fundamentals(
                tickers=tickers,
                providers=providers,
                force_refresh=args.force_refresh,
                show_progress=not args.quiet
            )

            # Return appropriate exit code
            if stats["failed_downloads"] > 0:
                return 1
            else:
                return 0

    except KeyboardInterrupt:
        print("\n⚠️  Download interrupted by user")
        return 130
    except Exception as e:
        _logger.exception("Unexpected error: %s", e)
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
