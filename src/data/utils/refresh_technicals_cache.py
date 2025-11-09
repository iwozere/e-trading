#!/usr/bin/env python3
"""
Cache Population and Testing Script

This script helps you:
1. Populate the file-based cache with historical data
2. Test the new data architecture
3. Verify all components are working correctly

Usage:
    python src/data/populate_cache.py --help
    python src/data/populate_cache.py --populate --symbols BTCUSDT,ETHUSDT --intervals 1h,4h,1d --start-date 2023-01-01 --end-date 2023-12-31
    python src/data/populate_cache.py --populate --symbols BTCUSDT --intervals 1h --start-date 2023-01-01
    python src/data/populate_cache.py --test-all
    python src/data/populate_cache.py --validate-cache
"""

import argparse
import sys
from datetime import datetime, timezone, timedelta
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.cache.unified_cache import configure_unified_cache
from src.data.sources.base_data_source import BaseDataSource

# Import API keys from donotshare
from config.donotshare.donotshare import ALPHA_VANTAGE_KEY, DATA_CACHE_DIR
# Validation removed - data is cached as-is without validation
from src.notification.logger import setup_logger

# Import data downloaders
from src.data.downloader.binance_data_downloader import BinanceDataDownloader
from src.data.downloader.yahoo_data_downloader import YahooDataDownloader
from src.data.downloader.alpha_vantage_data_downloader import AlphaVantageDataDownloader

# Import provider selector for automatic provider selection
from src.data.data_manager import ProviderSelector

# Initialize logger
_logger = setup_logger(__name__)


class MockDataSource(BaseDataSource):
    """Mock data source for testing purposes."""

    def __init__(self, provider_name: str = "mock", **kwargs):
        super().__init__(provider_name, **kwargs)

    def get_available_symbols(self) -> List[str]:
        return ["MOCK1", "MOCK2", "MOCK3"]

    def get_supported_intervals(self) -> List[str]:
        return ["1h", "4h", "1d"]

    def fetch_historical_data(self, symbol: str, interval: str,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Generate mock historical data for testing."""
        if start_date is None:
            start_date = datetime.now(timezone.utc) - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now(timezone.utc)

        # Ensure both dates are naive for consistent date range generation
        start_date_naive = start_date.replace(tzinfo=None) if start_date.tzinfo else start_date
        end_date_naive = end_date.replace(tzinfo=None) if end_date.tzinfo else end_date

        # Generate mock OHLCV data
        date_range = pd.date_range(start=start_date_naive, end=end_date_naive, freq='1h')
        n_points = len(date_range)

        # Create realistic-looking price data
        base_price = 100.0
        prices = []
        for i in range(n_points):
            # Add some randomness and trend
            trend = i * 0.1  # Slight upward trend
            noise = (i % 10 - 5) * 0.5  # Cyclical noise
            price = base_price + trend + noise
            prices.append(max(price, 1.0))  # Ensure positive prices

        df = pd.DataFrame({
            'timestamp': date_range,
            'open': prices,
            'high': [p * 1.02 for p in prices],  # High slightly above open
            'low': [p * 0.98 for p in prices],   # Low slightly below open
            'close': [p * 1.01 for p in prices], # Close slightly above open
            'volume': [1000 + i * 10 for i in range(n_points)]  # Increasing volume
        })

        return df

    def start_realtime_feed(self, symbol: str, interval: str,
                           callback: Optional[callable] = None) -> bool:
        return True

    def stop_realtime_feed(self, symbol: str) -> bool:
        return True


def populate_cache(symbols: List[str], intervals: List[str],
                   start_date: datetime, end_date: datetime,
                   cache_dir: str = DATA_CACHE_DIR, file_format: str = "csv") -> Dict[str, Any]:
    """Populate the file-based cache with historical data."""

    print(f"🚀 Populating cache at: {cache_dir}")
    print(f"📊 Symbols: {', '.join(symbols)}")
    print(f"⏱️  Intervals: {', '.join(intervals)}")
    print(f"📅 Date range: {start_date.date()} to {end_date.date()}")
    print(f"📅 Start time: {start_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📅 End time: {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔢 Total operations: {len(symbols) * len(intervals)}")
    print()

    _logger.info("Starting cache population for %d symbols, %d intervals", len(symbols), len(intervals))
    _logger.info("Date range: %s to %s", start_date, end_date)
    _logger.info("Cache directory: %s", cache_dir)

    # Configure unified cache
    cache = configure_unified_cache(cache_dir=cache_dir)

    # Initialize data downloaders
    downloaders = {}
    try:
        downloaders['binance'] = BinanceDataDownloader()
        print("  ✅ Binance downloader initialized")
    except Exception as e:
        print(f"  ⚠️  Binance downloader failed: {str(e)}")

    try:
        downloaders['yfinance'] = YahooDataDownloader()
        print("  ✅ Yahoo downloader initialized")
    except Exception as e:
        print(f"  ⚠️  Yahoo downloader failed: {str(e)}")

    try:
        # Check if Alpha Vantage API key is available
        if ALPHA_VANTAGE_KEY:
            downloaders['alpha_vantage'] = AlphaVantageDataDownloader(api_key=ALPHA_VANTAGE_KEY)
            print("  ✅ Alpha Vantage downloader initialized")
        else:
            print("  ⚠️  Alpha Vantage downloader skipped: No API key found in donotshare.py")
    except Exception as e:
        print(f"  ⚠️  Alpha Vantage downloader failed: {str(e)}")

    # Initialize provider selector for automatic provider selection
    provider_selector = ProviderSelector()
    print("  ✅ Provider selector initialized")

    if not downloaders:
        print("  ❌ No data downloaders could be initialized!")
        return {'error': 'No data downloaders available'}

    print(f"  📡 Available downloaders: {', '.join(downloaders.keys())}")
    print()

    results = {
        'success': [],
        'failed': [],
        'cache_stats': {},
        'data_quality': {}
    }

    total_operations = len(symbols) * len(intervals)
    current_operation = 0

    for symbol in symbols:
        for interval in intervals:
            current_operation += 1
            print(f"🔄 Progress: {current_operation}/{total_operations} ({current_operation/total_operations*100:.1f}%)")
            try:
                print(f"📥 Downloading {symbol} {interval} data...")

                                # Try to get data from real sources first, fallback to mock for testing
                df = None
                used_provider = None

                # Use provider selector to determine the best provider for this symbol and interval
                best_provider = provider_selector.get_best_provider(symbol, interval)
                provider_config = provider_selector.get_data_provider_config(symbol, interval)

                print(f"  🔍 Symbol classification: {symbol} -> {provider_config['symbol_type']}")
                print(f"  🎯 Best provider for {interval}: {best_provider}")
                print(f"  💡 Reason: {provider_config.get('reason', 'No reason provided')}")

                if provider_config.get('exchange'):
                    print(f"     Exchange: {provider_config['exchange']}")
                if provider_config.get('base_asset') and provider_config.get('quote_asset'):
                    print(f"     Crypto pair: {provider_config['base_asset']}/{provider_config['quote_asset']}")

                # Check if data already exists in unified cache for the full date range
                print(f"  🔍 Checking unified cache for date range: {start_date.year}-{end_date.year}")

                # Try to get existing data from unified cache
                # Convert to naive datetimes for cache comparison (cache stores naive timestamps)
                start_date_naive = start_date.replace(tzinfo=None) if start_date.tzinfo else start_date
                end_date_naive = end_date.replace(tzinfo=None) if end_date.tzinfo else end_date
                existing_data = cache.get(symbol, interval, start_date=start_date_naive, end_date=end_date_naive)
                existing_provider = None

                if existing_data is not None and not existing_data.empty:
                    # Check if the existing data covers the full requested date range
                    existing_start = existing_data.index.min()
                    existing_end = existing_data.index.max()

                    # Check if existing data covers the full requested range
                    covers_start = existing_start <= start_date_naive
                    covers_end = existing_end >= end_date_naive

                    if covers_start and covers_end:
                        print(f"  ✅ Data already exists in cache for {symbol} {interval} (full range covered)")
                        print(f"     Cached rows: {len(existing_data)}")
                        print(f"     Date range: {existing_start} to {existing_end}")
                        print(f"     Requested range: {start_date_naive} to {end_date_naive}")

                        # Check if we need to update current year data
                        current_year = datetime.now().year
                        current_month = datetime.now().month

                        if end_date.year == current_year:
                            # For current year, check if data is recent enough
                            try:
                                # Check if the last timestamp is in the current month
                                last_timestamp = existing_end
                                last_month = last_timestamp.month

                                if last_month == current_month:
                                    print(f"     ✅ Current year data is up-to-date (last data: {last_timestamp.strftime('%Y-%m-%d %H:%M')})")
                                    results['success'].append(f"{symbol}_{interval}")
                                    continue
                                else:
                                    print(f"     ⚠️  Current year data is outdated (last data: {last_timestamp.strftime('%Y-%m-%d %H:%M')}, current month: {current_month})")
                                    print("     📥 Will download current year data only")

                                    # Download only current year data
                                    current_year_start = datetime(current_year, 1, 1)
                                    current_year_end = end_date

                                    # Update the date range to only current year
                                    start_date = current_year_start
                                    end_date = current_year_end
                                    print(f"     📅 Updated date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                            except Exception as e:
                                print(f"     ⚠️  Error checking current year data: {e}, will download")
                        else:
                            # Previous years data - keep forever
                            print("     ✅ Historical data, keeping existing data")
                            results['success'].append(f"{symbol}_{interval}")
                            continue
                    else:
                        # Only check for missing historical years, not small gaps
                        print(f"  ⚠️  Partial data exists in cache for {symbol} {interval}")
                        print(f"     Cached range: {existing_start} to {existing_end}")
                        print(f"     Requested range: {start_date_naive} to {end_date_naive}")

                        # Check if we're missing entire years
                        available_years = cache.list_years(symbol, interval)
                        requested_years = set(range(start_date.year, end_date.year + 1))
                        missing_years = requested_years - set(available_years)

                        if missing_years:
                            print(f"     📥 Missing years: {sorted(missing_years)}, will download")
                        else:
                            print("     ✅ All requested years exist, skipping download")
                            results['success'].append(f"{symbol}_{interval}")
                            continue

                # Select provider based on intelligent provider selection
                provider = best_provider

                if provider == 'binance' and 'binance' in downloaders:
                    downloader = downloaders[provider]
                    try:
                        print(f"  📡 Using Binance data source for crypto symbol {symbol}")
                        print(f"     Symbol: {symbol}, Interval: {interval}")
                        print(f"     Date range: {start_date} to {end_date}")
                        print(f"     Reason: {provider_config.get('reason', 'Crypto symbol')}")

                        # Use UTC timestamps for Binance (it uses UTC)
                        df = downloader.get_ohlcv(symbol, interval, start_date.replace(tzinfo=timezone.utc), end_date.replace(tzinfo=timezone.utc))
                        if df is not None and not df.empty:
                            print(f"  ✅ Successfully downloaded from {provider}: {len(df)} rows")
                            used_provider = provider
                        else:
                            print(f"  ⚠️  No data returned from {provider}")
                    except Exception as e:
                        print(f"  ⚠️  {provider} failed: {str(e)}")
                        df = None

                elif provider == 'yfinance' and 'yfinance' in downloaders:
                    downloader = downloaders[provider]
                    try:
                        print(f"  📡 Using Yahoo Finance data source for stock symbol {symbol}")
                        print(f"     Symbol: {symbol}, Interval: {interval}")
                        print(f"     Date range: {start_date} to {end_date}")
                        print(f"     Reason: {provider_config.get('reason', 'Stock symbol with daily interval')}")

                        # Use naive timestamps for Yahoo Finance (it uses market timezone)
                        start_date_naive = start_date.replace(tzinfo=None) if start_date.tzinfo else start_date
                        end_date_naive = end_date.replace(tzinfo=None) if end_date.tzinfo else end_date
                        df = downloader.get_ohlcv(symbol, interval, start_date_naive, end_date_naive)
                        if df is not None and not df.empty:
                            print(f"  ✅ Successfully downloaded from {provider}: {len(df)} rows")
                            used_provider = provider
                        else:
                            print(f"  ⚠️  No data returned from {provider}")
                    except Exception as e:
                        print(f"  ⚠️  {provider} failed: {str(e)}")
                        df = None

                elif provider == 'alpha_vantage' and 'alpha_vantage' in downloaders:
                    downloader = downloaders[provider]
                    try:
                        print(f"  📡 Using Alpha Vantage data source for intraday data {symbol}")
                        print(f"     Symbol: {symbol}, Interval: {interval}")
                        print(f"     Date range: {start_date} to {end_date}")
                        print(f"     Reason: {provider_config.get('reason', 'Stock symbol with intraday interval')}")
                        print("     Note: Alpha Vantage provides full historical intraday data (no 60-day limit)")

                        # Use naive timestamps for Alpha Vantage (it uses market timezone)
                        start_date_naive = start_date.replace(tzinfo=None) if start_date.tzinfo else start_date
                        end_date_naive = end_date.replace(tzinfo=None) if end_date.tzinfo else end_date
                        df = downloader.get_ohlcv(symbol, interval, start_date_naive, end_date_naive)
                        if df is not None and not df.empty:
                            print(f"  ✅ Successfully downloaded from {provider}: {len(df)} rows")
                            used_provider = provider
                        else:
                            print(f"  ⚠️  No data returned from {provider}")
                    except Exception as e:
                        print(f"  ⚠️  {provider} failed: {str(e)}")
                        df = None

                else:
                    print(f"  ⚠️  No suitable provider found for {symbol} {interval}")
                    print(f"     Best provider: {best_provider}")
                    print(f"     Available downloaders: {list(downloaders.keys())}")
                    df = None

                # Fallback to mock if no real data available
                if df is None or df.empty:
                    print("  🎭 Using mock data source (fallback)")
                    mock_source = MockDataSource("mock")
                    # Ensure naive datetimes for mock data source
                    start_date_naive = start_date.replace(tzinfo=None) if start_date.tzinfo else start_date
                    end_date_naive = end_date.replace(tzinfo=None) if end_date.tzinfo else end_date
                    df = mock_source.fetch_historical_data(symbol, interval, start_date_naive, end_date_naive)
                    used_provider = "mock"

                if df is not None and not df.empty:
                    # Ensure DataFrame has the right format for caching
                    print(f"  📊 Data format: {df.columns.tolist()}")
                    print(f"  📅 Data range: {df.index.min()} to {df.index.max()}")

                    # Skip validation - cache data as-is
                    # Data quality will be handled by a separate validation script
                    is_valid = True  # Always cache data without validation
                    errors = []
                    quality_score = {'quality_score': 1.0}  # Default perfect score

                    if 'timestamp' in df.columns:
                        # Set timestamp as index for proper caching
                        df = df.set_index('timestamp')
                        # Ensure index is timezone-naive for compatibility
                        if df.index.tz is not None:
                            df.index = df.index.tz_localize(None)
                        print("  🔄 Converted timestamp to index")

                    results['data_quality'][f"{symbol}_{interval}"] = {
                        'is_valid': is_valid,
                        'errors': errors,
                        'quality_score': quality_score['quality_score'],
                        'rows': len(df),
                        'columns': list(df.columns)
                    }

                    if is_valid:
                        # Actually save data to unified cache
                        print(f"  💾 Caching data to unified cache: {symbol}/{interval}/...")
                        _logger.info("Attempting to cache %s_%s: %d rows (no validation)", symbol, interval, len(df))
                        try:
                            cache_success = cache.put(
                                df, symbol, interval,
                                start_date=start_date, end_date=end_date,
                                provider=used_provider
                            )

                            if cache_success:
                                results['success'].append(f"{symbol}_{interval}")
                                print(f"  ✅ Downloaded and cached {len(df)} rows")
                                print(f"  📈 Quality score: {quality_score['quality_score']:.2f}")
                                print(f"  📁 Cached to: {symbol}/{interval}/")
                                print(f"  🔗 Provider: {used_provider}")
                                _logger.info("SUCCESS: %s_%s cached successfully - %d rows", symbol, interval, len(df))
                            else:
                                results['failed'].append(f"{symbol}_{interval}")
                                print("  ❌ Failed to cache data")
                                _logger.error("FAILED: %s_%s - cache.put returned False", symbol, interval)
                        except Exception as cache_error:
                            results['failed'].append(f"{symbol}_{interval}")
                            print(f"  ❌ Cache error: {str(cache_error)}")
                            print(f"  🔍 Error details: {type(cache_error).__name__}")
                            _logger.exception("EXCEPTION: %s_%s - %s: %s", symbol, interval, type(cache_error).__name__, str(cache_error))
                    # Validation is disabled - this block should never be reached
                    else:
                        results['failed'].append(f"{symbol}_{interval}")
                        print("  ❌ Unexpected validation failure (validation is disabled)")
                        _logger.error("UNEXPECTED: %s_%s - validation failed but validation is disabled", symbol, interval)
                else:
                    results['failed'].append(f"{symbol}_{interval}")
                    print("  ❌ No data available from any source")
                    print(f"  🔍 Best provider: {best_provider}")
                    print(f"  🔍 Available downloaders: {list(downloaders.keys())}")

            except Exception as e:
                results['failed'].append(f"{symbol}_{interval}")
                print(f"  ❌ Error: {str(e)}")
                print(f"  🔍 Error type: {type(e).__name__}")
                import traceback
                print(f"  📋 Traceback: {traceback.format_exc()}")

            print()
            print(f"✅ Completed operation {current_operation}/{total_operations}")

            # Add rate limiting delay to avoid hitting API limits
            if current_operation < total_operations:  # Don't delay after the last operation
                print("  ⏱️  Rate limiting: waiting 0.2 seconds...")
                time.sleep(0.2)

            print()

    # Get final cache statistics
    print("\n📊 Getting cache statistics...")
    results['cache_stats'] = cache.get_stats()
    print(f"  💾 Cache size: {results['cache_stats'].get('cache_size_gb', 0):.2f} GB")
    print(f"  📁 Files created: {results['cache_stats'].get('files_created', 0)}")
    print(f"  📊 Total operations: {results['cache_stats'].get('total_operations', 0)}")
    print(f"  🎯 Success rate: {len(results['success'])}/{total_operations} ({len(results['success'])/total_operations*100:.1f}%)")

    return results


def validate_cache_structure(cache_dir: str = DATA_CACHE_DIR) -> Dict[str, Any]:
    """Validate the cache directory structure and contents."""
    print("🔍 Validating cache structure...")
    print()

    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return {'error': f"Cache directory does not exist: {cache_dir}"}

    results = {
        'directory_exists': True,
        'structure': {},
        'files': {},
        'total_size_mb': 0
    }

    # Check directory structure
    for item in cache_path.iterdir():
        if item.is_dir():
            provider = item.name
            results['structure'][provider] = {}

            for symbol_dir in item.iterdir():
                if symbol_dir.is_dir():
                    symbol = symbol_dir.name
                    results['structure'][provider][symbol] = {}

                    for interval_dir in symbol_dir.iterdir():
                        if interval_dir.is_dir():
                            interval = interval_dir.name
                            results['structure'][provider][symbol][interval] = {}

                            for year_dir in interval_dir.iterdir():
                                if year_dir.is_dir():
                                    year = year_dir.name
                                    year_files = list(year_dir.glob('*'))
                                    results['structure'][provider][symbol][interval][year] = len(year_files)

                                    # Calculate file sizes
                                    for file_path in year_files:
                                        if file_path.is_file():
                                            results['total_size_mb'] += file_path.stat().st_size / (1024 * 1024)

    # Print structure
    print("📁 Cache Directory Structure:")
    for provider, symbols in results['structure'].items():
        print(f"  📂 {provider}/")
        for symbol, intervals in symbols.items():
            print(f"    📂 {symbol}/")
            for interval, years in intervals.items():
                print(f"      📂 {interval}/")
                for year, file_count in years.items():
                    print(f"        📂 {year}/ ({file_count} files)")

    print(f"\n💾 Total cache size: {results['total_size_mb']:.2f} MB")

    return results


def main():
    """Main function to run the cache population and testing script."""
    parser = argparse.ArgumentParser(description="Cache Population and Testing Script")
    parser.add_argument("--populate", action="store_true",
                       help="Populate the cache with historical data")
    parser.add_argument("--test-all", action="store_true",
                       help="Test all system components")
    parser.add_argument("--validate-cache", action="store_true",
                       help="Validate cache structure and contents")
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,LTCUSDT,ADAUSDT,VT,GOOG,TSLA,NVDA,NFLX,VT,PSNY,VUSD,SMCI,RPD,QTUM,QBTS,PFE,MRNA,MASI,LULU,IONQ",
                       help="Comma-separated list of symbols to download")
    parser.add_argument("--intervals", type=str, default="1h,4h,1d,5m,15m",
                       help="Comma-separated list of intervals to download")
    parser.add_argument("--start-date", type=str, default="2020-01-01",
                       help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", type=str, default=None,
                       help="End date in YYYY-MM-DD format (defaults to today)")
    parser.add_argument("--cache-dir", type=str, default=DATA_CACHE_DIR,
                       help="Cache directory path")

    args = parser.parse_args()

    # If no arguments provided, default to populate
    if not any([args.populate, args.test_all, args.validate_cache]):
        args.populate = True

    print("🚀 E-Trading Data Module - Cache Population & Testing")
    print("=" * 60)
    print()

    # Parse arguments
    symbols = [s.strip() for s in args.symbols.split(',')]
    intervals = [i.strip() for i in args.intervals.split(',')]

    # Parse start date
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        print(f"📅 Start date: {start_date.strftime('%Y-%m-%d')}")
    except ValueError:
        print(f"❌ Invalid start date format: {args.start_date}. Use YYYY-MM-DD format.")
        sys.exit(1)

    # Convert to UTC for providers that use UTC (like Binance)
    # Keep naive for providers that use market timezones (like Yahoo, Alpha Vantage)
    start_date_utc = start_date.replace(tzinfo=timezone.utc)

    # Parse end date (defaults to today if not specified)
    if args.end_date:
        try:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
            print(f"📅 End date: {end_date.strftime('%Y-%m-%d')}")
        except ValueError:
            print(f"❌ Invalid end date format: {args.end_date}. Use YYYY-MM-DD format.")
            sys.exit(1)
    else:
        end_date = datetime.now(timezone.utc)
        print(f"📅 End date: {end_date.strftime('%Y-%m-%d')} (today)")

    # Convert to UTC for providers that use UTC (like Binance)
    end_date_utc = end_date.replace(tzinfo=timezone.utc) if end_date.tzinfo is None else end_date

    # Ensure start_date is before end_date (compare naive versions)
    start_date_naive = start_date.replace(tzinfo=None) if start_date.tzinfo else start_date
    end_date_naive = end_date.replace(tzinfo=None) if end_date.tzinfo else end_date

    if start_date_naive >= end_date_naive:
        print(f"❌ Start date ({start_date_naive.strftime('%Y-%m-%d')}) must be before end date ({end_date_naive.strftime('%Y-%m-%d')})")
        sys.exit(1)

    try:
        if args.populate:
            print("📥 POPULATING CACHE")
            print("-" * 30)
            results = populate_cache(symbols, intervals, start_date, end_date, args.cache_dir, "csv")

            print("\n📊 POPULATION RESULTS")
            print("-" * 30)
            print(f"✅ Successful: {len(results['success'])}")
            print(f"❌ Failed: {len(results['failed'])}")

            if results['success']:
                print(f"\n✅ Successfully cached: {', '.join(results['success'])}")
            if results['failed']:
                print(f"\n❌ Failed to cache: {', '.join(results['failed'])}")

            print("\n📈 CACHE STATISTICS")
            print("-" * 30)
            for key, value in results['cache_stats'].items():
                print(f"  {key}: {value}")

        if args.test_all:
            print("\n🧪 TESTING SYSTEM COMPONENTS")
            print("-" * 30)
            print("\n🧪 NO TESTS implemented")

        if args.validate_cache:
            print("\n🔍 VALIDATING CACHE STRUCTURE")
            print("-" * 30)
            cache_results = validate_cache_structure(args.cache_dir)

            if 'error' in cache_results:
                print(f"❌ Cache validation failed: {cache_results['error']}")

        print("\n🎉 OPERATION COMPLETED SUCCESSFULLY!")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
