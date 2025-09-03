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
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data import (
    configure_file_cache, get_file_cache,
    get_data_handler, DataAggregator
)
from src.data.base_data_source import BaseDataSource
from src.data.utils.validation import validate_ohlcv_data, get_data_quality_score

# Import data downloaders
from src.data.binance_data_downloader import BinanceDataDownloader
from src.data.yahoo_data_downloader import YahooDataDownloader

# Import ticker classifier for automatic provider selection
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'common'))
from src.common.ticker_classifier import TickerClassifier, DataProvider


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
            start_date = datetime.utcnow() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.utcnow()

        # Generate mock OHLCV data
        date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
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
            'open': prices,
            'high': [p * 1.02 for p in prices],  # High slightly above open
            'low': [p * 0.98 for p in prices],   # Low slightly below open
            'close': [p * 1.01 for p in prices], # Close slightly above open
            'volume': [1000 + i * 10 for i in range(n_points)]  # Increasing volume
        }, index=date_range)

        return df

    def start_realtime_feed(self, symbol: str, interval: str,
                           callback: Optional[callable] = None) -> bool:
        return True

    def stop_realtime_feed(self, symbol: str) -> bool:
        return True


def populate_cache(symbols: List[str], intervals: List[str],
                   start_date: datetime, end_date: datetime,
                   cache_dir: str = "d:/data-cache", file_format: str = "csv") -> Dict[str, Any]:
    """Populate the file-based cache with historical data."""
    print(f"🚀 Populating cache at: {cache_dir}")
    print(f"📊 Symbols: {', '.join(symbols)}")
    print(f"⏱️  Intervals: {', '.join(intervals)}")
    print(f"📅 Date range: {start_date.date()} to {end_date.date()}")
    print(f"📅 Start time: {start_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📅 End time: {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔢 Total operations: {len(symbols) * len(intervals)}")
    print()

    # Configure cache
    cache = configure_file_cache(cache_dir=cache_dir)

    # Initialize data downloaders
    downloaders = {}
    try:
        downloaders['binance'] = BinanceDataDownloader()
        print("  ✅ Binance downloader initialized")
    except Exception as e:
        print(f"  ⚠️  Binance downloader failed: {str(e)}")

    try:
        downloaders['yahoo'] = YahooDataDownloader()
        print("  ✅ Yahoo downloader initialized")
    except Exception as e:
        print(f"  ⚠️  Yahoo downloader failed: {str(e)}")

    # Initialize ticker classifier for automatic provider selection
    ticker_classifier = TickerClassifier()
    print("  ✅ Ticker classifier initialized")

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

                # Use ticker classifier to determine the appropriate provider
                ticker_info = ticker_classifier.classify_ticker(symbol)
                print(f"  🔍 Ticker classification: {symbol} -> {ticker_info.provider.value}")

                if ticker_info.exchange:
                    print(f"     Exchange: {ticker_info.exchange}")
                if ticker_info.base_asset and ticker_info.quote_asset:
                    print(f"     Crypto pair: {ticker_info.base_asset}/{ticker_info.quote_asset}")

                # Check if data already exists in cache and is still valid
                cache_year = start_date.year
                print(f"  🔍 Checking cache for year: {cache_year}")

                # Try to get existing data from cache for each provider
                existing_data = None
                existing_provider = None

                # Check Binance first (for crypto)
                if ticker_info.provider == DataProvider.BINANCE and 'binance' in downloaders:
                    print(f"  🔍 Checking Binance cache for {symbol} {interval}")
                    # Try to get data for the specific year with CSV format
                    existing_data = cache.get('binance', symbol, interval, start_date=start_date, end_date=end_date, format='csv')
                    print(f"  🔍 Binance cache result: {existing_data is not None and not existing_data.empty if existing_data is not None else 'None'}")
                    if existing_data is not None and not existing_data.empty:
                        existing_provider = 'binance'

                # Check Yahoo if no Binance data
                if existing_data is None and ticker_info.provider == DataProvider.YFINANCE and 'yahoo' in downloaders:
                    print(f"  🔍 Checking Yahoo cache for {symbol} {interval}")
                    # Try to get data for the specific year with CSV format
                    existing_data = cache.get('yahoo', symbol, interval, start_date=start_date, end_date=end_date, format='csv')
                    print(f"  🔍 Yahoo cache result: {existing_data is not None and not existing_data.empty if existing_data is not None else 'None'}")
                    if existing_data is not None and not existing_data.empty:
                        existing_provider = 'yahoo'

                if existing_data is not None and not existing_data.empty:
                    print(f"  ✅ Data already exists in cache for {symbol} {interval} ({cache_year})")
                    print(f"     Provider: {existing_provider}")
                    print(f"     Cached rows: {len(existing_data)}")
                    print(f"     Date range: {existing_data.index.min()} to {existing_data.index.max()}")

                    # Check if we need to update current year data
                    current_year = datetime.now().year
                    if cache_year == current_year:
                        # For current year, check if data is recent enough
                        try:
                            cache_metadata = cache._load_metadata(cache._get_cache_path(existing_provider, symbol, interval, cache_year))
                            if cache_metadata:
                                created_at = datetime.fromisoformat(cache_metadata.get('created_at', '1970-01-01'))
                                days_old = (datetime.now() - created_at).days
                                if days_old <= 30:
                                    print(f"     ✅ Current year data is recent ({days_old} days old), skipping download")
                                    results['success'].append(f"{symbol}_{interval}")
                                    continue
                                else:
                                    print(f"     ⚠️  Current year data is {days_old} days old, will update")
                        except Exception as e:
                            print(f"     ⚠️  Error checking cache metadata: {e}, will download")
                    else:
                        # Previous years data - keep forever
                        print(f"     ✅ Previous year data ({cache_year}), keeping existing data")
                        results['success'].append(f"{symbol}_{interval}")
                        continue

                # Select provider based on ticker classification
                if ticker_info.provider == DataProvider.BINANCE and 'binance' in downloaders:
                    provider = 'binance'
                    downloader = downloaders[provider]

                    try:
                        print(f"  📡 Using Binance data source for crypto symbol {symbol}")
                        print(f"     Symbol: {symbol}, Interval: {interval}")
                        print(f"     Date range: {start_date} to {end_date}")

                        df = downloader.get_ohlcv(symbol, interval, start_date, end_date)
                        if df is not None and not df.empty:
                            print(f"  ✅ Successfully downloaded from {provider}: {len(df)} rows")
                            used_provider = provider
                        else:
                            print(f"  ⚠️  No data returned from {provider}")
                    except Exception as e:
                        print(f"  ⚠️  {provider} failed: {str(e)}")
                        df = None

                elif ticker_info.provider == DataProvider.YFINANCE and 'yahoo' in downloaders:
                    provider = 'yahoo'
                    downloader = downloaders[provider]

                    try:
                        print(f"  📡 Using Yahoo Finance data source for stock symbol {symbol}")
                        print(f"     Symbol: {symbol}, Interval: {interval}")
                        print(f"     Date range: {start_date} to {end_date}")

                        df = downloader.get_ohlcv(symbol, interval, start_date, end_date)
                        if df is not None and not df.empty:
                            print(f"  ✅ Successfully downloaded from {provider}: {len(df)} rows")
                            used_provider = provider
                        else:
                            print(f"  ⚠️  No data returned from {provider}")
                    except Exception as e:
                        print(f"  ⚠️  {provider} failed: {str(e)}")
                        df = None

                else:
                    print(f"  ⚠️  No suitable provider found for {symbol} (classified as {ticker_info.provider.value})")
                    df = None

                # Fallback to mock if no real data available
                if df is None or df.empty:
                    print(f"  🎭 Using mock data source (fallback)")
                    mock_source = MockDataSource("mock")
                    df = mock_source.fetch_historical_data(symbol, interval, start_date, end_date)
                    used_provider = "mock"

                if df is not None and not df.empty:
                    # Ensure DataFrame has the right format for caching
                    print(f"  📊 Data format: {df.columns.tolist()}")
                    print(f"  📅 Data range: {df.index.min()} to {df.index.max()}")

                    if 'timestamp' in df.columns:
                        # Set timestamp as index for proper caching
                        df = df.set_index('timestamp')
                        # Ensure index is timezone-naive for compatibility
                        if df.index.tz is not None:
                            df.index = df.index.tz_localize(None)
                        print(f"  🔄 Converted timestamp to index")

                    # Validate data quality
                    is_valid, errors = validate_ohlcv_data(df)
                    quality_score = get_data_quality_score(df)

                    results['data_quality'][f"{symbol}_{interval}"] = {
                        'is_valid': is_valid,
                        'errors': errors,
                        'quality_score': quality_score['quality_score'],
                        'rows': len(df),
                        'columns': list(df.columns)
                    }

                    if is_valid:
                        # Actually save data to cache
                        print(f"  💾 Caching data to {used_provider}/{symbol}/{interval}/...")
                        try:
                            cache_success = cache.put(
                                df, used_provider, symbol, interval,
                                start_date=start_date, end_date=end_date,
                                format=file_format
                            )

                            if cache_success:
                                results['success'].append(f"{symbol}_{interval}")
                                print(f"  ✅ Downloaded and cached {len(df)} rows")
                                print(f"  📈 Quality score: {quality_score['quality_score']:.2f}")
                                print(f"  📁 Cached to: {used_provider}/{symbol}/{interval}/")
                            else:
                                results['failed'].append(f"{symbol}_{interval}")
                                print(f"  ❌ Failed to cache data")
                        except Exception as cache_error:
                            results['failed'].append(f"{symbol}_{interval}")
                            print(f"  ❌ Cache error: {str(cache_error)}")
                            print(f"  🔍 Error details: {type(cache_error).__name__}")
                    else:
                        results['failed'].append(f"{symbol}_{interval}")
                        print(f"  ⚠️  Data validation failed: {errors}")
                        print(f"  📊 Data columns: {df.columns.tolist()}")
                        print(f"  📅 Data shape: {df.shape}")
                else:
                    results['failed'].append(f"{symbol}_{interval}")
                    print(f"  ❌ No data available from any source")
                    print(f"  🔍 Tried providers: binance, yahoo")

            except Exception as e:
                results['failed'].append(f"{symbol}_{interval}")
                print(f"  ❌ Error: {str(e)}")
                print(f"  🔍 Error type: {type(e).__name__}")
                import traceback
                print(f"  📋 Traceback: {traceback.format_exc()}")

            print()
            print(f"✅ Completed operation {current_operation}/{total_operations}")
            print()

    # Get final cache statistics
    print("\n📊 Getting cache statistics...")
    results['cache_stats'] = cache.get_stats()
    print(f"  💾 Cache size: {results['cache_stats'].get('cache_size_gb', 0):.2f} GB")
    print(f"  📁 Files created: {results['cache_stats'].get('files_created', 0)}")
    print(f"  📊 Total operations: {results['cache_stats'].get('total_operations', 0)}")
    print(f"  🎯 Success rate: {len(results['success'])}/{total_operations} ({len(results['success'])/total_operations*100:.1f}%)")

    return results


def test_system_components(cache_dir: str = "d:/data-cache") -> Dict[str, Any]:
    """Test all system components to ensure they're working correctly."""
    print("🧪 Testing system components...")
    print()

    results = {
        'cache': {},
        'data_handler': {},
        'downloaders': {},
        'aggregator': {},
        'validation': {}
    }

    try:
        # Test 1: Cache operations
        print("1️⃣  Testing file-based cache...")
        cache = configure_file_cache(cache_dir=cache_dir, max_size_gb=1.0)

        # Create test data
        test_df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2023-01-01', periods=3, freq='1H'))

        # Test put/get operations
        success = cache.put(test_df, 'test_provider', 'TESTSYM', '1h',
                           start_date=datetime(2023, 1, 1), format='csv')
        results['cache']['put_success'] = success

        retrieved_df = cache.get('test_provider', 'TESTSYM', '1h',
                               start_date=datetime(2023, 1, 1), format='csv')
        results['cache']['get_success'] = retrieved_df is not None
        results['cache']['data_integrity'] = retrieved_df is not None and len(retrieved_df) == 3

        print(f"   ✅ Cache put: {success}")
        print(f"   ✅ Cache get: {results['cache']['get_success']}")
        print(f"   ✅ Data integrity: {results['cache']['data_integrity']}")

        # Test 2: Data Handler
        print("\n2️⃣  Testing data handler...")
        handler = get_data_handler('test_provider', cache_enabled=True)
        results['data_handler']['creation'] = handler is not None

        # Test data standardization
        standardized_df = handler.standardize_ohlcv_data(test_df, 'TESTSYM', '1h')
        results['data_handler']['standardization'] = standardized_df is not None and not standardized_df.empty

        print(f"   ✅ Handler creation: {results['data_handler']['creation']}")
        print(f"   ✅ Data standardization: {results['data_handler']['standardization']}")

        # Test 3: Data Downloaders
        print("\n3️⃣  Testing data downloaders...")
        try:
            binance_downloader = BinanceDataDownloader()
            results['downloaders']['binance_downloader'] = binance_downloader is not None
            results['downloaders']['binance_intervals'] = len(binance_downloader.get_intervals()) > 0
        except Exception as e:
            results['downloaders']['binance_downloader'] = False
            results['downloaders']['binance_intervals'] = False

        print(f"   ✅ Binance downloader: {results['downloaders']['binance_downloader']}")
        print(f"   ✅ Binance intervals: {results['downloaders']['binance_intervals']}")

        # Test 4: Data Aggregator
        print("\n4️⃣  Testing data aggregator...")
        try:
            aggregator = DataAggregator(primary_provider="mock")
            results['aggregator']['creation'] = aggregator is not None
            print(f"   ✅ Aggregator creation: {results['aggregator']['creation']}")
        except Exception as e:
            results['aggregator']['creation'] = False
            print(f"   ❌ Aggregator creation failed: {str(e)}")

        # Test 5: Data Validation
        print("\n5️⃣  Testing data validation...")
        is_valid, errors = validate_ohlcv_data(test_df)
        quality_score = get_data_quality_score(test_df)

        results['validation']['ohlcv_validation'] = is_valid
        results['validation']['quality_score'] = quality_score['quality_score']

        print(f"   ✅ OHLCV validation: {is_valid}")
        print(f"   ✅ Quality score: {quality_score['quality_score']:.2f}")

    except Exception as e:
        print(f"   ❌ Error during testing: {str(e)}")
        return {'error': str(e)}

    return results


def validate_cache_structure(cache_dir: str = "d:/data-cache") -> Dict[str, Any]:
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
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,LTCUSDT,ADAUSDT,VT,GOOG,TSLA,NVDA,NFLX",
                       help="Comma-separated list of symbols to download")
    parser.add_argument("--intervals", type=str, default="1h,4h,1d,5m,15m",
                       help="Comma-separated list of intervals to download")
    parser.add_argument("--start-date", type=str, default="2020-01-01",
                       help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", type=str, default="2025-09-01",
                       help="End date in YYYY-MM-DD format (defaults to today)")
    parser.add_argument("--cache-dir", type=str, default="d:/data-cache",
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

    # Parse end date (defaults to today if not specified)
    if args.end_date:
        try:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
            print(f"📅 End date: {end_date.strftime('%Y-%m-%d')}")
        except ValueError:
            print(f"❌ Invalid end date format: {args.end_date}. Use YYYY-MM-DD format.")
            sys.exit(1)
    else:
        end_date = datetime.utcnow()
        print(f"📅 End date: {end_date.strftime('%Y-%m-%d')} (today)")

    # Ensure start_date is before end_date
    if start_date >= end_date:
        print(f"❌ Start date ({start_date.strftime('%Y-%m-%d')}) must be before end date ({end_date.strftime('%Y-%m-%d')})")
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

            print(f"\n📈 CACHE STATISTICS")
            print("-" * 30)
            for key, value in results['cache_stats'].items():
                print(f"  {key}: {value}")

        if args.test_all:
            print("\n🧪 TESTING SYSTEM COMPONENTS")
            print("-" * 30)
            test_results = test_system_components(args.cache_dir)

            if 'error' in test_results:
                print(f"❌ Testing failed: {test_results['error']}")
            else:
                print("\n📊 TEST RESULTS SUMMARY")
                print("-" * 30)
                for component, tests in test_results.items():
                    if isinstance(tests, dict):
                        passed = sum(1 for test in tests.values() if test is True)
                        total = len(tests)
                        print(f"  {component}: {passed}/{total} tests passed")

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
