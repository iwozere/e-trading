"""
Demonstration of Intelligent Cached Data Downloaders

This script demonstrates how to use the new cached data downloaders that provide:
1. Automatic cache checking
2. Smart gap detection
3. Minimal server requests
4. Seamless data access

Usage Examples:
- First-time data download (no cache)
- Partial cache hits with gap filling
- Complete cache hits (no server requests)
- Mixed scenarios with multiple symbols
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.cached_downloader_factory import (
    create_cached_binance_downloader,
    create_cached_yahoo_downloader,
    get_cached_downloader_factory
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def demo_binance_caching():
    """Demonstrate Binance data caching with different scenarios."""
    print("\n" + "="*60)
    print("BINANCE DATA CACHING DEMONSTRATION")
    print("="*60)

    # Create cached Binance downloader
    binance_downloader = create_cached_binance_downloader()

    # Scenario 1: First-time download (no cache)
    print("\n📥 SCENARIO 1: First-time download (no cache)")
    print("-" * 50)

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 31)

    print(f"Requesting BTCUSDT 1d data from {start_date.date()} to {end_date.date()}")
    print("Expected: Download from server, cache data")

    data1 = binance_downloader.get_ohlcv("BTCUSDT", "1d", start_date, end_date)
    print(f"✅ Downloaded: {len(data1)} rows")
    print(f"📊 Date range: {data1.index.min().date()} to {data1.index.max().date()}")

    # Scenario 2: Same request (cache hit)
    print("\n📥 SCENARIO 2: Same request (cache hit)")
    print("-" * 50)

    print("Requesting same data again...")
    print("Expected: Return from cache, no server request")

    data2 = binance_downloader.get_ohlcv("BTCUSDT", "1d", start_date, end_date)
    print(f"✅ Retrieved from cache: {len(data2)} rows")
    print(f"📊 Date range: {data2.index.min().date()} to {data2.index.max().date()}")

    # Scenario 3: Extended range (partial cache hit)
    print("\n📥 SCENARIO 3: Extended range (partial cache hit)")
    print("-" * 50)

    extended_end = datetime(2024, 2, 15)
    print(f"Requesting BTCUSDT 1d data from {start_date.date()} to {extended_end.date()}")
    print("Expected: Use cached data for Jan, download only Feb 1-15")

    data3 = binance_downloader.get_ohlcv("BTCUSDT", "1d", start_date, extended_end)
    print(f"✅ Extended data: {len(data3)} rows")
    print(f"📊 Date range: {data3.index.min().date()} to {data3.index.max().date()}")

    # Scenario 4: Different interval (new cache entry)
    print("\n📥 SCENARIO 4: Different interval (new cache entry)")
    print("-" * 50)

    print("Requesting BTCUSDT 4h data for same period...")
    print("Expected: Download from server, create new cache entry")

    data4 = binance_downloader.get_ohlcv("BTCUSDT", "4h", start_date, end_date)
    print(f"✅ Downloaded 4h data: {len(data4)} rows")
    print(f"📊 Date range: {data4.index.min().date()} to {data4.index.max().date()}")

    return binance_downloader


def demo_yahoo_caching():
    """Demonstrate Yahoo Finance data caching."""
    print("\n" + "="*60)
    print("YAHOO FINANCE DATA CACHING DEMONSTRATION")
    print("="*60)

    # Create cached Yahoo downloader
    yahoo_downloader = create_cached_yahoo_downloader()

    # Scenario 1: Stock data download
    print("\n📥 SCENARIO 1: Stock data download")
    print("-" * 50)

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 31)

    print(f"Requesting AAPL 1d data from {start_date.date()} to {end_date.date()}")

    data1 = yahoo_downloader.get_ohlcv("AAPL", "1d", start_date, end_date)
    print(f"✅ Downloaded: {len(data1)} rows")
    print(f"📊 Date range: {data1.index.min().date()} to {data1.index.max().date()}")

    # Scenario 2: Different symbol (new cache entry)
    print("\n📥 SCENARIO 2: Different symbol (new cache entry)")
    print("-" * 50)

    print("Requesting TSLA 1d data for same period...")

    data2 = yahoo_downloader.get_ohlcv("TSLA", "1d", start_date, end_date)
    print(f"✅ Downloaded TSLA data: {len(data2)} rows")
    print(f"📊 Date range: {data2.index.min().date()} to {data2.index.max().date()}")

    # Scenario 3: Cache hit for existing data
    print("\n📥 SCENARIO 3: Cache hit for existing data")
    print("-" * 50)

    print("Requesting AAPL data again...")
    print("Expected: Return from cache, no server request")

    data3 = yahoo_downloader.get_ohlcv("AAPL", "1d", start_date, end_date)
    print(f"✅ Retrieved from cache: {len(data3)} rows")

    return yahoo_downloader


def demo_mixed_scenarios():
    """Demonstrate mixed scenarios with multiple providers."""
    print("\n" + "="*60)
    print("MIXED SCENARIOS DEMONSTRATION")
    print("="*60)

    # Get the factory to manage multiple downloaders
    factory = get_cached_downloader_factory()

    # Create multiple cached downloaders
    binance = factory.create_binance_downloader()
    yahoo = factory.create_yahoo_downloader()

    # Scenario: Mixed data requests
    print("\n📥 SCENARIO: Mixed data requests")
    print("-" * 50)

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 31)

    symbols_and_intervals = [
        ("BTCUSDT", "1d", binance),
        ("ETHUSDT", "1d", binance),
        ("AAPL", "1d", yahoo),
        ("GOOGL", "1d", yahoo),
    ]

    for symbol, interval, downloader in symbols_and_intervals:
        print(f"\nRequesting {symbol} {interval} data...")

        try:
            data = downloader.get_ohlcv(symbol, interval, start_date, end_date)
            print(f"✅ {symbol} {interval}: {len(data)} rows")
            print(f"   Date range: {data.index.min().date()} to {data.index.max().date()}")
        except Exception as e:
            print(f"❌ {symbol} {interval}: Error - {e}")

    return factory


def demo_cache_statistics():
    """Demonstrate cache statistics and management."""
    print("\n" + "="*60)
    print("CACHE STATISTICS AND MANAGEMENT")
    print("="*60)

    factory = get_cached_downloader_factory()

    # Get cache statistics
    stats = factory.get_cache_stats()

    print("\n📊 CACHE STATISTICS:")
    print(f"   Cache directory: {stats['cache_dir']}")
    print(f"   Total files: {stats['file_count']}")
    print(f"   Total directories: {stats['directory_count']}")
    print(f"   Cache size: {stats['cache_size_gb']:.4f} GB")
    print(f"   Max size: {stats['max_size_gb']} GB")
    print(f"   Hits: {stats['hits']}")
    print(f"   Misses: {stats['misses']}")
    print(f"   Hit rate: {stats['hit_rate']:.2%}")

    # Get all cached downloaders
    downloaders = factory.get_all_cached_downloaders()
    print(f"\n🔧 ACTIVE CACHED DOWNLOADERS:")
    for name, downloader in downloaders.items():
        print(f"   {name}: {downloader.__class__.__name__}")

    return factory


def main():
    """Run all demonstrations."""
    print("🚀 INTELLIGENT CACHED DATA DOWNLOADERS DEMONSTRATION")
    print("=" * 80)
    print("This demo shows how the new caching system works with different scenarios.")
    print("=" * 80)

    try:
        # Run all demonstrations
        binance_downloader = demo_binance_caching()
        yahoo_downloader = demo_yahoo_caching()
        factory = demo_mixed_scenarios()
        demo_cache_statistics()

        print("\n" + "="*80)
        print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nKey Benefits Demonstrated:")
        print("✅ Automatic cache checking before server requests")
        print("✅ Smart gap detection and partial downloads")
        print("✅ Seamless data merging and caching")
        print("✅ Minimal server requests and API usage")
        print("✅ Professional error handling and logging")
        print("✅ Support for all data downloader types")

        print("\nNext Steps:")
        print("1. Use create_cached_*_downloader() functions for your projects")
        print("2. The cache automatically handles data gaps and updates")
        print("3. Monitor cache statistics with factory.get_cache_stats()")
        print("4. Clear specific cache entries with factory.clear_cache()")

    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        _logger.exception("Demonstration failed")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
