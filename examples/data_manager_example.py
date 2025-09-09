#!/usr/bin/env python3
"""
DataManager Usage Example
========================

This example demonstrates how to use the new unified DataManager
for retrieving financial data with automatic provider selection,
caching, and failover support.

The DataManager provides a single, simple interface for all data operations,
abstracting away the complexity of provider selection, rate limiting,
caching, and error handling.
"""

import os
import sys
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import DataManager, get_data_manager

# Import cache directory setting
from config.donotshare.donotshare import DATA_CACHE_DIR


def main():
    """Demonstrate DataManager usage."""

    print("🚀 DataManager Example")
    print("=" * 50)

    # Initialize DataManager
    print("\n1. Initializing DataManager...")
    dm = get_data_manager(
        cache_dir=DATA_CACHE_DIR,
        config_path="config/data/provider_rules.yaml"
    )
    print("✅ DataManager initialized successfully")

    # Example 1: Get cryptocurrency data (will use Binance)
    print("\n2. Getting cryptocurrency data (BTCUSDT)...")
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        btc_data = dm.get_ohlcv(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=start_date,
            end_date=end_date
        )

        print(f"✅ Retrieved {len(btc_data)} rows of BTCUSDT data")
        print(f"   Date range: {btc_data.index[0]} to {btc_data.index[-1]}")
        print(f"   Columns: {list(btc_data.columns)}")
        print(f"   Sample data:")
        print(btc_data.head(3))

    except Exception as e:
        print(f"❌ Error getting BTCUSDT data: {e}")

    # Example 2: Get stock data (will use Yahoo Finance)
    print("\n3. Getting stock data (AAPL)...")
    try:
        aapl_data = dm.get_ohlcv(
            symbol="AAPL",
            timeframe="1d",
            start_date=start_date,
            end_date=end_date
        )

        print(f"✅ Retrieved {len(aapl_data)} rows of AAPL data")
        print(f"   Date range: {aapl_data.index[0]} to {aapl_data.index[-1]}")
        print(f"   Sample data:")
        print(aapl_data.head(3))

    except Exception as e:
        print(f"❌ Error getting AAPL data: {e}")

    # Example 3: Demonstrate caching (second request should be faster)
    print("\n4. Demonstrating caching (second request)...")
    try:
        import time

        start_time = time.time()
        btc_data_cached = dm.get_ohlcv(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=start_date,
            end_date=end_date
        )
        cache_time = time.time() - start_time

        print(f"✅ Cached request completed in {cache_time:.3f} seconds")
        print(f"   Retrieved {len(btc_data_cached)} rows (should be same as before)")

    except Exception as e:
        print(f"❌ Error with cached request: {e}")

    # Example 4: Get live feed
    print("\n5. Creating live data feed...")
    try:
        live_feed = dm.get_live_feed(
            symbol="BTCUSDT",
            timeframe="1m",
            lookback_bars=100
        )

        if live_feed:
            print(f"✅ Created live feed for BTCUSDT 1m")
            print(f"   Feed type: {type(live_feed).__name__}")
            print(f"   Lookback bars: 100")
        else:
            print("❌ Failed to create live feed")

    except Exception as e:
        print(f"❌ Error creating live feed: {e}")

    # Example 5: Show cache statistics
    print("\n6. Cache statistics...")
    try:
        cache_stats = dm.get_cache_stats()
        print(f"✅ Cache statistics:")
        for key, value in cache_stats.items():
            print(f"   {key}: {value}")

    except Exception as e:
        print(f"❌ Error getting cache stats: {e}")

    # Example 6: Demonstrate provider selection
    print("\n7. Provider selection examples...")
    try:
        # Test different symbol types
        test_symbols = [
            ("BTCUSDT", "crypto"),
            ("AAPL", "stock"),
            ("ETHUSDT", "crypto"),
            ("MSFT", "stock")
        ]

        for symbol, expected_type in test_symbols:
            provider = dm.provider_selector.get_best_provider(symbol, "1d")
            print(f"   {symbol} ({expected_type}): {provider}")

    except Exception as e:
        print(f"❌ Error testing provider selection: {e}")

    print("\n🎉 DataManager example completed!")
    print("\nKey benefits demonstrated:")
    print("✅ Single interface for all data operations")
    print("✅ Automatic provider selection based on symbol type")
    print("✅ Transparent caching with performance benefits")
    print("✅ Live feed integration")
    print("✅ Comprehensive error handling and logging")


def demonstrate_failover():
    """Demonstrate provider failover functionality."""

    print("\n🔄 Provider Failover Example")
    print("=" * 50)

    dm = get_data_manager()

    # Test failover for a symbol that might have issues
    print("Testing provider failover for problematic symbol...")

    try:
        # This might fail with primary provider, demonstrating failover
        data = dm.get_ohlcv(
            symbol="INVALID_SYMBOL",
            timeframe="1d",
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now()
        )
        print("✅ Data retrieved successfully")

    except Exception as e:
        print(f"❌ All providers failed (expected for invalid symbol): {e}")


if __name__ == "__main__":
    main()

    # Uncomment to test failover
    # demonstrate_failover()
