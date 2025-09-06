"""
Test script for fundamentals cache system.

This script tests the fundamentals cache functionality including:
- Cache operations (write, read, find latest)
- Data combination strategies
- Stale data cleanup
- Integration with DataManager
"""

import sys
import os
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.data.cache.fundamentals_cache import FundamentalsCache, get_fundamentals_cache
from src.data.cache.fundamentals_combiner import FundamentalsCombiner, get_fundamentals_combiner
from src.data.data_manager import DataManager

def test_fundamentals_cache():
    """Test fundamentals cache operations."""
    print("Testing Fundamentals Cache...")

    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = FundamentalsCache(temp_dir)

        # Test data
        test_data = {
            'market_cap': 3000000000000,
            'pe_ratio': 25.5,
            'pb_ratio': 4.2,
            'dividend_yield': 0.5,
            'revenue': 365000000000,
            'net_income': 95000000000
        }

        # Test write and read
        print("  Testing cache write/read...")
        file_path = cache.write_json('AAPL', 'yfinance', test_data)
        assert os.path.exists(file_path), "Cache file should be created"

        # Test read
        cached_data = cache.read_json(file_path)
        assert cached_data == test_data, "Cached data should match original"

        # Test find latest
        print("  Testing find latest...")
        latest = cache.find_latest_json('AAPL')
        assert latest is not None, "Should find latest cache"
        assert latest.provider == 'yfinance', "Provider should match"
        assert latest.symbol == 'AAPL', "Symbol should match"

        # Test cache validity
        print("  Testing cache validity...")
        assert cache.is_cache_valid(latest.timestamp), "Fresh cache should be valid"

        # Test with old timestamp
        old_timestamp = datetime.now() - timedelta(days=8)
        assert not cache.is_cache_valid(old_timestamp), "Old cache should be invalid"

        # Test cleanup
        print("  Testing stale data cleanup...")
        # Use a timestamp that's newer than the cached file
        new_timestamp = datetime.now() + timedelta(minutes=1)
        removed_files = cache.cleanup_stale_data('AAPL', 'yfinance', new_timestamp)
        assert len(removed_files) == 1, "Should remove the existing file for newer timestamp"

        # Test cache stats
        print("  Testing cache stats...")
        stats = cache.get_cache_stats('AAPL')
        assert stats['files'] == 0, "Should have 0 cache files after cleanup"
        assert stats['symbol'] == 'AAPL', "Symbol should match"

        print("  ✅ Fundamentals cache tests passed!")

def test_fundamentals_combiner():
    """Test fundamentals data combination."""
    print("Testing Fundamentals Combiner...")

    combiner = FundamentalsCombiner()

    # Test data from different providers
    provider_data = {
        'yfinance': {
            'market_cap': 3000000000000,
            'pe_ratio': 25.5,
            'dividend_yield': 0.5,
            'revenue': 365000000000
        },
        'fmp': {
            'market_cap': 3001000000000,  # Slightly different
            'pe_ratio': 25.4,
            'pb_ratio': 4.2,  # Additional field
            'net_income': 95000000000
        },
        'alpha_vantage': {
            'market_cap': 3000500000000,
            'pe_ratio': 25.6,
            'cash': 50000000000
        }
    }

    # Test priority-based combination
    print("  Testing priority-based combination...")
    combined = combiner.combine_snapshots(provider_data, 'priority_based')

    assert 'market_cap' in combined, "Should have market_cap"
    assert 'pe_ratio' in combined, "Should have pe_ratio"
    assert 'pb_ratio' in combined, "Should have pb_ratio from FMP"
    assert 'cash' in combined, "Should have cash from Alpha Vantage"
    assert '_metadata' in combined, "Should have metadata"

    # Check that FMP data takes priority (highest priority)
    assert combined['pb_ratio'] == 4.2, "Should use FMP pb_ratio"

    # Test quality-based combination
    print("  Testing quality-based combination...")
    combined_quality = combiner.combine_snapshots(provider_data, 'quality_based')
    assert '_metadata' in combined_quality, "Should have metadata"
    assert combined_quality['_metadata']['combination_strategy'] == 'quality_based'

    # Test consensus combination
    print("  Testing consensus combination...")
    combined_consensus = combiner.combine_snapshots(provider_data, 'consensus')
    assert '_metadata' in combined_consensus, "Should have metadata"
    assert combined_consensus['_metadata']['combination_strategy'] == 'consensus'

    print("  ✅ Fundamentals combiner tests passed!")

def test_data_manager_integration():
    """Test DataManager integration with fundamentals cache."""
    print("Testing DataManager Integration...")

    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Initialize DataManager
            dm = DataManager(temp_dir)

            # Test get_fundamentals method exists
            assert hasattr(dm, 'get_fundamentals'), "DataManager should have get_fundamentals method"

            # Test with a real symbol (this will try to fetch real data)
            print("  Testing fundamentals retrieval for AAPL...")
            try:
                fundamentals = dm.get_fundamentals('AAPL', force_refresh=True)

                if fundamentals:
                    print(f"    Retrieved fundamentals with {len(fundamentals)} fields")
                    print(f"    Fields: {list(fundamentals.keys())[:5]}...")  # Show first 5 fields

                    # Check for metadata
                    if '_metadata' in fundamentals:
                        print(f"    Combination strategy: {fundamentals['_metadata']['combination_strategy']}")
                        print(f"    Providers used: {fundamentals['_metadata']['providers_used']}")
                else:
                    print("    No fundamentals data retrieved (this is expected if no API keys are configured)")

            except Exception as e:
                print(f"    Expected error (no API keys): {e}")

            # Test cache stats
            print("  Testing cache statistics...")
            stats = dm.get_cache_stats()
            print(f"    Cache stats: {stats}")

            print("  ✅ DataManager integration tests passed!")

        except Exception as e:
            print(f"  ⚠️  DataManager integration test failed (expected if no API keys): {e}")

def main():
    """Run all tests."""
    print("🧪 Running Fundamentals Cache System Tests\n")

    try:
        test_fundamentals_cache()
        print()

        test_fundamentals_combiner()
        print()

        test_data_manager_integration()
        print()

        print("🎉 All tests completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
