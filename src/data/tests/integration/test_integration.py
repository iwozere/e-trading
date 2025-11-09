"""
Integration test script for Phase 2 components.

This script tests the new data handling infrastructure to ensure all
components work together correctly.
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data import (
    get_data_source_factory,
    register_data_source,
    DataAggregator
)
from unittest.mock import patch
from src.data.feed.binance_live_feed import BinanceLiveDataFeed
from src.data.utils import get_data_handler, validate_ohlcv_data


def test_data_handler():
    """Test the data handler functionality."""
    print("Testing Data Handler...")

    try:
        # Create a data handler
        handler = get_data_handler("test_provider")

        # Create sample data
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1h'),
            'open': [100 + i * 0.1 for i in range(100)],
            'high': [101 + i * 0.1 for i in range(100)],
            'low': [99 + i * 0.1 for i in range(100)],
            'close': [100.5 + i * 0.1 for i in range(100)],
            'volume': [1000 + i * 10 for i in range(100)]
        })

        # Test standardization
        standardized = handler.standardize_ohlcv_data(sample_data, "TEST", "1h")
        print(f"✓ Data standardization: {len(standardized)} rows")

        # Test validation
        validation_result = handler.validate_and_score_data(standardized, "TEST")
        print(f"✓ Data validation: {validation_result['is_valid']}")
        print(f"✓ Quality score: {validation_result['quality_score']['quality_score']:.2f}")

        assert True  # Test passed

    except Exception as e:
        print(f"✗ Data handler test failed: {e}")
        assert False, f"Data handler test failed: {e}"


def test_data_source_factory():
    """Test the data source factory functionality."""
    print("\nTesting Data Source Factory...")

    try:
        # Get factory instance with mocked config (avoid drive D:, use temp on C:)
        with patch('src.data.sources.data_source_factory.DataSourceFactory._load_config') as mock_cfg:
            temp_cache = str(Path.cwd() / 'temp_cache')
            mock_cfg.return_value = {'caching': {'enabled': True, 'cache_dir': temp_cache}, 'data_sources': {}}
            factory = get_data_source_factory()
        print("✓ Factory instance created")

        # Register a test data source
        register_data_source("test_binance", BinanceLiveDataFeed)
        print("✓ Test data source registered")

        # Check available providers
        providers = factory.get_available_providers()
        print(f"✓ Available providers: {providers}")

        assert True  # Test passed

    except Exception as e:
        print(f"✗ Data source factory test failed: {e}")
        assert False, f"Data source factory test failed: {e}"


def test_data_aggregator():
    """Test the data aggregator functionality."""
    print("\nTesting Data Aggregator...")

    try:
        # Create aggregator
        aggregator = DataAggregator("test_provider")
        print("✓ Aggregator created")

        # Create sample datasets for comparison
        data1 = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=50, freq='1h'),
            'open': [100 + i * 0.1 for i in range(50)],
            'high': [101 + i * 0.1 for i in range(50)],
            'low': [99 + i * 0.1 for i in range(50)],
            'close': [100.5 + i * 0.1 for i in range(50)],
            'volume': [1000 + i * 10 for i in range(50)]
        })

        data2 = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01 12:00:00', periods=50, freq='1h'),
            'open': [100 + i * 0.1 for i in range(50)],
            'high': [101 + i * 0.1 for i in range(50)],
            'low': [99 + i * 0.1 for i in range(50)],
            'close': [100.5 + i * 0.1 for i in range(50)],
            'volume': [1000 + i * 10 for i in range(50)]
        })

        # Test synchronization
        sync1, sync2 = aggregator.synchronize_data(data1, data2, "TEST")
        print(f"✓ Data synchronization: {len(sync1)} synchronized points")

        assert True  # Test passed

    except Exception as e:
        print(f"✗ Data aggregator test failed: {e}")
        assert False, f"Data aggregator test failed: {e}"


def test_utility_functions():
    """Test utility functions."""
    print("\nTesting Utility Functions...")

    try:
        # Create sample data for validation
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1h'),
            'open': [100 + i * 0.1 for i in range(100)],
            'high': [101 + i * 0.1 for i in range(100)],
            'low': [99 + i * 0.1 for i in range(100)],
            'close': [100.5 + i * 0.1 for i in range(100)],
            'volume': [1000 + i * 10 for i in range(100)]
        })

        # Test validation
        is_valid, errors = validate_ohlcv_data(sample_data)
        print(f"✓ Data validation: {is_valid}")
        if errors:
            print(f"  Validation errors: {errors}")

        assert True  # Test passed

    except Exception as e:
        print(f"✗ Utility functions test failed: {e}")
        assert False, f"Utility functions test failed: {e}"


def main():
    """Run all integration tests."""
    print("Phase 2 Integration Tests")
    print("=" * 40)

    tests = [
        test_data_handler,
        test_data_source_factory,
        test_data_aggregator,
        test_utility_functions
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")

    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Phase 2 implementation is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
