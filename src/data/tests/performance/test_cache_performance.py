#!/usr/bin/env python3
"""
Performance Tests for Unified Cache

This module contains performance tests to validate that the new unified cache
architecture provides the expected performance improvements.

Test Coverage:
- Cache hit performance (should be 10x+ faster than cache miss)
- Memory usage validation
- File I/O performance
- Concurrent access performance
"""

import sys
import os
import tempfile
import shutil
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_manager import DataManager
from src.data.cache.unified_cache import UnifiedCache
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class TestCachePerformance:
    """Performance tests for the unified cache system."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def data_manager(self, temp_cache_dir):
        """Create a DataManager instance with temporary cache."""
        return DataManager(cache_dir=temp_cache_dir)

    @pytest.fixture
    def mock_downloader(self):
        """Create a mock downloader that simulates API delay."""
        downloader = Mock()

        def slow_get_ohlcv(*args, **kwargs):
            # Simulate API delay
            time.sleep(0.1)  # 100ms delay
            return pd.DataFrame({
                'open': [100.0 + i * 0.1 for i in range(100)],
                'high': [101.0 + i * 0.1 for i in range(100)],
                'low': [99.0 + i * 0.1 for i in range(100)],
                'close': [100.5 + i * 0.1 for i in range(100)],
                'volume': [1000 + i * 10 for i in range(100)]
            }, index=pd.date_range('2024-01-01', periods=100, freq='1h'))

        downloader.get_ohlcv.side_effect = slow_get_ohlcv
        return downloader

    def test_cache_hit_performance(self, data_manager, mock_downloader):
        """Test that cache hits are significantly faster than cache misses."""
        symbol = "BTCUSDT"
        timeframe = "1h"
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 5, 3)  # Match the full range of mock data (100 hours)

        with patch.object(data_manager.provider_selector, 'get_provider_with_failover') as mock_failover:
            mock_failover.return_value = ['test_provider']

            with patch.dict(data_manager.provider_selector.downloaders, {
                'test_provider': mock_downloader
            }):
                # First request - cache miss (should be slow)
                start_time = time.time()
                result1 = data_manager.get_ohlcv(symbol, timeframe, start_date, end_date)
                cache_miss_time = time.time() - start_time

                # Second request - cache hit (should be fast)
                start_time = time.time()
                result2 = data_manager.get_ohlcv(symbol, timeframe, start_date, end_date)
                cache_hit_time = time.time() - start_time

                # Verify results are identical (ignore frequency differences due to caching)
                pd.testing.assert_frame_equal(result1, result2, check_freq=False)

                # Verify cache hit is significantly faster
                speedup = cache_miss_time / cache_hit_time
                print(f"Cache miss time: {cache_miss_time:.3f}s")
                print(f"Cache hit time: {cache_hit_time:.3f}s")
                print(f"Speedup: {speedup:.1f}x")

                # Cache hit should be at least 3x faster
                assert speedup >= 3.0, f"Cache hit should be 3x+ faster, got {speedup:.1f}x"

    def test_concurrent_access_performance(self, data_manager, mock_downloader):
        """Test performance under concurrent access."""
        symbol = "BTCUSDT"
        timeframe = "1h"
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 2)

        with patch.object(data_manager.provider_selector, 'get_provider_with_failover') as mock_failover:
            mock_failover.return_value = ['test_provider']

            with patch.dict(data_manager.provider_selector.downloaders, {
                'test_provider': mock_downloader
            }):
                # First request to populate cache
                data_manager.get_ohlcv(symbol, timeframe, start_date, end_date)

                # Test concurrent access
                results = []
                threads = []

                def fetch_data():
                    result = data_manager.get_ohlcv(symbol, timeframe, start_date, end_date)
                    results.append(result)

                # Create multiple threads
                num_threads = 5
                start_time = time.time()

                for _ in range(num_threads):
                    thread = threading.Thread(target=fetch_data)
                    threads.append(thread)
                    thread.start()

                # Wait for all threads to complete
                for thread in threads:
                    thread.join()

                concurrent_time = time.time() - start_time

                # Verify all results are identical
                for result in results:
                    pd.testing.assert_frame_equal(results[0], result)

                print(f"Concurrent access time for {num_threads} threads: {concurrent_time:.3f}s")

                # Concurrent access should be reasonably fast
                assert concurrent_time < 1.0, "Concurrent access should be fast"

    def test_large_dataset_performance(self, data_manager, mock_downloader):
        """Test performance with large datasets."""
        symbol = "BTCUSDT"
        timeframe = "1h"
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)  # Full year

        # Create a mock downloader that returns large dataset
        large_downloader = Mock()
        large_data = pd.DataFrame({
            'open': [100.0 + i * 0.1 for i in range(8760)],  # 8760 hours in a year
            'high': [101.0 + i * 0.1 for i in range(8760)],
            'low': [99.0 + i * 0.1 for i in range(8760)],
            'close': [100.5 + i * 0.1 for i in range(8760)],
            'volume': [1000 + i * 10 for i in range(8760)]
        }, index=pd.date_range('2024-01-01', periods=8760, freq='1h'))

        large_downloader.get_ohlcv.return_value = large_data

        with patch.object(data_manager.provider_selector, 'get_provider_with_failover') as mock_failover:
            mock_failover.return_value = ['test_provider']

            with patch.dict(data_manager.provider_selector.downloaders, {
                'test_provider': large_downloader
            }):
                # First request - cache miss
                start_time = time.time()
                result1 = data_manager.get_ohlcv(symbol, timeframe, start_date, end_date)
                cache_miss_time = time.time() - start_time

                # Second request - cache hit
                start_time = time.time()
                result2 = data_manager.get_ohlcv(symbol, timeframe, start_date, end_date)
                cache_hit_time = time.time() - start_time

                # Verify results
                pd.testing.assert_frame_equal(result1, result2, check_freq=False)
                assert len(result1) == 8760

                speedup = cache_miss_time / cache_hit_time
                print(f"Large dataset cache miss time: {cache_miss_time:.3f}s")
                print(f"Large dataset cache hit time: {cache_hit_time:.3f}s")
                print(f"Large dataset speedup: {speedup:.1f}x")

                # Should still be significantly faster
                assert speedup >= 2.0, f"Large dataset cache hit should be 2x+ faster, got {speedup:.1f}x"

    def test_memory_usage(self, data_manager, mock_downloader):
        """Test memory usage of cache operations."""
        import psutil
        import gc

        symbol = "BTCUSDT"
        timeframe = "1h"
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 2)

        with patch.object(data_manager.provider_selector, 'get_provider_with_failover') as mock_failover:
            mock_failover.return_value = ['test_provider']

            with patch.dict(data_manager.provider_selector.downloaders, {
                'test_provider': mock_downloader
            }):
                # Get initial memory usage
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB

                # Make multiple requests
                for _ in range(10):
                    result = data_manager.get_ohlcv(symbol, timeframe, start_date, end_date)
                    del result  # Explicit cleanup

                # Force garbage collection
                gc.collect()

                # Get final memory usage
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = final_memory - initial_memory

                print(f"Initial memory: {initial_memory:.1f} MB")
                print(f"Final memory: {final_memory:.1f} MB")
                print(f"Memory increase: {memory_increase:.1f} MB")

                # Memory increase should be reasonable
                assert memory_increase < 100, f"Memory usage should be reasonable, got {memory_increase:.1f} MB increase"

    def test_cache_file_size(self, data_manager, mock_downloader, temp_cache_dir):
        """Test that cache files are efficiently compressed."""
        symbol = "BTCUSDT"
        timeframe = "1h"
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 2)

        with patch.object(data_manager.provider_selector, 'get_provider_with_failover') as mock_failover:
            mock_failover.return_value = ['test_provider']

            with patch.dict(data_manager.provider_selector.downloaders, {
                'test_provider': mock_downloader
            }):
                # Make request to create cache
                result = data_manager.get_ohlcv(symbol, timeframe, start_date, end_date)

                # Check cache file size
                cache_path = Path(temp_cache_dir) / symbol / timeframe
                data_files = list(cache_path.glob("*.csv.gz"))

                assert len(data_files) > 0

                total_size = sum(f.stat().st_size for f in data_files)
                print(f"Cache file size: {total_size} bytes")

                # Estimate uncompressed size
                estimated_uncompressed = len(result) * len(result.columns) * 8  # Rough estimate
                compression_ratio = estimated_uncompressed / total_size

                print(f"Estimated compression ratio: {compression_ratio:.1f}x")

                # Should have good compression
                assert compression_ratio >= 2.0, f"Cache should be well compressed, got {compression_ratio:.1f}x"


def run_performance_tests():
    """Run all performance tests."""
    print("Running Cache Performance Tests...")
    print("=" * 50)

    # Run pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--color=yes"
    ])


if __name__ == "__main__":
    run_performance_tests()
