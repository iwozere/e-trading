"""
Unit tests for file-based cache system.

Tests the FileBasedCache class and related components to ensure
proper functionality of the hierarchical cache structure.
"""

import unittest
import tempfile
import shutil
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.data.utils.file_based_cache import (
    FileBasedCache, FileCacheCompressor, FileCacheInvalidationStrategy,
    TimeBasedInvalidation, VersionBasedInvalidation,
    FileCacheMetrics
)


class TestFileCacheMetrics(unittest.TestCase):
    """Test FileCacheMetrics class."""

    def setUp(self):
        """Set up test fixtures."""
        self.metrics = FileCacheMetrics()

    def test_initial_state(self):
        """Test initial state of metrics."""
        self.assertEqual(self.metrics.hits, 0)
        self.assertEqual(self.metrics.misses, 0)
        self.assertEqual(self.metrics.sets, 0)
        self.assertEqual(self.metrics.deletes, 0)
        self.assertEqual(self.metrics.errors, 0)

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        # No operations
        self.assertEqual(self.metrics.hit_rate, 0.0)

        # All hits
        self.metrics.hits = 10
        self.assertEqual(self.metrics.hit_rate, 1.0)

        # Mixed hits and misses
        self.metrics.misses = 5
        self.assertEqual(self.metrics.hit_rate, 2/3)  # 10/(10+5)

    def test_total_operations(self):
        """Test total operations calculation."""
        self.metrics.hits = 5
        self.metrics.misses = 3
        self.metrics.sets = 2
        self.metrics.deletes = 1
        self.assertEqual(self.metrics.total_operations, 11)


class TestFileCacheCompressor(unittest.TestCase):
    """Test FileCacheCompressor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.compressor = FileCacheCompressor(compression_level=3)

    def test_compression_decompression(self):
        """Test compression and decompression."""
        test_data = b"This is test data for compression testing"

        compressed = self.compressor.compress(test_data)
        decompressed = self.compressor.decompress(compressed)

        self.assertEqual(test_data, decompressed)
        self.assertLess(len(compressed), len(test_data))

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        original = b"x" * 1000
        compressed = self.compressor.compress(original)

        ratio = self.compressor.get_compression_ratio(original, compressed)
        self.assertLess(ratio, 1.0)  # Should be compressed
        self.assertGreater(ratio, 0.0)  # Should be positive


class TestCacheInvalidationStrategies(unittest.TestCase):
    """Test cache invalidation strategies."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test.txt"
        self.test_file.write_text("test")

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_time_based_invalidation(self):
        """Test time-based invalidation."""
        strategy = TimeBasedInvalidation(max_age_hours=1)

        # File is new, should not invalidate
        self.assertFalse(strategy.should_invalidate(self.test_file, {}))

        # Make file old
        old_time = time.time() - (2 * 3600)  # 2 hours ago
        os.utime(self.test_file, (old_time, old_time))

        # File is old, should invalidate
        self.assertTrue(strategy.should_invalidate(self.test_file, {}))

    def test_version_based_invalidation(self):
        """Test version-based invalidation."""
        strategy = VersionBasedInvalidation(current_version="2.0.0")

        # Same version, should not invalidate
        metadata = {"version": "2.0.0"}
        self.assertFalse(strategy.should_invalidate(self.test_file, metadata))

        # Different version, should invalidate
        metadata = {"version": "1.0.0"}
        self.assertTrue(strategy.should_invalidate(self.test_file, metadata))


class TestFileBasedCache(unittest.TestCase):
    """Test FileBasedCache class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = FileBasedCache(
            cache_dir=self.temp_dir,
            max_size_gb=1.0,
            retention_days=7,
            compression_enabled=False  # Disable for testing
        )

        # Create test data
        self.test_df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [102.0, 103.0, 104.0],
            'low': [99.0, 100.0, 101.0],
            'close': [101.0, 102.0, 103.0],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2023-01-01', periods=3, freq='D'))

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_cache_initialization(self):
        """Test cache initialization."""
        self.assertEqual(self.cache.cache_dir, Path(self.temp_dir))
        self.assertEqual(self.cache.max_size_gb, 1.0)
        self.assertEqual(self.cache.retention_days, 7)
        self.assertFalse(self.cache.compression_enabled)

    def test_get_cache_path(self):
        """Test cache path generation."""
        path = self.cache._get_cache_path("binance", "BTCUSDT", "1h", 2023)
        expected = Path(self.temp_dir) / "binance" / "BTCUSDT" / "1h" / "2023"
        self.assertEqual(path, expected)
        self.assertTrue(path.exists())

    def test_put_and_get_data(self):
        """Test putting and getting data."""
        # Put data
        success = self.cache.put(
            self.test_df, "binance", "BTCUSDT", "1h",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 3)
        )
        self.assertTrue(success)

        # Get data
        retrieved_df = self.cache.get(
            "binance", "BTCUSDT", "1h",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 3)
        )

        self.assertIsNotNone(retrieved_df)
        pd.testing.assert_frame_equal(self.test_df, retrieved_df)

    def test_get_nonexistent_data(self):
        """Test getting non-existent data."""
        df = self.cache.get("nonexistent", "SYMBOL", "1h")
        self.assertIsNone(df)
        self.assertEqual(self.cache.metrics.misses, 1)

    def test_delete_data(self):
        """Test deleting data."""
        # Put data first
        self.cache.put(self.test_df, "binance", "BTCUSDT", "1h")

        # Delete data
        success = self.cache.delete("binance", "BTCUSDT", "1h", 2023)
        self.assertTrue(success)

        # Verify data is gone
        df = self.cache.get("binance", "BTCUSDT", "1h")
        self.assertIsNone(df)

    def test_clear_cache(self):
        """Test clearing cache."""
        # Put data for multiple providers
        self.cache.put(self.test_df, "binance", "BTCUSDT", "1h")
        self.cache.put(self.test_df, "yahoo", "AAPL", "1d")

        # Clear specific provider
        success = self.cache.clear(provider="binance")
        self.assertTrue(success)

        # Verify binance data is gone but yahoo remains
        self.assertIsNone(self.cache.get("binance", "BTCUSDT", "1h"))
        self.assertIsNotNone(self.cache.get("yahoo", "AAPL", "1d"))

    def test_get_stats(self):
        """Test getting cache statistics."""
        # Put some data
        self.cache.put(self.test_df, "binance", "BTCUSDT", "1h")

        stats = self.cache.get_stats()

        self.assertIn('hits', stats)
        self.assertIn('misses', stats)
        self.assertIn('sets', stats)
        self.assertIn('cache_dir', stats)
        self.assertIn('cache_size_gb', stats)
        self.assertIn('file_count', stats)
        self.assertIn('directory_count', stats)

    def test_get_cache_info(self):
        """Test getting cache information."""
        # Put data for multiple years
        self.cache.put(self.test_df, "binance", "BTCUSDT", "1h")

        # Create data for another year
        df_2024 = self.test_df.copy()
        df_2024.index = pd.date_range('2024-01-01', periods=3, freq='D')
        self.cache.put(df_2024, "binance", "BTCUSDT", "1h")

        info = self.cache.get_cache_info("binance", "BTCUSDT", "1h")

        self.assertEqual(info['provider'], "binance")
        self.assertEqual(info['symbol'], "BTCUSDT")
        self.assertEqual(info['interval'], "1h")
        self.assertIn(2023, info['years_available'])
        self.assertIn(2024, info['years_available'])
        self.assertGreater(info['total_rows'], 0)

    def test_cleanup_old_files(self):
        """Test cleanup of old files."""
        # Put data
        self.cache.put(self.test_df, "binance", "BTCUSDT", "1h")

        # Make file old
        cache_path = self.cache._get_cache_path("binance", "BTCUSDT", "1h", 2023)
        data_path = self.cache._get_data_path(cache_path)
        old_time = time.time() - (10 * 24 * 3600)  # 10 days ago
        os.utime(data_path, (old_time, old_time))

        # Clean up old files
        deleted_count = self.cache.cleanup_old_files()
        self.assertGreater(deleted_count, 0)

        # Verify data is gone
        df = self.cache.get("binance", "BTCUSDT", "1h")
        self.assertIsNone(df)

    def test_csv_format(self):
        """Test CSV format support."""
        # Put data in CSV format
        success = self.cache.put(
            self.test_df, "binance", "BTCUSDT", "1h",
            format="csv"
        )
        self.assertTrue(success)

        # Get data in CSV format
        retrieved_df = self.cache.get(
            "binance", "BTCUSDT", "1h",
            format="csv"
        )

        self.assertIsNotNone(retrieved_df)
        pd.testing.assert_frame_equal(self.test_df, retrieved_df)

    def test_date_filtering(self):
        """Test date range filtering."""
        # Put data
        self.cache.put(self.test_df, "binance", "BTCUSDT", "1h")

        # Get data with date range
        start_date = datetime(2023, 1, 2)
        end_date = datetime(2023, 1, 3)

        filtered_df = self.cache.get(
            "binance", "BTCUSDT", "1h",
            start_date=start_date,
            end_date=end_date
        )

        self.assertIsNotNone(filtered_df)
        self.assertEqual(len(filtered_df), 2)
        self.assertTrue(all(filtered_df.index >= start_date))
        self.assertTrue(all(filtered_df.index <= end_date))


class TestGlobalFunctions(unittest.TestCase):
    """Test global cache functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_get_file_cache(self):
        """Test get_file_cache function."""
        from src.data.utils.file_based_cache import get_file_cache, _file_cache_instance

        # Reset global instance
        _file_cache_instance = None

        # Get cache instance
        cache = get_file_cache(
            cache_dir=self.temp_dir,
            max_size_gb=5.0,
            retention_days=14,
            compression_enabled=True
        )

        self.assertIsInstance(cache, FileBasedCache)
        self.assertEqual(cache.cache_dir, Path(self.temp_dir))
        self.assertEqual(cache.max_size_gb, 5.0)
        self.assertEqual(cache.retention_days, 14)
        self.assertTrue(cache.compression_enabled)

    def test_configure_file_cache(self):
        """Test configure_file_cache function."""
        from src.data.utils.file_based_cache import configure_file_cache, _file_cache_instance

        # Reset global instance
        _file_cache_instance = None

        # Configure cache
        cache = configure_file_cache(
            cache_dir=self.temp_dir,
            max_size_gb=10.0,
            retention_days=30,
            compression_enabled=False
        )

        self.assertIsInstance(cache, FileBasedCache)
        self.assertEqual(cache.max_size_gb, 10.0)
        self.assertEqual(cache.retention_days, 30)
        self.assertFalse(cache.compression_enabled)


if __name__ == '__main__':
    unittest.main()
