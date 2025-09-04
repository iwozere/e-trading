#!/usr/bin/env python3
"""
Unit tests for enhanced CSV cache functionality.
"""

import unittest
import tempfile
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.data.utils.file_based_cache import (
    CSVFormatConventions, SafeCSVAppender, SmartDataAppender,
    CacheMetadata, FileBasedCache
)


class TestCSVFormatConventions(unittest.TestCase):
    """Test CSV format conventions validation and standardization."""

    def setUp(self):
        """Set up test data."""
        self.valid_df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='h'),
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [101.0, 102.0, 103.0, 104.0, 105.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [101.0, 102.0, 103.0, 104.0, 105.0],
            'volume': [1000.0, 1100.0, 1200.0, 1300.0, 1400.0]
        })

        self.invalid_df_missing_cols = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='h'),
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [101.0, 102.0, 103.0, 104.0, 105.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [101.0, 102.0, 103.0, 104.0, 105.0]
            # Missing volume column
        })

        self.invalid_df_wrong_types = pd.DataFrame({
            'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'open': [100.0, 101.0, 102.0],
            'high': [101.0, 102.0, 103.0],
            'low': [99.0, 100.0, 101.0],
            'close': [101.0, 102.0, 103.0],
            'volume': [1000.0, 1100.0, 1200.0]
        })

    def test_validate_dataframe_valid(self):
        """Test validation of valid DataFrame."""
        self.assertTrue(CSVFormatConventions.validate_dataframe(self.valid_df))

    def test_validate_dataframe_missing_columns(self):
        """Test validation of DataFrame with missing columns."""
        self.assertFalse(CSVFormatConventions.validate_dataframe(self.invalid_df_missing_cols))

    def test_validate_dataframe_wrong_types(self):
        """Test validation of DataFrame with wrong data types."""
        self.assertFalse(CSVFormatConventions.validate_dataframe(self.invalid_df_wrong_types))

    def test_validate_dataframe_empty(self):
        """Test validation of empty DataFrame."""
        empty_df = pd.DataFrame()
        self.assertFalse(CSVFormatConventions.validate_dataframe(empty_df))

    def test_standardize_dataframe_with_index(self):
        """Test standardization of DataFrame with datetime index."""
        df_with_index = self.valid_df.set_index('timestamp')
        standardized = CSVFormatConventions.standardize_dataframe(df_with_index, 'test_provider')

        self.assertIn('timestamp', standardized.columns)
        self.assertIn('provider_download_ts', standardized.columns)
        self.assertEqual(len(standardized), len(df_with_index))

    def test_standardize_dataframe_adds_missing_columns(self):
        """Test that missing columns are added with defaults."""
        df_missing_volume = self.valid_df.drop(columns=['volume'])
        standardized = CSVFormatConventions.standardize_dataframe(df_missing_volume, 'test_provider')

        self.assertIn('volume', standardized.columns)
        self.assertTrue(all(standardized['volume'] == 0.0))

    def test_standardize_dataframe_adds_provider_timestamp(self):
        """Test that provider download timestamp is added."""
        standardized = CSVFormatConventions.standardize_dataframe(self.valid_df, 'test_provider')

        self.assertIn('provider_download_ts', standardized.columns)
        self.assertTrue(all(pd.notna(standardized['provider_download_ts'])))

    def test_standardize_dataframe_removes_duplicates(self):
        """Test that duplicates are removed."""
        df_with_duplicates = pd.concat([self.valid_df, self.valid_df.iloc[[0]]])
        standardized = CSVFormatConventions.standardize_dataframe(df_with_duplicates, 'test_provider')

        self.assertEqual(len(standardized), len(self.valid_df))


class TestSafeCSVAppender(unittest.TestCase):
    """Test safe CSV appending functionality."""

    def setUp(self):
        """Set up test data."""
        self.test_df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='h'),
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [101.0, 102.0, 103.0, 104.0, 105.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [101.0, 102.0, 103.0, 104.0, 105.0],
            'volume': [1000.0, 1100.0, 1200.0, 1300.0, 1400.0]
        })

        self.new_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01 05:00:00', periods=3, freq='h'),
            'open': [105.0, 106.0, 107.0],
            'high': [106.0, 107.0, 108.0],
            'low': [104.0, 105.0, 106.0],
            'close': [106.0, 107.0, 108.0],
            'volume': [1500.0, 1600.0, 1700.0]
        })

    def test_append_to_csv_new_file(self):
        """Test appending to non-existent file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.csv"

            success = SafeCSVAppender.append_to_csv(file_path, self.test_df)
            self.assertTrue(success)
            self.assertTrue(file_path.exists())

            # Verify data was written correctly
            loaded_df = pd.read_csv(file_path, parse_dates=['timestamp'])
            self.assertEqual(len(loaded_df), len(self.test_df))

    def test_append_to_csv_existing_file(self):
        """Test appending to existing file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.csv"

            # Create initial file
            self.test_df.to_csv(file_path, index=False)

            # Append new data
            success = SafeCSVAppender.append_to_csv(file_path, self.new_data)
            self.assertTrue(success)

            # Verify combined data
            loaded_df = pd.read_csv(file_path, parse_dates=['timestamp'])
            self.assertEqual(len(loaded_df), len(self.test_df) + len(self.new_data))

    def test_append_to_csv_with_backup(self):
        """Test that backup is created when appending."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.csv"

            # Create initial file
            self.test_df.to_csv(file_path, index=False)

            # Append with backup
            success = SafeCSVAppender.append_to_csv(file_path, self.new_data, backup_existing=True)
            self.assertTrue(success)

            # Check backup was created
            backup_path = file_path.with_suffix('.csv.backup')
            self.assertTrue(backup_path.exists())

            # Verify backup contains original data
            backup_df = pd.read_csv(backup_path, parse_dates=['timestamp'])
            self.assertEqual(len(backup_df), len(self.test_df))

    def test_append_to_csv_removes_duplicates(self):
        """Test that duplicates are removed when appending."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.csv"

            # Create initial file
            self.test_df.to_csv(file_path, index=False)

            # Append data with overlap (last row of test_df overlaps with first row of new_data)
            overlapping_data = pd.DataFrame({
                'timestamp': [self.test_df['timestamp'].iloc[-1]],  # Same timestamp as last row
                'open': [999.0],  # Different value
                'high': [999.0],
                'low': [999.0],
                'close': [999.0],
                'volume': [999.0]
            })

            success = SafeCSVAppender.append_to_csv(file_path, overlapping_data)
            self.assertTrue(success)

            # Verify no duplicates
            loaded_df = pd.read_csv(file_path, parse_dates=['timestamp'])
            self.assertEqual(len(loaded_df), len(self.test_df))  # Should not increase


class TestSmartDataAppender(unittest.TestCase):
    """Test smart data appending functionality."""

    def setUp(self):
        """Set up test data."""
        self.existing_df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='h'),
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [101.0, 102.0, 103.0, 104.0, 105.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [101.0, 102.0, 103.0, 104.0, 105.0],
            'volume': [1000.0, 1100.0, 1200.0, 1300.0, 1400.0]
        })

        self.new_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01 05:00:00', periods=3, freq='h'),
            'open': [105.0, 106.0, 107.0],
            'high': [106.0, 107.0, 108.0],
            'low': [104.0, 105.0, 106.0],
            'close': [106.0, 107.0, 108.0],
            'volume': [1500.0, 1600.0, 1700.0]
        })

        self.overlapping_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01 04:00:00', periods=3, freq='h'),  # Overlaps with existing
            'open': [999.0, 999.0, 999.0],
            'high': [999.0, 999.0, 999.0],
            'low': [999.0, 999.0, 999.0],
            'close': [999.0, 999.0, 999.0],
            'volume': [999.0, 999.0, 999.0]
        })

    def test_append_new_data_new_file(self):
        """Test appending to non-existent file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.csv"
            metadata = CacheMetadata(
                provider="test",
                symbol="TEST",
                interval="1h",
                year=2023
            )

            success, rows_added = SmartDataAppender.append_new_data(file_path, self.new_data, metadata)
            self.assertTrue(success)
            self.assertEqual(rows_added, len(self.new_data))

            # Verify file was created
            self.assertTrue(file_path.exists())

    def test_append_new_data_only_new(self):
        """Test that only new data is appended."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.csv"

            # Create existing file
            self.existing_df.to_csv(file_path, index=False)

            metadata = CacheMetadata(
                provider="test",
                symbol="TEST",
                interval="1h",
                year=2023
            )

            success, rows_added = SmartDataAppender.append_new_data(file_path, self.new_data, metadata)
            self.assertTrue(success)
            self.assertEqual(rows_added, len(self.new_data))

            # Verify combined data
            loaded_df = pd.read_csv(file_path, parse_dates=['timestamp'])
            self.assertEqual(len(loaded_df), len(self.existing_df) + len(self.new_data))

    def test_append_new_data_no_new_data(self):
        """Test appending when no new data exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.csv"

            # Create existing file
            self.existing_df.to_csv(file_path, index=False)

            metadata = CacheMetadata(
                provider="test",
                symbol="TEST",
                interval="1h",
                year=2023
            )

            # Try to append data that's all before existing data
            old_data = pd.DataFrame({
                'timestamp': pd.date_range('2022-12-31', periods=3, freq='h'),
                'open': [90.0, 91.0, 92.0],
                'high': [91.0, 92.0, 93.0],
                'low': [89.0, 90.0, 91.0],
                'close': [91.0, 92.0, 93.0],
                'volume': [900.0, 910.0, 920.0]
            })

            success, rows_added = SmartDataAppender.append_new_data(file_path, old_data, metadata)
            self.assertTrue(success)
            self.assertEqual(rows_added, 0)  # No new rows added

            # Verify file unchanged
            loaded_df = pd.read_csv(file_path, parse_dates=['timestamp'])
            self.assertEqual(len(loaded_df), len(self.existing_df))

    def test_append_new_data_with_overlap(self):
        """Test appending with overlapping data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.csv"

            # Create existing file
            self.existing_df.to_csv(file_path, index=False)

            metadata = CacheMetadata(
                provider="test",
                symbol="TEST",
                interval="1h",
                year=2023
            )

            success, rows_added = SmartDataAppender.append_new_data(file_path, self.overlapping_data, metadata)
            self.assertTrue(success)
            self.assertGreater(rows_added, 0)  # Some new data should be added

            # Verify no duplicates
            loaded_df = pd.read_csv(file_path, parse_dates=['timestamp'])
            self.assertEqual(len(loaded_df), len(loaded_df.drop_duplicates(subset=['timestamp'])))


class TestCacheMetadata(unittest.TestCase):
    """Test cache metadata functionality."""

    def setUp(self):
        """Set up test data."""
        self.test_df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='h'),
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [101.0, 102.0, 103.0, 104.0, 105.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [101.0, 102.0, 103.0, 104.0, 105.0],
            'volume': [1000.0, 1100.0, 1200.0, 1300.0, 1400.0]
        })

    def test_cache_metadata_creation(self):
        """Test cache metadata creation."""
        metadata = CacheMetadata(
            provider="test",
            symbol="TEST",
            interval="1h",
            year=2023
        )

        self.assertEqual(metadata.provider, "test")
        self.assertEqual(metadata.symbol, "TEST")
        self.assertEqual(metadata.interval, "1h")
        self.assertEqual(metadata.year, 2023)
        self.assertIsNotNone(metadata.created_at)
        self.assertIsNotNone(metadata.last_modified)

    def test_cache_metadata_update_integrity_info(self):
        """Test updating integrity information."""
        metadata = CacheMetadata(
            provider="test",
            symbol="TEST",
            interval="1h",
            year=2023
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.csv"
            self.test_df.to_csv(file_path, index=False)

            metadata.update_integrity_info(self.test_df, file_path)

            self.assertEqual(metadata.row_count, len(self.test_df))
            self.assertEqual(metadata.first_timestamp, self.test_df['timestamp'].min().isoformat())
            self.assertEqual(metadata.last_timestamp, self.test_df['timestamp'].max().isoformat())
            self.assertGreater(metadata.file_size_bytes, 0)

    def test_cache_metadata_validate_integrity(self):
        """Test metadata integrity validation."""
        metadata = CacheMetadata(
            provider="test",
            symbol="TEST",
            interval="1h",
            year=2023
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.csv"
            self.test_df.to_csv(file_path, index=False)

            metadata.update_integrity_info(self.test_df, file_path)

            # Should be valid
            self.assertTrue(metadata.validate_integrity(file_path))

            # Modify file to make it invalid
            with open(file_path, 'a') as f:
                f.write("\n2023-01-01 06:00:00,999,999,999,999,999\n")

            # Should be invalid now
            self.assertFalse(metadata.validate_integrity(file_path))


class TestEnhancedFileBasedCache(unittest.TestCase):
    """Test enhanced file-based cache functionality."""

    def setUp(self):
        """Set up test cache."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = FileBasedCache(
            cache_dir=self.temp_dir,
            compression_enabled=False,
            default_format="csv"
        )

        self.test_df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='h'),
            'open': [100.0 + i for i in range(10)],
            'high': [101.0 + i for i in range(10)],
            'low': [99.0 + i for i in range(10)],
            'close': [101.0 + i for i in range(10)],
            'volume': [1000.0 + i * 100 for i in range(10)]
        })

    def tearDown(self):
        """Clean up test cache."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_put_with_csv_format(self):
        """Test putting data with CSV format."""
        success = self.cache.put(
            self.test_df, "test_provider", "TEST", "1h", format="csv"
        )
        self.assertTrue(success)

        # Verify data was stored
        retrieved_df = self.cache.get("test_provider", "TEST", "1h", format="csv")
        self.assertIsNotNone(retrieved_df)
        self.assertEqual(len(retrieved_df), len(self.test_df))

    def test_put_with_append_mode(self):
        """Test putting data in append mode."""
        # First put
        success = self.cache.put(
            self.test_df, "test_provider", "TEST", "1h", format="csv", append_mode=True
        )
        self.assertTrue(success)

        # Second put with new data
        new_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01 10:00:00', periods=5, freq='h'),
            'open': [110.0 + i for i in range(5)],
            'high': [111.0 + i for i in range(5)],
            'low': [109.0 + i for i in range(5)],
            'close': [111.0 + i for i in range(5)],
            'volume': [2000.0 + i * 100 for i in range(5)]
        })

        success = self.cache.put(
            new_data, "test_provider", "TEST", "1h", format="csv", append_mode=True
        )
        self.assertTrue(success)

        # Verify combined data
        retrieved_df = self.cache.get("test_provider", "TEST", "1h", format="csv")
        self.assertIsNotNone(retrieved_df)
        self.assertEqual(len(retrieved_df), len(self.test_df) + len(new_data))

    def test_csv_format_validation(self):
        """Test that CSV format validation is no longer enforced (validation removed)."""
        # Create DataFrame with wrong data types - should now succeed since validation was removed
        invalid_df = pd.DataFrame({
            'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03'],  # String timestamps
            'open': [100.0, 101.0, 102.0],
            'high': [101.0, 102.0, 103.0],
            'low': [99.0, 100.0, 101.0],
            'close': [101.0, 102.0, 103.0],
            'volume': [1000.0, 1100.0, 1200.0]
        })

        success = self.cache.put(
            invalid_df, "test_provider", "TEST", "1h", format="csv"
        )
        # Should now succeed since validation was removed from cache system
        self.assertTrue(success)

    def test_cache_info_with_enhanced_metadata(self):
        """Test that cache info includes enhanced metadata."""
        success = self.cache.put(
            self.test_df, "test_provider", "TEST", "1h", format="csv"
        )
        self.assertTrue(success)

        info = self.cache.get_cache_info("test_provider", "TEST", "1h")
        self.assertIn('years_available', info)
        self.assertIn('total_rows', info)
        self.assertIn('total_size_bytes', info)
        self.assertIn('last_updated', info)


if __name__ == '__main__':
    unittest.main()
