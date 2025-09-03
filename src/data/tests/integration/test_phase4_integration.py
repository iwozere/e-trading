"""
Phase 4 Integration Tests.

Comprehensive integration tests for the complete data module functionality,
including file-based caching, data handling, and all Phase 1-3 features.
"""

import sys
import os
import tempfile
import shutil
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import unittest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.data import (
    # Core components
    BaseDataSource, DataSourceFactory, DataAggregator,
    get_data_source_factory, register_data_source, create_data_source,

    # Utilities
    get_data_handler, validate_ohlcv_data, get_data_quality_score,
    RateLimiter, DataCache,

    # Phase 3: Advanced Features
    FileBasedCache, get_file_cache, configure_file_cache,
    StreamMultiplexer, get_stream_multiplexer, create_stream_config,
    LazyDataLoader, ParallelProcessor, MemoryOptimizer, PerformanceMonitor,
    get_performance_monitor, get_memory_optimizer, get_data_compressor,
    optimize_dataframe_performance, compress_dataframe_efficiently,

    # Specific implementations
    BinanceLiveFeed, BinanceDataFeed
)

from src.data.utils.file_based_cache import (
    TimeBasedInvalidation, VersionBasedInvalidation
)


class MockDataSource(BaseDataSource):
    """Mock data source for testing."""

    def __init__(self, provider_name: str = "mock", **kwargs):
        super().__init__(provider_name, **kwargs)
        self.data = {}

    def get_available_symbols(self):
        return ["MOCK1", "MOCK2", "MOCK3"]

    def get_supported_intervals(self):
        return ["1m", "5m", "1h", "1d"]

    def fetch_historical_data(self, symbol: str, interval: str,
                            start_date=None, end_date=None, limit=None):
        # Generate mock data
        if symbol not in self.data:
            self.data[symbol] = {}

        if interval not in self.data[symbol]:
            # Create mock OHLCV data
            dates = pd.date_range(
                start=start_date or datetime.now() - timedelta(days=30),
                end=end_date or datetime.now(),
                freq=interval
            )

            self.data[symbol][interval] = pd.DataFrame({
                'open': np.random.uniform(100, 200, len(dates)),
                'high': np.random.uniform(200, 300, len(dates)),
                'low': np.random.uniform(50, 100, len(dates)),
                'close': np.random.uniform(100, 200, len(dates)),
                'volume': np.random.uniform(1000, 10000, len(dates))
            }, index=dates)

        return self.data[symbol][interval]

    def start_realtime_feed(self, symbol: str, interval: str, callback=None):
        return True

    def stop_realtime_feed(self, symbol: str):
        return True


class TestPhase4Integration(unittest.TestCase):
    """Integration tests for Phase 4 complete system."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.temp_dir, "cache")

        # Configure file-based cache
        self.cache = configure_file_cache(
            cache_dir=self.cache_dir,
            max_size_gb=1.0,
            retention_days=7,
            compression_enabled=True,
            invalidation_strategies=[
                TimeBasedInvalidation(max_age_hours=24),
                VersionBasedInvalidation(current_version="1.0.0")
            ]
        )

        # Create data source factory
        self.factory = get_data_source_factory()

        # Register mock data source
        register_data_source("mock", MockDataSource)

        # Create mock data source
        self.mock_source = create_data_source("mock", cache_enabled=True)

        # Create data aggregator
        self.aggregator = DataAggregator(primary_provider="mock")

        # Create data handler
        self.data_handler = get_data_handler("mock", cache_enabled=True)

        # Create performance monitor
        self.performance_monitor = get_performance_monitor()

        # Create memory optimizer
        self.memory_optimizer = get_memory_optimizer()

        # Create data compressor
        self.data_compressor = get_data_compressor()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_complete_data_flow(self):
        """Test complete data flow from source to cache."""
        symbol = "MOCK1"
        interval = "1h"
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()

        # Step 1: Fetch data from source
        with self.performance_monitor.start_operation("data_fetch"):
            df = self.mock_source.fetch_historical_data(
                symbol, interval, start_date, end_date
            )

        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)

        # Step 2: Validate data
        is_valid, errors = validate_ohlcv_data(df)
        self.assertTrue(is_valid, f"Data validation failed: {errors}")

        quality_score = get_data_quality_score(df)
        self.assertGreater(quality_score['quality_score'], 0.8)

        # Step 3: Optimize data
        optimized_df = optimize_dataframe_performance(df)
        self.assertIsNotNone(optimized_df)

        # Step 4: Cache data
        cache_success = self.cache.put(
            df, "mock", symbol, interval,
            start_date=start_date, end_date=end_date
        )
        self.assertTrue(cache_success)

        # Step 5: Retrieve from cache
        cached_df = self.cache.get(
            "mock", symbol, interval,
            start_date=start_date, end_date=end_date
        )

        self.assertIsNotNone(cached_df)
        pd.testing.assert_frame_equal(df, cached_df)

        # Step 6: Check cache statistics
        stats = self.cache.get_stats()
        self.assertGreater(stats['hits'], 0)
        self.assertGreater(stats['sets'], 0)

    def test_data_source_factory_integration(self):
        """Test data source factory integration."""
        # Test factory operations
        sources = self.factory.get_all_data_sources()
        self.assertIn("mock", sources)

        available_providers = self.factory.get_available_providers()
        self.assertIn("mock", available_providers)

        # Test health status
        health = self.factory.get_health_status()
        self.assertIn("mock", health)
        self.assertTrue(health["mock"]["healthy"])

        # Test data quality reports
        reports = self.factory.get_data_quality_reports(
            ["MOCK1", "MOCK2"], "1h"
        )
        self.assertIn("MOCK1", reports)
        self.assertIn("MOCK2", reports)

    def test_data_aggregator_integration(self):
        """Test data aggregator integration."""
        # Create multiple data sources
        source1 = create_data_source("mock", cache_enabled=True)
        source2 = create_data_source("mock", cache_enabled=True)

        # Test aggregation with multiple sources
        aggregated_df = self.aggregator.aggregate_data(
            "MOCK1", "1h", ["mock", "mock"],
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now()
        )

        self.assertIsNotNone(aggregated_df)
        self.assertGreater(len(aggregated_df), 0)

        # Test data comparison
        comparison = self.aggregator.compare_data_sources(
            "MOCK1", "1h", ["mock", "mock"]
        )

        self.assertIn("consistency_score", comparison)
        self.assertIn("recommendations", comparison)

    def test_data_handler_integration(self):
        """Test data handler integration."""
        # Test data standardization
        raw_df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='H'),
            'open': [100, 101, 102, 103, 104],
            'high': [102, 103, 104, 105, 106],
            'low': [99, 100, 101, 102, 103],
            'close': [101, 102, 103, 104, 105],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })

        standardized_df = self.data_handler.standardize_ohlcv_data(
            raw_df, "TEST", "1h", timestamp_col="timestamp"
        )

        self.assertIsNotNone(standardized_df)
        self.assertTrue(isinstance(standardized_df.index, pd.DatetimeIndex))

        # Test data validation and scoring
        validation_result = self.data_handler.validate_and_score_data(
            standardized_df, "TEST"
        )

        self.assertIn("is_valid", validation_result)
        self.assertIn("quality_score", validation_result)
        self.assertTrue(validation_result["is_valid"])

        # Test data caching
        cache_success = self.data_handler.cache_data(
            standardized_df, "TEST", "1h"
        )
        self.assertTrue(cache_success)

        # Test data retrieval
        cached_df = self.data_handler.get_cached_data("TEST", "1h")
        self.assertIsNotNone(cached_df)

    def test_performance_optimization_integration(self):
        """Test performance optimization integration."""
        # Create large test dataset
        large_df = pd.DataFrame({
            'open': np.random.uniform(100, 200, 10000),
            'high': np.random.uniform(200, 300, 10000),
            'low': np.random.uniform(50, 100, 10000),
            'close': np.random.uniform(100, 200, 10000),
            'volume': np.random.uniform(1000, 10000, 10000),
            'category': np.random.choice(['A', 'B', 'C'], 10000)
        }, index=pd.date_range('2023-01-01', periods=10000, freq='H'))

        # Test memory optimization
        with self.performance_monitor.start_operation("memory_optimization"):
            optimized_df = self.memory_optimizer.optimize_dataframe(large_df)

        # Check memory reduction
        original_memory = large_df.memory_usage(deep=True).sum()
        optimized_memory = optimized_df.memory_usage(deep=True).sum()

        self.assertLess(optimized_memory, original_memory)

        # Test data compression
        with self.performance_monitor.start_operation("data_compression"):
            compressed_data = self.data_compressor.compress_dataframe(
                optimized_df, format="parquet"
            )

        self.assertIsNotNone(compressed_data)
        self.assertGreater(len(compressed_data), 0)

        # Test decompression
        decompressed_df = self.data_compressor.decompress_dataframe(
            compressed_data, format="parquet"
        )

        pd.testing.assert_frame_equal(optimized_df, decompressed_df)

    def test_lazy_loading_integration(self):
        """Test lazy loading integration."""
        # Create test file
        test_file = os.path.join(self.temp_dir, "test_data.parquet")
        test_df = pd.DataFrame({
            'open': np.random.uniform(100, 200, 1000),
            'high': np.random.uniform(200, 300, 1000),
            'low': np.random.uniform(50, 100, 1000),
            'close': np.random.uniform(100, 200, 1000),
            'volume': np.random.uniform(1000, 10000, 1000)
        }, index=pd.date_range('2023-01-01', periods=1000, freq='H'))

        test_df.to_parquet(test_file)

        # Test lazy loading
        lazy_loader = LazyDataLoader(test_file, chunk_size=100)

        chunk_count = 0
        total_rows = 0

        for chunk in lazy_loader.iter_chunks():
            chunk_count += 1
            total_rows += len(chunk)
            self.assertLessEqual(len(chunk), 100)

        self.assertEqual(total_rows, 1000)
        self.assertGreater(chunk_count, 1)

    def test_parallel_processing_integration(self):
        """Test parallel processing integration."""
        # Create test data
        test_df = pd.DataFrame({
            'open': np.random.uniform(100, 200, 1000),
            'high': np.random.uniform(200, 300, 1000),
            'low': np.random.uniform(50, 100, 1000),
            'close': np.random.uniform(100, 200, 1000),
            'volume': np.random.uniform(1000, 10000, 1000)
        }, index=pd.date_range('2023-01-01', periods=1000, freq='H'))

        # Define processing function
        def process_chunk(chunk):
            chunk['sma_20'] = chunk['close'].rolling(20).mean()
            chunk['rsi'] = 100 - (100 / (1 + chunk['close'].pct_change().rolling(14).mean()))
            return chunk

        # Test parallel processing
        with self.performance_monitor.start_operation("parallel_processing"):
            processed_df = ParallelProcessor(max_workers=2).process_dataframe(
                test_df, process_chunk
            )

        self.assertIsNotNone(processed_df)
        self.assertIn('sma_20', processed_df.columns)
        self.assertIn('rsi', processed_df.columns)

    def test_streaming_integration(self):
        """Test streaming integration."""
        # Create stream configuration
        config = create_stream_config(
            url="wss://test.example.com",
            max_connections=2,
            reconnect_interval=5.0,
            message_timeout=30.0
        )

        # Create stream multiplexer
        multiplexer = get_stream_multiplexer()

        # Test stream management
        success = multiplexer.add_stream("test_stream", config)
        self.assertTrue(success)

        # Test stream start/stop
        start_success = multiplexer.start()
        self.assertTrue(start_success)

        multiplexer.stop()

    def test_cache_hierarchical_structure(self):
        """Test the hierarchical cache structure."""
        # Test different providers, symbols, intervals, and years
        test_cases = [
            ("binance", "BTCUSDT", "1h", 2023),
            ("yahoo", "AAPL", "1d", 2023),
            ("mock", "TEST", "5m", 2024),
        ]

        for provider, symbol, interval, year in test_cases:
            # Create test data
            test_df = pd.DataFrame({
                'open': [100.0, 101.0, 102.0],
                'high': [102.0, 103.0, 104.0],
                'low': [99.0, 100.0, 101.0],
                'close': [101.0, 102.0, 103.0],
                'volume': [1000, 1100, 1200]
            }, index=pd.date_range(f'{year}-01-01', periods=3, freq='D'))

            # Cache data
            cache_success = self.cache.put(
                test_df, provider, symbol, interval,
                start_date=datetime(year, 1, 1),
                end_date=datetime(year, 1, 3)
            )
            self.assertTrue(cache_success)

            # Verify cache structure
            cache_path = self.cache._get_cache_path(provider, symbol, interval, year)
            self.assertTrue(cache_path.exists())

            # Verify data file exists
            data_path = self.cache._get_data_path(cache_path)
            self.assertTrue(data_path.exists())

            # Verify metadata file exists
            metadata_path = self.cache._get_metadata_path(cache_path)
            self.assertTrue(metadata_path.exists())

            # Retrieve and verify data
            retrieved_df = self.cache.get(
                provider, symbol, interval,
                start_date=datetime(year, 1, 1),
                end_date=datetime(year, 1, 3)
            )

            self.assertIsNotNone(retrieved_df)
            pd.testing.assert_frame_equal(test_df, retrieved_df)

        # Test cache info
        for provider, symbol, interval, year in test_cases:
            info = self.cache.get_cache_info(provider, symbol, interval)
            self.assertEqual(info['provider'], provider)
            self.assertEqual(info['symbol'], symbol)
            self.assertEqual(info['interval'], interval)
            self.assertIn(year, info['years_available'])

    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        # Test invalid data handling
        invalid_df = pd.DataFrame({
            'open': [100.0, np.nan, 102.0],  # Invalid data
            'high': [102.0, 103.0, 104.0],
            'low': [99.0, 100.0, 101.0],
            'close': [101.0, 102.0, 103.0],
            'volume': [1000, 1100, 1200]
        })

        # Validation should catch invalid data
        is_valid, errors = validate_ohlcv_data(invalid_df)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)

        # Test cache error handling
        # Try to cache with invalid provider
        cache_success = self.cache.put(
            invalid_df, "", "TEST", "1h"  # Empty provider
        )
        self.assertFalse(cache_success)

        # Test factory error handling
        # Try to create non-existent data source
        non_existent_source = create_data_source("non_existent")
        self.assertIsNone(non_existent_source)

    def test_performance_monitoring(self):
        """Test performance monitoring integration."""
        # Test operation timing
        with self.performance_monitor.start_operation("test_operation") as metrics:
            time.sleep(0.1)  # Simulate work
            metrics.add_metric("rows_processed", 1000)
            metrics.add_metric("memory_used_mb", 50.5)

        # Test performance metrics retrieval
        metrics_data = self.performance_monitor.get_metrics()
        self.assertIn("test_operation", metrics_data)

        operation_metrics = metrics_data["test_operation"]
        self.assertGreater(operation_metrics.duration_ms, 0)
        self.assertEqual(operation_metrics.metrics["rows_processed"], 1000)
        self.assertEqual(operation_metrics.metrics["memory_used_mb"], 50.5)

        # Test performance summary
        summary = self.performance_monitor.get_performance_summary()
        self.assertIn("total_operations", summary)
        self.assertIn("avg_duration_ms", summary)
        self.assertIn("total_duration_ms", summary)


def run_phase4_integration_tests():
    """Run all Phase 4 integration tests."""
    print("Running Phase 4 Integration Tests...")
    print("=" * 50)

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPhase4Integration)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 50)
    print("Phase 4 Integration Test Results:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")

    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = run_phase4_integration_tests()
    sys.exit(0 if success else 1)
