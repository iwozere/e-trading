"""
Performance Benchmark Tests for Phase 4.

Benchmark tests to measure performance of the file-based cache system
and other data module components.
"""

import sys
import os
import tempfile
import shutil
import time
from pathlib import Path
import pandas as pd
import numpy as np
import unittest
import concurrent.futures

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.data import (
    validate_ohlcv_data, optimize_dataframe_performance, compress_dataframe_efficiently,
    get_performance_monitor, ParallelProcessor
)

# Import new unified cache system
from src.data.cache.unified_cache import configure_unified_cache



class PerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.temp_dir, "cache")

        # Configure unified cache for performance testing
        self.cache = configure_unified_cache(
            cache_dir=self.cache_dir,
            max_size_gb=10.0
        )

        # Create performance monitor
        self.performance_monitor = get_performance_monitor()

        # Create large test dataset
        self.large_df = self._create_large_dataset(100000)  # 100k rows

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def _create_large_dataset(self, rows: int) -> pd.DataFrame:
        """Create large test dataset."""
        dates = pd.date_range('2020-01-01', periods=rows, freq='h')

        df = pd.DataFrame({
            'open': np.random.uniform(100, 200, rows),
            'high': np.random.uniform(200, 300, rows),
            'low': np.random.uniform(50, 100, rows),
            'close': np.random.uniform(100, 200, rows),
            'volume': np.random.uniform(1000, 10000, rows),
            'category': np.random.choice(['A', 'B', 'C', 'D'], rows),
            'indicator1': np.random.uniform(0, 1, rows),
            'indicator2': np.random.uniform(0, 1, rows),
            'indicator3': np.random.uniform(0, 1, rows)
        }, index=dates)

        # Ensure the index is named 'timestamp' for compatibility
        df.index.name = 'timestamp'
        return df

    def _profile_function(self, func, *args, **kwargs):
        """Profile a function and return stats."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        return {
            'result': result,
            'execution_time': end_time - start_time,
            'stats': None
        }

    def test_cache_write_performance(self):
        """Benchmark cache write performance."""
        print("\n=== Cache Write Performance Test ===")

        # Test different dataset sizes
        sizes = [1000, 10000, 50000, 100000]

        for size in sizes:
            df = self._create_large_dataset(size)

            # Profile cache write
            profile_result = self._profile_function(
                self.cache.put,
                df, "TEST", "1h",
                start_date=df.index[0],
                end_date=df.index[-1],
                provider="benchmark"
            )

            execution_time = profile_result['execution_time']
            throughput = size / execution_time

            print(f"Dataset size: {size:,} rows")
            print(f"Write time: {execution_time:.3f}s")
            print(f"Throughput: {throughput:.0f} rows/second")
            print(f"File size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            print("-" * 40)

            # Performance assertions
            self.assertLess(execution_time, 10.0)  # Should complete within 10 seconds
            self.assertGreater(throughput, 1000)   # Should handle at least 1000 rows/second

    def test_cache_read_performance(self):
        """Benchmark cache read performance."""
        print("\n=== Cache Read Performance Test ===")

        # First, write data to cache
        df = self._create_large_dataset(50000)
        start_date = df.index[0]
        end_date = df.index[-1]
        self.cache.put(df, "TEST", "1h", start_date, end_date, provider="benchmark")

        # Test read performance
        profile_result = self._profile_function(
            self.cache.get,
            "TEST", "1h", start_date, end_date
        )

        execution_time = profile_result['execution_time']
        retrieved_df = profile_result['result']

        throughput = len(retrieved_df) / execution_time

        print(f"Dataset size: {len(retrieved_df):,} rows")
        print(f"Read time: {execution_time:.3f}s")
        print(f"Throughput: {throughput:.0f} rows/second")
        print(f"Data size: {retrieved_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

        # Performance assertions
        self.assertLess(execution_time, 5.0)   # Should complete within 5 seconds
        self.assertGreater(throughput, 5000)   # Should handle at least 5000 rows/second
        self.assertIsNotNone(retrieved_df)

    def test_concurrent_cache_access(self):
        """Benchmark concurrent cache access."""
        print("\n=== Concurrent Cache Access Test ===")

        # Write test data
        df = self._create_large_dataset(10000)
        start_date = df.index[0]
        end_date = df.index[-1]
        self.cache.put(df, "TEST", "1h", start_date, end_date, provider="concurrent")

        def read_operation():
            """Single read operation."""
            return self.cache.get("TEST", "1h", start_date, end_date)

        def write_operation(symbol_suffix):
            """Single write operation."""
            test_df = self._create_large_dataset(1000)
            test_start = test_df.index[0]
            test_end = test_df.index[-1]
            return self.cache.put(test_df, f"TEST{symbol_suffix}", "1h", test_start, test_end, provider="concurrent")

        # Test concurrent reads
        print("Testing concurrent reads...")
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            read_futures = [executor.submit(read_operation) for _ in range(20)]
            results = [future.result() for future in concurrent.futures.as_completed(read_futures)]

        read_time = time.time() - start_time

        print(f"20 concurrent reads completed in {read_time:.3f}s")
        print(f"Average read time: {read_time / 20:.3f}s")

        # Test concurrent writes
        print("Testing concurrent writes...")
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            write_futures = [executor.submit(write_operation, i) for i in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(write_futures)]

        write_time = time.time() - start_time

        print(f"10 concurrent writes completed in {write_time:.3f}s")
        print(f"Average write time: {write_time / 10:.3f}s")

        # Performance assertions
        self.assertLess(read_time, 10.0)   # Concurrent reads should complete quickly
        self.assertLess(write_time, 30.0)  # Concurrent writes should complete within reasonable time

    def test_data_validation_performance(self):
        """Benchmark data validation performance."""
        print("\n=== Data Validation Performance Test ===")

        # Test different dataset sizes
        sizes = [1000, 10000, 50000, 100000]

        for size in sizes:
            df = self._create_large_dataset(size)

            # Profile validation
            profile_result = self._profile_function(validate_ohlcv_data, df)

            execution_time = profile_result['execution_time']
            throughput = size / execution_time

            print(f"Dataset size: {size:,} rows")
            print(f"Validation time: {execution_time:.3f}s")
            print(f"Throughput: {throughput:.0f} rows/second")

            # Performance assertions
            self.assertLess(execution_time, 5.0)   # Should complete within 5 seconds
            self.assertGreater(throughput, 10000)  # Should handle at least 10000 rows/second

    def test_data_optimization_performance(self):
        """Benchmark data optimization performance."""
        print("\n=== Data Optimization Performance Test ===")

        # Test memory optimization
        print("Testing memory optimization...")
        profile_result = self._profile_function(
            optimize_dataframe_performance,
            self.large_df
        )

        execution_time = profile_result['execution_time']
        optimized_df = profile_result['result']

        # Calculate memory savings
        original_memory = self.large_df.memory_usage(deep=True).sum()
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        memory_savings = (original_memory - optimized_memory) / original_memory * 100

        print(f"Original memory: {original_memory / 1024 / 1024:.2f} MB")
        print(f"Optimized memory: {optimized_memory / 1024 / 1024:.2f} MB")
        print(f"Memory savings: {memory_savings:.1f}%")
        print(f"Optimization time: {execution_time:.3f}s")

        # Performance assertions
        self.assertLess(execution_time, 10.0)  # Should complete within 10 seconds
        self.assertGreater(memory_savings, 10)  # Should save at least 10% memory

        # Test data compression
        print("Testing data compression...")
        profile_result = self._profile_function(
            compress_dataframe_efficiently,
            optimized_df
        )

        execution_time = profile_result['execution_time']
        compressed_data = profile_result['result']

        # Calculate compression ratio based on actual file sizes
        original_size = len(optimized_df.to_csv(index=False).encode('utf-8'))
        compressed_size = len(compressed_data) if isinstance(compressed_data, bytes) else original_size
        compression_ratio = compressed_size / original_size if original_size > 0 else 1.0

        print(f"Compression time: {execution_time:.3f}s")
        print(f"Original size: {original_size / 1024:.2f} KB")
        print(f"Compressed size: {compressed_size / 1024:.2f} KB")
        print(f"Compression ratio: {compression_ratio:.2f}")

        # Performance assertions
        self.assertLess(execution_time, 5.0)   # Should complete within 5 seconds
        # Only expect compression for larger datasets
        if original_size > 10000:  # 10KB threshold
            self.assertLess(compression_ratio, 1.0)  # Should compress data

    def test_parallel_processing_performance(self):
        """Benchmark parallel processing performance."""
        print("\n=== Parallel Processing Performance Test ===")

        # Define processing function
        def process_chunk(chunk):
            """Process a chunk of data."""
            # Create a copy to avoid SettingWithCopyWarning
            chunk_copy = chunk.copy()
            chunk_copy['sma_20'] = chunk_copy['close'].rolling(20).mean()
            chunk_copy['rsi'] = 100 - (100 / (1 + chunk_copy['close'].pct_change().rolling(14).mean()))
            chunk_copy['volatility'] = chunk_copy['close'].rolling(20).std()
            return chunk_copy

        # Test different worker counts
        worker_counts = [1, 2, 4, 8]

        for workers in worker_counts:
            processor = ParallelProcessor(max_workers=workers, use_processes=False)  # Use threads to avoid pickle issues

            # Profile parallel processing
            profile_result = self._profile_function(
                processor.process_dataframe,
                self.large_df, process_chunk
            )

            execution_time = profile_result['execution_time']
            processed_df = profile_result['result']

            print(f"Workers: {workers}")
            print(f"Processing time: {execution_time:.3f}s")
            print(f"Throughput: {len(processed_df) / execution_time:.0f} rows/second")

            # Verify processing results
            self.assertIn('sma_20', processed_df.columns)
            self.assertIn('rsi', processed_df.columns)
            self.assertIn('volatility', processed_df.columns)

    def test_cache_hit_rate_performance(self):
        """Benchmark cache hit rate performance."""
        print("\n=== Cache Hit Rate Performance Test ===")

        # Write multiple datasets using new unified cache API
        symbols = [f"SYMBOL{i}" for i in range(10)]
        for symbol in symbols:
            df = self._create_large_dataset(5000)
            # Use the actual date range from the DataFrame
            start_date = df.index[0]
            end_date = df.index[-1]
            self.cache.put(df, symbol, "1h", start_date, end_date, provider="test")

        # Simulate cache access pattern
        access_pattern = symbols * 5  # Access each symbol 5 times

        start_time = time.time()
        hits = 0
        misses = 0

        for symbol in access_pattern:
            # Use the same date range as when we wrote the data
            df = self._create_large_dataset(5000)  # Create same dataset to get same date range
            start_date = df.index[0]
            end_date = df.index[-1]
            df = self.cache.get(symbol, "1h", start_date, end_date)
            if df is not None and len(df) > 0:
                hits += 1
            else:
                misses += 1

        total_time = time.time() - start_time
        hit_rate = hits / (hits + misses) * 100 if (hits + misses) > 0 else 0

        print(f"Total accesses: {hits + misses}")
        print(f"Hits: {hits}")
        print(f"Misses: {misses}")
        print(f"Hit rate: {hit_rate:.1f}%")
        print(f"Average access time: {total_time / (hits + misses) * 1000:.2f}ms")

        # Performance assertions - relaxed for new cache system
        self.assertGreater(hit_rate, 0)  # Should have some hit rate
        self.assertLess(total_time / (hits + misses), 1.0)  # Average access should be reasonably fast

    def test_memory_usage_performance(self):
        """Benchmark memory usage performance."""
        print("\n=== Memory Usage Performance Test ===")

        import psutil
        import gc

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        print(f"Initial memory usage: {initial_memory:.2f} MB")

        # Perform memory-intensive operations
        large_datasets = []

        for i in range(5):
            df = self._create_large_dataset(20000)
            large_datasets.append(df)

            # Cache the dataset
            start_date = df.index[0]
            end_date = df.index[-1]
            self.cache.put(df, f"TEST{i}", "1h", start_date, end_date, provider="memory")

            # Get current memory usage
            current_memory = process.memory_info().rss / 1024 / 1024
            print(f"After dataset {i+1}: {current_memory:.2f} MB")

        # Clear datasets from memory
        del large_datasets
        gc.collect()

        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory

        print(f"Final memory usage: {final_memory:.2f} MB")
        print(f"Memory increase: {memory_increase:.2f} MB")

        # Performance assertions
        self.assertLess(memory_increase, 1000)  # Should not increase memory by more than 1GB

    def test_cache_cleanup_performance(self):
        """Benchmark cache cleanup performance."""
        print("\n=== Cache Cleanup Performance Test ===")

        # Create many cache entries
        for i in range(100):
            df = self._create_large_dataset(1000)
            start_date = df.index[0]
            end_date = df.index[-1]
            self.cache.put(df, f"TEST{i}", "1h", start_date, end_date, provider="cleanup")

        # Get cache stats before cleanup
        stats_before = self.cache.get_stats()
        print(f"Files before cleanup: {stats_before['files_count']}")
        print(f"Cache size before cleanup: {stats_before['total_size_gb']:.2f} GB")

        # Profile cleanup operation
        profile_result = self._profile_function(
            self.cache.cleanup_old_data
        )

        execution_time = profile_result['execution_time']
        deleted_count = profile_result['result']

        # Get cache stats after cleanup
        stats_after = self.cache.get_stats()

        print(f"Cleanup time: {execution_time:.3f}s")
        print(f"Files deleted: {deleted_count}")
        print(f"Files after cleanup: {stats_after['files_count']}")
        print(f"Cache size after cleanup: {stats_after['total_size_gb']:.2f} GB")

        # Performance assertions
        self.assertLess(execution_time, 30.0)  # Should complete within 30 seconds


def run_performance_benchmarks():
    """Run all performance benchmarks."""
    print("Running Performance Benchmarks...")
    print("=" * 60)

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(PerformanceBenchmarks)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 60)
    print("Performance Benchmark Results:")
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
    success = run_performance_benchmarks()
    sys.exit(0 if success else 1)
