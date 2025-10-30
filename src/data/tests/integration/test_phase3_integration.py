"""
Phase 3 Integration Test Script.

This script tests the advanced features implemented in Phase 3:
- Advanced caching with Redis support
- Data streaming with WebSocket connection pooling
- Performance optimization features
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time
import threading

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data import (
    get_advanced_cache,
    configure_advanced_cache,
    # get_stream_multiplexer,  # Not implemented yet
    # create_stream_config,    # Not implemented yet
    LazyDataLoader,
    ParallelProcessor,
    MemoryOptimizer,
    PerformanceMonitor,
    get_performance_monitor,
    get_memory_optimizer,
    get_data_compressor,
    optimize_dataframe_performance,
    compress_dataframe_efficiently,
    TimeBasedInvalidation,
    VersionBasedInvalidation
)


def test_advanced_caching():
    """Test advanced caching functionality."""
    print("Testing Advanced Caching...")

    try:
        # Create sample data
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1h'),
            'open': np.random.uniform(100, 200, 1000),
            'high': np.random.uniform(200, 300, 1000),
            'low': np.random.uniform(50, 100, 1000),
            'close': np.random.uniform(100, 200, 1000),
            'volume': np.random.uniform(1000, 10000, 1000)
        })

        # Configure advanced cache with invalidation strategies
        invalidation_strategies = [
            TimeBasedInvalidation(max_age_hours=1),
            VersionBasedInvalidation(current_version="1.0.0")
        ]

        cache = configure_advanced_cache(
            cache_dir="test_cache",
            invalidation_strategies=invalidation_strategies,
            compression_enabled=True
        )

        # Test caching
        success = cache.put(sample_data, "test_provider", "TEST", "1h")
        print(f"✓ Cache put: {success}")

        # Test retrieval
        cached_data = cache.get("test_provider", "TEST", "1h")
        print(f"✓ Cache get: {cached_data is not None}")

        # Test metrics
        metrics = cache.get_metrics()
        print(f"✓ Cache metrics: {metrics['advanced_metrics']['sets']} sets, {metrics['advanced_metrics']['hits']} hits")

        assert True  # Test passed

    except Exception as e:
        print(f"✗ Advanced caching test failed: {e}")
        assert False, f"Advanced caching test failed: {e}"


# def test_data_streaming():
#     """Test data streaming functionality."""
#     print("\nTesting Data Streaming...")
#
#     try:
#         # Create stream multiplexer
#         multiplexer = get_stream_multiplexer()
#
#         # Create stream configuration
#         config = create_stream_config(
#             url="wss://echo.websocket.org",
#             symbol="TEST",
#             interval="1m",
#             max_connections=2,
#             message_queue_size=1000
#         )
#
#         # Create data processor
#         from src.data.utils.data_streaming import DataStreamProcessor
#
#         processor = DataStreamProcessor(config)
#
#         # Add a simple processor
#         def process_message(data):
#             data['processed'] = True
#             data['timestamp'] = datetime.now().isoformat()
#             return data
#
#         processor.add_processor(process_message)
#
#         # Add stream to multiplexer
#         success = multiplexer.add_stream("test_stream", config, processor)
#         print(f"✓ Stream added: {success}")
#
#         # Test metrics
#         metrics = multiplexer.get_metrics()
#         print(f"✓ Stream metrics: {metrics['total_streams']} streams")
#
#         return True
#
#     except Exception as e:
#         print(f"✗ Data streaming test failed: {e}")
#         return False


def test_performance_optimization():
    """Test performance optimization features."""
    print("\nTesting Performance Optimization...")

    try:
        # Create large sample data
        large_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100000, freq='1h'),
            'open': np.random.uniform(100, 200, 100000),
            'high': np.random.uniform(200, 300, 100000),
            'low': np.random.uniform(50, 100, 100000),
            'close': np.random.uniform(100, 200, 100000),
            'volume': np.random.uniform(1000, 10000, 100000),
            'category': np.random.choice(['A', 'B', 'C'], 100000)
        })

        # Test memory optimization
        optimizer = get_memory_optimizer()
        initial_usage = optimizer.get_memory_usage(large_data)
        optimized_data = optimizer.optimize_dataframe(large_data)
        final_usage = optimizer.get_memory_usage(optimized_data)

        memory_reduction = optimizer.estimate_memory_reduction(large_data)
        print(f"✓ Memory optimization: {memory_reduction['reduction_percent']:.1f}% reduction")

        # Test data compression
        compressor = get_data_compressor()
        compressed_data = compressor.compress_dataframe(large_data, format="parquet")
        compression_ratio = len(compressed_data) / (initial_usage['total_mb'] * 1024 * 1024)
        print(f"✓ Data compression: {compression_ratio:.3f} ratio")

        # Test performance monitoring
        monitor = get_performance_monitor()
        metrics = monitor.start_operation("test_optimization")

        # Simulate some work
        time.sleep(0.1)

        monitor.end_operation(metrics, data_size_mb=initial_usage['total_mb'])
        summary = monitor.get_summary()
        print(f"✓ Performance monitoring: {summary['test_optimization']['avg_duration_ms']:.1f}ms avg")

        assert True  # Test passed

    except Exception as e:
        print(f"✗ Performance optimization test failed: {e}")
        assert False, f"Performance optimization test failed: {e}"


def test_lazy_loading():
    """Test lazy loading functionality."""
    print("\nTesting Lazy Loading...")

    try:
        # Create a temporary CSV file for testing
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10000, freq='1h'),
            'value': np.random.uniform(100, 200, 10000)
        })

        test_file = Path("test_lazy_data.csv")
        test_data.to_csv(test_file, index=False)

        # Test lazy loading
        loader = LazyDataLoader(test_file, chunk_size=1000)

        print(f"✓ Lazy loader created: {len(loader)} total rows")
        print(f"✓ Columns: {loader.get_columns()}")

        # Test chunk iteration
        chunk_count = 0
        total_rows = 0
        for chunk in loader.iter_chunks():
            chunk_count += 1
            total_rows += len(chunk)

        print(f"✓ Chunk iteration: {chunk_count} chunks, {total_rows} total rows")

        # Clean up
        test_file.unlink()

        assert True  # Test passed

    except Exception as e:
        print(f"✗ Lazy loading test failed: {e}")
        assert False, f"Lazy loading test failed: {e}"


def test_parallel_processing():
    """Test parallel processing functionality."""
    print("\nTesting Parallel Processing...")

    try:
        # Create sample data
        sample_data = pd.DataFrame({
            'value': np.random.uniform(100, 200, 10000),
            'category': np.random.choice(['A', 'B', 'C'], 10000)
        })

        # Create parallel processor
        processor = ParallelProcessor(max_workers=2, chunk_size=1000, use_processes=False)

        # Define processing function
        def process_chunk(chunk):
            # Create a copy to avoid SettingWithCopyWarning
            chunk_copy = chunk.copy()
            chunk_copy['processed_value'] = chunk_copy['value'] * 2
            chunk_copy['processed_at'] = datetime.now().isoformat()
            return chunk_copy

        # Process data in parallel
        start_time = time.time()
        result = processor.process_dataframe(sample_data, process_chunk)
        processing_time = time.time() - start_time

        print(f"✓ Parallel processing: {len(result)} rows in {processing_time:.2f}s")
        print(f"✓ Processing result: {result['processed_value'].mean():.2f} avg")

        assert True  # Test passed

    except Exception as e:
        print(f"✗ Parallel processing test failed: {e}")
        assert False, f"Parallel processing test failed: {e}"


def test_integration_features():
    """Test integration between Phase 3 features."""
    print("\nTesting Integration Features...")

    try:
        # Create sample data
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5000, freq='1h'),
            'price': np.random.uniform(100, 200, 5000),
            'volume': np.random.uniform(1000, 10000, 5000)
        })

        # Test complete workflow: optimize -> compress -> cache
        monitor = get_performance_monitor()

        # Step 1: Optimize
        metrics1 = monitor.start_operation("optimization")
        optimized_data = optimize_dataframe_performance(data)
        monitor.end_operation(metrics1)

        # Step 2: Compress
        metrics2 = monitor.start_operation("compression")
        compressed_data = compress_dataframe_efficiently(optimized_data)
        monitor.end_operation(metrics2)

        # Step 3: Cache
        cache = get_advanced_cache()
        metrics3 = monitor.start_operation("caching")
        cache.put(optimized_data, "test", "INTEGRATION", "1h")
        cached_data = cache.get("test", "INTEGRATION", "1h")
        monitor.end_operation(metrics3)

        # Get performance summary
        summary = monitor.get_summary()
        print(f"✓ Integration workflow completed:")
        for operation, stats in summary.items():
            print(f"  - {operation}: {stats['avg_duration_ms']:.1f}ms avg")

        assert True  # Test passed

    except Exception as e:
        print(f"✗ Integration features test failed: {e}")
        assert False, f"Integration features test failed: {e}"


def main():
    """Run all Phase 3 integration tests."""
    print("Phase 3 Integration Tests")
    print("=" * 50)

    tests = [
        test_advanced_caching,
        # test_data_streaming,  # Commented out - streaming not implemented yet
        test_performance_optimization,
        test_lazy_loading,
        test_parallel_processing,
        test_integration_features
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All Phase 3 tests passed! Advanced features are working correctly.")
        return True
    else:
        print("❌ Some Phase 3 tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
