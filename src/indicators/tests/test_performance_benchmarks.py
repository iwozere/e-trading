"""
Comprehensive performance benchmarks for unified indicator service.

Benchmarks unified service against legacy implementations, tests batch processing
performance and scalability, and measures memory usage and concurrent request handling.
"""

import pytest
import pandas as pd
import numpy as np
import time
import asyncio
import psutil
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.indicators.service import IndicatorService
from src.indicators.models import IndicatorBatchConfig, IndicatorSpec, TickerIndicatorsRequest


class TestPerformanceBenchmarks:
    """Performance benchmarks for the unified indicator service."""

    @pytest.fixture
    def small_dataset(self):
        """Small dataset for quick benchmarks (50 days)."""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D', tz='UTC')
        np.random.seed(42)

        close_prices = 100 * np.exp(np.cumsum(np.random.randn(50) * 0.02))

        return pd.DataFrame({
            'open': close_prices * (1 + np.random.randn(50) * 0.005),
            'high': close_prices * (1 + np.abs(np.random.randn(50)) * 0.01),
            'low': close_prices * (1 - np.abs(np.random.randn(50)) * 0.01),
            'close': close_prices,
            'volume': np.random.randint(1000000, 10000000, 50)
        }, index=dates)

    @pytest.fixture
    def medium_dataset(self):
        """Medium dataset for realistic benchmarks (252 days - 1 year)."""
        dates = pd.date_range(start='2023-01-01', periods=252, freq='B', tz='UTC')
        np.random.seed(42)

        close_prices = 100 * np.exp(np.cumsum(np.random.randn(252) * 0.015))

        return pd.DataFrame({
            'open': close_prices * (1 + np.random.randn(252) * 0.003),
            'high': close_prices * (1 + np.abs(np.random.randn(252)) * 0.008),
            'low': close_prices * (1 - np.abs(np.random.randn(252)) * 0.008),
            'close': close_prices,
            'volume': np.random.randint(1000000, 50000000, 252)
        }, index=dates)

    @pytest.fixture
    def large_dataset(self):
        """Large dataset for stress testing (1260 days - 5 years)."""
        dates = pd.date_range(start='2019-01-01', periods=1260, freq='B', tz='UTC')
        np.random.seed(42)

        close_prices = 100 * np.exp(np.cumsum(np.random.randn(1260) * 0.012))

        return pd.DataFrame({
            'open': close_prices * (1 + np.random.randn(1260) * 0.002),
            'high': close_prices * (1 + np.abs(np.random.randn(1260)) * 0.006),
            'low': close_prices * (1 - np.abs(np.random.randn(1260)) * 0.006),
            'close': close_prices,
            'volume': np.random.randint(1000000, 100000000, 1260)
        }, index=dates)

    def measure_execution_time(self, func, *args, **kwargs):
        """Measure execution time of a function."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time

    def measure_memory_usage(self, func, *args, **kwargs):
        """Measure memory usage during function execution."""
        process = psutil.Process(os.getpid())

        # Get initial memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Execute function
        result = func(*args, **kwargs)

        # Get peak memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = peak_memory - initial_memory

        return result, memory_delta

    def test_single_indicator_performance(self, medium_dataset):
        """Benchmark single indicator computation performance."""
        service = IndicatorService()

        indicators_to_test = [
            ("rsi", {"timeperiod": 14}),
            ("ema", {"timeperiod": 20}),
            ("sma", {"timeperiod": 20}),
            ("macd", {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}),
            ("bbands", {"timeperiod": 20}),
            ("atr", {"timeperiod": 14}),
            ("adx", {"timeperiod": 14}),
            ("stoch", {"fastk_period": 14, "slowk_period": 3, "slowd_period": 3})
        ]

        performance_results = {}

        for indicator_name, params in indicators_to_test:
            config = IndicatorBatchConfig(
                indicators=[IndicatorSpec(name=indicator_name, output=indicator_name, params=params)]
            )

            result, execution_time = self.measure_execution_time(
                service.compute, medium_dataset, config
            )

            performance_results[indicator_name] = {
                'execution_time': execution_time,
                'data_points': len(medium_dataset),
                'throughput': len(medium_dataset) / execution_time if execution_time > 0 else float('inf')
            }

            # Performance assertions
            assert execution_time < 1.0, f"{indicator_name} took too long: {execution_time:.3f}s"
            assert isinstance(result, pd.DataFrame)
            assert indicator_name in result.columns or any(indicator_name in col for col in result.columns)

        # Print performance summary
        print("\n=== Single Indicator Performance ===")
        for indicator, metrics in performance_results.items():
            print(f"{indicator:10s}: {metrics['execution_time']:.3f}s ({metrics['throughput']:.0f} points/s)")

    def test_multiple_indicators_performance(self, medium_dataset):
        """Benchmark multiple indicators computed together."""
        service = IndicatorService()

        # Test with increasing number of indicators
        indicator_sets = [
            ["rsi"],
            ["rsi", "ema"],
            ["rsi", "ema", "macd"],
            ["rsi", "ema", "macd", "bbands"],
            ["rsi", "ema", "macd", "bbands", "atr", "adx", "stoch"]
        ]

        performance_results = {}

        for i, indicators in enumerate(indicator_sets):
            config = IndicatorBatchConfig(
                indicators=[IndicatorSpec(name=ind, output=ind) for ind in indicators]
            )

            result, execution_time = self.measure_execution_time(
                service.compute, medium_dataset, config
            )

            performance_results[f"{len(indicators)}_indicators"] = {
                'execution_time': execution_time,
                'indicator_count': len(indicators),
                'throughput_per_indicator': execution_time / len(indicators) if len(indicators) > 0 else 0
            }

            # Performance should scale reasonably
            assert execution_time < len(indicators) * 0.5, f"Too slow for {len(indicators)} indicators: {execution_time:.3f}s"

        print("\n=== Multiple Indicators Performance ===")
        for test_name, metrics in performance_results.items():
            print(f"{test_name:15s}: {metrics['execution_time']:.3f}s ({metrics['throughput_per_indicator']:.3f}s per indicator)")

    def test_dataset_size_scaling(self, small_dataset, medium_dataset, large_dataset):
        """Test performance scaling with dataset size."""
        service = IndicatorService()

        datasets = [
            ("small", small_dataset),
            ("medium", medium_dataset),
            ("large", large_dataset)
        ]

        config = IndicatorBatchConfig(
            indicators=[
                IndicatorSpec(name="rsi", output="rsi"),
                IndicatorSpec(name="ema", output="ema"),
                IndicatorSpec(name="macd", output="macd")
            ]
        )

        scaling_results = {}

        for dataset_name, dataset in datasets:
            result, execution_time = self.measure_execution_time(
                service.compute, dataset, config
            )

            scaling_results[dataset_name] = {
                'execution_time': execution_time,
                'data_points': len(dataset),
                'throughput': len(dataset) / execution_time if execution_time > 0 else float('inf')
            }

            # Should complete within reasonable time
            max_time = len(dataset) * 0.001  # 1ms per data point max
            assert execution_time < max_time, f"{dataset_name} dataset too slow: {execution_time:.3f}s"

        print("\n=== Dataset Size Scaling ===")
        for dataset_name, metrics in scaling_results.items():
            print(f"{dataset_name:8s}: {metrics['data_points']:4d} points, {metrics['execution_time']:.3f}s ({metrics['throughput']:.0f} points/s)")

    def test_memory_usage_scaling(self, small_dataset, medium_dataset, large_dataset):
        """Test memory usage with different dataset sizes."""
        service = IndicatorService()

        datasets = [
            ("small", small_dataset),
            ("medium", medium_dataset),
            ("large", large_dataset)
        ]

        config = IndicatorBatchConfig(
            indicators=[
                IndicatorSpec(name="rsi", output="rsi"),
                IndicatorSpec(name="ema", output="ema"),
                IndicatorSpec(name="macd", output="macd"),
                IndicatorSpec(name="bbands", output="bbands")
            ]
        )

        memory_results = {}

        for dataset_name, dataset in datasets:
            result, memory_delta = self.measure_memory_usage(
                service.compute, dataset, config
            )

            memory_results[dataset_name] = {
                'memory_delta': memory_delta,
                'data_points': len(dataset),
                'memory_per_point': memory_delta / len(dataset) if len(dataset) > 0 else 0
            }

            # Memory usage should be reasonable
            assert memory_delta < 100, f"{dataset_name} dataset uses too much memory: {memory_delta:.1f}MB"

        print("\n=== Memory Usage Scaling ===")
        for dataset_name, metrics in memory_results.items():
            print(f"{dataset_name:8s}: {metrics['memory_delta']:6.1f}MB ({metrics['memory_per_point']:.3f}MB per point)")

    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, medium_dataset):
        """Test batch processing performance with multiple tickers."""
        service = IndicatorService()

        ticker_counts = [1, 5, 10, 20]

        with patch('src.common.get_ohlcv', return_value=medium_dataset):
            batch_results = {}

            for ticker_count in ticker_counts:
                tickers = [f"TICKER_{i}" for i in range(ticker_count)]

                start_time = time.perf_counter()
                results = await service.compute_batch(
                    tickers=tickers,
                    indicators=["rsi", "ema", "macd"],
                    timeframe="1D",
                    period="1Y"
                )
                end_time = time.perf_counter()

                execution_time = end_time - start_time

                batch_results[ticker_count] = {
                    'execution_time': execution_time,
                    'throughput': ticker_count / execution_time if execution_time > 0 else float('inf'),
                    'results_count': len(results)
                }

                # Should complete all tickers
                assert len(results) == ticker_count

                # Should be reasonably fast
                assert execution_time < ticker_count * 0.5, f"Batch processing too slow: {execution_time:.3f}s for {ticker_count} tickers"

        print("\n=== Batch Processing Performance ===")
        for ticker_count, metrics in batch_results.items():
            print(f"{ticker_count:2d} tickers: {metrics['execution_time']:.3f}s ({metrics['throughput']:.1f} tickers/s)")

    def test_concurrent_request_handling(self, medium_dataset):
        """Test performance under concurrent requests."""
        service = IndicatorService()

        config = IndicatorBatchConfig(
            indicators=[
                IndicatorSpec(name="rsi", output="rsi"),
                IndicatorSpec(name="ema", output="ema")
            ]
        )

        def compute_indicators():
            return service.compute(medium_dataset, config)

        # Test with different concurrency levels
        concurrency_levels = [1, 2, 4, 8]

        for concurrency in concurrency_levels:
            start_time = time.perf_counter()

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(compute_indicators) for _ in range(concurrency)]
                results = [future.result() for future in as_completed(futures)]

            end_time = time.perf_counter()
            execution_time = end_time - start_time

            # All requests should complete successfully
            assert len(results) == concurrency
            for result in results:
                assert isinstance(result, pd.DataFrame)
                assert "rsi" in result.columns or any("rsi" in col for col in result.columns)

            # Should handle concurrent requests efficiently
            max_time = concurrency * 0.5  # Allow 0.5s per concurrent request
            assert execution_time < max_time, f"Concurrent processing too slow: {execution_time:.3f}s for {concurrency} requests"

            print(f"Concurrency {concurrency:2d}: {execution_time:.3f}s total ({execution_time/concurrency:.3f}s per request)")

    def test_adapter_performance_comparison(self, medium_dataset):
        """Compare performance between different adapters."""
        from src.indicators.adapters.ta_lib_adapter import TaLibAdapter
        from src.indicators.adapters.pandas_ta_adapter import PandasTaAdapter

        adapters = [
            ("TA-Lib", TaLibAdapter()),
            ("pandas-ta", PandasTaAdapter())
        ]

        inputs = {
            'close': medium_dataset['close'],
            'high': medium_dataset['high'],
            'low': medium_dataset['low'],
            'volume': medium_dataset['volume']
        }

        # Test common indicators
        indicators_to_test = [
            ("rsi", {"timeperiod": 14}, {"length": 14}),
            ("ema", {"timeperiod": 20}, {"length": 20}),
            ("sma", {"timeperiod": 20}, {"length": 20})
        ]

        adapter_results = {}

        for adapter_name, adapter in adapters:
            adapter_results[adapter_name] = {}

            for indicator_name, ta_params, pta_params in indicators_to_test:
                if adapter.supports(indicator_name):
                    params = ta_params if "TA-Lib" in adapter_name else pta_params

                    result, execution_time = self.measure_execution_time(
                        adapter.compute, indicator_name, medium_dataset, inputs, params
                    )

                    adapter_results[adapter_name][indicator_name] = {
                        'execution_time': execution_time,
                        'throughput': len(medium_dataset) / execution_time if execution_time > 0 else float('inf')
                    }

        print("\n=== Adapter Performance Comparison ===")
        for indicator_name, _, _ in indicators_to_test:
            print(f"\n{indicator_name.upper()}:")
            for adapter_name in adapter_results:
                if indicator_name in adapter_results[adapter_name]:
                    metrics = adapter_results[adapter_name][indicator_name]
                    print(f"  {adapter_name:12s}: {metrics['execution_time']:.3f}s ({metrics['throughput']:.0f} points/s)")

    def test_recommendation_engine_performance(self, medium_dataset):
        """Test recommendation engine performance."""
        service = IndicatorService()

        request = TickerIndicatorsRequest(
            ticker="AAPL",
            indicators=["rsi", "ema", "macd", "bbands"],
            include_recommendations=True
        )

        with patch('src.common.get_ohlcv', return_value=medium_dataset):
            # Test with recommendations
            result_with_rec, time_with_rec = self.measure_execution_time(
                lambda: asyncio.run(service.compute_for_ticker(request))
            )

            # Test without recommendations
            request.include_recommendations = False
            result_without_rec, time_without_rec = self.measure_execution_time(
                lambda: asyncio.run(service.compute_for_ticker(request))
            )

            # Recommendations should not add significant overhead
            overhead = time_with_rec - time_without_rec
            assert overhead < 0.1, f"Recommendation overhead too high: {overhead:.3f}s"

            print(f"\n=== Recommendation Engine Performance ===")
            print(f"Without recommendations: {time_without_rec:.3f}s")
            print(f"With recommendations:    {time_with_rec:.3f}s")
            print(f"Overhead:                {overhead:.3f}s ({overhead/time_without_rec*100:.1f}%)")

    def test_configuration_loading_performance(self):
        """Test configuration loading and parameter retrieval performance."""
        from src.indicators.config_manager import UnifiedConfigManager

        # Test config manager initialization
        start_time = time.perf_counter()
        config_manager = UnifiedConfigManager()
        init_time = time.perf_counter() - start_time

        # Test parameter retrieval
        indicators = ["rsi", "ema", "macd", "bbands", "atr", "adx", "stoch"]

        start_time = time.perf_counter()
        for indicator in indicators:
            params = config_manager.get_indicator_parameters(indicator)
            assert isinstance(params, dict)
        retrieval_time = time.perf_counter() - start_time

        # Should be very fast
        assert init_time < 0.1, f"Config initialization too slow: {init_time:.3f}s"
        assert retrieval_time < 0.01, f"Parameter retrieval too slow: {retrieval_time:.3f}s"

        print(f"\n=== Configuration Performance ===")
        print(f"Initialization: {init_time:.3f}s")
        print(f"Parameter retrieval: {retrieval_time:.3f}s ({len(indicators)} indicators)")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])