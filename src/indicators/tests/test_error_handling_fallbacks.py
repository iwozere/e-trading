"""
Comprehensive error handling and fallback mechanism tests.

Tests adapter failures, data quality issues, and recovery strategies.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from datetime import datetime
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.indicators.service import IndicatorService, IndicatorServiceError, DataError
from src.indicators.models import IndicatorBatchConfig, IndicatorSpec


class TestErrorHandlingAndFallbacks:
    """Test comprehensive error handling and fallback mechanisms."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [102, 103, 104, 105, 106],
            'low': [99, 100, 101, 102, 103],
            'close': [101, 102, 103, 104, 105],
            'volume': [1000, 1100, 1200, 1300, 1400]
        }, index=pd.date_range('2024-01-01', periods=5, freq='D', tz='UTC'))

    def test_adapter_computation_failure_recovery(self, sample_data):
        """Test service recovers from adapter computation failures."""
        service = IndicatorService()

        config = IndicatorBatchConfig(
            indicators=[IndicatorSpec(name="rsi", output="rsi")]
        )

        # Mock primary adapter failure
        with patch.object(service._ta_lib_adapter, 'compute', side_effect=Exception("TA-Lib failed")):
            # Should attempt fallback to pandas_ta
            with patch.object(service._pandas_ta_adapter, 'supports', return_value=True):
                with patch.object(service._pandas_ta_adapter, 'compute') as mock_compute:
                    mock_compute.return_value = {"value": pd.Series([50.0] * 5)}

                    result = service.compute(sample_data, config)

                    assert isinstance(result, pd.DataFrame)
                    assert "rsi" in result.columns
                    mock_compute.assert_called_once()

    def test_all_adapters_fail(self, sample_data):
        """Test behavior when all adapters fail."""
        service = IndicatorService()

        config = IndicatorBatchConfig(
            indicators=[IndicatorSpec(name="rsi", output="rsi")]
        )

        # Mock all adapters to fail
        with patch.object(service._ta_lib_adapter, 'compute', side_effect=Exception("TA-Lib failed")):
            with patch.object(service._pandas_ta_adapter, 'compute', side_effect=Exception("pandas_ta failed")):

                with pytest.raises(IndicatorServiceError):
                    service.compute(sample_data, config)

    def test_data_quality_issues(self):
        """Test handling of various data quality issues."""
        service = IndicatorService()

        # Test with NaN values
        nan_data = pd.DataFrame({
            'open': [100, np.nan, 102, 103, 104],
            'high': [102, 103, np.nan, 105, 106],
            'low': [99, 100, 101, np.nan, 103],
            'close': [101, 102, 103, 104, np.nan],
            'volume': [1000, 1100, 1200, 1300, 1400]
        }, index=pd.date_range('2024-01-01', periods=5, freq='D', tz='UTC'))

        config = IndicatorBatchConfig(
            indicators=[IndicatorSpec(name="rsi", output="rsi")]
        )

        # Should handle NaN values gracefully
        result = service.compute(nan_data, config)
        assert isinstance(result, pd.DataFrame)
        assert "rsi" in result.columns

    def test_invalid_ohlc_relationships(self):
        """Test handling of invalid OHLC relationships."""
        service = IndicatorService()

        # Create data with invalid OHLC (high < low)
        invalid_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [99, 100, 101],  # High < Open (invalid)
            'low': [102, 103, 104],  # Low > Open (invalid)
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3, freq='D', tz='UTC'))

        config = IndicatorBatchConfig(
            indicators=[IndicatorSpec(name="atr", output="atr")]
        )

        # Should either correct the data or handle gracefully
        try:
            result = service.compute(invalid_data, config)
            assert isinstance(result, pd.DataFrame)
        except (ValueError, DataError):
            # Acceptable to raise error for invalid data
            pass

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        service = IndicatorService()

        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        config = IndicatorBatchConfig(
            indicators=[IndicatorSpec(name="rsi", output="rsi")]
        )

        with pytest.raises((ValueError, DataError)):
            service.compute(empty_df, config)

    def test_insufficient_data_periods(self):
        """Test handling when data has insufficient periods for indicator."""
        service = IndicatorService()

        # Only 3 data points, but RSI needs 14+
        insufficient_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [102, 103, 104],
            'low': [99, 100, 101],
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3, freq='D', tz='UTC'))

        config = IndicatorBatchConfig(
            indicators=[IndicatorSpec(name="rsi", output="rsi", params={"timeperiod": 14})]
        )

        result = service.compute(insufficient_data, config)

        # Should return DataFrame with mostly NaN values
        assert isinstance(result, pd.DataFrame)
        assert "rsi" in result.columns
        assert result["rsi"].isna().sum() >= 2  # Most values should be NaN

    def test_missing_required_columns(self):
        """Test handling when required columns are missing."""
        service = IndicatorService()

        # Missing 'high' column required for ATR
        incomplete_data = pd.DataFrame({
            'open': [100, 101, 102],
            'low': [99, 100, 101],
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3, freq='D', tz='UTC'))

        config = IndicatorBatchConfig(
            indicators=[IndicatorSpec(name="atr", output="atr")]
        )

        with pytest.raises((KeyError, DataError)):
            service.compute(incomplete_data, config)

    def test_invalid_parameter_handling(self, sample_data):
        """Test handling of invalid indicator parameters."""
        service = IndicatorService()

        # Invalid timeperiod (negative)
        config = IndicatorBatchConfig(
            indicators=[IndicatorSpec(name="rsi", output="rsi", params={"timeperiod": -1})]
        )

        with pytest.raises((ValueError, IndicatorServiceError)):
            service.compute(sample_data, config)

    def test_unsupported_indicator_handling(self, sample_data):
        """Test handling of unsupported indicators."""
        service = IndicatorService()

        config = IndicatorBatchConfig(
            indicators=[IndicatorSpec(name="nonexistent_indicator", output="fake")]
        )

        with pytest.raises((ValueError, IndicatorServiceError)):
            service.compute(sample_data, config)

    def test_adapter_timeout_handling(self, sample_data):
        """Test handling of adapter timeouts."""
        service = IndicatorService()

        config = IndicatorBatchConfig(
            indicators=[IndicatorSpec(name="rsi", output="rsi")]
        )

        # Mock slow adapter
        def slow_compute(*args, **kwargs):
            import time
            time.sleep(2)  # Simulate slow computation
            return {"value": pd.Series([50.0] * 5)}

        with patch.object(service._ta_lib_adapter, 'compute', side_effect=slow_compute):
            # Should handle timeout appropriately
            start_time = datetime.now()
            try:
                result = service.compute(sample_data, config, timeout=1.0)
                # If completed, should be within reasonable time
                end_time = datetime.now()
                assert (end_time - start_time).total_seconds() < 3.0
            except (TimeoutError, IndicatorServiceError):
                # Acceptable to timeout
                pass

    def test_memory_pressure_handling(self):
        """Test handling of memory pressure during computation."""
        service = IndicatorService()

        # Create very large dataset
        large_size = 10000
        large_data = pd.DataFrame({
            'open': np.random.randn(large_size) + 100,
            'high': np.random.randn(large_size) + 102,
            'low': np.random.randn(large_size) + 98,
            'close': np.random.randn(large_size) + 101,
            'volume': np.random.randint(1000, 10000, large_size)
        }, index=pd.date_range('2020-01-01', periods=large_size, freq='D', tz='UTC'))

        config = IndicatorBatchConfig(
            indicators=[
                IndicatorSpec(name="rsi", output="rsi"),
                IndicatorSpec(name="ema", output="ema"),
                IndicatorSpec(name="macd", output="macd")
            ]
        )

        # Should handle large dataset without memory issues
        try:
            result = service.compute(large_data, config)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == large_size
        except MemoryError:
            # Acceptable to fail with memory error on very large datasets
            pass

    def test_concurrent_failure_handling(self, sample_data):
        """Test handling of failures in concurrent operations."""
        service = IndicatorService()

        # Mock some adapters to fail randomly
        def random_failure(*args, **kwargs):
            import random
            if random.random() < 0.5:
                raise Exception("Random failure")
            return {"value": pd.Series([50.0] * 5)}

        config = IndicatorBatchConfig(
            indicators=[
                IndicatorSpec(name="rsi", output="rsi"),
                IndicatorSpec(name="ema", output="ema"),
                IndicatorSpec(name="sma", output="sma")
            ]
        )

        with patch.object(service._ta_lib_adapter, 'compute', side_effect=random_failure):
            # Should handle partial failures gracefully
            try:
                result = service.compute(sample_data, config)
                # Some indicators might succeed
                assert isinstance(result, pd.DataFrame)
            except IndicatorServiceError:
                # Acceptable if all fail
                pass

    def test_circuit_breaker_pattern(self, sample_data):
        """Test circuit breaker pattern for repeated failures."""
        service = IndicatorService()

        config = IndicatorBatchConfig(
            indicators=[IndicatorSpec(name="rsi", output="rsi")]
        )

        # Mock repeated failures
        failure_count = 0
        def counting_failure(*args, **kwargs):
            nonlocal failure_count
            failure_count += 1
            raise Exception(f"Failure #{failure_count}")

        with patch.object(service._ta_lib_adapter, 'compute', side_effect=counting_failure):
            with patch.object(service._pandas_ta_adapter, 'compute', side_effect=counting_failure):

                # Multiple attempts should eventually trigger circuit breaker
                for i in range(3):
                    with pytest.raises(IndicatorServiceError):
                        service.compute(sample_data, config)

                # Circuit breaker should prevent excessive retries
                assert failure_count <= 10  # Should not retry indefinitely


if __name__ == '__main__':
    pytest.main([__file__, '-v'])