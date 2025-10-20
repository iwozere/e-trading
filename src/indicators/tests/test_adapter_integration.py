"""
Comprehensive integration tests for all indicator adapters.

Tests adapters with real market data, cross-adapter consistency,
error handling, and fallback mechanisms.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, Mock
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.indicators.adapters.ta_lib_adapter import TaLibAdapter
from src.indicators.adapters.pandas_ta_adapter import PandasTaAdapter
from src.indicators.adapters.fundamentals_adapter import FundamentalsAdapter
from src.indicators.adapters.base import BaseAdapter


class TestAdapterIntegration:
    """Integration tests for adapter functionality with real-world scenarios."""

    @pytest.fixture
    def realistic_market_data(self):
        """Create realistic market data for integration testing."""
        # Generate 252 trading days (1 year) of realistic price data
        dates = pd.date_range(start='2023-01-01', periods=252, freq='B', tz='UTC')
        np.random.seed(42)

        # Simulate realistic price movements with trends and volatility
        base_price = 150.0
        trend = np.linspace(0, 0.2, 252)  # 20% annual growth trend
        volatility = 0.02

        returns = np.random.randn(252) * volatility + trend / 252
        log_prices = np.cumsum(returns)
        close_prices = base_price * np.exp(log_prices)

        # Generate OHLC data with realistic relationships
        daily_ranges = np.abs(np.random.randn(252)) * 0.01 + 0.005

        df = pd.DataFrame({
            'open': close_prices * (1 + np.random.randn(252) * 0.002),
            'high': close_prices * (1 + daily_ranges),
            'low': close_prices * (1 - daily_ranges),
            'close': close_prices,
            'volume': np.random.lognormal(15, 0.5, 252).astype(int)
        }, index=dates)

        # Ensure OHLC relationships
        df['high'] = df[['high', 'close', 'open']].max(axis=1)
        df['low'] = df[['low', 'close', 'open']].min(axis=1)

        return df

    @pytest.fixture
    def input_series(self, realistic_market_data):
        """Create input series from market data."""
        return {
            'close': realistic_market_data['close'],
            'open': realistic_market_data['open'],
            'high': realistic_market_data['high'],
            'low': realistic_market_data['low'],
            'volume': realistic_market_data['volume']
        }

    def test_ta_lib_adapter_with_real_data(self, realistic_market_data, input_series):
        """Test TA-Lib adapter with realistic market data."""
        adapter = TaLibAdapter()

        # Test multiple indicators
        indicators_to_test = [
            ('rsi', {'timeperiod': 14}),
            ('ema', {'timeperiod': 20}),
            ('macd', {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9}),
            ('bbands', {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2}),
            ('atr', {'timeperiod': 14}),
            ('adx', {'timeperiod': 14}),
            ('stoch', {'fastk_period': 14, 'slowk_period': 3, 'slowd_period': 3})
        ]

        for indicator_name, params in indicators_to_test:
            with pytest.subTest(indicator=indicator_name):
                if adapter.supports(indicator_name):
                    result = adapter.compute(indicator_name, realistic_market_data, input_series, params)

                    assert isinstance(result, dict)
                    assert len(result) > 0

                    # Verify all outputs are Series or arrays with correct length
                    for output_name, output_data in result.items():
                        if isinstance(output_data, pd.Series):
                            assert len(output_data) == len(realistic_market_data)
                        elif isinstance(output_data, np.ndarray):
                            assert len(output_data) == len(realistic_market_data)

    def test_pandas_ta_adapter_with_real_data(self, realistic_market_data, input_series):
        """Test pandas-ta adapter with realistic market data."""
        adapter = PandasTaAdapter()

        indicators_to_test = [
            ('rsi', {'length': 14}),
            ('ema', {'length': 20}),
            ('sma', {'length': 20}),
            ('bbands', {'length': 20}),
            ('stoch', {'k': 14, 'd': 3})
        ]

        for indicator_name, params in indicators_to_test:
            with pytest.subTest(indicator=indicator_name):
                if adapter.supports(indicator_name):
                    result = adapter.compute(indicator_name, realistic_market_data, input_series, params)

                    assert isinstance(result, dict)
                    assert len(result) > 0

                    # Verify outputs are pandas Series
                    for output_name, output_data in result.items():
                        assert isinstance(output_data, pd.Series)
                        assert len(output_data) == len(realistic_market_data)

    def test_cross_adapter_consistency(self, realistic_market_data, input_series):
        """Test consistency between different adapters for same indicators."""
        ta_lib = TaLibAdapter()
        pandas_ta = PandasTaAdapter()

        # Test indicators available in both adapters
        common_indicators = [
            ('rsi', {'timeperiod': 14}, {'length': 14}),
            ('ema', {'timeperiod': 20}, {'length': 20}),
            ('sma', {'timeperiod': 20}, {'length': 20})
        ]

        for indicator_name, ta_params, pta_params in common_indicators:
            with pytest.subTest(indicator=indicator_name):
                if ta_lib.supports(indicator_name) and pandas_ta.supports(indicator_name):
                    ta_result = ta_lib.compute(indicator_name, realistic_market_data, input_series, ta_params)
                    pta_result = pandas_ta.compute(indicator_name, realistic_market_data, input_series, pta_params)

                    # Compare primary output values
                    ta_values = ta_result.get('value', list(ta_result.values())[0])
                    pta_values = pta_result.get('value', list(pta_result.values())[0])

                    if isinstance(ta_values, np.ndarray):
                        ta_values = pd.Series(ta_values, index=realistic_market_data.index)

                    # Compare non-NaN values (allow small differences)
                    valid_mask = ~(ta_values.isna() | pta_values.isna())
                    if valid_mask.sum() > 10:  # Need sufficient data points
                        diff = (ta_values[valid_mask] - pta_values[valid_mask]).abs()
                        max_diff = diff.max()

                        # Allow reasonable tolerance for numerical differences
                        tolerance = max(ta_values[valid_mask].std() * 0.01, 0.1)
                        assert max_diff < tolerance, f"{indicator_name}: Max difference {max_diff} exceeds tolerance {tolerance}"

    def test_fundamentals_adapter_integration(self):
        """Test fundamentals adapter with mock data."""
        mock_fundamentals = Mock()
        mock_fundamentals.pe_ratio = 18.5
        mock_fundamentals.forward_pe = 16.2
        mock_fundamentals.price_to_book = 3.1
        mock_fundamentals.return_on_equity = 0.22
        mock_fundamentals.debt_to_equity = 0.45

        def mock_getter(ticker, provider=None):
            return mock_fundamentals

        adapter = FundamentalsAdapter(fundamentals_getter=mock_getter)

        # Test multiple fundamental indicators
        fundamental_indicators = ['pe', 'forward_pe', 'pb', 'roe', 'de_ratio']

        for indicator in fundamental_indicators:
            with pytest.subTest(indicator=indicator):
                if adapter.supports(indicator):
                    result = adapter.compute(
                        indicator,
                        pd.DataFrame(),
                        {},
                        {'ticker': 'AAPL', 'provider': None}
                    )

                    assert 'value' in result
                    assert isinstance(result['value'], pd.Series)
                    assert len(result['value']) == 1
                    assert not pd.isna(result['value'].iloc[0])

    def test_adapter_error_handling(self, realistic_market_data, input_series):
        """Test adapter error handling and recovery."""
        adapter = TaLibAdapter()

        # Test with insufficient data
        small_data = realistic_market_data.head(5)
        small_inputs = {k: v.head(5) for k, v in input_series.items()}

        try:
            result = adapter.compute('rsi', small_data, small_inputs, {'timeperiod': 14})
            # Should either succeed with mostly NaN or raise appropriate error
            if 'value' in result:
                values = result['value']
                if isinstance(values, np.ndarray):
                    assert np.isnan(values).sum() >= 3  # Most values should be NaN
                else:
                    assert values.isna().sum() >= 3
        except (ValueError, IndexError):
            # Acceptable to raise error with insufficient data
            pass

        # Test with missing required inputs
        incomplete_inputs = {k: v for k, v in input_series.items() if k != 'high'}

        with pytest.raises(KeyError):
            adapter.compute('atr', realistic_market_data, incomplete_inputs, {'timeperiod': 14})

    def test_adapter_fallback_mechanisms(self, realistic_market_data, input_series):
        """Test adapter fallback when primary computation fails."""
        adapter = TaLibAdapter()

        # Mock a computation failure
        original_compute = adapter._compute_indicator

        def failing_compute(*args, **kwargs):
            raise Exception("Computation failed")

        with patch.object(adapter, '_compute_indicator', side_effect=failing_compute):
            # Should handle the failure gracefully
            try:
                result = adapter.compute('rsi', realistic_market_data, input_series, {'timeperiod': 14})
                # If fallback succeeds, should get valid result
                assert isinstance(result, dict)
            except Exception as e:
                # If no fallback available, should raise informative error
                assert "failed" in str(e).lower() or "error" in str(e).lower()

    def test_multi_output_indicator_integration(self, realistic_market_data, input_series):
        """Test multi-output indicators work correctly across adapters."""
        adapters = [TaLibAdapter(), PandasTaAdapter()]

        multi_output_indicators = [
            ('macd', {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9}),
            ('bbands', {'timeperiod': 20}),
            ('stoch', {'fastk_period': 14, 'slowk_period': 3, 'slowd_period': 3})
        ]

        for adapter in adapters:
            for indicator_name, params in multi_output_indicators:
                with pytest.subTest(adapter=type(adapter).__name__, indicator=indicator_name):
                    if adapter.supports(indicator_name):
                        result = adapter.compute(indicator_name, realistic_market_data, input_series, params)

                        # Should have multiple outputs
                        assert len(result) > 1, f"{indicator_name} should have multiple outputs"

                        # All outputs should have same length
                        lengths = []
                        for output_data in result.values():
                            if isinstance(output_data, pd.Series):
                                lengths.append(len(output_data))
                            elif isinstance(output_data, np.ndarray):
                                lengths.append(len(output_data))

                        assert len(set(lengths)) == 1, "All outputs should have same length"

    def test_adapter_performance_characteristics(self, realistic_market_data, input_series):
        """Test adapter performance with realistic data sizes."""
        adapter = TaLibAdapter()

        # Test with different data sizes
        data_sizes = [50, 100, 252, 500]

        for size in data_sizes:
            with pytest.subTest(size=size):
                test_data = realistic_market_data.head(size)
                test_inputs = {k: v.head(size) for k, v in input_series.items()}

                start_time = datetime.now()
                result = adapter.compute('rsi', test_data, test_inputs, {'timeperiod': 14})
                end_time = datetime.now()

                computation_time = (end_time - start_time).total_seconds()

                # Should complete within reasonable time
                assert computation_time < 1.0, f"Computation took too long: {computation_time}s"

                # Should produce valid results
                assert 'value' in result
                values = result['value']
                if isinstance(values, np.ndarray):
                    valid_count = np.sum(~np.isnan(values))
                else:
                    valid_count = values.notna().sum()

                # Should have reasonable number of valid values
                expected_valid = max(0, size - 14)  # RSI needs 14 periods
                assert valid_count >= expected_valid * 0.8  # Allow some tolerance

    def test_adapter_memory_efficiency(self, realistic_market_data, input_series):
        """Test adapters handle large datasets efficiently."""
        adapter = TaLibAdapter()

        # Create larger dataset
        large_data = pd.concat([realistic_market_data] * 5, ignore_index=True)
        large_data.index = pd.date_range(
            start='2020-01-01',
            periods=len(large_data),
            freq='B',
            tz='UTC'
        )

        large_inputs = {
            k: pd.concat([v] * 5, ignore_index=True)
            for k, v in input_series.items()
        }
        for k, v in large_inputs.items():
            v.index = large_data.index

        # Should handle large dataset without memory issues
        result = adapter.compute('rsi', large_data, large_inputs, {'timeperiod': 14})

        assert 'value' in result
        assert len(result['value']) == len(large_data)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])