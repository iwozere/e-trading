# ---------------------------------------------------------------------------
# tests/test_adapters.py
# Comprehensive test suite for all indicator adapters
# ---------------------------------------------------------------------------
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

# Assuming your project structure - adjust imports as needed
from src.indicators.adapters.base import BaseAdapter
from src.indicators.adapters.ta_lib_adapter import TaLibAdapter
from src.indicators.adapters.pandas_ta_adapter import PandasTaAdapter
from src.indicators.adapters.fundamentals_adapter import FundamentalsAdapter


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV data for testing technical indicators."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D', tz='UTC')
    np.random.seed(42)

    # Generate realistic price data
    base_price = 100
    returns = np.random.randn(100) * 0.02
    close = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'open': close * (1 + np.random.randn(100) * 0.005),
        'high': close * (1 + np.abs(np.random.randn(100)) * 0.01),
        'low': close * (1 - np.abs(np.random.randn(100)) * 0.01),
        'close': close,
        'volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)

    # Ensure high >= close >= low
    df['high'] = df[['high', 'close']].max(axis=1)
    df['low'] = df[['low', 'close']].min(axis=1)

    return df


@pytest.fixture
def sample_inputs(sample_ohlcv_df):
    """Create input series dict from OHLCV data."""
    return {
        'close': sample_ohlcv_df['close'],
        'open': sample_ohlcv_df['open'],
        'high': sample_ohlcv_df['high'],
        'low': sample_ohlcv_df['low'],
        'volume': sample_ohlcv_df['volume']
    }


@pytest.fixture
def mock_fundamentals():
    """Mock fundamentals object for testing."""
    class MockFundamentals:
        pe_ratio = 15.5
        forward_pe = 14.2
        price_to_book = 2.3
        price_to_sales = 1.8
        peg_ratio = 1.2
        return_on_equity = 0.18
        return_on_assets = 0.08
        debt_to_equity = 0.6
        current_ratio = 2.1
        quick_ratio = 1.5
        dividend_yield = 0.025
        payout_ratio = 0.45
        market_cap = 50000000000
        enterprise_value = 55000000000

    return MockFundamentals()


@pytest.fixture
def fundamentals_getter(mock_fundamentals):
    """Create a mock fundamentals getter function."""
    def getter(ticker: str, provider: str = None):
        return mock_fundamentals
    return getter


# ---------------------------------------------------------------------------
# TA-Lib Adapter Tests
# ---------------------------------------------------------------------------

class TestTaLibAdapter:

    @pytest.fixture
    def adapter(self):
        return TaLibAdapter()

    def test_supports_known_indicators(self, adapter):
        """Test that adapter reports support for known indicators."""
        assert adapter.supports('rsi')
        assert adapter.supports('ema')
        assert adapter.supports('sma')
        assert adapter.supports('macd')
        assert adapter.supports('bbands')
        assert adapter.supports('atr')
        assert adapter.supports('adx')
        assert adapter.supports('stoch')
        assert adapter.supports('obv')

    def test_does_not_support_unknown(self, adapter):
        """Test that adapter rejects unknown indicators."""
        assert not adapter.supports('unknown_indicator')
        assert not adapter.supports('fake_ta')

    def test_rsi_computation(self, adapter, sample_ohlcv_df, sample_inputs):
        """Test RSI calculation returns valid Series."""
        result = adapter.compute('rsi', sample_ohlcv_df, sample_inputs, {'timeperiod': 14})

        assert 'value' in result
        assert isinstance(result['value'], (pd.Series, np.ndarray))

        # Convert to Series if needed
        if isinstance(result['value'], np.ndarray):
            values = result['value']
        else:
            values = result['value'].values

        # RSI should be between 0 and 100
        valid_values = values[~np.isnan(values)]
        assert len(valid_values) > 0
        assert np.all(valid_values >= 0)
        assert np.all(valid_values <= 100)

    def test_ema_computation(self, adapter, sample_ohlcv_df, sample_inputs):
        """Test EMA calculation."""
        result = adapter.compute('ema', sample_ohlcv_df, sample_inputs, {'timeperiod': 20})

        assert 'value' in result
        values = result['value'] if isinstance(result['value'], np.ndarray) else result['value'].values

        # EMA should have some valid values
        assert np.sum(~np.isnan(values)) > 50

        # EMA should be close to price range
        close_values = sample_inputs['close'].values
        valid_ema = values[~np.isnan(values)]
        assert np.min(valid_ema) >= np.min(close_values) * 0.9
        assert np.max(valid_ema) <= np.max(close_values) * 1.1

    def test_macd_multi_output(self, adapter, sample_ohlcv_df, sample_inputs):
        """Test MACD returns multiple outputs correctly."""
        result = adapter.compute('macd', sample_ohlcv_df, sample_inputs,
                                {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9})

        assert 'macd' in result
        assert 'signal' in result
        assert 'hist' in result

        # All outputs should have same length
        macd_len = len(result['macd']) if isinstance(result['macd'], np.ndarray) else len(result['macd'])
        signal_len = len(result['signal']) if isinstance(result['signal'], np.ndarray) else len(result['signal'])
        hist_len = len(result['hist']) if isinstance(result['hist'], np.ndarray) else len(result['hist'])

        assert macd_len == signal_len == hist_len

    def test_bbands_multi_output(self, adapter, sample_ohlcv_df, sample_inputs):
        """Test Bollinger Bands returns upper, middle, lower."""
        result = adapter.compute('bbands', sample_ohlcv_df, sample_inputs,
                                {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2})

        assert 'upper' in result
        assert 'middle' in result
        assert 'lower' in result

        # Convert to arrays for comparison
        upper = result['upper'] if isinstance(result['upper'], np.ndarray) else result['upper'].values
        middle = result['middle'] if isinstance(result['middle'], np.ndarray) else result['middle'].values
        lower = result['lower'] if isinstance(result['lower'], np.ndarray) else result['lower'].values

        # Where all are valid, upper >= middle >= lower
        valid_mask = ~(np.isnan(upper) | np.isnan(middle) | np.isnan(lower))
        assert np.all(upper[valid_mask] >= middle[valid_mask])
        assert np.all(middle[valid_mask] >= lower[valid_mask])

    def test_stoch_multi_output(self, adapter, sample_ohlcv_df, sample_inputs):
        """Test Stochastic returns K and D lines."""
        result = adapter.compute('stoch', sample_ohlcv_df, sample_inputs,
                                {'fastk_period': 14, 'slowk_period': 3, 'slowd_period': 3})

        assert 'k' in result
        assert 'd' in result

        k_vals = result['k'] if isinstance(result['k'], np.ndarray) else result['k'].values
        d_vals = result['d'] if isinstance(result['d'], np.ndarray) else result['d'].values

        # Stochastic should be between 0 and 100
        valid_k = k_vals[~np.isnan(k_vals)]
        valid_d = d_vals[~np.isnan(d_vals)]

        assert np.all(valid_k >= 0) and np.all(valid_k <= 100)
        assert np.all(valid_d >= 0) and np.all(valid_d <= 100)

    def test_atr_with_hlc(self, adapter, sample_ohlcv_df, sample_inputs):
        """Test ATR uses high, low, close inputs."""
        result = adapter.compute('atr', sample_ohlcv_df, sample_inputs, {'timeperiod': 14})

        assert 'value' in result
        values = result['value'] if isinstance(result['value'], np.ndarray) else result['value'].values

        # ATR should be positive
        valid_values = values[~np.isnan(values)]
        assert len(valid_values) > 0
        assert np.all(valid_values >= 0)

    def test_obv_with_volume(self, adapter, sample_ohlcv_df, sample_inputs):
        """Test OBV uses close and volume."""
        result = adapter.compute('obv', sample_ohlcv_df, sample_inputs, {})

        assert 'value' in result
        # OBV should have mostly non-NaN values
        values = result['value'] if isinstance(result['value'], np.ndarray) else result['value'].values
        assert np.sum(~np.isnan(values)) > 80


# ---------------------------------------------------------------------------
# Pandas-TA Adapter Tests
# ---------------------------------------------------------------------------

class TestPandasTaAdapter:

    @pytest.fixture
    def adapter(self):
        return PandasTaAdapter()

    def test_supports_common_indicators(self, adapter):
        """Test support for common indicators."""
        assert adapter.supports('rsi')
        assert adapter.supports('ema')
        assert adapter.supports('sma')
        assert adapter.supports('bbands')
        assert adapter.supports('stoch')

    def test_rsi_computation(self, adapter, sample_ohlcv_df, sample_inputs):
        """Test RSI via pandas_ta."""
        result = adapter.compute('rsi', sample_ohlcv_df, sample_inputs, {'length': 14})

        assert 'value' in result
        assert isinstance(result['value'], pd.Series)

        # Check values are in valid range
        valid_values = result['value'].dropna()
        assert len(valid_values) > 0
        assert valid_values.min() >= 0
        assert valid_values.max() <= 100

    def test_bbands_multi_output(self, adapter, sample_ohlcv_df, sample_inputs):
        """Test pandas_ta bbands returns multiple outputs."""
        result = adapter.compute('bbands', sample_ohlcv_df, sample_inputs, {'length': 20})

        assert 'upper' in result
        assert 'middle' in result
        assert 'lower' in result

        # Verify band ordering
        valid_idx = ~(result['upper'].isna() | result['middle'].isna() | result['lower'].isna())
        assert (result['upper'][valid_idx] >= result['middle'][valid_idx]).all()
        assert (result['middle'][valid_idx] >= result['lower'][valid_idx]).all()

    def test_stoch_multi_output(self, adapter, sample_ohlcv_df, sample_inputs):
        """Test pandas_ta stochastic."""
        result = adapter.compute('stoch', sample_ohlcv_df, sample_inputs, {'k': 14, 'd': 3})

        assert 'k' in result
        assert 'd' in result

        # Check values are in valid range
        valid_k = result['k'].dropna()
        valid_d = result['d'].dropna()

        assert len(valid_k) > 0 and len(valid_d) > 0
        assert valid_k.between(0, 100).all()
        assert valid_d.between(0, 100).all()


# ---------------------------------------------------------------------------
# Fundamentals Adapter Tests
# ---------------------------------------------------------------------------

class TestFundamentalsAdapter:

    @pytest.fixture
    def adapter(self, fundamentals_getter):
        return FundamentalsAdapter(fundamentals_getter=fundamentals_getter)

    def test_supports_fundamental_metrics(self, adapter):
        """Test support for fundamental indicators."""
        assert adapter.supports('pe')
        assert adapter.supports('forward_pe')
        assert adapter.supports('pb')
        assert adapter.supports('ps')
        assert adapter.supports('peg')
        assert adapter.supports('roe')
        assert adapter.supports('roa')
        assert adapter.supports('de_ratio')
        assert adapter.supports('current_ratio')
        assert adapter.supports('quick_ratio')
        assert adapter.supports('div_yield')
        assert adapter.supports('payout_ratio')

    def test_does_not_support_technical(self, adapter):
        """Test that fundamentals adapter rejects technical indicators."""
        assert not adapter.supports('rsi')
        assert not adapter.supports('macd')

    def test_pe_computation(self, adapter, sample_ohlcv_df):
        """Test P/E ratio retrieval."""
        result = adapter.compute('pe', sample_ohlcv_df, {},
                                {'ticker': 'AAPL', 'provider': None})

        assert 'value' in result
        assert isinstance(result['value'], pd.Series)
        assert len(result['value']) == 1
        assert result['value'].iloc[0] == 15.5

    def test_pb_computation(self, adapter, sample_ohlcv_df):
        """Test P/B ratio retrieval."""
        result = adapter.compute('pb', sample_ohlcv_df, {},
                                {'ticker': 'AAPL', 'provider': None})

        assert 'value' in result
        assert result['value'].iloc[0] == 2.3

    def test_roe_computation(self, adapter, sample_ohlcv_df):
        """Test ROE retrieval."""
        result = adapter.compute('roe', sample_ohlcv_df, {},
                                {'ticker': 'AAPL', 'provider': None})

        assert 'value' in result
        assert result['value'].iloc[0] == 0.18

    def test_all_fundamental_fields(self, adapter, sample_ohlcv_df, mock_fundamentals):
        """Test all fundamental fields map correctly."""
        for field_name, attr_name in adapter.FIELD_MAP.items():
            result = adapter.compute(field_name, sample_ohlcv_df, {},
                                    {'ticker': 'AAPL', 'provider': None})

            expected = getattr(mock_fundamentals, attr_name)
            assert result['value'].iloc[0] == expected, \
                f"Field {field_name} -> {attr_name} mismatch"


# ---------------------------------------------------------------------------
# Cross-Adapter Consistency Tests
# ---------------------------------------------------------------------------

class TestAdapterConsistency:
    """Test that adapters produce consistent results for the same indicator."""

    def test_rsi_consistency(self, sample_ohlcv_df, sample_inputs):
        """Test RSI is similar between TA-Lib and pandas_ta."""
        ta_lib = TaLibAdapter()
        pandas_ta = PandasTaAdapter()

        ta_result = ta_lib.compute('rsi', sample_ohlcv_df, sample_inputs, {'timeperiod': 14})
        pta_result = pandas_ta.compute('rsi', sample_ohlcv_df, sample_inputs, {'length': 14})

        # Convert to Series if needed
        ta_values = ta_result['value'] if isinstance(ta_result['value'], pd.Series) else pd.Series(ta_result['value'], index=sample_ohlcv_df.index)
        pta_values = pta_result['value']

        # Compare non-NaN values (allow small differences due to implementation)
        valid_idx = ~(ta_values.isna() | pta_values.isna())
        if valid_idx.sum() > 0:
            diff = (ta_values[valid_idx] - pta_values[valid_idx]).abs()
            assert diff.max() < 5.0, "RSI values differ significantly between adapters"


# ---------------------------------------------------------------------------
# Edge Cases and Error Handling
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_dataframe(self):
        """Test adapters handle empty DataFrames gracefully."""
        adapter = TaLibAdapter()
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        empty_inputs = {k: pd.Series(dtype=float) for k in ['close', 'high', 'low', 'volume']}

        # Should not crash, but may return NaN
        try:
            result = adapter.compute('rsi', empty_df, empty_inputs, {'timeperiod': 14})
            assert 'value' in result
        except (ValueError, IndexError) as e:
            # Acceptable to raise error on empty data
            pass

    def test_insufficient_data(self, sample_ohlcv_df):
        """Test indicators with insufficient data return mostly NaN."""
        adapter = TaLibAdapter()

        # Take only 5 rows, RSI needs 14+
        small_df = sample_ohlcv_df.head(5)
        small_inputs = {
            'close': small_df['close'],
            'high': small_df['high'],
            'low': small_df['low']
        }

        result = adapter.compute('rsi', small_df, small_inputs, {'timeperiod': 14})
        values = result['value'] if isinstance(result['value'], np.ndarray) else result['value'].values

        # Should be mostly or all NaN
        assert np.isnan(values).sum() >= 4

    def test_missing_required_input(self, sample_ohlcv_df):
        """Test adapter handles missing required inputs."""
        adapter = TaLibAdapter()

        # Try to compute ATR without 'high' input
        incomplete_inputs = {
            'close': sample_ohlcv_df['close'],
            'low': sample_ohlcv_df['low']
        }

        with pytest.raises(KeyError):
            adapter.compute('atr', sample_ohlcv_df, incomplete_inputs, {'timeperiod': 14})


# ---------------------------------------------------------------------------
# Integration Test
# ---------------------------------------------------------------------------

class TestAdapterIntegration:
    """Test adapters work together in a realistic scenario."""

    def test_multiple_indicators(self, sample_ohlcv_df, sample_inputs):
        """Test computing multiple indicators on same data."""
        adapter = TaLibAdapter()

        indicators = [
            ('rsi', {'timeperiod': 14}),
            ('ema', {'timeperiod': 20}),
            ('macd', {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9}),
            ('bbands', {'timeperiod': 20}),
        ]

        results = {}
        for name, params in indicators:
            result = adapter.compute(name, sample_ohlcv_df, sample_inputs, params)
            results[name] = result

        # All should succeed
        assert len(results) == 4
        assert 'value' in results['rsi']
        assert 'macd' in results['macd']
        assert 'upper' in results['bbands']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])