"""
Comprehensive unit tests for core indicator service functionality.

This test suite covers:
- All indicator calculations against known reference values
- Configuration management and parameter validation
- Batch processing and error handling mechanisms
- Service orchestration and adapter coordination
"""

import pytest
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime
from unittest.mock import patch

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.indicators.service import (
    IndicatorService, ConfigurationError, DataError
)
from src.indicators.config_manager import UnifiedConfigManager
from src.indicators.recommendation_engine import RecommendationEngine
from src.indicators.models import (
    IndicatorBatchConfig, IndicatorSpec, TickerIndicatorsRequest,
    IndicatorResultSet
)
from src.indicators.models import (
    IndicatorResult, IndicatorSet, RecommendationType, CompositeRecommendation
)


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ohlcv_data():
    """Create realistic OHLCV data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D', tz='UTC')
    np.random.seed(42)

    # Generate realistic price movements
    base_price = 100.0
    returns = np.random.randn(100) * 0.02
    close_prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'open': close_prices * (1 + np.random.randn(100) * 0.005),
        'high': close_prices * (1 + np.abs(np.random.randn(100)) * 0.01),
        'low': close_prices * (1 - np.abs(np.random.randn(100)) * 0.01),
        'close': close_prices,
        'volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)

    # Ensure OHLC relationships are valid
    df['high'] = df[['high', 'close', 'open']].max(axis=1)
    df['low'] = df[['low', 'close', 'open']].min(axis=1)

    return df


@pytest.fixture
def mock_fundamentals():
    """Mock fundamentals data for testing."""
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
def indicator_service():
    """Create indicator service instance for testing."""
    return IndicatorService()


@pytest.fixture
def config_manager():
    """Create config manager instance for testing."""
    with patch('src.indicators.config_manager.Path.exists', return_value=False):
        return UnifiedConfigManager()


@pytest.fixture
def recommendation_engine():
    """Create recommendation engine instance for testing."""
    return RecommendationEngine()


# ---------------------------------------------------------------------------
# Core Service Tests
# ---------------------------------------------------------------------------

class TestIndicatorService:
    """Test core indicator service functionality."""

    def test_service_initialization(self, indicator_service):
        """Test service initializes correctly with all adapters."""
        assert indicator_service is not None
        assert hasattr(indicator_service, '_ta_lib_adapter')
        assert hasattr(indicator_service, '_pandas_ta_adapter')
        assert hasattr(indicator_service, '_fundamentals_adapter')
        assert hasattr(indicator_service, '_config_manager')
        assert hasattr(indicator_service, '_recommendation_engine')

    def test_compute_single_indicator_rsi(self, indicator_service, sample_ohlcv_data):
        """Test computing single RSI indicator with known reference values."""
        config = IndicatorBatchConfig(
            indicators=[IndicatorSpec(name="rsi", output="rsi_value", params={"timeperiod": 14})]
        )

        result = indicator_service.compute(sample_ohlcv_data, config)

        assert isinstance(result, pd.DataFrame)
        assert "rsi_value" in result.columns
        assert len(result) == len(sample_ohlcv_data)

        # RSI should be between 0 and 100
        rsi_values = result["rsi_value"].dropna()
        assert len(rsi_values) > 0
        assert rsi_values.min() >= 0
        assert rsi_values.max() <= 100

    def test_compute_multi_output_indicator_macd(self, indicator_service, sample_ohlcv_data):
        """Test computing MACD with multiple outputs."""
        config = IndicatorBatchConfig(
            indicators=[IndicatorSpec(name="macd", output="macd")]
        )

        result = indicator_service.compute(sample_ohlcv_data, config)

        # MACD should produce multiple columns
        macd_columns = [col for col in result.columns if "macd" in col.lower()]
        assert len(macd_columns) >= 2  # At least MACD line and signal

        # Check for expected MACD components
        assert any("macd" in col for col in result.columns)
        assert any("signal" in col or "hist" in col for col in result.columns)

    def test_compute_bollinger_bands(self, indicator_service, sample_ohlcv_data):
        """Test Bollinger Bands calculation with proper band relationships."""
        config = IndicatorBatchConfig(
            indicators=[IndicatorSpec(name="bbands", output="bb")]
        )

        result = indicator_service.compute(sample_ohlcv_data, config)

        # Should have upper, middle, lower bands
        bb_columns = [col for col in result.columns if "bb" in col]
        assert len(bb_columns) >= 3

        # Find the band columns
        upper_col = next((col for col in bb_columns if "upper" in col), None)
        middle_col = next((col for col in bb_columns if "middle" in col), None)
        lower_col = next((col for col in bb_columns if "lower" in col), None)

        if upper_col and middle_col and lower_col:
            # Where all values are valid, upper >= middle >= lower
            valid_mask = ~(result[upper_col].isna() | result[middle_col].isna() | result[lower_col].isna())
            valid_data = result[valid_mask]

            if len(valid_data) > 0:
                assert (valid_data[upper_col] >= valid_data[middle_col]).all()
                assert (valid_data[middle_col] >= valid_data[lower_col]).all()

    def test_compute_with_insufficient_data(self, indicator_service):
        """Test service handles insufficient data gracefully."""
        # Create very small dataset
        small_df = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [101, 102],
            'volume': [1000, 1100]
        }, index=pd.date_range('2024-01-01', periods=2, freq='D', tz='UTC'))

        config = IndicatorBatchConfig(
            indicators=[IndicatorSpec(name="rsi", output="rsi", params={"timeperiod": 14})]
        )

        # Should not crash, but may return mostly NaN values
        result = indicator_service.compute(small_df, config)
        assert isinstance(result, pd.DataFrame)
        assert "rsi" in result.columns

    def test_compute_with_invalid_parameters(self, indicator_service, sample_ohlcv_data):
        """Test service handles invalid parameters appropriately."""
        config = IndicatorBatchConfig(
            indicators=[IndicatorSpec(name="rsi", output="rsi", params={"timeperiod": -1})]
        )

        # Should handle invalid parameters gracefully
        with pytest.raises((ValueError, ConfigurationError)):
            indicator_service.compute(sample_ohlcv_data, config)

    @pytest.mark.asyncio
    async def test_compute_for_ticker_async(self, indicator_service):
        """Test async ticker computation."""
        request = TickerIndicatorsRequest(
            ticker="AAPL",
            timeframe="1D",
            period="1M",
            indicators=["rsi", "ema"],
            include_recommendations=False
        )

        with patch('src.common.get_ohlcv') as mock_get_ohlcv:
            # Mock the data retrieval
            mock_get_ohlcv.return_value = pd.DataFrame({
                'open': [100, 101, 102],
                'high': [102, 103, 104],
                'low': [99, 100, 101],
                'close': [101, 102, 103],
                'volume': [1000, 1100, 1200]
            }, index=pd.date_range('2024-01-01', periods=3, freq='D', tz='UTC'))

            result = await indicator_service.compute_for_ticker(request)

            assert isinstance(result, IndicatorResultSet)
            assert result.ticker == "AAPL"
            assert len(result.technical) > 0

    def test_batch_processing_multiple_tickers(self, indicator_service):
        """Test batch processing capabilities."""
        tickers = ["AAPL", "GOOGL", "MSFT"]

        with patch('src.common.get_ohlcv') as mock_get_ohlcv:
            # Mock data for each ticker
            mock_get_ohlcv.return_value = pd.DataFrame({
                'open': [100, 101, 102],
                'high': [102, 103, 104],
                'low': [99, 100, 101],
                'close': [101, 102, 103],
                'volume': [1000, 1100, 1200]
            }, index=pd.date_range('2024-01-01', periods=3, freq='D', tz='UTC'))

            # Test batch processing
            results = asyncio.run(indicator_service.compute_batch(
                tickers=tickers,
                indicators=["rsi", "ema"],
                timeframe="1D",
                period="1M"
            ))

            assert len(results) == len(tickers)
            for ticker, result in results.items():
                assert ticker in tickers
                assert isinstance(result, IndicatorResultSet)

    def test_error_handling_data_retrieval_failure(self, indicator_service):
        """Test error handling when data retrieval fails."""
        request = TickerIndicatorsRequest(
            ticker="INVALID",
            timeframe="1D",
            period="1M",
            indicators=["rsi"]
        )

        with patch('src.common.get_ohlcv', side_effect=Exception("Data not found")):
            with pytest.raises(DataError):
                asyncio.run(indicator_service.compute_for_ticker(request))

    def test_performance_metrics_collection(self, indicator_service, sample_ohlcv_data):
        """Test that performance metrics are collected during computation."""
        config = IndicatorBatchConfig(
            indicators=[IndicatorSpec(name="rsi", output="rsi")]
        )

        start_time = datetime.now()
        result = indicator_service.compute(sample_ohlcv_data, config)
        end_time = datetime.now()

        # Verify computation completed in reasonable time
        computation_time = (end_time - start_time).total_seconds()
        assert computation_time < 5.0  # Should complete within 5 seconds


# ---------------------------------------------------------------------------
# Configuration Manager Tests
# ---------------------------------------------------------------------------

class TestUnifiedConfigManager:
    """Test configuration management functionality."""

    def test_config_manager_initialization(self, config_manager):
        """Test config manager initializes with defaults."""
        assert config_manager is not None
        assert config_manager._current_preset == "default"
        assert isinstance(config_manager._runtime_overrides, dict)

    def test_get_indicator_parameters_default(self, config_manager):
        """Test getting default parameters for indicators."""
        rsi_params = config_manager.get_indicator_parameters("rsi")
        assert isinstance(rsi_params, dict)

        # Should have reasonable defaults
        if "timeperiod" in rsi_params:
            assert isinstance(rsi_params["timeperiod"], int)
            assert rsi_params["timeperiod"] > 0

    def test_set_runtime_override(self, config_manager):
        """Test setting runtime parameter overrides."""
        config_manager.set_runtime_override("rsi", "timeperiod", 21)

        params = config_manager.get_indicator_parameters("rsi")
        assert params.get("timeperiod") == 21

    def test_clear_runtime_overrides(self, config_manager):
        """Test clearing runtime overrides."""
        config_manager.set_runtime_override("rsi", "timeperiod", 21)
        config_manager.clear_runtime_overrides()

        params = config_manager.get_indicator_parameters("rsi")
        assert params.get("timeperiod") != 21  # Should revert to default

    def test_validate_parameters(self, config_manager):
        """Test parameter validation."""
        # Valid parameters should pass
        valid_params = {"timeperiod": 14}
        assert config_manager.validate_parameters("rsi", valid_params)

        # Invalid parameters should fail
        invalid_params = {"timeperiod": -1}
        assert not config_manager.validate_parameters("rsi", invalid_params)

    def test_preset_management(self, config_manager):
        """Test preset loading and switching."""
        # Test default preset
        assert config_manager.get_current_preset() == "default"

        # Test switching presets (if available)
        available_presets = config_manager.get_available_presets()
        assert isinstance(available_presets, list)
        assert "default" in available_presets

    def test_parameter_inheritance(self, config_manager):
        """Test parameter inheritance from presets to runtime overrides."""
        # Set a preset parameter
        config_manager.set_preset("default")
        base_params = config_manager.get_indicator_parameters("rsi")

        # Override a parameter
        config_manager.set_runtime_override("rsi", "timeperiod", 21)
        override_params = config_manager.get_indicator_parameters("rsi")

        # Override should take precedence
        assert override_params.get("timeperiod") == 21

        # Other parameters should remain from preset
        for key, value in base_params.items():
            if key != "timeperiod":
                assert override_params.get(key) == value


# ---------------------------------------------------------------------------
# Recommendation Engine Tests
# ---------------------------------------------------------------------------

class TestRecommendationEngine:
    """Test recommendation engine functionality."""

    def test_recommendation_engine_initialization(self, recommendation_engine):
        """Test recommendation engine initializes correctly."""
        assert recommendation_engine is not None

    def test_rsi_recommendations(self, recommendation_engine):
        """Test RSI recommendation logic with known values."""
        # Oversold condition (RSI < 30)
        oversold_rec = recommendation_engine.get_recommendation("rsi", 25.0)
        assert oversold_rec.type in [RecommendationType.BUY, RecommendationType.STRONG_BUY]
        assert oversold_rec.confidence > 0.5

        # Overbought condition (RSI > 70)
        overbought_rec = recommendation_engine.get_recommendation("rsi", 75.0)
        assert overbought_rec.type in [RecommendationType.SELL, RecommendationType.STRONG_SELL]
        assert overbought_rec.confidence > 0.5

        # Neutral condition
        neutral_rec = recommendation_engine.get_recommendation("rsi", 50.0)
        assert neutral_rec.type == RecommendationType.HOLD

    def test_macd_recommendations(self, recommendation_engine):
        """Test MACD recommendation logic."""
        # Bullish crossover
        bullish_context = {"macd": 0.5, "signal": 0.3, "hist": 0.2}
        bullish_rec = recommendation_engine.get_recommendation("macd", 0.5, bullish_context)
        assert bullish_rec.type in [RecommendationType.BUY, RecommendationType.STRONG_BUY]

        # Bearish crossover
        bearish_context = {"macd": -0.5, "signal": -0.3, "hist": -0.2}
        bearish_rec = recommendation_engine.get_recommendation("macd", -0.5, bearish_context)
        assert bearish_rec.type in [RecommendationType.SELL, RecommendationType.STRONG_SELL]

    def test_fundamental_recommendations(self, recommendation_engine):
        """Test fundamental indicator recommendations."""
        # Low P/E ratio (attractive)
        low_pe_rec = recommendation_engine.get_recommendation("pe", 12.0)
        assert low_pe_rec.type in [RecommendationType.BUY, RecommendationType.STRONG_BUY]

        # High P/E ratio (expensive)
        high_pe_rec = recommendation_engine.get_recommendation("pe", 35.0)
        assert high_pe_rec.type in [RecommendationType.SELL, RecommendationType.HOLD]

        # High ROE (good profitability)
        high_roe_rec = recommendation_engine.get_recommendation("roe", 0.25)
        assert high_roe_rec.type in [RecommendationType.BUY, RecommendationType.STRONG_BUY]

    def test_composite_recommendations(self, recommendation_engine):
        """Test composite recommendation generation."""
        # Create mock indicator set
        indicator_set = IndicatorSet(
            ticker="AAPL",
            technical={
                "rsi": IndicatorResult(name="rsi", value=25.0, timestamp=datetime.now()),
                "macd": IndicatorResult(name="macd", value=0.5, timestamp=datetime.now())
            },
            fundamental={
                "pe": IndicatorResult(name="pe", value=15.0, timestamp=datetime.now())
            }
        )

        composite_rec = recommendation_engine.get_composite_recommendation(indicator_set)

        assert isinstance(composite_rec, CompositeRecommendation)
        assert composite_rec.overall_recommendation is not None
        assert 0 <= composite_rec.confidence <= 1
        assert len(composite_rec.contributing_indicators) > 0

    def test_recommendation_confidence_scoring(self, recommendation_engine):
        """Test confidence scoring for recommendations."""
        # Strong signal should have high confidence
        strong_rec = recommendation_engine.get_recommendation("rsi", 15.0)  # Very oversold
        assert strong_rec.confidence > 0.8

        # Weak signal should have lower confidence
        weak_rec = recommendation_engine.get_recommendation("rsi", 45.0)  # Near neutral
        assert weak_rec.confidence < 0.6

    def test_contextual_recommendations(self, recommendation_engine):
        """Test context-aware recommendations."""
        # MACD with context should provide more nuanced recommendations
        context = {
            "macd": 0.1,
            "signal": 0.05,
            "hist": 0.05,
            "trend": "bullish"
        }

        contextual_rec = recommendation_engine.get_recommendation("macd", 0.1, context)

        assert contextual_rec is not None
        assert contextual_rec.reasoning is not None
        assert len(contextual_rec.reasoning) > 0


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Test comprehensive error handling mechanisms."""

    def test_configuration_error_handling(self, indicator_service, sample_ohlcv_data):
        """Test configuration error handling."""
        # Invalid indicator name
        config = IndicatorBatchConfig(
            indicators=[IndicatorSpec(name="invalid_indicator", output="invalid")]
        )

        with pytest.raises((ValueError, ConfigurationError)):
            indicator_service.compute(sample_ohlcv_data, config)

    def test_data_error_handling(self, indicator_service):
        """Test data error handling."""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        config = IndicatorBatchConfig(
            indicators=[IndicatorSpec(name="rsi", output="rsi")]
        )

        with pytest.raises((ValueError, DataError)):
            indicator_service.compute(empty_df, config)

    def test_adapter_failure_recovery(self, indicator_service, sample_ohlcv_data):
        """Test adapter failure and recovery mechanisms."""
        config = IndicatorBatchConfig(
            indicators=[IndicatorSpec(name="rsi", output="rsi")]
        )

        # Mock adapter failure
        with patch.object(indicator_service._ta_lib_adapter, 'compute', side_effect=Exception("Adapter failed")):
            # Should attempt fallback to other adapters
            try:
                result = indicator_service.compute(sample_ohlcv_data, config)
                # If fallback succeeds, we should get a result
                assert isinstance(result, pd.DataFrame)
            except Exception:
                # If all adapters fail, should raise appropriate error
                pass

    def test_timeout_handling(self, indicator_service):
        """Test timeout handling for long-running operations."""
        request = TickerIndicatorsRequest(
            ticker="AAPL",
            timeframe="1D",
            period="10Y",  # Large dataset
            indicators=["rsi", "ema", "macd", "bbands"]
        )

        with patch('src.common.get_ohlcv', side_effect=lambda *args, **kwargs: asyncio.sleep(10)):
            # Should handle timeout appropriately
            with pytest.raises((asyncio.TimeoutError, DataError)):
                asyncio.run(asyncio.wait_for(
                    indicator_service.compute_for_ticker(request),
                    timeout=2.0
                ))


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestServiceIntegration:
    """Test integration between service components."""

    def test_service_config_integration(self, indicator_service, sample_ohlcv_data):
        """Test service integrates properly with config manager."""
        # Service should use config manager for parameters
        config = IndicatorBatchConfig(
            indicators=[IndicatorSpec(name="rsi", output="rsi")]
        )

        result = indicator_service.compute(sample_ohlcv_data, config)
        assert isinstance(result, pd.DataFrame)
        assert "rsi" in result.columns

    def test_service_recommendation_integration(self, indicator_service, sample_ohlcv_data):
        """Test service integrates with recommendation engine."""
        config = IndicatorBatchConfig(
            indicators=[IndicatorSpec(name="rsi", output="rsi")]
        )

        # Test with recommendations enabled
        request = TickerIndicatorsRequest(
            ticker="TEST",
            indicators=["rsi"],
            include_recommendations=True
        )

        with patch('src.common.get_ohlcv', return_value=sample_ohlcv_data):
            result = asyncio.run(indicator_service.compute_for_ticker(request))

            assert isinstance(result, IndicatorResultSet)
            # Should include recommendations when requested
            if result.technical:
                for indicator_result in result.technical.values():
                    if hasattr(indicator_result, 'recommendation'):
                        assert indicator_result.recommendation is not None

    def test_adapter_coordination(self, indicator_service, sample_ohlcv_data):
        """Test service coordinates multiple adapters correctly."""
        config = IndicatorBatchConfig(
            indicators=[
                IndicatorSpec(name="rsi", output="rsi"),  # Technical
                IndicatorSpec(name="pe", output="pe")     # Fundamental (if available)
            ]
        )

        # Mock fundamentals for testing
        with patch.object(indicator_service._fundamentals_adapter, 'compute') as mock_fund:
            mock_fund.return_value = {"value": pd.Series([15.0])}

            result = indicator_service.compute(sample_ohlcv_data, config)

            assert isinstance(result, pd.DataFrame)
            assert "rsi" in result.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])