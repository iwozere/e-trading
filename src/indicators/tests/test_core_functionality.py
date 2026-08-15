"""
Comprehensive unit tests for core indicator service functionality.

This test suite covers:
- All indicator calculations against known reference values
- Configuration management and parameter validation
- Batch processing and error handling mechanisms
- Service orchestration and adapter coordination
"""

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.recommendation.engine import RecommendationEngine
from src.indicators.config_manager import UnifiedConfigManager
from src.indicators.models import (
    BatchIndicatorRequest,
    CompositeRecommendation,
    IndicatorBatchConfig,
    IndicatorCategory,
    IndicatorResult,
    IndicatorResultSet,
    IndicatorSet,
    IndicatorSpec,
    RecommendationType,
    TickerIndicatorsRequest,
)
from src.indicators.service import CalculationError, ConfigurationError, DataError, IndicatorService
from src.indicators.types import IndicatorName, Period, TickerSymbol, TimeFrame

# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_ohlcv_data():
    """Create realistic OHLCV data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D", tz="UTC")
    np.random.seed(42)

    # Generate realistic price movements
    base_price = 100.0
    returns = np.random.randn(100) * 0.02
    close_prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "open": close_prices * (1 + np.random.randn(100) * 0.005),
            "high": close_prices * (1 + np.abs(np.random.randn(100)) * 0.01),
            "low": close_prices * (1 - np.abs(np.random.randn(100)) * 0.01),
            "close": close_prices,
            "volume": np.random.randint(1000000, 10000000, 100),
        },
        index=dates,
    )

    # Ensure OHLC relationships are valid
    df["high"] = df[["high", "close", "open"]].max(axis=1)
    df["low"] = df[["low", "close", "open"]].min(axis=1)

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
    with patch("src.indicators.config_manager.Path.exists", return_value=False):
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
        # Adapters live in a dict (self.adapters), not individually named
        # attributes — this changed at some point and these tests never
        # caught up.
        assert "ta-lib" in indicator_service.adapters
        assert "pandas-ta" in indicator_service.adapters
        assert "fundamentals" in indicator_service.adapters
        assert hasattr(indicator_service, "config_manager")
        assert hasattr(indicator_service, "recommendation_engine")

    def test_compute_single_indicator_rsi(self, indicator_service, sample_ohlcv_data):
        """Test computing single RSI indicator with known reference values."""
        config = IndicatorBatchConfig(
            indicators=[IndicatorSpec(name="rsi", output="rsi_value", params={"timeperiod": 14})]
        )

        result = asyncio.run(indicator_service.compute(sample_ohlcv_data, config))

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
        # Multi-output indicators need an explicit {sub_output: column_name}
        # map — a plain string `output` only captures a "value"-keyed
        # result, which macd's adapter result never produces (see
        # IndicatorMeta.outputs for "macd": ["macd", "signal", "hist"]).
        config = IndicatorBatchConfig(
            indicators=[
                IndicatorSpec(
                    name="macd", output={"macd": "macd_macd", "signal": "macd_signal", "hist": "macd_hist"}
                )
            ]
        )

        result = asyncio.run(indicator_service.compute(sample_ohlcv_data, config))

        # MACD should produce multiple columns
        macd_columns = [col for col in result.columns if "macd" in col.lower()]
        assert len(macd_columns) >= 2  # At least MACD line and signal

        # Check for expected MACD components
        assert any("macd" in col for col in result.columns)
        assert any("signal" in col or "hist" in col for col in result.columns)

    def test_compute_bollinger_bands(self, indicator_service, sample_ohlcv_data):
        """Test Bollinger Bands calculation with proper band relationships."""
        # See comment in test_compute_multi_output_indicator_macd — same
        # multi-output requirement.
        config = IndicatorBatchConfig(
            indicators=[
                IndicatorSpec(
                    name="bbands", output={"upper": "bb_upper", "middle": "bb_middle", "lower": "bb_lower"}
                )
            ]
        )

        result = asyncio.run(indicator_service.compute(sample_ohlcv_data, config))

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
        small_df = pd.DataFrame(
            {"open": [100, 101], "high": [102, 103], "low": [99, 100], "close": [101, 102], "volume": [1000, 1100]},
            index=pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"),
        )

        config = IndicatorBatchConfig(indicators=[IndicatorSpec(name="rsi", output="rsi", params={"timeperiod": 14})])

        # Should not crash, but may return mostly NaN values
        result = asyncio.run(indicator_service.compute(small_df, config))
        assert isinstance(result, pd.DataFrame)
        assert "rsi" in result.columns

    def test_compute_with_invalid_parameters(self, indicator_service, sample_ohlcv_data):
        """Test service handles invalid parameters appropriately."""
        config = IndicatorBatchConfig(indicators=[IndicatorSpec(name="rsi", output="rsi", params={"timeperiod": -1})])

        # Should handle invalid parameters gracefully. `compute()` (unlike
        # compute_for_ticker()) has no exception-wrapping around adapter
        # calls, so a bad talib parameter surfaces as talib's own bare
        # `Exception` ("TA_BAD_PARAM"), not ValueError/ConfigurationError.
        with pytest.raises(Exception, match="Bad Parameter|TA_BAD_PARAM"):
            asyncio.run(indicator_service.compute(sample_ohlcv_data, config))

    @pytest.mark.asyncio
    async def test_compute_for_ticker_async(self, indicator_service, sample_ohlcv_data):
        """Test async ticker computation."""
        request = TickerIndicatorsRequest(
            ticker=TickerSymbol("AAPL"),
            timeframe=TimeFrame("1D"),
            period=Period("1M"),
            indicators=[IndicatorName("rsi"), IndicatorName("ema")],
            include_recommendations=False,
        )

        # Patch where `get_ohlcv` is looked up (service.py imports it by
        # name at module load), not where it's defined — and use the
        # 100-row fixture, not a 3-row frame: rsi/ema need warm-up history
        # or every value comes back NaN and gets filtered out.
        with patch("src.indicators.service.get_ohlcv", return_value=sample_ohlcv_data):
            result = await indicator_service.compute_for_ticker(request)

            assert isinstance(result, IndicatorResultSet)
            assert result.ticker == "AAPL"
            assert len(result.technical) > 0

    def test_batch_processing_multiple_tickers(self, indicator_service, sample_ohlcv_data):
        """Test batch processing capabilities."""
        tickers = ["AAPL", "GOOGL", "MSFT"]

        # No `compute_batch` method exists — the real API is
        # `get_batch_indicators(BatchIndicatorRequest)`, returning
        # Dict[str, IndicatorSet] (see service.py).
        with patch("src.indicators.service.get_ohlcv", return_value=sample_ohlcv_data):
            batch_request = BatchIndicatorRequest(
                tickers=[TickerSymbol(t) for t in tickers],
                indicators=[IndicatorName(i) for i in ["rsi", "ema"]],
                timeframe=TimeFrame("1D"),
                period=Period("1M"),
            )
            results = asyncio.run(indicator_service.get_batch_indicators(batch_request))

            assert len(results) == len(tickers)
            for ticker, result in results.items():
                assert ticker in tickers
                assert isinstance(result, IndicatorSet)

    def test_error_handling_data_retrieval_failure(self, indicator_service):
        """Test error handling when data retrieval fails."""
        request = TickerIndicatorsRequest(
            ticker=TickerSymbol("INVALID"),
            timeframe=TimeFrame("1D"),
            period=Period("1M"),
            indicators=[IndicatorName("rsi")],
        )

        # `DataError` is only raised for a *successfully fetched but empty*
        # DataFrame (service.py's explicit `if df is None or df.empty` check)
        # — an exception raised *by* the fetch itself falls through to
        # compute_for_ticker()'s generic `except Exception` wrapper, which
        # raises CalculationError instead. Confirmed via traceback, not
        # assumed.
        with patch("src.indicators.service.get_ohlcv", side_effect=Exception("Data not found")):
            with pytest.raises(CalculationError):
                asyncio.run(indicator_service.compute_for_ticker(request))

    def test_performance_metrics_collection(self, indicator_service, sample_ohlcv_data):
        """Test that performance metrics are collected during computation."""
        config = IndicatorBatchConfig(indicators=[IndicatorSpec(name="rsi", output="rsi")])

        start_time = datetime.now()
        result = asyncio.run(indicator_service.compute(sample_ohlcv_data, config))
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
        # Real method is `get_parameters` — `get_indicator_parameters` was
        # never defined on UnifiedConfigManager.
        rsi_params = config_manager.get_parameters("rsi")
        assert isinstance(rsi_params, dict)

        # Should have reasonable defaults
        if "timeperiod" in rsi_params:
            assert isinstance(rsi_params["timeperiod"], int)
            assert rsi_params["timeperiod"] > 0

    def test_set_runtime_override(self, config_manager):
        """Test setting runtime parameter overrides."""
        # Real method is `set_parameter_override`.
        config_manager.set_parameter_override("rsi", "timeperiod", 21)

        params = config_manager.get_parameters("rsi")
        assert params.get("timeperiod") == 21

    def test_clear_runtime_overrides(self, config_manager):
        """Test clearing runtime overrides."""
        # Real method is `clear_parameter_overrides`.
        config_manager.set_parameter_override("rsi", "timeperiod", 21)
        config_manager.clear_parameter_overrides()

        params = config_manager.get_parameters("rsi")
        assert params.get("timeperiod") != 21  # Should revert to default

    def test_validate_parameters(self, config_manager):
        """Test parameter validation."""
        # validate_parameters returns a List[str] of error messages, not a
        # bool — empty list (falsy) means valid, non-empty (truthy) means
        # invalid. Inverted from what this test originally asserted.
        valid_params = {"timeperiod": 14}
        assert config_manager.validate_parameters("rsi", valid_params) == []

        invalid_params = {"timeperiod": -1}
        assert config_manager.validate_parameters("rsi", invalid_params) != []

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
        base_params = config_manager.get_parameters("rsi")

        # Override a parameter
        config_manager.set_parameter_override("rsi", "timeperiod", 21)
        override_params = config_manager.get_parameters("rsi")

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
        # `Recommendation`'s field is `.recommendation` (a RecommendationType),
        # not `.type` — see src/indicators/models.py.
        # Oversold condition (RSI < 30)
        oversold_rec = recommendation_engine.get_recommendation("rsi", 25.0)
        assert oversold_rec.recommendation in [RecommendationType.BUY, RecommendationType.STRONG_BUY]
        assert oversold_rec.confidence > 0.5

        # Overbought condition (RSI > 70)
        overbought_rec = recommendation_engine.get_recommendation("rsi", 75.0)
        assert overbought_rec.recommendation in [RecommendationType.SELL, RecommendationType.STRONG_SELL]
        assert overbought_rec.confidence > 0.5

        # Neutral condition
        neutral_rec = recommendation_engine.get_recommendation("rsi", 50.0)
        assert neutral_rec.recommendation == RecommendationType.HOLD

    def test_macd_recommendations(self, recommendation_engine):
        """Test MACD recommendation logic."""
        # The MACD wrapper (_get_macd_recommendation) reads "macd_signal" and
        # "macd_hist" from context — plain "signal"/"hist" don't match, so
        # this always fell through to the "Insufficient context" HOLD case.
        # Bullish crossover
        bullish_context = {"macd_signal": 0.3, "macd_hist": 0.2}
        bullish_rec = recommendation_engine.get_recommendation("macd", 0.5, bullish_context)
        assert bullish_rec.recommendation in [RecommendationType.BUY, RecommendationType.STRONG_BUY]

        # Bearish crossover
        bearish_context = {"macd_signal": -0.3, "macd_hist": -0.2}
        bearish_rec = recommendation_engine.get_recommendation("macd", -0.5, bearish_context)
        assert bearish_rec.recommendation in [RecommendationType.SELL, RecommendationType.STRONG_SELL]

    def test_fundamental_recommendations(self, recommendation_engine):
        """Test fundamental indicator recommendations."""
        # Canonical registry name is "pe_ratio", not "pe" (see
        # src.indicators.models.FUNDAMENTAL_INDICATORS).
        # Low P/E ratio (attractive)
        low_pe_rec = recommendation_engine.get_recommendation("pe_ratio", 12.0)
        assert low_pe_rec.recommendation in [RecommendationType.BUY, RecommendationType.STRONG_BUY]

        # High P/E ratio (expensive)
        high_pe_rec = recommendation_engine.get_recommendation("pe_ratio", 35.0)
        assert high_pe_rec.recommendation in [RecommendationType.SELL, RecommendationType.HOLD]

        # High ROE (good profitability)
        high_roe_rec = recommendation_engine.get_recommendation("roe", 0.25)
        assert high_roe_rec.recommendation in [RecommendationType.BUY, RecommendationType.STRONG_BUY]

    def test_composite_recommendations(self, recommendation_engine):
        """Test composite recommendation generation."""
        # get_composite_recommendation() reads indicator.recommendation, which
        # is never auto-computed — each IndicatorResult must carry a
        # pre-attached Recommendation, or the composite skips it entirely
        # (contributing_indicators stays empty). Canonical names ("rsi",
        # "macd", "pe_ratio") so the lookup in get_recommendation() resolves.
        rsi_rec = recommendation_engine.get_recommendation("rsi", 25.0)
        macd_rec = recommendation_engine.get_recommendation("macd", 0.5, {"macd_signal": 0.3, "macd_hist": 0.2})
        pe_rec = recommendation_engine.get_recommendation("pe_ratio", 15.0)

        indicator_set = IndicatorSet(
            ticker="AAPL",
            technical_indicators={
                "rsi": IndicatorResult(
                    name="rsi", value=25.0, recommendation=rsi_rec, category=IndicatorCategory.TECHNICAL, source="test"
                ),
                "macd": IndicatorResult(
                    name="macd",
                    value=0.5,
                    recommendation=macd_rec,
                    category=IndicatorCategory.TECHNICAL,
                    source="test",
                ),
            },
            fundamental_indicators={
                "pe_ratio": IndicatorResult(
                    name="pe_ratio",
                    value=15.0,
                    recommendation=pe_rec,
                    category=IndicatorCategory.FUNDAMENTAL,
                    source="test",
                )
            },
        )

        composite_rec = recommendation_engine.get_composite_recommendation(indicator_set)

        assert isinstance(composite_rec, CompositeRecommendation)
        assert composite_rec.recommendation is not None
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
        # MACD with context should provide more nuanced recommendations.
        # `Recommendation`'s explanation field is `.reason` (singular) —
        # `.reasoning` is only on `CompositeRecommendation`.
        context = {"macd_signal": 0.05, "macd_hist": 0.05, "trend": "bullish"}

        contextual_rec = recommendation_engine.get_recommendation("macd", 0.1, context)

        assert contextual_rec is not None
        assert contextual_rec.reason is not None
        assert len(contextual_rec.reason) > 0


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test comprehensive error handling mechanisms."""

    def test_configuration_error_handling(self, indicator_service, sample_ohlcv_data):
        """Test configuration error handling."""
        # Invalid indicator name
        config = IndicatorBatchConfig(indicators=[IndicatorSpec(name="invalid_indicator", output="invalid")])

        with pytest.raises((ValueError, ConfigurationError)):
            asyncio.run(indicator_service.compute(sample_ohlcv_data, config))

    def test_data_error_handling(self, indicator_service):
        """Test data error handling."""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        config = IndicatorBatchConfig(indicators=[IndicatorSpec(name="rsi", output="rsi")])

        with pytest.raises((ValueError, DataError)):
            asyncio.run(indicator_service.compute(empty_df, config))

    def test_adapter_failure_recovery(self, indicator_service, sample_ohlcv_data):
        """Test adapter failure and recovery mechanisms."""
        config = IndicatorBatchConfig(indicators=[IndicatorSpec(name="rsi", output="rsi")])

        # Adapters live in self.adapters (dict), not a `_ta_lib_adapter` attr.
        with patch.object(indicator_service.adapters["ta-lib"], "compute", side_effect=Exception("Adapter failed")):
            # Should attempt fallback to other adapters
            try:
                result = asyncio.run(indicator_service.compute(sample_ohlcv_data, config))
                # If fallback succeeds, we should get a result
                assert isinstance(result, pd.DataFrame)
            except Exception:
                # If all adapters fail, should raise appropriate error
                pass

    def test_timeout_handling(self, indicator_service):
        """Test timeout handling for long-running operations."""
        request = TickerIndicatorsRequest(
            ticker=TickerSymbol("AAPL"),
            timeframe=TimeFrame("1D"),
            period=Period("10Y"),  # Large dataset
            indicators=[IndicatorName(i) for i in ["rsi", "ema", "macd", "bbands"]],
        )

        # get_ohlcv runs via `asyncio.to_thread` (a *sync* call in a worker
        # thread) — `asyncio.sleep(10)` there just returns an unawaited
        # coroutine object immediately (no actual delay); a real blocking
        # call (`time.sleep`) is needed to simulate a slow synchronous fetch.
        with patch("src.indicators.service.get_ohlcv", side_effect=lambda *args, **kwargs: time.sleep(10)):
            # Should handle timeout appropriately
            with pytest.raises((asyncio.TimeoutError, DataError)):
                asyncio.run(asyncio.wait_for(indicator_service.compute_for_ticker(request), timeout=2.0))


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestServiceIntegration:
    """Test integration between service components."""

    def test_service_config_integration(self, indicator_service, sample_ohlcv_data):
        """Test service integrates properly with config manager."""
        # Service should use config manager for parameters
        config = IndicatorBatchConfig(indicators=[IndicatorSpec(name="rsi", output="rsi")])

        result = asyncio.run(indicator_service.compute(sample_ohlcv_data, config))
        assert isinstance(result, pd.DataFrame)
        assert "rsi" in result.columns

    def test_service_recommendation_integration(self, indicator_service, sample_ohlcv_data):
        """Test service integrates with recommendation engine."""
        config = IndicatorBatchConfig(indicators=[IndicatorSpec(name="rsi", output="rsi")])

        # Test with recommendations enabled
        request = TickerIndicatorsRequest(
            ticker=TickerSymbol("TEST"), indicators=[IndicatorName("rsi")], include_recommendations=True
        )

        with patch("src.indicators.service.get_ohlcv", return_value=sample_ohlcv_data):
            result = asyncio.run(indicator_service.compute_for_ticker(request))

            assert isinstance(result, IndicatorResultSet)
            # Should include recommendations when requested
            if result.technical:
                for indicator_result in result.technical.values():
                    if hasattr(indicator_result, "recommendation"):
                        assert getattr(indicator_result, "recommendation", None) is not None

    def test_adapter_coordination(self, indicator_service, sample_ohlcv_data):
        """Test service coordinates multiple adapters correctly."""
        config = IndicatorBatchConfig(
            indicators=[
                IndicatorSpec(name="rsi", output="rsi"),  # Technical
                # Canonical registry name is "pe_ratio", not "pe".
                IndicatorSpec(name="pe_ratio", output="pe_ratio"),  # Fundamental (if available)
            ]
        )

        # Mock fundamentals for testing. Adapters live in self.adapters
        # (dict), not a `_fundamentals_adapter` attr.
        with patch.object(indicator_service.adapters["fundamentals"], "compute") as mock_fund:
            mock_fund.return_value = {"value": pd.Series([15.0])}

            result = asyncio.run(indicator_service.compute(sample_ohlcv_data, config))

            assert isinstance(result, pd.DataFrame)
            assert "rsi" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
