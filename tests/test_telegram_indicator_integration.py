#!/usr/bin/env python3
"""
Test Telegram Indicator Service Integration
Tests the integration between telegram business logic and IndicatorService using service mocks.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pytest
import asyncio
from tests.fixtures.service_fixtures import (
    business_logic_with_mocks, telegram_service_mock, indicator_service_mock,
    sample_indicator_result, setup_user_in_mock, setup_indicator_data_in_mock
)
from src.indicators.models import TickerIndicatorsRequest, IndicatorResultSet, IndicatorValue


class TestTelegramIndicatorIntegration:
    """Test telegram business logic indicator service integration using service mocks."""

    @pytest.mark.asyncio
    async def test_calculate_indicators_for_ticker_success(self, business_logic_with_mocks):
        """Test successful indicator calculation using service mocks."""
        business_logic, telegram_service_mock, indicator_service_mock = business_logic_with_mocks

        # Setup indicator data in mock
        setup_indicator_data_in_mock(
            indicator_service_mock,
            "AAPL",
            technical_indicators={"RSI": 65.5, "MACD": 1.23},
            fundamental_indicators={"PE": 25.4}
        )

        # Test the method
        result = await business_logic.calculate_indicators_for_ticker(
            ticker="AAPL",
            indicators=["RSI", "MACD", "PE"],
            timeframe="1d",
            period="1y"
        )

        # Verify results
        assert result["status"] == "ok"
        assert result["ticker"] == "AAPL"
        assert "RSI" in result["technical"]
        assert "PE" in result["fundamental"]
        assert result["technical"]["RSI"].value == 65.5

        # Verify service was called
        assert indicator_service_mock.get_call_count("compute_for_ticker") == 1

    @pytest.mark.asyncio
    async def test_calculate_indicators_for_ticker_error(self, business_logic_with_mocks):
        """Test error handling in indicator calculation using service mocks."""
        business_logic, telegram_service_mock, indicator_service_mock = business_logic_with_mocks

        # Configure mock to raise exception
        indicator_service_mock.configure_error("compute_for_ticker", Exception("API key missing"))

        # Test the method
        result = await business_logic.calculate_indicators_for_ticker(
            ticker="AAPL",
            indicators=["RSI"]
        )

        # Verify error handling
        assert result["status"] == "error"
        assert "AAPL" in result.get("message", "") or "API" in result.get("message", "")

        # Verify service was called despite error
        assert indicator_service_mock.get_call_count("compute_for_ticker") == 1

    def test_convert_telegram_indicator_request_basic(self, business_logic_with_mocks):
        """Test conversion of telegram parameters to indicator request using service mocks."""
        business_logic, _, _ = business_logic_with_mocks

        telegram_params = {
            "ticker": "aapl",
            "indicators": "RSI,MACD,SMA",
            "interval": "1h",
            "period": "6mo",
            "provider": "yahoo"
        }

        request = business_logic.convert_telegram_indicator_request(telegram_params)

        assert request.ticker == "AAPL"
        assert "RSI" in request.indicators
        assert "MACD" in request.indicators
        assert "SMA" in request.indicators
        assert request.timeframe == "1h"
        assert request.period == "6mo"
        assert request.provider == "yahoo"

    def test_convert_telegram_indicator_request_list_format(self, business_logic_with_mocks):
        """Test conversion with indicators as list using service mocks."""
        business_logic, _, _ = business_logic_with_mocks

        telegram_params = {
            "ticker": "TSLA",
            "indicators": ["rsi", "bb", "ma"],
        }

        request = business_logic.convert_telegram_indicator_request(telegram_params)

        assert request.ticker == "TSLA"
        assert "RSI" in request.indicators
        assert "BollingerBands" in request.indicators  # BB mapped to BollingerBands
        assert "SMA" in request.indicators  # MA mapped to SMA

    def test_convert_telegram_indicator_request_defaults(self, business_logic_with_mocks):
        """Test conversion with default values using service mocks."""
        business_logic, _, _ = business_logic_with_mocks

        telegram_params = {
            "ticker": "BTC"
        }

        request = business_logic.convert_telegram_indicator_request(telegram_params)

        assert request.ticker == "BTC"
        assert len(request.indicators) > 0  # Should have default indicators
        assert request.timeframe == "1d"  # Default
        assert request.period == "1y"  # Default

    def test_convert_telegram_indicator_request_invalid(self, business_logic_with_mocks):
        """Test conversion with invalid parameters using service mocks."""
        business_logic, _, _ = business_logic_with_mocks

        telegram_params = {}  # Missing ticker

        with pytest.raises(ValueError, match="Ticker is required"):
            business_logic.convert_telegram_indicator_request(telegram_params)

    def test_handle_indicator_service_error_api_key(self, business_logic_with_mocks):
        """Test error handling for API key errors using service mocks."""
        business_logic, _, _ = business_logic_with_mocks

        error = Exception("API key missing or invalid")

        result = business_logic.handle_indicator_service_error(error, "test_context")

        assert result["status"] == "error"
        assert "API configuration" in result["message"]

    def test_handle_indicator_service_error_rate_limit(self, business_logic_with_mocks):
        """Test error handling for rate limit errors using service mocks."""
        business_logic, _, _ = business_logic_with_mocks

        error = Exception("Rate limit exceeded")

        result = business_logic.handle_indicator_service_error(error, "test_context")

        assert result["status"] == "error"
        assert "Rate limit" in result["message"]

    def test_handle_indicator_service_error_ticker_not_found(self, business_logic_with_mocks):
        """Test error handling for invalid ticker errors using service mocks."""
        business_logic, _, _ = business_logic_with_mocks

        error = Exception("Ticker not found")

        result = business_logic.handle_indicator_service_error(error, "test_context")

        assert result["status"] == "error"
        assert "Ticker symbol not found" in result["message"]

    def test_extract_indicator_values(self, business_logic_with_mocks, sample_indicator_result):
        """Test extraction of indicator values for display using service mocks."""
        business_logic, _, _ = business_logic_with_mocks

        formatted = business_logic.extract_indicator_values(sample_indicator_result)

        assert "RSI" in formatted
        assert "SMA" in formatted
        assert "PE" in formatted
        assert formatted["RSI"]["type"] == "technical"
        assert formatted["PE"]["type"] == "fundamental"
        assert formatted["RSI"]["value"] == 65.5
        assert "65.50" in formatted["RSI"]["formatted"]
        assert "$150.25" in formatted["SMA"]["formatted"]

    def test_format_indicator_value_rsi(self, business_logic_with_mocks):
        """Test formatting of RSI values using service mocks."""
        business_logic, _, _ = business_logic_with_mocks

        formatted = business_logic._format_indicator_value("RSI", 65.5432)
        assert formatted == "65.54"

    def test_format_indicator_value_price(self, business_logic_with_mocks):
        """Test formatting of price-based values using service mocks."""
        business_logic, _, _ = business_logic_with_mocks

        formatted = business_logic._format_indicator_value("SMA", 150.2567)
        assert formatted == "$150.26"

    def test_format_indicator_value_generic(self, business_logic_with_mocks):
        """Test formatting of generic numeric values using service mocks."""
        business_logic, _, _ = business_logic_with_mocks

        formatted = business_logic._format_indicator_value("MACD", 1.23456)
        assert formatted == "1.2346"


if __name__ == "__main__":
    pytest.main([__file__])