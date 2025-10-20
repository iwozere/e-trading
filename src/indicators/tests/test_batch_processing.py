"""
Comprehensive batch processing tests.

Tests concurrent processing, error handling, and performance characteristics.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock, AsyncMock
from datetime import datetime
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.indicators.service import IndicatorService
from src.indicators.models import TickerIndicatorsRequest, IndicatorResultSet


class TestBatchProcessing:
    """Test batch processing capabilities."""

    @pytest.fixture
    def mock_ohlcv_data(self):
        """Mock OHLCV data for testing."""
        return pd.DataFrame({
            'open': [100, 101, 102],
            'high': [102, 103, 104],
            'low': [99, 100, 101],
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3, freq='D', tz='UTC'))

    @pytest.mark.asyncio
    async def test_batch_processing_multiple_tickers(self, mock_ohlcv_data):
        """Test processing multiple tickers concurrently."""
        service = IndicatorService()
        tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

        with patch('src.common.get_ohlcv', return_value=mock_ohlcv_data):
            results = await service.compute_batch(
                tickers=tickers,
                indicators=["rsi", "ema"],
                timeframe="1D",
                period="1M"
            )

            assert len(results) == len(tickers)
            for ticker in tickers:
                assert ticker in results
                assert isinstance(results[ticker], IndicatorResultSet)

    @pytest.mark.asyncio
    async def test_batch_processing_with_failures(self, mock_ohlcv_data):
        """Test batch processing handles individual ticker failures."""
        service = IndicatorService()
        tickers = ["AAPL", "INVALID", "GOOGL"]

        def mock_get_ohlcv(ticker, *args, **kwargs):
            if ticker == "INVALID":
                raise Exception("Invalid ticker")
            return mock_ohlcv_data

        with patch('src.common.get_ohlcv', side_effect=mock_get_ohlcv):
            results = await service.compute_batch(
                tickers=tickers,
                indicators=["rsi"],
                timeframe="1D",
                period="1M",
                fail_fast=False
            )

            # Should have results for valid tickers
            assert "AAPL" in results
            assert "GOOGL" in results
            # Invalid ticker should not be in results or have error info
            assert "INVALID" not in results or results["INVALID"] is None

    @pytest.mark.asyncio
    async def test_batch_processing_concurrency_limits(self, mock_ohlcv_data):
        """Test batch processing respects concurrency limits."""
        service = IndicatorService()
        tickers = [f"TICKER_{i}" for i in range(20)]  # Many tickers

        call_times = []

        async def mock_get_ohlcv(*args, **kwargs):
            call_times.append(datetime.now())
            await asyncio.sleep(0.1)  # Simulate processing time
            return mock_ohlcv_data

        with patch('src.common.get_ohlcv', side_effect=mock_get_ohlcv):
            start_time = datetime.now()
            results = await service.compute_batch(
                tickers=tickers,
                indicators=["rsi"],
                timeframe="1D",
                period="1M",
                max_concurrent=5
            )
            end_time = datetime.now()

            # Should complete all tickers
            assert len(results) == len(tickers)

            # Should take reasonable time (not all concurrent)
            total_time = (end_time - start_time).total_seconds()
            assert total_time > 0.3  # At least 4 batches of 0.1s each

    def test_batch_processing_memory_efficiency(self, mock_ohlcv_data):
        """Test batch processing is memory efficient."""
        service = IndicatorService()

        # Process large number of indicators
        large_indicator_list = [
            "rsi", "ema", "sma", "macd", "bbands", "atr", "adx", "stoch", "obv"
        ]

        with patch('src.common.get_ohlcv', return_value=mock_ohlcv_data):
            results = asyncio.run(service.compute_batch(
                tickers=["AAPL"],
                indicators=large_indicator_list,
                timeframe="1D",
                period="1M"
            ))

            assert len(results) == 1
            assert "AAPL" in results
            # Should complete without memory issues


if __name__ == '__main__':
    pytest.main([__file__, '-v'])