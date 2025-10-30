"""
Unit tests for AsyncStocktwitsAdapter.

Tests cover:
- Message fetching with various scenarios
- Summary generation and sentiment scoring
- Error handling and retry logic
- Health monitoring and circuit breaker
- Rate limiting and concurrency
"""
import pytest
import pytest_asyncio
import asyncio
import aiohttp
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.common.sentiments.adapters.async_stocktwits import AsyncStocktwitsAdapter
from src.common.sentiments.adapters.base_adapter import AdapterStatus


class TestAsyncStocktwitsAdapter:
    """Test suite for AsyncStocktwitsAdapter."""

    @pytest_asyncio.fixture
    async def adapter(self):
        """Create adapter instance for testing."""
        # Create a mock session to avoid real HTTP requests
        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_session.close = AsyncMock()

        adapter = AsyncStocktwitsAdapter(
            concurrency=2,
            rate_limit_delay=0.01,
            max_retries=1,
            session=mock_session
        )
        yield adapter
        await adapter.close()

    @pytest.fixture
    def mock_session(self):
        """Create mock aiohttp session."""
        session = Mock(spec=aiohttp.ClientSession)
        return session

    @pytest.fixture
    def sample_stocktwits_response(self):
        """Sample StockTwits API response."""
        return {
            "messages": [
                {
                    "id": 123456,
                    "body": "AAPL to the moon! 🚀 Buy and hold!",
                    "created_at": "2023-01-01T12:00:00Z",
                    "user": {
                        "id": 789,
                        "username": "trader123",
                        "followers": 1500
                    },
                    "likes": 25,
                    "replies": 5
                },
                {
                    "id": 123457,
                    "body": "AAPL looks bearish, might sell soon",
                    "created_at": "2023-01-01T11:30:00Z",
                    "user": {
                        "id": 790,
                        "username": "bear_trader",
                        "followers": 800
                    },
                    "likes": 10,
                    "replies": 2
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_fetch_messages_success(self, adapter, sample_stocktwits_response):
        """Test successful message fetching."""
        with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = sample_stocktwits_response

            messages = await adapter.fetch_messages("AAPL", limit=10)

            assert len(messages) == 2
            assert messages[0]["id"] == "123456"
            assert messages[0]["body"] == "AAPL to the moon! 🚀 Buy and hold!"
            assert messages[0]["provider"] == "stocktwits"
            assert messages[0]["user"]["username"] == "trader123"
            assert messages[0]["likes"] == 25
            assert messages[0]["replies"] == 5
            assert messages[0]["retweets"] == 0  # StockTwits doesn't have retweets

    @pytest.mark.asyncio
    async def test_fetch_messages_empty_response(self, adapter):
        """Test handling of empty API response."""
        with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"messages": []}

            messages = await adapter.fetch_messages("INVALID")

            assert messages == []

    @pytest.mark.asyncio
    async def test_fetch_messages_invalid_ticker(self, adapter):
        """Test error handling for invalid ticker."""
        with pytest.raises(ValueError, match="Ticker cannot be empty"):
            await adapter.fetch_messages("")

        with pytest.raises(ValueError, match="Ticker cannot be empty"):
            await adapter.fetch_messages("   ")

    @pytest.mark.asyncio
    async def test_fetch_messages_malformed_data(self, adapter):
        """Test handling of malformed message data."""
        malformed_response = {
            "messages": [
                {"id": 123, "body": "Valid message"},  # Valid
                {"body": "Missing ID"},  # Invalid - no ID
                {"id": 124},  # Valid but missing body
                None,  # Invalid - null message
            ]
        }

        with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = malformed_response

            messages = await adapter.fetch_messages("AAPL")

            # Should only return valid messages
            assert len(messages) == 2
            assert messages[0]["id"] == "123"
            assert messages[1]["id"] == "124"
            assert messages[1]["body"] == ""  # Empty body should be handled

    @pytest.mark.asyncio
    async def test_fetch_summary_success(self, adapter, sample_stocktwits_response):
        """Test successful summary generation."""
        with patch.object(adapter, 'fetch_messages', new_callable=AsyncMock) as mock_fetch:
            # Create messages with clear sentiment
            messages = [
                {"body": "AAPL to the moon! 🚀 Buy and hold!"},  # Bullish
                {"body": "AAPL looks bearish, might sell soon"},  # Bearish
                {"body": "AAPL neutral comment"},  # Neutral
                {"body": "Another bullish rocket 🚀 comment"},  # Bullish
            ]
            mock_fetch.return_value = messages

            summary = await adapter.fetch_summary("AAPL")

            assert summary["mentions"] == 4
            assert summary["bullish"] == 2
            assert summary["bearish"] == 1
            assert summary["neutral"] == 1
            assert summary["sentiment_score"] == 0.25  # (2-1)/4 = 0.25
            assert summary["provider"] == "stocktwits"
            assert "timestamp" in summary

    @pytest.mark.asyncio
    async def test_fetch_summary_no_messages(self, adapter):
        """Test summary generation with no messages."""
        with patch.object(adapter, 'fetch_messages', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = []

            summary = await adapter.fetch_summary("INVALID")

            assert summary["mentions"] == 0
            assert summary["bullish"] == 0
            assert summary["bearish"] == 0
            assert summary["neutral"] == 0
            assert summary["sentiment_score"] == 0.0

    @pytest.mark.asyncio
    async def test_fetch_summary_sentiment_scoring(self, adapter):
        """Test sentiment scoring algorithm."""
        test_cases = [
            # (messages, expected_score)
            ([{"body": "bull moon rocket 🚀"}], 1.0),  # All bullish
            ([{"body": "bear short sell dump"}], -1.0),  # All bearish
            ([{"body": "neutral comment"}], 0.0),  # All neutral
            ([{"body": "bull"}, {"body": "bear"}], 0.0),  # Equal bull/bear
        ]

        for messages, expected_score in test_cases:
            with patch.object(adapter, 'fetch_messages', new_callable=AsyncMock) as mock_fetch:
                mock_fetch.return_value = messages

                summary = await adapter.fetch_summary("TEST")
                assert summary["sentiment_score"] == expected_score

    @pytest.mark.asyncio
    async def test_get_with_retry_success(self, adapter):
        """Test successful HTTP request with retry logic."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"test": "data"})
        mock_response.raise_for_status = Mock()
        mock_response.request_info = Mock()
        mock_response.history = []

        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_response
        mock_context_manager.__aexit__.return_value = None

        import aiohttp
        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_session.get = Mock(return_value=mock_context_manager)
        adapter._session = mock_session

        result = await adapter._get_with_retry("/test", params={"param": "value"})

        assert result == {"test": "data"}
        assert adapter._health_info.status == AdapterStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_get_with_retry_rate_limit(self, adapter):
        """Test handling of rate limit (429) responses."""
        # First call returns 429, second call succeeds
        mock_response_429 = Mock()
        mock_response_429.status = 429
        mock_response_429.headers = {'retry-after': '0.1'}  # Shorter delay for testing
        mock_response_429.request_info = Mock()
        mock_response_429.history = []

        mock_response_200 = Mock()
        mock_response_200.status = 200
        mock_response_200.json = AsyncMock(return_value={"success": True})
        mock_response_200.raise_for_status = Mock()
        mock_response_200.request_info = Mock()
        mock_response_200.history = []

        mock_context_manager_429 = AsyncMock()
        mock_context_manager_429.__aenter__.return_value = mock_response_429
        mock_context_manager_429.__aexit__.return_value = None

        mock_context_manager_200 = AsyncMock()
        mock_context_manager_200.__aenter__.return_value = mock_response_200
        mock_context_manager_200.__aexit__.return_value = None

        import aiohttp
        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_session.get = Mock(side_effect=[mock_context_manager_429, mock_context_manager_200])
        adapter._session = mock_session

        result = await adapter._get_with_retry("/test")

        assert result == {"success": True}
        # Should have made 2 calls (retry after 429)
        assert mock_session.get.call_count == 2

    @pytest.mark.asyncio
    async def test_get_with_retry_server_error(self, adapter):
        """Test handling of server errors (5xx)."""
        mock_response_500 = Mock()
        mock_response_500.status = 500
        mock_response_500.request_info = Mock()
        mock_response_500.history = []

        mock_response_200 = Mock()
        mock_response_200.status = 200
        mock_response_200.json = AsyncMock(return_value={"recovered": True})
        mock_response_200.raise_for_status = Mock()

        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.side_effect = [
            mock_response_500, mock_response_200
        ]
        adapter._session = mock_session

        result = await adapter._get_with_retry("/test")

        assert result == {"recovered": True}

    @pytest.mark.asyncio
    async def test_get_with_retry_max_retries_exceeded(self, adapter):
        """Test behavior when max retries are exceeded."""
        mock_response = Mock()
        mock_response.status = 500
        mock_response.request_info = Mock()
        mock_response.history = []

        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        adapter._session = mock_session

        result = await adapter._get_with_retry("/test")

        assert result is None
        assert adapter._health_info.status == AdapterStatus.DEGRADED
        # Should have made max_retries + 1 calls
        assert mock_session.get.call_count == adapter.max_retries + 1

    @pytest.mark.asyncio
    async def test_get_with_retry_timeout(self, adapter):
        """Test handling of timeout errors."""
        mock_session = AsyncMock()
        mock_session.get.side_effect = asyncio.TimeoutError("Request timeout")
        adapter._session = mock_session

        result = await adapter._get_with_retry("/test")

        assert result is None
        assert adapter._health_info.status == AdapterStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_health_check(self, adapter):
        """Test health check functionality."""
        health = await adapter.health_check()

        assert health.status == AdapterStatus.HEALTHY
        assert health.failure_count == 0
        assert health.error_message is None

    @pytest.mark.asyncio
    async def test_health_update_on_failure(self, adapter):
        """Test health status updates on failures."""
        # Simulate multiple failures
        error = Exception("Test error")
        for i in range(3):
            adapter._update_health_failure(error)

        health = await adapter.health_check()

        assert health.status == AdapterStatus.DEGRADED
        assert health.failure_count == 3
        assert health.error_message == "Test error"
        assert health.last_failure is not None

    @pytest.mark.asyncio
    async def test_health_update_on_success(self, adapter):
        """Test health status updates on success."""
        # First simulate failure
        adapter._update_health_failure(Exception("Test error"))

        # Then simulate success
        adapter._update_health_success(100.0)

        health = await adapter.health_check()

        assert health.status == AdapterStatus.HEALTHY
        assert health.failure_count == 0
        assert health.error_message is None
        assert health.response_time_ms == 100.0
        assert health.last_success is not None

    @pytest.mark.asyncio
    async def test_is_healthy(self, adapter):
        """Test is_healthy method."""
        assert adapter.is_healthy() is True

        # Simulate failures to make it degraded
        for _ in range(3):
            adapter._update_health_failure(Exception("Test"))

        assert adapter.is_healthy() is True  # Degraded is still considered healthy

        # Simulate more failures to make it failed
        for _ in range(3):
            adapter._update_health_failure(Exception("Test"))

        assert adapter.is_healthy() is False  # Failed is not healthy

    @pytest.mark.asyncio
    async def test_concurrency_limiting(self, adapter):
        """Test that concurrency is properly limited."""
        # Create adapter with concurrency limit of 1
        limited_adapter = AsyncStocktwitsAdapter(concurrency=1, rate_limit_delay=0.01)

        try:
            call_times = []

            async def mock_slow_request(*args, **kwargs):
                call_times.append(asyncio.get_event_loop().time())
                await asyncio.sleep(0.1)  # Simulate slow request
                return {"messages": []}

            with patch.object(limited_adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
                mock_get.side_effect = mock_slow_request

                # Start multiple concurrent requests
                tasks = [
                    limited_adapter.fetch_messages("AAPL"),
                    limited_adapter.fetch_messages("TSLA"),
                    limited_adapter.fetch_messages("MSFT")
                ]

                await asyncio.gather(*tasks)

                # Verify requests were serialized (not truly concurrent)
                assert len(call_times) == 3
                # With concurrency=1, requests should be serialized
                time_diffs = [call_times[i+1] - call_times[i] for i in range(len(call_times)-1)]
                assert all(diff >= 0.05 for diff in time_diffs)  # Should have gaps due to serialization

        finally:
            await limited_adapter.close()

    @pytest.mark.asyncio
    async def test_close_cleanup(self, adapter):
        """Test proper resource cleanup on close."""
        # Set up a session
        adapter._session = Mock(spec=aiohttp.ClientSession)
        adapter._session.close = AsyncMock()

        await adapter.close()

        # Verify session was closed
        adapter._session.close.assert_called_once()
        assert adapter._session is None

    @pytest.mark.asyncio
    async def test_close_with_provided_session(self):
        """Test that provided sessions are not closed."""
        external_session = Mock(spec=aiohttp.ClientSession)
        external_session.close = AsyncMock()

        adapter = AsyncStocktwitsAdapter(session=external_session)

        await adapter.close()

        # External session should not be closed
        external_session.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetch_messages_limit_enforcement(self, adapter):
        """Test that message limit is properly enforced."""
        # Create response with many messages
        large_response = {
            "messages": [
                {"id": i, "body": f"Message {i}", "user": {"id": i, "username": f"user{i}"}}
                for i in range(50)
            ]
        }

        with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = large_response

            messages = await adapter.fetch_messages("AAPL", limit=10)

            # Should respect the limit
            assert len(messages) <= 10

    @pytest.mark.asyncio
    async def test_ticker_normalization(self, adapter):
        """Test that tickers are properly normalized."""
        with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"messages": []}

            # Test various ticker formats
            await adapter.fetch_messages("  aapl  ")  # Should be normalized to AAPL
            await adapter.fetch_messages("tsla")      # Should be normalized to TSLA

            # Verify the API was called with normalized tickers
            calls = mock_get.call_args_list
            assert "/streams/symbol/AAPL.json" in calls[0][0][0]
            assert "/streams/symbol/TSLA.json" in calls[1][0][0]