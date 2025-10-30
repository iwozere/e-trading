"""
Unit tests for AsyncTrendsAdapter.

Tests cover:
- Google Trends API integration and rate limiting
- Search volume analysis and trend detection
- Geographic distribution analysis
- Related queries sentiment analysis
- Proxy support and user agent rotation
"""
import pytest
import pytest_asyncio
import asyncio
import aiohttp
import random
import time
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.common.sentiments.adapters.async_trends import AsyncTrendsAdapter
from src.common.sentiments.adapters.base_adapter import AdapterStatus


class TestAsyncTrendsAdapter:
    """Test suite for AsyncTrendsAdapter."""

    @pytest_asyncio.fixture
    async def adapter(self):
        """Create adapter instance for testing."""
        # Create a mock session to avoid real HTTP requests
        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_session.close = AsyncMock()

        adapter = AsyncTrendsAdapter(
            concurrency=1,
            rate_limit_delay=0.01,
            max_retries=1,
            session=mock_session
        )
        yield adapter
        await adapter.close()

    @pytest.fixture
    def sample_interest_over_time_response(self):
        """Sample Google Trends interest over time response."""
        return {
            "default": {
                "timelineData": [
                    {
                        "time": "1672531200",  # 2023-01-01
                        "formattedTime": "Jan 1, 2023",
                        "value": [75]
                    },
                    {
                        "time": "1672617600",  # 2023-01-02
                        "formattedTime": "Jan 2, 2023",
                        "value": [82]
                    },
                    {
                        "time": "1672704000",  # 2023-01-03
                        "formattedTime": "Jan 3, 2023",
                        "value": [68]
                    }
                ]
            }
        }

    @pytest.fixture
    def sample_related_queries_response(self):
        """Sample Google Trends related queries response."""
        return {
            "default": {
                "rankedList": [
                    {
                        "rankedKeyword": [
                            {"query": "AAPL stock price", "value": 100},
                            {"query": "AAPL buy recommendation", "value": 85},
                            {"query": "AAPL earnings report", "value": 70},
                            {"query": "AAPL sell signal", "value": 45},
                            {"query": "AAPL crash prediction", "value": 30}
                        ]
                    }
                ]
            }
        }

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test adapter initialization."""
        adapter = AsyncTrendsAdapter(
            concurrency=2,
            rate_limit_delay=1.5,
            proxy_list=['http://proxy1:8080', 'http://proxy2:8080']
        )

        try:
            assert adapter.concurrency == 2
            assert adapter.rate_limit_delay == 1.5
            assert len(adapter.proxy_list) == 2
            assert adapter.trends_base == "https://trends.google.com/trends/api"
            assert adapter.requests_per_hour == 30
            assert len(adapter.regions) > 0
            assert len(adapter.user_agents) > 0
        finally:
            await adapter.close()

    def test_get_headers(self, adapter):
        """Test header generation with random user agent."""
        headers1 = adapter._get_headers()
        headers2 = adapter._get_headers()

        # Should contain required headers
        assert 'User-Agent' in headers1
        assert 'Accept' in headers1
        assert 'Accept-Language' in headers1

        # User agent should be from the predefined list
        assert headers1['User-Agent'] in adapter.user_agents

    def test_rate_limit_checking(self, adapter):
        """Test rate limit checking functionality."""
        # Initially should be within limits
        assert adapter._check_rate_limit() is True

        # Simulate many requests
        current_time = time.time()
        adapter._request_times = [current_time - i for i in range(30)]

        # Should now be at limit
        assert adapter._check_rate_limit() is False

        # Simulate old requests (outside window)
        adapter._request_times = [current_time - 4000 for _ in range(30)]

        # Should be within limits again
        assert adapter._check_rate_limit() is True

    def test_record_request(self, adapter):
        """Test request recording for rate limiting."""
        initial_count = len(adapter._request_times)

        adapter._record_request()

        assert len(adapter._request_times) == initial_count + 1

    def test_build_trends_query(self, adapter):
        """Test Google Trends query building."""
        query = adapter._build_trends_query("AAPL")

        # Should contain ticker in various formats
        assert "AAPL" in query
        assert "AAPL%20stock" in query or "AAPL+stock" in query
        assert "AAPL%20price" in query or "AAPL+price" in query

    def test_analyze_search_sentiment(self, adapter):
        """Test sentiment analysis of search queries."""
        test_queries = [
            "AAPL buy recommendation",  # bullish
            "AAPL stock price target",  # bullish
            "AAPL sell signal",         # bearish
            "AAPL crash prediction",    # bearish
            "AAPL earnings report",     # neutral
            "AAPL financial news"       # neutral
        ]

        sentiment = adapter._analyze_search_sentiment(test_queries)

        assert sentiment['bullish'] == 2
        assert sentiment['bearish'] == 2
        assert sentiment['neutral'] == 2

    @pytest.mark.asyncio
    async def test_fetch_interest_over_time_success(self, adapter, sample_interest_over_time_response):
        """Test successful interest over time fetching."""
        with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = sample_interest_over_time_response

            result = await adapter._fetch_interest_over_time("AAPL", "today 7-d", "US")

            assert result == sample_interest_over_time_response
            mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_related_queries_success(self, adapter, sample_related_queries_response):
        """Test successful related queries fetching."""
        with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = sample_related_queries_response

            result = await adapter._fetch_related_queries("AAPL", "US")

            assert result == sample_related_queries_response
            mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_messages_success(self, adapter, sample_interest_over_time_response):
        """Test successful message fetching from trends data."""
        with patch.object(adapter, '_fetch_interest_over_time', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = sample_interest_over_time_response

            messages = await adapter.fetch_messages("AAPL", limit=10)

            # Should return data points from multiple regions (up to 3)
            assert len(messages) > 0

            # Check first message structure
            if messages:
                msg = messages[0]
                assert msg["provider"] == "trends"
                assert "search_volume" in msg
                assert "region" in msg
                assert "timeframe" in msg
                assert msg["user"]["region"] in adapter.regions.values()

    @pytest.mark.asyncio
    async def test_fetch_messages_with_since_ts(self, adapter, sample_interest_over_time_response):
        """Test fetch messages with since_ts parameter affecting timeframe."""
        with patch.object(adapter, '_fetch_interest_over_time', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = sample_interest_over_time_response

            # Test different timeframes based on since_ts
            current_time = time.time()

            # 1 day ago
            since_ts_1d = current_time - 86400
            await adapter.fetch_messages("AAPL", since_ts=since_ts_1d)

            # 7 days ago
            since_ts_7d = current_time - (7 * 86400)
            await adapter.fetch_messages("AAPL", since_ts=since_ts_7d)

            # 30 days ago
            since_ts_30d = current_time - (30 * 86400)
            await adapter.fetch_messages("AAPL", since_ts=since_ts_30d)

            # Should have made multiple calls with different timeframes
            assert mock_fetch.call_count >= 3

    @pytest.mark.asyncio
    async def test_fetch_messages_invalid_ticker(self, adapter):
        """Test error handling for invalid ticker."""
        with pytest.raises(ValueError, match="Ticker cannot be empty"):
            await adapter.fetch_messages("")

        with pytest.raises(ValueError, match="Ticker cannot be empty"):
            await adapter.fetch_messages("   ")

    @pytest.mark.asyncio
    async def test_fetch_messages_no_data(self, adapter):
        """Test handling when no trends data is available."""
        with patch.object(adapter, '_fetch_interest_over_time', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = None

            messages = await adapter.fetch_messages("AAPL")

            assert messages == []

    @pytest.mark.asyncio
    async def test_fetch_summary_success(self, adapter, sample_interest_over_time_response, sample_related_queries_response):
        """Test successful summary generation."""
        with patch.object(adapter, 'fetch_messages', new_callable=AsyncMock) as mock_messages:
            with patch.object(adapter, '_fetch_related_queries', new_callable=AsyncMock) as mock_queries:
                # Mock trend data with rising pattern
                mock_messages.return_value = [
                    {"search_volume": 60, "region": "US", "created_at": "2023-01-01T10:00:00"},
                    {"search_volume": 70, "region": "US", "created_at": "2023-01-01T11:00:00"},
                    {"search_volume": 80, "region": "US", "created_at": "2023-01-01T12:00:00"},
                    {"search_volume": 90, "region": "GB", "created_at": "2023-01-01T13:00:00"},
                ]

                mock_queries.return_value = sample_related_queries_response

                summary = await adapter.fetch_summary("AAPL")

                assert summary["mentions"] == 4
                assert summary["total_search_volume"] == 300  # 60+70+80+90
                assert summary["avg_search_volume"] == 75.0   # 300/4
                assert summary["trend_direction"] == 1        # Rising trend
                assert summary["provider"] == "trends"
                assert "regional_distribution" in summary
                assert "related_queries" in summary
                assert len(summary["related_queries"]) > 0

                # Check sentiment from related queries
                assert summary["bullish"] == 2  # "buy recommendation" and "stock price"
                assert summary["bearish"] == 2  # "sell signal" and "crash prediction"
                assert summary["neutral"] == 1  # "earnings report"

    @pytest.mark.asyncio
    async def test_fetch_summary_falling_trend(self, adapter):
        """Test summary with falling search volume trend."""
        with patch.object(adapter, 'fetch_messages', new_callable=AsyncMock) as mock_messages:
            with patch.object(adapter, '_fetch_related_queries', new_callable=AsyncMock) as mock_queries:
                # Mock trend data with falling pattern
                mock_messages.return_value = [
                    {"search_volume": 90, "region": "US", "created_at": "2023-01-01T10:00:00"},
                    {"search_volume": 80, "region": "US", "created_at": "2023-01-01T11:00:00"},
                    {"search_volume": 70, "region": "US", "created_at": "2023-01-01T12:00:00"},
                    {"search_volume": 60, "region": "US", "created_at": "2023-01-01T13:00:00"},
                ]

                mock_queries.return_value = None

                summary = await adapter.fetch_summary("AAPL")

                assert summary["trend_direction"] == -1  # Falling trend
                assert summary["sentiment_score"] < 0    # Should be negative due to falling trend

    @pytest.mark.asyncio
    async def test_fetch_summary_no_data(self, adapter):
        """Test summary generation with no data."""
        with patch.object(adapter, 'fetch_messages', new_callable=AsyncMock) as mock_messages:
            with patch.object(adapter, '_fetch_related_queries', new_callable=AsyncMock) as mock_queries:
                mock_messages.return_value = []
                mock_queries.return_value = None

                summary = await adapter.fetch_summary("AAPL")

                assert summary["mentions"] == 0
                assert summary["total_search_volume"] == 0
                assert summary["avg_search_volume"] == 0.0
                assert summary["trend_direction"] == 0
                assert summary["bullish"] == 0
                assert summary["bearish"] == 0
                assert summary["neutral"] == 0

    @pytest.mark.asyncio
    async def test_get_with_retry_success(self, adapter):
        """Test successful HTTP request with retry logic."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value='{"test": "data"}')
        mock_response.request_info = Mock()
        mock_response.history = []

        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        adapter._session = mock_session

        result = await adapter._get_with_retry("https://test.com", {"param": "value"})

        assert result == {"test": "data"}
        assert adapter._health_info.status == AdapterStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_get_with_retry_jsonp_response(self, adapter):
        """Test handling of JSONP response format."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=')]}\'\n{"test": "data"}')
        mock_response.request_info = Mock()
        mock_response.history = []

        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        adapter._session = mock_session

        result = await adapter._get_with_retry("https://test.com")

        assert result == {"test": "data"}

    @pytest.mark.asyncio
    async def test_get_with_retry_rate_limit(self, adapter):
        """Test handling of rate limit (429) responses."""
        # First call returns 429, second call succeeds
        mock_response_429 = Mock()
        mock_response_429.status = 429
        mock_response_429.request_info = Mock()
        mock_response_429.history = []

        mock_response_200 = Mock()
        mock_response_200.status = 200
        mock_response_200.text = AsyncMock(return_value='{"success": true}')

        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.side_effect = [
            mock_response_429, mock_response_200
        ]
        adapter._session = mock_session

        result = await adapter._get_with_retry("https://test.com")

        assert result == {"success": True}

    @pytest.mark.asyncio
    async def test_get_with_retry_forbidden(self, adapter):
        """Test handling of forbidden (403) responses."""
        mock_response = Mock()
        mock_response.status = 403
        mock_response.request_info = Mock()
        mock_response.history = []

        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        adapter._session = mock_session

        result = await adapter._get_with_retry("https://test.com")

        # Should return None for forbidden access
        assert result is None

    @pytest.mark.asyncio
    async def test_get_with_retry_invalid_json(self, adapter):
        """Test handling of invalid JSON responses."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value='invalid json')
        mock_response.request_info = Mock()
        mock_response.history = []

        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        adapter._session = mock_session

        result = await adapter._get_with_retry("https://test.com")

        # Should return None for invalid JSON
        assert result is None

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

        adapter = AsyncTrendsAdapter(session=external_session)

        await adapter.close()

        # External session should not be closed
        external_session.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_proxy_usage(self):
        """Test that proxy list is used when provided."""
        proxy_list = ['http://proxy1:8080', 'http://proxy2:8080']
        adapter = AsyncTrendsAdapter(proxy_list=proxy_list)

        try:
            assert adapter.proxy_list == proxy_list

            # Mock session to verify proxy usage
            mock_session = AsyncMock()
            adapter._session = mock_session

            mock_response = Mock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value='{"test": "data"}')
            mock_session.get.return_value.__aenter__.return_value = mock_response

            await adapter._get_with_retry("https://test.com")

            # Verify get was called with proxy parameter
            call_kwargs = mock_session.get.call_args[1]
            assert 'proxy' in call_kwargs

        finally:
            await adapter.close()

    @pytest.mark.asyncio
    async def test_user_agent_rotation(self, adapter):
        """Test that user agents are rotated."""
        headers1 = adapter._get_headers()
        headers2 = adapter._get_headers()
        headers3 = adapter._get_headers()

        # All should have user agents from the predefined list
        user_agents = [headers1['User-Agent'], headers2['User-Agent'], headers3['User-Agent']]
        for ua in user_agents:
            assert ua in adapter.user_agents

    @pytest.mark.asyncio
    async def test_regional_data_processing(self, adapter):
        """Test processing of regional trend data."""
        trend_data = [
            {"search_volume": 50, "region": "US"},
            {"search_volume": 60, "region": "US"},
            {"search_volume": 40, "region": "GB"},
            {"search_volume": 30, "region": "CA"},
        ]

        with patch.object(adapter, 'fetch_messages', new_callable=AsyncMock) as mock_messages:
            with patch.object(adapter, '_fetch_related_queries', new_callable=AsyncMock) as mock_queries:
                mock_messages.return_value = trend_data
                mock_queries.return_value = None

                summary = await adapter.fetch_summary("AAPL")

                regional_dist = summary["regional_distribution"]
                assert "US" in regional_dist
                assert "GB" in regional_dist
                assert "CA" in regional_dist

                # US should have 2 data points with total volume 110
                assert regional_dist["US"]["count"] == 2
                assert regional_dist["US"]["total_volume"] == 110

    @pytest.mark.asyncio
    async def test_sentiment_score_calculation(self, adapter):
        """Test sentiment score calculation combining trend and query sentiment."""
        # Test rising trend with bullish queries
        trend_data = [
            {"search_volume": 50, "region": "US"},  # older data
            {"search_volume": 80, "region": "US"},  # recent data (higher)
        ]

        related_queries_response = {
            "default": {
                "rankedList": [
                    {
                        "rankedKeyword": [
                            {"query": "AAPL buy recommendation", "value": 100},  # bullish
                            {"query": "AAPL stock price target", "value": 85},   # bullish
                        ]
                    }
                ]
            }
        }

        with patch.object(adapter, 'fetch_messages', new_callable=AsyncMock) as mock_messages:
            with patch.object(adapter, '_fetch_related_queries', new_callable=AsyncMock) as mock_queries:
                mock_messages.return_value = trend_data
                mock_queries.return_value = related_queries_response

                summary = await adapter.fetch_summary("AAPL")

                # Should have positive sentiment (rising trend + bullish queries)
                assert summary["sentiment_score"] > 0
                assert summary["trend_direction"] == 1
                assert summary["bullish"] == 2
                assert summary["bearish"] == 0
