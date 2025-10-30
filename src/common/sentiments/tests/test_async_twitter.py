"""
Unit tests for AsyncTwitterAdapter.

Tests cover:
- Twitter API v2 integration and authentication
- Tweet fetching with engagement metrics
- Hashtag and mention tracking
- Rate limiting and error handling
- Sentiment analysis and summary generation
"""
import pytest
import pytest_asyncio
import asyncio
import aiohttp
import os
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from pathlib import Path
import sys
import time

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.common.sentiments.adapters.async_twitter import AsyncTwitterAdapter
from src.common.sentiments.adapters.base_adapter import AdapterStatus


class TestAsyncTwitterAdapter:
    """Test suite for AsyncTwitterAdapter."""

    @pytest_asyncio.fixture
    async def adapter(self):
        """Create adapter instance for testing."""
        # Create a mock session to avoid real HTTP requests
        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_session.close = AsyncMock()

        adapter = AsyncTwitterAdapter(
            bearer_token="test_token",
            concurrency=2,
            rate_limit_delay=0.01,
            max_retries=1,
            session=mock_session
        )
        yield adapter
        await adapter.close()

    @pytest.fixture
    def sample_twitter_response(self):
        """Sample Twitter API v2 response."""
        return {
            "data": [
                {
                    "id": "1234567890",
                    "text": "AAPL to the moon! 🚀 Great earnings report, buying more shares #AAPL",
                    "created_at": "2023-01-01T12:00:00.000Z",
                    "author_id": "user123",
                    "public_metrics": {
                        "like_count": 150,
                        "retweet_count": 45,
                        "reply_count": 20,
                        "quote_count": 10
                    },
                    "entities": {
                        "hashtags": [{"tag": "AAPL"
}],
                        "mentions": [{"username": "elonmusk"}]
                    }
                },
                {
                    "id": "1234567891",
                    "text": "AAPL looks bearish, might crash soon. Selling my position #bearish",
                    "created_at": "2023-01-01T11:30:00.000Z",
                    "author_id": "user456",
                    "public_metrics": {
                        "like_count": 75,
                        "retweet_count": 15,
                        "reply_count": 8,
                        "quote_count": 3
                    },
                    "entities": {
                        "hashtags": [{"tag": "bearish"}],
                        "mentions": []
                    }
                }
            ],
            "includes": {
                "users": [
                    {
                        "id": "user123",
                        "username": "bullish_trader",
                        "name": "Bull Trader",
                        "public_metrics": {
                            "followers_count": 5000
                        },
                        "verified": True
                    },
                    {
                        "id": "user456",
                        "username": "bear_trader",
                        "name": "Bear Trader",
                        "public_metrics": {
                            "followers_count": 2000
                        },
                        "verified": False
                    }
                ]
            }
        }

    @pytest.mark.asyncio
    async def test_initialization_with_token(self):
        """Test adapter initialization with bearer token."""
        adapter = AsyncTwitterAdapter(bearer_token="test_token")

        try:
            assert adapter.bearer_token == "test_token"
            assert adapter.base_url == "https://api.twitter.com/2"
            assert adapter.search_rate_limit == 300
            assert adapter.search_window == 900  # 15 minutes
        finally:
            await adapter.close()

    @pytest.mark.asyncio
    async def test_initialization_without_token(self):
        """Test adapter initialization without bearer token."""
        with patch.dict(os.environ, {}, clear=True):
            adapter = AsyncTwitterAdapter()

            try:
                assert adapter.bearer_token is None
            finally:
                await adapter.close()

    def test_get_headers_success(self, adapter):
        """Test authentication header generation."""
        headers = adapter._get_headers()

        assert headers['Authorization'] == 'Bearer test_token'
        assert headers['Content-Type'] == 'application/json'

    def test_get_headers_no_token(self):
        """Test header generation without token raises error."""
        adapter = AsyncTwitterAdapter(bearer_token=None)

        try:
            with pytest.raises(ValueError, match="Twitter bearer token not configured"):
                adapter._get_headers()
        finally:
            asyncio.run(adapter.close())

    def test_build_search_query(self, adapter):
        """Test search query building."""
        query = adapter._build_search_query("AAPL")

        expected_parts = ["$AAPL", "#AAPL", '"AAPL"']
        assert all(part in query for part in expected_parts)
        assert "-is:retweet" in query
        assert "lang:en" in query

    def test_rate_limit_checking(self, adapter):
        """Test rate limit checking functionality."""
        # Initially should be within limits
        assert adapter._check_rate_limit() is True

        # Simulate many requests
        import time
        current_time = time.time()
        adapter._search_requests = [current_time - i for i in range(300)]

        # Should now be at limit
        assert adapter._check_rate_limit() is False

        # Simulate old requests (outside window)
        adapter._search_requests = [current_time - 1000 for _ in range(300)]

        # Should be within limits again
        assert adapter._check_rate_limit() is True

    def test_record_request(self, adapter):
        """Test request recording for rate limiting."""
        initial_count = len(adapter._search_requests)

        adapter._record_request()

        assert len(adapter._search_requests) == initial_count + 1

    @pytest.mark.asyncio
    async def test_fetch_messages_success(self, adapter, sample_twitter_response):
        """Test successful tweet fetching."""
        with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = sample_twitter_response

            messages = await adapter.fetch_messages("AAPL", limit=10)

            assert len(messages) == 2

            # Check first tweet
            tweet1 = messages[0]
            assert tweet1["id"] == "1234567890"
            assert "AAPL to the moon!" in tweet1["body"]
            assert tweet1["provider"] == "twitter"
            assert tweet1["user"]["username"] == "bullish_trader"
            assert tweet1["user"]["verified"] is True
            assert tweet1["user"]["followers"] == 5000
            assert tweet1["likes"] == 150
            assert tweet1["retweets"] == 45
            assert tweet1["replies"] == 20
            assert tweet1["quotes"] == 10
            assert "AAPL" in tweet1["hashtags"]
            assert "elonmusk" in tweet1["mentions"]

            # Check second tweet
            tweet2 = messages[1]
            assert tweet2["id"] == "1234567891"
            assert "bearish" in tweet2["body"]
            assert tweet2["user"]["verified"] is False

    @pytest.mark.asyncio
    async def test_fetch_messages_no_token(self):
        """Test fetch messages without bearer token."""
        adapter = AsyncTwitterAdapter(bearer_token=None)

        try:
            messages = await adapter.fetch_messages("AAPL")
            assert messages == []
        finally:
            await adapter.close()

    @pytest.mark.asyncio
    async def test_fetch_messages_empty_response(self, adapter):
        """Test handling of empty API response."""
        with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"data": []}

            messages = await adapter.fetch_messages("AAPL")

            assert messages == []

    @pytest.mark.asyncio
    async def test_fetch_messages_no_payload(self, adapter):
        """Test handling when no payload is received."""
        with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            messages = await adapter.fetch_messages("AAPL")

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
        """Test handling of malformed tweet data."""
        malformed_response = {
            "data": [
                {"id": "123", "text": "Valid tweet", "author_id": "user1"},  # Valid
                {"text": "Missing ID", "author_id": "user2"},  # Invalid - no ID
                {"id": "124"},  # Valid but missing text
                None,  # Invalid - null tweet
            ],
            "includes": {
                "users": [
                    {"id": "user1", "username": "user1", "public_metrics": {"followers_count": 100}},
                    {"id": "user2", "username": "user2", "public_metrics": {"followers_count": 200}}
                ]
            }
        }

        with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = malformed_response

            messages = await adapter.fetch_messages("AAPL")

            # Should only return valid messages
            assert len(messages) == 2
            assert messages[0]["id"] == "123"
            assert messages[1]["id"] == "124"

    @pytest.mark.asyncio
    async def test_fetch_messages_with_since_ts(self, adapter, sample_twitter_response):
        """Test fetch messages with since_ts parameter."""
        with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = sample_twitter_response

            since_ts = 1672531200  # 2023-01-01 00:00:00
            await adapter.fetch_messages("AAPL", since_ts=since_ts)

            # Verify start_time parameter was added
            call_args = mock_get.call_args[1]
            assert 'start_time' in call_args['params']
            assert call_args['params']['start_time'].startswith('2023-01-01')

    @pytest.mark.asyncio
    async def test_fetch_messages_limit_enforcement(self, adapter, sample_twitter_response):
        """Test that message limit is properly enforced."""
        with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = sample_twitter_response

            messages = await adapter.fetch_messages("AAPL", limit=1)

            # Should respect the limit
            assert len(messages) <= 1

    @pytest.mark.asyncio
    async def test_fetch_summary_success(self, adapter):
        """Test successful summary generation."""
        # Create messages with clear sentiment patterns
        messages = [
            {"body": "AAPL to the moon! 🚀 bullish rocket", "user": {"followers": 1000, "verified": True},
             "likes": 100, "retweets": 50, "replies": 20, "quotes": 10, "hashtags": ["AAPL", "bullish"]},
            {"body": "AAPL bearish outlook, might crash", "user": {"followers": 500, "verified": False},
             "likes": 30, "retweets": 10, "replies": 5, "quotes": 2, "hashtags": ["bearish"]},
            {"body": "AAPL neutral analysis", "user": {"followers": 200, "verified": False},
             "likes": 10, "retweets": 2, "replies": 1, "quotes": 0, "hashtags": []},
            {"body": "Another AAPL bullish rocket comment", "user": {"followers": 2000, "verified": True},
             "likes": 200, "retweets": 80, "replies": 30, "quotes": 15, "hashtags": ["rocket"]},
        ]

        with patch.object(adapter, 'fetch_messages', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = messages

            summary = await adapter.fetch_summary("AAPL")

            assert summary["mentions"] == 4
            assert summary["bullish"] == 2  # Two bullish tweets
            assert summary["bearish"] == 1   # One bearish tweet
            assert summary["neutral"] == 1   # One neutral tweet
            assert summary["sentiment_score"] == 0.25  # (2-1)/4 = 0.25
            assert summary["provider"] == "twitter"
            assert summary["verified_tweets"] == 2
            assert summary["verified_ratio"] == 0.5
            assert "timestamp" in summary

            # Check engagement metrics
            assert summary["total_engagement"] > 0
            assert summary["avg_engagement"] > 0
            assert summary["avg_followers"] == 925.0  # (1000+500+200+2000)/4

            # Check hashtag tracking
            assert "top_hashtags" in summary
            assert len(summary["top_hashtags"]) > 0

    @pytest.mark.asyncio
    async def test_fetch_summary_no_messages(self, adapter):
        """Test summary generation with no messages."""
        with patch.object(adapter, 'fetch_messages', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = []

            summary = await adapter.fetch_summary("AAPL")

            assert summary["mentions"] == 0
            assert summary["bullish"] == 0
            assert summary["bearish"] == 0
            assert summary["neutral"] == 0
            assert summary["sentiment_score"] == 0.0
            assert summary["total_engagement"] == 0
            assert summary["avg_engagement"] == 0.0
            assert summary["verified_tweets"] == 0

    @pytest.mark.asyncio
    async def test_fetch_summary_sentiment_keywords(self, adapter):
        """Test sentiment analysis with various keyword combinations."""
        test_cases = [
            # (tweet_text, expected_sentiment)
            ("AAPL moon rocket 🚀 bullish", "bullish"),
            ("AAPL bearish crash dump", "bearish"),
            ("AAPL neutral analysis report", "neutral"),
            ("AAPL moon but also crash", "neutral"),  # Mixed = neutral
            ("AAPL diamond hands hold", "bullish"),
            ("AAPL puts short sell", "bearish"),
        ]

        for tweet_text, expected_sentiment in test_cases:
            messages = [{"body": tweet_text, "user": {"followers": 100, "verified": False},
                       "likes": 10, "retweets": 1, "replies": 0, "quotes": 0, "hashtags": []}]

            with patch.object(adapter, 'fetch_messages', new_callable=AsyncMock) as mock_fetch:
                mock_fetch.return_value = messages

                summary = await adapter.fetch_summary("TEST")

                if expected_sentiment == "bullish":
                    assert summary["bullish"] == 1 and summary["bearish"] == 0
                elif expected_sentiment == "bearish":
                    assert summary["bearish"] == 1 and summary["bullish"] == 0
                else:  # neutral
                    assert summary["neutral"] == 1

    @pytest.mark.asyncio
    async def test_get_with_retry_success(self, adapter):
        """Test successful HTTP request with retry logic."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"test": "data"})
        mock_response.raise_for_status = Mock()
        mock_response.request_info = Mock()
        mock_response.history = []

        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        adapter._session = mock_session

        result = await adapter._get_with_retry("https://test.com", {"param": "value"})

        assert result == {"test": "data"}
        assert adapter._health_info.status == AdapterStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_get_with_retry_rate_limit(self, adapter):
        """Test handling of rate limit (429) responses."""
        # First call returns 429, second call succeeds
        mock_response_429 = Mock()
        mock_response_429.status = 429
        mock_response_429.headers = {'x-rate-limit-reset': str(int(time.time()) + 60)}
        mock_response_429.request_info = Mock()
        mock_response_429.history = []

        mock_response_200 = Mock()
        mock_response_200.status = 200
        mock_response_200.json = AsyncMock(return_value={"success": True})
        mock_response_200.raise_for_status = Mock()

        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.side_effect = [
            mock_response_429, mock_response_200
        ]
        adapter._session = mock_session

        result = await adapter._get_with_retry("https://test.com")

        assert result == {"success": True}

    @pytest.mark.asyncio
    async def test_get_with_retry_authentication_error(self, adapter):
        """Test handling of authentication errors (401)."""
        mock_response = Mock()
        mock_response.status = 401
        mock_response.request_info = Mock()
        mock_response.history = []

        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        adapter._session = mock_session

        with pytest.raises(aiohttp.ClientResponseError, match="Authentication failed"):
            await adapter._get_with_retry("https://test.com")

    @pytest.mark.asyncio
    async def test_get_with_retry_no_token(self):
        """Test request without bearer token."""
        adapter = AsyncTwitterAdapter(bearer_token=None)

        try:
            result = await adapter._get_with_retry("https://test.com")
            assert result is None
        finally:
            await adapter.close()

    @pytest.mark.asyncio
    async def test_concurrency_limiting(self):
        """Test that concurrency is properly limited."""
        adapter = AsyncTwitterAdapter(bearer_token="test", concurrency=1, rate_limit_delay=0.01)

        try:
            call_times = []

            async def mock_slow_request(*args, **kwargs):
                call_times.append(asyncio.get_event_loop().time())
                await asyncio.sleep(0.1)  # Simulate slow request
                return {"data": []}

            with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
                mock_get.side_effect = mock_slow_request

                # Start multiple concurrent requests
                tasks = [
                    adapter.fetch_messages("AAPL"),
                    adapter.fetch_messages("TSLA"),
                    adapter.fetch_messages("MSFT")
                ]

                await asyncio.gather(*tasks)

                # Verify requests were serialized (not truly concurrent)
                assert len(call_times) == 3
                # With concurrency=1, requests should be serialized
                time_diffs = [call_times[i+1] - call_times[i] for i in range(len(call_times)-1)]
                assert all(diff >= 0.05 for diff in time_diffs)

        finally:
            await adapter.close()

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

        adapter = AsyncTwitterAdapter(bearer_token="test", session=external_session)

        await adapter.close()

        # External session should not be closed
        external_session.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_ticker_normalization(self, adapter):
        """Test that tickers are properly normalized."""
        with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"data": []}

            # Test various ticker formats
            await adapter.fetch_messages("  aapl  ")  # Should be normalized to AAPL
            await adapter.fetch_messages("tsla")      # Should be normalized to TSLA

            # Verify the API was called with normalized tickers
            calls = mock_get.call_args_list
            # Check that the query contains the normalized ticker
            assert "$AAPL" in calls[0][1]['params']['query']
            assert "$TSLA" in calls[1][1]['params']['query']

    @pytest.mark.asyncio
    async def test_hashtag_and_mention_extraction(self, adapter):
        """Test extraction of hashtags and mentions from tweets."""
        response_with_entities = {
            "data": [
                {
                    "id": "123",
                    "text": "Test tweet",
                    "author_id": "user1",
                    "entities": {
                        "hashtags": [{"tag": "AAPL"}, {"tag": "stocks"}],
                        "mentions": [{"username": "elonmusk"}, {"username": "tim_cook"}]
                    }
                }
            ],
            "includes": {
                "users": [{"id": "user1", "username": "testuser", "public_metrics": {"followers_count": 100}}]
            }
        }

        with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = response_with_entities

            messages = await adapter.fetch_messages("AAPL")

            assert len(messages) == 1
            tweet = messages[0]
            assert "AAPL" in tweet["hashtags"]
            assert "stocks" in tweet["hashtags"]
            assert "elonmusk" in tweet["mentions"]
            assert "tim_cook" in tweet["mentions"]

    @pytest.mark.asyncio
    async def test_engagement_metrics_calculation(self, adapter):
        """Test calculation of engagement metrics in summary."""
        messages = [
            {
                "body": "Test tweet 1", "user": {"followers": 1000, "verified": True},
                "likes": 100, "retweets": 50, "replies": 20, "quotes": 10, "hashtags": []
            },
            {
                "body": "Test tweet 2", "user": {"followers": 500, "verified": False},
                "likes": 50, "retweets": 25, "replies": 10, "quotes": 5, "hashtags": []
            }
        ]

        with patch.object(adapter, 'fetch_messages', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = messages

            summary = await adapter.fetch_summary("TEST")

            # Engagement = likes + (retweets * 2) + (replies * 1.5) + (quotes * 1.5)
            # Tweet 1: 100 + (50*2) + (20*1.5) + (10*1.5) = 100 + 100 + 30 + 15 = 245
            # Tweet 2: 50 + (25*2) + (10*1.5) + (5*1.5) = 50 + 50 + 15 + 7.5 = 122.5
            # Total: 367.5, Average: 183.75

            assert summary["total_engagement"] == 367
            assert abs(summary["avg_engagement"] - 183.75) < 1.0  # Allow for rounding
            assert summary["avg_followers"] == 750.0  # (1000 + 500) / 2
            assert summary["verified_tweets"] == 1
            assert summary["verified_ratio"] == 0.5

import time