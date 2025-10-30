"""
Unit tests for AsyncPushshiftAdapter.

Tests cover:
- Submission and comment fetching
- Message normalization and aggregation
- Summary generation with sentiment analysis
- Error handling and retry logic
- Health monitoring and rate limiting
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

from src.common.sentiments.adapters.async_pushshift import AsyncPushshiftAdapter
from src.common.sentiments.adapters.base_adapter import AdapterStatus


class TestAsyncPushshiftAdapter:
    """Test suite for AsyncPushshiftAdapter."""

    @pytest_asyncio.fixture
    async def adapter(self):
        """Create adapter instance for testing."""
        # Create a mock session to avoid real HTTP requests
        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_session.close = AsyncMock()

        adapter = AsyncPushshiftAdapter(
            concurrency=2,
            rate_limit_delay=0.01,
            max_retries=1,
            session=mock_session
        )
        yield adapter
        await adapter.close()

    @pytest.fixture
    def sample_submissions(self):
        """Sample Reddit submissions from Pushshift API."""
        return [
            {
                "id": "sub1",
                "title": "AAPL to the moon! 🚀",
                "selftext": "Great earnings, buying more shares",
                "author": "bullish_trader",
                "author_fullname": "t2_user1",
                "created_utc": 1672531200,
                "score": 150,
                "num_comments": 25
            },
            {
                "id": "sub2",
                "title": "AAPL bearish outlook",
                "selftext": "Might crash soon, selling my position",
                "author": "bear_trader",
                "author_fullname": "t2_user2",
                "created_utc": 1672527600,
                "score": 75,
                "num_comments": 10
            }
        ]

    @pytest.fixture
    def sample_comments(self):
        """Sample Reddit comments from Pushshift API."""
        return [
            {
                "id": "comm1",
                "body": "AAPL diamond hands! Hold strong 💎",
                "author": "diamond_hands",
                "author_fullname": "t2_user3",
                "created_utc": 1672534800,
                "score": 50
            },
            {
                "id": "comm2",
                "body": "AAPL looks like it will dump hard",
                "author": "short_seller",
                "author_fullname": "t2_user4",
                "created_utc": 1672531800,
                "score": 20
            }
        ]

    @pytest.mark.asyncio
    async def test_fetch_submissions_success(self, adapter, sample_submissions):
        """Test successful submission fetching."""
        with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = sample_submissions

            submissions = await adapter.fetch_submissions("AAPL", since_ts=None, limit=10)

            assert len(submissions) == 2
            assert submissions[0]["id"] == "sub1"
            assert submissions[0]["title"] == "AAPL to the moon! 🚀"
            assert submissions[1]["id"] == "sub2"

            # Verify API call parameters
            mock_get.assert_called_once_with(
                "submission",
                {"q": "AAPL OR $AAPL", "size": 10}
            )

    @pytest.mark.asyncio
    async def test_fetch_submissions_with_timestamp(self, adapter, sample_submissions):
        """Test submission fetching with since_ts parameter."""
        with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = sample_submissions

            since_ts = 1672531200
            await adapter.fetch_submissions("AAPL", since_ts=since_ts, limit=10)

            # Verify timestamp was included in API call
            mock_get.assert_called_once_with(
                "submission",
                {"q": "AAPL OR $AAPL", "size": 10, "after": since_ts}
            )

    @pytest.mark.asyncio
    async def test_fetch_comments_success(self, adapter, sample_comments):
        """Test successful comment fetching."""
        with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = sample_comments

            comments = await adapter.fetch_comments("AAPL", since_ts=None, limit=10)

            assert len(comments) == 2
            assert comments[0]["id"] == "comm1"
            assert comments[0]["body"] == "AAPL diamond hands! Hold strong 💎"
            assert comments[1]["id"] == "comm2"

            # Verify API call parameters
            mock_get.assert_called_once_with(
                "comment",
                {"q": "AAPL OR $AAPL", "size": 10}
            )

    @pytest.mark.asyncio
    async def test_fetch_messages_success(self, adapter, sample_submissions, sample_comments):
        """Test successful message fetching (submissions + comments)."""
        with patch.object(adapter, 'fetch_submissions', new_callable=AsyncMock) as mock_subs:
            with patch.object(adapter, 'fetch_comments', new_callable=AsyncMock) as mock_comms:
                mock_subs.return_value = sample_submissions
                mock_comms.return_value = sample_comments

                messages = await adapter.fetch_messages("AAPL", limit=100)

                assert len(messages) == 4  # 2 submissions + 2 comments

                # Check submission normalization
                sub_messages = [m for m in messages if m.get("type") == "submission"]
                assert len(sub_messages) == 2
                assert sub_messages[0]["id"] == "sub1"
                assert sub_messages[0]["body"] == "AAPL to the moon! 🚀 Great earnings, buying more shares"
                assert sub_messages[0]["provider"] == "reddit"
                assert sub_messages[0]["user"]["username"] == "bullish_trader"
                assert sub_messages[0]["likes"] == 150
                assert sub_messages[0]["replies"] == 25
                assert sub_messages[0]["retweets"] == 0

                # Check comment normalization
                comm_messages = [m for m in messages if m.get("type") == "comment"]
                assert len(comm_messages) == 2
                assert comm_messages[0]["id"] == "comm1"
                assert comm_messages[0]["body"] == "AAPL diamond hands! Hold strong 💎"
                assert comm_messages[0]["provider"] == "reddit"
                assert comm_messages[0]["user"]["username"] == "diamond_hands"
                assert comm_messages[0]["likes"] == 50
                assert comm_messages[0]["replies"] == 0

    @pytest.mark.asyncio
    async def test_fetch_messages_limit_distribution(self, adapter):
        """Test that message limit is properly distributed between submissions and comments."""
        with patch.object(adapter, 'fetch_submissions', new_callable=AsyncMock) as mock_subs:
            with patch.object(adapter, 'fetch_comments', new_callable=AsyncMock) as mock_comms:
                mock_subs.return_value = []
                mock_comms.return_value = []

                await adapter.fetch_messages("AAPL", limit=90)

                # Verify limit distribution (1/3 for submissions, 2/3 for comments)
                sub_limit = mock_subs.call_args[1]['limit']
                comm_limit = mock_comms.call_args[1]['limit']

                assert sub_limit == 30  # min(90 // 3, 100) = 30
                assert comm_limit == 60  # 90 - 30 = 60

    @pytest.mark.asyncio
    async def test_fetch_mentions_summary_success(self, adapter, sample_submissions, sample_comments):
        """Test successful mentions summary generation."""
        with patch.object(adapter, 'fetch_submissions', new_callable=AsyncMock) as mock_subs:
            with patch.object(adapter, 'fetch_comments', new_callable=AsyncMock) as mock_comms:
                mock_subs.return_value = sample_submissions
                mock_comms.return_value = sample_comments

                summary = await adapter.fetch_mentions_summary("AAPL", since_ts=None)

                assert summary["mentions"] == 4  # 2 submissions + 2 comments
                assert summary["pos"] == 2  # "moon", "diamond" keywords
                assert summary["neg"] == 2  # "bearish", "crash", "dump" keywords
                assert summary["neutral"] == 0
                assert summary["sentiment_score"] == 0.0  # (2-2)/4 = 0
                assert summary["unique_authors"] == 4  # All different authors
                assert summary["provider"] == "reddit"
                assert "timestamp" in summary

    @pytest.mark.asyncio
    async def test_fetch_summary_alias(self, adapter):
        """Test that fetch_summary is an alias for fetch_mentions_summary."""
        with patch.object(adapter, 'fetch_mentions_summary', new_callable=AsyncMock) as mock_mentions:
            mock_mentions.return_value = {"test": "data"}

            result = await adapter.fetch_summary("AAPL", since_ts=123456)

            assert result == {"test": "data"}
            mock_mentions.assert_called_once_with("AAPL", 123456)

    @pytest.mark.asyncio
    async def test_sentiment_analysis_keywords(self, adapter):
        """Test sentiment analysis with various keyword combinations."""
        test_cases = [
            # (submissions, comments, expected_pos, expected_neg, expected_neutral)
            ([{"title": "AAPL moon rocket 🚀", "selftext": ""}], [], 1, 0, 0),
            ([{"title": "AAPL crash dump", "selftext": ""}], [], 0, 1, 0),
            ([{"title": "AAPL neutral comment", "selftext": ""}], [], 0, 0, 1),
            ([{"title": "AAPL moon", "selftext": "but also crash"}], [], 0, 0, 1),  # Mixed = neutral
        ]

        for subs, comms, exp_pos, exp_neg, exp_neutral in test_cases:
            with patch.object(adapter, 'fetch_submissions', new_callable=AsyncMock) as mock_subs:
                with patch.object(adapter, 'fetch_comments', new_callable=AsyncMock) as mock_comms:
                    mock_subs.return_value = subs
                    mock_comms.return_value = comms

                    summary = await adapter.fetch_mentions_summary("TEST", since_ts=None)

                    assert summary["pos"] == exp_pos
                    assert summary["neg"] == exp_neg
                    assert summary["neutral"] == exp_neutral

    @pytest.mark.asyncio
    async def test_unique_authors_calculation(self, adapter):
        """Test unique author counting."""
        submissions = [
            {"author": "user1", "title": "Test", "selftext": ""},
            {"author": "user2", "title": "Test", "selftext": ""},
            {"author": "user1", "title": "Test", "selftext": ""},  # Duplicate
        ]
        comments = [
            {"author": "user3", "body": "Test"},
            {"author": "user1", "body": "Test"},  # Duplicate across types
            {"author": None, "body": "Test"},  # No author
        ]

        with patch.object(adapter, 'fetch_submissions', new_callable=AsyncMock) as mock_subs:
            with patch.object(adapter, 'fetch_comments', new_callable=AsyncMock) as mock_comms:
                mock_subs.return_value = submissions
                mock_comms.return_value = comments

                summary = await adapter.fetch_mentions_summary("AAPL", since_ts=None)

                # Should count unique authors: user1, user2, user3 = 3
                assert summary["unique_authors"] == 3

    @pytest.mark.asyncio
    async def test_get_with_retry_success(self, adapter):
        """Test successful HTTP request with retry logic."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": [{"test": "item"}]})
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

        result = await adapter._get_with_retry("submission", params={"q": "AAPL"})

        assert result == [{"test": "item"}]
        assert adapter._health_info.status == AdapterStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_get_with_retry_empty_data(self, adapter):
        """Test handling of empty data response."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": []})
        mock_response.raise_for_status = Mock()

        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        adapter._session = mock_session

        result = await adapter._get_with_retry("submission", {"q": "INVALID"})

        assert result == []

    @pytest.mark.asyncio
    async def test_get_with_retry_malformed_response(self, adapter):
        """Test handling of malformed API response."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"error": "Invalid request"})  # No "data" field
        mock_response.raise_for_status = Mock()

        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        adapter._session = mock_session

        result = await adapter._get_with_retry("submission", {"q": "AAPL"})

        assert result == []  # Should return empty list when "data" field is missing

    @pytest.mark.asyncio
    async def test_invalid_ticker_validation(self, adapter):
        """Test validation of ticker parameters."""
        with pytest.raises(ValueError, match="Ticker cannot be empty"):
            await adapter.fetch_submissions("")

        with pytest.raises(ValueError, match="Ticker cannot be empty"):
            await adapter.fetch_comments("   ")

        with pytest.raises(ValueError, match="Ticker cannot be empty"):
            await adapter.fetch_mentions_summary(None)

    @pytest.mark.asyncio
    async def test_ticker_normalization(self, adapter):
        """Test that tickers are properly normalized."""
        with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = []

            # Test various ticker formats
            await adapter.fetch_submissions("  aapl  ")
            await adapter.fetch_comments("tsla")

            # Verify the API was called with normalized tickers
            calls = mock_get.call_args_list
            assert calls[0][1]["q"] == "AAPL OR $AAPL"
            assert calls[1][1]["q"] == "TSLA OR $TSLA"

    @pytest.mark.asyncio
    async def test_limit_enforcement(self, adapter):
        """Test that API limits are properly enforced."""
        with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = []

            # Test with large limit
            await adapter.fetch_submissions("AAPL", since_ts=None, limit=1000)

            # Should be capped at 500 (API limit)
            call_args = mock_get.call_args[1]
            assert call_args["size"] == 500

    @pytest.mark.asyncio
    async def test_error_handling_in_processing(self, adapter):
        """Test error handling during message processing."""
        # Create malformed data that will cause processing errors
        malformed_submissions = [
            {"id": "valid", "title": "Valid submission", "selftext": ""},
            {"title": "Missing ID"},  # No ID
            None,  # Null submission
            {"id": "partial"},  # Missing other fields
        ]

        with patch.object(adapter, 'fetch_submissions', new_callable=AsyncMock) as mock_subs:
            with patch.object(adapter, 'fetch_comments', new_callable=AsyncMock) as mock_comms:
                mock_subs.return_value = malformed_submissions
                mock_comms.return_value = []

                messages = await adapter.fetch_messages("AAPL")

                # Should handle errors gracefully and return valid messages
                valid_messages = [m for m in messages if m.get("id")]
                assert len(valid_messages) >= 1  # At least the valid one

    @pytest.mark.asyncio
    async def test_message_sorting(self, adapter):
        """Test that messages are sorted by creation time."""
        submissions = [
            {"id": "old", "title": "Old", "selftext": "", "created_utc": 1000},
            {"id": "new", "title": "New", "selftext": "", "created_utc": 2000},
        ]
        comments = [
            {"id": "middle", "body": "Middle", "created_utc": 1500},
        ]

        with patch.object(adapter, 'fetch_submissions', new_callable=AsyncMock) as mock_subs:
            with patch.object(adapter, 'fetch_comments', new_callable=AsyncMock) as mock_comms:
                mock_subs.return_value = submissions
                mock_comms.return_value = comments

                messages = await adapter.fetch_messages("AAPL")

                # Should be sorted by created_utc in descending order
                timestamps = [m.get("created_at", 0) for m in messages]
                assert timestamps == sorted(timestamps, reverse=True)

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
    async def test_rate_limiting_delay(self, adapter):
        """Test that rate limiting delays are applied."""
        with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
            with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                mock_get.return_value = []

                await adapter.fetch_submissions("AAPL", since_ts=None, limit=10)

                # Should have called sleep for rate limiting
                mock_sleep.assert_called_with(adapter.rate_limit_delay)

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, adapter):
        """Test concurrent request handling."""
        with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = []

            # Make multiple concurrent requests
            tasks = [
                adapter.fetch_submissions("AAPL", since_ts=None, limit=10),
                adapter.fetch_submissions("TSLA", since_ts=None, limit=10),
                adapter.fetch_submissions("MSFT", since_ts=None, limit=10)
            ]

            await asyncio.gather(*tasks)

            # All requests should have completed
            assert mock_get.call_count == 3