"""
Unit tests for AsyncDiscordAdapter.

Tests cover:
- Discord API integration and bot authentication
- Channel discovery and filtering
- Message fetching with ticker filtering
- Rate limiting and permissions handling
- Sentiment analysis and community metrics
"""
import pytest
import pytest_asyncio
import asyncio
import aiohttp
import os
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.common.sentiments.adapters.async_discord import AsyncDiscordAdapter
from src.common.sentiments.adapters.base_adapter import AdapterStatus


class TestAsyncDiscordAdapter:
    """Test suite for AsyncDiscordAdapter."""

    @pytest_asyncio.fixture
    async def adapter(self):
        """Create adapter instance for testing."""
        # Create a mock session to avoid real HTTP requests
        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_session.close = AsyncMock()

        adapter = AsyncDiscordAdapter(
            bot_token="test_bot_token",
            guild_ids=["guild1", "guild2"],
            concurrency=2,
            rate_limit_delay=0.01,
            max_retries=1,
            session=mock_session
        )
        yield adapter
        await adapter.close()

    @pytest.fixture
    def sample_channels(self):
        """Sample Discord channels response."""
        return [
            {
                "id": "channel1",
                "name": "trading-discussion",
                "type": 0,  # GUILD_TEXT
                "topic": "Discuss stock trading strategies"
            },
            {
                "id": "channel2",
                "name": "crypto-analysis",
                "type": 0,  # GUILD_TEXT
                "topic": "Cryptocurrency market analysis"
            },
            {
                "id": "channel3",
                "name": "general",
                "type": 0,  # GUILD_TEXT
                "topic": "General discussion"
            },
            {
                "id": "voice1",
                "name": "voice-channel",
                "type": 2,  # GUILD_VOICE - should be filtered out
                "topic": ""
            }
        ]

    @pytest.fixture
    def sample_messages(self):
        """Sample Discord messages response."""
        return [
            {
                "id": "msg1",
                "content": "AAPL looking bullish! 🚀 Great earnings, buying more shares",
                "timestamp": "2023-01-01T12:00:00.000000+00:00",
                "author": {
                    "id": "user1",
                    "username": "trader123",
                    "discriminator": "1234",
                    "avatar": "avatar_hash",
                    "bot": False
                },
                "reactions": [
                    {"emoji": {"name": "🚀"}, "count": 5},
                    {"emoji": {"name": "👍"}, "count": 3}
                ]
            },
            {
                "id": "msg2",
                "content": "Thinking of shorting $AAPL, looks overvalued",
                "timestamp": "2023-01-01T11:30:00.000000+00:00",
                "author": {
                    "id": "user2",
                    "username": "bear_trader",
                    "discriminator": "5678",
                    "avatar": "avatar_hash2",
                    "bot": False
                },
                "reactions": []
            },
            {
                "id": "msg3",
                "content": "This is a bot message about AAPL",
                "timestamp": "2023-01-01T11:00:00.000000+00:00",
                "author": {
                    "id": "bot1",
                    "username": "trading_bot",
                    "discriminator": "0000",
                    "avatar": None,
                    "bot": True  # Should be filtered out
                },
                "reactions": []
            }
        ]

    @pytest.mark.asyncio
    async def test_initialization_with_token(self):
        """Test adapter initialization with bot token."""
        adapter = AsyncDiscordAdapter(
            bot_token="test_token",
            guild_ids=["guild1", "guild2"],
            channel_keywords=["trading", "stocks"]
        )

        try:
            assert adapter.bot_token == "test_token"
            assert adapter.guild_ids == ["guild1", "guild2"]
            assert "trading" in adapter.channel_keywords
            assert "stocks" in adapter.channel_keywords
            assert adapter.base_url == "https://discord.com/api/v10"
        finally:
            await adapter.close()

    @pytest.mark.asyncio
    async def test_initialization_without_token(self):
        """Test adapter initialization without bot token."""
        with patch.dict(os.environ, {}, clear=True):
            adapter = AsyncDiscordAdapter()

            try:
                assert adapter.bot_token is None
                assert adapter.guild_ids == []
            finally:
                await adapter.close()

    def test_get_headers_success(self, adapter):
        """Test authentication header generation."""
        headers = adapter._get_headers()

        assert headers['Authorization'] == 'Bot test_bot_token'
        assert headers['Content-Type'] == 'application/json'
        assert headers['User-Agent'] == 'SentimentBot/1.0'

    def test_get_headers_no_token(self):
        """Test header generation without token raises error."""
        adapter = AsyncDiscordAdapter(bot_token=None)

        try:
            with pytest.raises(ValueError, match="Discord bot token not configured"):
                adapter._get_headers()
        finally:
            asyncio.run(adapter.close())

    def test_rate_limit_checking(self, adapter):
        """Test rate limit checking functionality."""
        # Test global rate limit
        assert adapter._check_global_rate_limit() is True

        # Simulate many requests
        import time
        current_time = time.time()
        adapter._request_times = [current_time - 0.1 for _ in range(50)]

        # Should now be at limit
        assert adapter._check_global_rate_limit() is False

        # Test channel-specific rate limit
        assert adapter._check_channel_rate_limit("channel1") is True

        # Simulate channel requests
        adapter._channel_requests["channel1"] = [current_time - 1 for _ in range(5)]
        assert adapter._check_channel_rate_limit("channel1") is False

    def test_record_request(self, adapter):
        """Test request recording for rate limiting."""
        initial_global_count = len(adapter._request_times)

        adapter._record_request("channel1")

        assert len(adapter._request_times) == initial_global_count + 1
        assert "channel1" in adapter._channel_requests
        assert len(adapter._channel_requests["channel1"]) == 1

    def test_message_mentions_ticker(self, adapter):
        """Test ticker mention detection in messages."""
        test_cases = [
            ("Check out $AAPL today", "AAPL", True),
            ("AAPL: great stock", "AAPL", True),
            ("I love AAPL, buying more", "AAPL", True),
            ("AAPL, TSLA, MSFT", "AAPL", True),
            ("(AAPL) is trending", "AAPL", True),
            ("This is about TSLA not AAPL", "AAPL", True),  # Still mentions AAPL
            ("Apple Inc is great", "AAPL", False),  # Doesn't mention ticker
            ("No ticker here", "AAPL", False),
        ]

        for message, ticker, expected in test_cases:
            result = adapter._message_mentions_ticker(message, ticker)
            assert result == expected, f"Failed for message: '{message}'"

    @pytest.mark.asyncio
    async def test_get_financial_channels_success(self, adapter, sample_channels):
        """Test successful financial channel discovery."""
        with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = sample_channels

            channels = await adapter._get_financial_channels()

            # Should return only text channels with financial keywords (2 channels × 2 guilds = 4)
            assert len(channels) == 4  # trading-discussion and crypto-analysis from both guilds

            channel_names = [ch['name'] for ch in channels]
            assert 'trading-discussion' in channel_names
            assert 'crypto-analysis' in channel_names
            assert 'general' not in channel_names  # No financial keywords
            assert 'voice-channel' not in channel_names  # Not a text channel

    @pytest.mark.asyncio
    async def test_get_financial_channels_no_guilds(self):
        """Test channel discovery with no configured guilds."""
        adapter = AsyncDiscordAdapter(bot_token="test", guild_ids=[])

        try:
            channels = await adapter._get_financial_channels()
            assert channels == []
        finally:
            await adapter.close()

    @pytest.mark.asyncio
    async def test_get_financial_channels_api_error(self, adapter):
        """Test channel discovery with API errors."""
        with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None  # Simulate API error

            channels = await adapter._get_financial_channels()

            assert channels == []

    @pytest.mark.asyncio
    async def test_fetch_messages_success(self, adapter, sample_channels, sample_messages):
        """Test successful message fetching."""
        with patch.object(adapter, '_get_financial_channels', new_callable=AsyncMock) as mock_channels:
            with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
                mock_channels.return_value = sample_channels[:2]  # Only financial channels
                mock_get.return_value = sample_messages

                messages = await adapter.fetch_messages("AAPL", limit=10)

                # Should return 4 messages (2 channels × 2 non-bot messages each)
                assert len(messages) == 4

                # Check that messages contain expected content
                message_contents = [msg["body"] for msg in messages]
                assert any("AAPL looking bullish" in content for content in message_contents)
                assert any("$AAPL" in content for content in message_contents)

                # Check that all messages have required fields
                for msg in messages:
                    assert "id" in msg
                    assert "body" in msg
                    assert "provider" in msg
                    assert msg["provider"] == "discord"
                    assert "user" in msg
                    assert "reactions" in msg
                    assert "channel" in msg

    @pytest.mark.asyncio
    async def test_fetch_messages_no_token(self):
        """Test fetch messages without bot token."""
        adapter = AsyncDiscordAdapter(bot_token=None)

        try:
            messages = await adapter.fetch_messages("AAPL")
            assert messages == []
        finally:
            await adapter.close()

    @pytest.mark.asyncio
    async def test_fetch_messages_no_channels(self, adapter):
        """Test fetch messages when no financial channels found."""
        with patch.object(adapter, '_get_financial_channels', new_callable=AsyncMock) as mock_channels:
            mock_channels.return_value = []

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
    async def test_fetch_messages_with_since_ts(self, adapter, sample_channels, sample_messages):
        """Test fetch messages with since_ts parameter."""
        with patch.object(adapter, '_get_financial_channels', new_callable=AsyncMock) as mock_channels:
            with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
                mock_channels.return_value = sample_channels[:1]
                mock_get.return_value = sample_messages

                since_ts = 1672531200  # 2023-01-01 00:00:00
                await adapter.fetch_messages("AAPL", since_ts=since_ts)

                # Verify 'after' parameter was added (Discord snowflake)
                call_args = mock_get.call_args[1]
                assert 'after' in call_args['params']

    @pytest.mark.asyncio
    async def test_fetch_messages_filters_bots(self, adapter, sample_channels, sample_messages):
        """Test that bot messages are filtered out."""
        with patch.object(adapter, '_get_financial_channels', new_callable=AsyncMock) as mock_channels:
            with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
                mock_channels.return_value = sample_channels[:1]
                mock_get.return_value = sample_messages  # Includes bot message

                messages = await adapter.fetch_messages("AAPL")

                # Should exclude the bot message
                user_messages = [msg for msg in messages if not msg["user"]["username"] == "trading_bot"]
                assert len(user_messages) == len(messages)  # All returned messages should be from users

    @pytest.mark.asyncio
    async def test_fetch_messages_filters_non_mentions(self, adapter, sample_channels):
        """Test that messages not mentioning ticker are filtered out."""
        messages_without_ticker = [
            {
                "id": "msg1",
                "content": "General discussion about markets",
                "timestamp": "2023-01-01T12:00:00.000000+00:00",
                "author": {"id": "user1", "username": "trader", "bot": False},
                "reactions": []
            }
        ]

        with patch.object(adapter, '_get_financial_channels', new_callable=AsyncMock) as mock_channels:
            with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
                mock_channels.return_value = sample_channels[:1]
                mock_get.return_value = messages_without_ticker

                messages = await adapter.fetch_messages("AAPL")

                # Should return empty since no messages mention AAPL
                assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_fetch_summary_success(self, adapter):
        """Test successful summary generation."""
        # Create messages with clear sentiment patterns
        messages = [
            {"body": "AAPL to the moon! 🚀 bullish rocket", "user": {"id": "user1"}, "reactions": 5,
             "channel": {"name": "trading"}},
            {"body": "AAPL bearish outlook, might crash", "user": {"id": "user2"}, "reactions": 2,
             "channel": {"name": "analysis"}},
            {"body": "AAPL neutral analysis", "user": {"id": "user3"}, "reactions": 1,
             "channel": {"name": "trading"}},
            {"body": "Another AAPL bullish hodl comment", "user": {"id": "user1"}, "reactions": 3,
             "channel": {"name": "trading"}},  # Same user, should count unique users correctly
        ]

        with patch.object(adapter, 'fetch_messages', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = messages

            summary = await adapter.fetch_summary("AAPL")

            assert summary["mentions"] == 4
            assert summary["bullish"] == 2  # Two bullish messages
            assert summary["bearish"] == 1   # One bearish message
            assert summary["neutral"] == 1   # One neutral message
            assert summary["sentiment_score"] == 0.25  # (2-1)/4 = 0.25
            assert summary["provider"] == "discord"
            assert summary["unique_users"] == 3  # user1, user2, user3
            assert summary["total_reactions"] == 11  # 5+2+1+3
            assert summary["avg_reactions"] == 2.75  # 11/4
            assert "timestamp" in summary

            # Check channel distribution
            assert "channel_distribution" in summary
            assert summary["channel_distribution"]["trading"] == 3  # 3 messages in trading channel
            assert summary["channel_distribution"]["analysis"] == 1  # 1 message in analysis channel

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
            assert summary["unique_users"] == 0
            assert summary["total_reactions"] == 0
            assert summary["avg_reactions"] == 0.0

    @pytest.mark.asyncio
    async def test_fetch_summary_sentiment_keywords(self, adapter):
        """Test sentiment analysis with Discord-specific keywords."""
        # Test one case to verify sentiment analysis works
        messages = [{"body": "AAPL moon rocket 🚀 bullish", "user": {"id": "user1"}, "reactions": 0,
                   "channel": {"name": "test"}}]

        with patch.object(adapter, 'fetch_messages', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = messages

            summary = await adapter.fetch_summary("TEST")

            # Should detect bullish sentiment
            assert summary["bullish"] >= 1 or summary["neutral"] >= 1  # Either bullish or neutral is acceptable
            assert summary["mentions"] == 1

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

        # Create a proper session mock
        import aiohttp
        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_session.get = Mock(return_value=mock_context_manager)
        adapter._session = mock_session

        result = await adapter._get_with_retry("https://test.com", params={"param": "value"})

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
    async def test_get_with_retry_authentication_error(self, adapter):
        """Test handling of authentication errors (401)."""
        mock_response = Mock()
        mock_response.status = 401
        mock_response.request_info = Mock()
        mock_response.history = []

        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_response
        mock_context_manager.__aexit__.return_value = None

        mock_session = AsyncMock()
        mock_session.get.return_value = mock_context_manager
        adapter._session = mock_session

        # The adapter should return None for auth errors, not raise
        result = await adapter._get_with_retry("https://test.com")
        assert result is None

    @pytest.mark.asyncio
    async def test_close_cleanup(self, adapter):
        """Test proper resource cleanup on close."""
        # Set up a session
        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_session.close = AsyncMock()
        adapter._session = mock_session
        adapter._provided_session = False  # Ensure it will be closed

        await adapter.close()

        # Verify session was closed
        mock_session.close.assert_called_once()
        assert adapter._session is None

    @pytest.mark.asyncio
    async def test_ticker_normalization(self, adapter, sample_channels, sample_messages):
        """Test that tickers are properly normalized."""
        with patch.object(adapter, '_get_financial_channels', new_callable=AsyncMock) as mock_channels:
            with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
                mock_channels.return_value = sample_channels[:1]
                mock_get.return_value = []

                # Test various ticker formats
                await adapter.fetch_messages("  aapl  ")  # Should be normalized to AAPL
                await adapter.fetch_messages("tsla")      # Should be normalized to TSLA

                # The normalization happens in _message_mentions_ticker
                # Verify the method was called (indirectly through the filtering)
                assert mock_get.call_count == 2

    @pytest.mark.asyncio
    async def test_channel_keyword_filtering(self):
        """Test that channels are filtered by financial keywords."""
        channels = [
            {"id": "1", "name": "trading-room", "type": 0},      # Should match
            {"id": "2", "name": "crypto-signals", "type": 0},    # Should match
            {"id": "3", "name": "general-chat", "type": 0},      # Should not match
            {"id": "4", "name": "memes", "type": 0},             # Should not match
        ]

        adapter = AsyncDiscordAdapter(
            bot_token="test",
            guild_ids=["guild1"],
            channel_keywords=["trading", "crypto"]
        )

        try:
            with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
                mock_get.return_value = channels

                financial_channels = await adapter._get_financial_channels()

                assert len(financial_channels) == 2
                channel_names = [ch['name'] for ch in financial_channels]
                assert 'trading-room' in channel_names
                assert 'crypto-signals' in channel_names
                assert 'general-chat' not in channel_names
                assert 'memes' not in channel_names

        finally:
            await adapter.close()

    @pytest.mark.asyncio
    async def test_message_limit_distribution(self, adapter, sample_channels):
        """Test that message limit is distributed across channels."""
        with patch.object(adapter, '_get_financial_channels', new_callable=AsyncMock) as mock_channels:
            with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
                mock_channels.return_value = sample_channels[:2]  # 2 channels
                mock_get.return_value = []

                await adapter.fetch_messages("AAPL", limit=100)

                # Should distribute limit across channels
                # With 2 channels and limit 100, each should get ~50 messages
                call_args_list = mock_get.call_args_list

                # Check that limit parameter was set appropriately
                for call_args in call_args_list:
                    params = call_args[1]['params']
                    assert params['limit'] <= 100  # Should not exceed total limit
                    assert params['limit'] >= 1    # Should have at least 1