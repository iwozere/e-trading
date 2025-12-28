# src/common/sentiments/adapters/async_discord.py
"""
Async Discord adapter for sentiment data collection.

Provides:
- async fetch_messages(ticker, since_ts=None, limit=200)
- async fetch_summary(ticker, since_ts=None)

Features:
- Discord server monitoring for financial channels
- Channel-specific sentiment analysis
- Real-time message processing capabilities
- Rate limit handling and permissions management
"""
import asyncio
import aiohttp
import os
from typing import List, Dict, Optional, Any
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.common.sentiments.adapters.base_adapter import BaseSentimentAdapter
from src.common.sentiments.processing.heuristic_analyzer import HeuristicSentimentAnalyzer

_logger = setup_logger(__name__)


class AsyncDiscordAdapter(BaseSentimentAdapter):
    """
    Async Discord sentiment adapter using Discord API.

    Supports message collection from financial Discord channels,
    sentiment analysis, and community engagement metrics.
    """

    def __init__(self, name: str = "discord", session: Optional[aiohttp.ClientSession] = None,
                 concurrency: int = 2, rate_limit_delay: float = 1.0, max_retries: int = 3,
                 bot_token: Optional[str] = None, guild_ids: Optional[List[str]] = None):
        super().__init__(name, concurrency, rate_limit_delay)
        self._provided_session = session is not None
        self._session = session
        self.max_retries = max_retries
        self._consecutive_failures = 0
        self._analyzer = HeuristicSentimentAnalyzer()

        # Discord API configuration
        self.bot_token = bot_token or os.getenv('DISCORD_BOT_TOKEN')
        self.base_url = "https://discord.com/api/v10"

        # Guild and channel configuration
        self.guild_ids = guild_ids or os.getenv('DISCORD_GUILD_IDS', '').split(',')
        self.guild_ids = [gid.strip() for gid in self.guild_ids if gid.strip()]

        # Keywords to identify financial channels
        self.channel_keywords = self._analyzer.get_discord_channel_keywords()

        # Rate limiting - Discord API limits
        self.global_rate_limit = 50  # requests per second
        self.channel_rate_limit = 5  # requests per 5 seconds per channel
        self._request_times = []
        self._channel_requests: Dict[str, List[float]] = {}

        # Cache for channel information
        self._channel_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_expiry = 3600  # 1 hour

        if not self.bot_token:
            _logger.warning("Discord bot token not provided - adapter will not function")

    def _get_headers(self) -> Dict[str, str]:
        """Get authentication headers for Discord API."""
        if not self.bot_token:
            raise ValueError("Discord bot token not configured")

        return {
            'Authorization': f'Bot {self.bot_token}',
            'Content-Type': 'application/json',
            'User-Agent': 'SentimentBot/1.0'
        }

    def _check_global_rate_limit(self) -> bool:
        """Check if we're within global rate limits."""
        current_time = time.time()

        # Remove old requests outside the 1-second window
        self._request_times = [
            req_time for req_time in self._request_times
            if current_time - req_time < 1.0
        ]

        return len(self._request_times) < self.global_rate_limit

    def _check_channel_rate_limit(self, channel_id: str) -> bool:
        """Check if we're within rate limits for a specific channel."""
        current_time = time.time()

        if channel_id not in self._channel_requests:
            self._channel_requests[channel_id] = []

        # Remove old requests outside the 5-second window
        self._channel_requests[channel_id] = [
            req_time for req_time in self._channel_requests[channel_id]
            if current_time - req_time < 5.0
        ]

        return len(self._channel_requests[channel_id]) < self.channel_rate_limit

    def _record_request(self, channel_id: Optional[str] = None) -> None:
        """Record a new API request for rate limiting."""
        current_time = time.time()
        self._request_times.append(current_time)

        if channel_id:
            if channel_id not in self._channel_requests:
                self._channel_requests[channel_id] = []
            self._channel_requests[channel_id].append(current_time)

    async def _get_with_retry(self, url: str, params: Optional[dict] = None,
                             channel_id: Optional[str] = None, timeout: int = 30) -> Optional[dict]:
        """Make HTTP request with exponential backoff retry logic."""
        if not self.bot_token:
            _logger.error("Discord bot token not configured")
            return None

        if not self._session:
            self._session = aiohttp.ClientSession()

        last_exception = None

        for attempt in range(self.max_retries + 1):
            # Check rate limits before making request
            if not self._check_global_rate_limit():
                _logger.debug("Discord global rate limit reached, waiting")
                await asyncio.sleep(1.0)
                continue

            if channel_id and not self._check_channel_rate_limit(channel_id):
                _logger.debug("Discord channel rate limit reached for %s, waiting", channel_id)
                await asyncio.sleep(5.0)
                continue

            async with self.semaphore:
                try:
                    start_time = time.time()
                    headers = self._get_headers()

                    async with self._session.get(url, params=params, headers=headers, timeout=timeout) as resp:
                        response_time_ms = (time.time() - start_time) * 1000
                        self._record_request(channel_id)

                        if resp.status == 429:
                            # Rate limited - check headers for retry time
                            retry_after = resp.headers.get('retry-after')
                            if retry_after:
                                wait_time = float(retry_after)
                            else:
                                wait_time = self.rate_limit_delay * (2 ** attempt)

                            _logger.warning("Discord 429 rate limit (attempt %d/%d) - sleeping %.2fs",
                                          attempt + 1, self.max_retries + 1, wait_time)
                            await asyncio.sleep(wait_time)

                            if attempt < self.max_retries:
                                continue
                            else:
                                raise aiohttp.ClientResponseError(
                                    request_info=resp.request_info,
                                    history=resp.history,
                                    status=resp.status,
                                    message="Rate limit exceeded after retries"
                                )

                        if resp.status == 401:
                            _logger.error("Discord authentication failed - check bot token")
                            raise aiohttp.ClientResponseError(
                                request_info=resp.request_info,
                                history=resp.history,
                                status=resp.status,
                                message="Authentication failed"
                            )

                        if resp.status == 403:
                            _logger.warning("Discord access forbidden - check bot permissions")
                            return None

                        if resp.status >= 500:
                            # Server error - retry with backoff
                            if attempt < self.max_retries:
                                backoff_delay = self.rate_limit_delay * (2 ** attempt)
                                _logger.warning("Discord server error %d (attempt %d/%d) - retrying in %.2fs",
                                              resp.status, attempt + 1, self.max_retries + 1, backoff_delay)
                                await asyncio.sleep(backoff_delay)
                                continue

                        resp.raise_for_status()
                        data = await resp.json()

                        # Success - reset failure counter and update health
                        self._consecutive_failures = 0
                        self._update_health_success(response_time_ms)

                        return data

                except asyncio.TimeoutError as e:
                    last_exception = e
                    if attempt < self.max_retries:
                        backoff_delay = self.rate_limit_delay * (2 ** attempt)
                        _logger.warning("Discord timeout (attempt %d/%d) - retrying in %.2fs",
                                      attempt + 1, self.max_retries + 1, backoff_delay)
                        await asyncio.sleep(backoff_delay)
                        continue

                except (aiohttp.ClientError, aiohttp.ClientResponseError) as e:
                    last_exception = e
                    if attempt < self.max_retries and not isinstance(e, aiohttp.ClientResponseError) or (
                        isinstance(e, aiohttp.ClientResponseError) and e.status >= 500
                    ):
                        backoff_delay = self.rate_limit_delay * (2 ** attempt)
                        _logger.warning("Discord client error (attempt %d/%d) - retrying in %.2fs: %s",
                                      attempt + 1, self.max_retries + 1, backoff_delay, e)
                        await asyncio.sleep(backoff_delay)
                        continue
                    else:
                        # Client error that shouldn't be retried (4xx except 429)
                        break

                except Exception as e:
                    last_exception = e
                    _logger.debug("Discord unexpected error (attempt %d/%d): %s",
                                attempt + 1, self.max_retries + 1, e)
                    if attempt < self.max_retries:
                        backoff_delay = self.rate_limit_delay * (2 ** attempt)
                        await asyncio.sleep(backoff_delay)
                        continue
                    break

        # All retries failed
        self._consecutive_failures += 1
        if last_exception:
            self._update_health_failure(last_exception)
            _logger.error("Discord request failed after %d attempts: %s %s",
                         self.max_retries + 1, url, last_exception)

        return None

    async def _get_financial_channels(self) -> List[Dict[str, Any]]:
        """Get list of financial-related channels from configured guilds."""
        channels = []

        for guild_id in self.guild_ids:
            if not guild_id:
                continue

            try:
                url = f"{self.base_url}/guilds/{guild_id}/channels"
                guild_channels = await self._get_with_retry(url)

                if not guild_channels:
                    continue

                for channel in guild_channels:
                    # Only process text channels
                    if channel.get('type') != 0:  # 0 = GUILD_TEXT
                        continue

                    channel_name = channel.get('name', '').lower()

                    # Check if channel name contains financial keywords
                    if any(keyword in channel_name for keyword in self.channel_keywords):
                        channels.append({
                            'id': channel['id'],
                            'name': channel['name'],
                            'guild_id': guild_id,
                            'topic': channel.get('topic', '')
                        })

                await asyncio.sleep(self.rate_limit_delay)

            except Exception as e:
                _logger.warning("Failed to get channels for guild %s: %s", guild_id, e)
                continue

        _logger.debug("Found %d financial channels across %d guilds", len(channels), len(self.guild_ids))
        return channels

    def _message_mentions_ticker(self, message_content: str, ticker: str) -> bool:
        """Check if message mentions the ticker symbol."""
        import re
        content_lower = message_content.lower()
        ticker_lower = ticker.lower()

        # Check various formats
        patterns = [
            f'\\${ticker_lower}\\b',  # Cashtag
            f'#{ticker_lower}\\b',  # Hashtag
            f'\\b{ticker_lower}\\b',  # Word boundary
            f'\\({ticker_lower}\\)',  # In parentheses
        ]

        return any(re.search(pattern, content_lower) for pattern in patterns)

    async def fetch_messages(self, ticker: str, since_ts: Optional[int] = None, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Fetch individual Discord messages for a ticker.

        Args:
            ticker: Stock ticker symbol
            since_ts: Unix timestamp to fetch messages since
            limit: Maximum number of messages to fetch

        Returns:
            List of normalized message dictionaries
        """
        if not ticker or not ticker.strip():
            raise ValueError("Ticker cannot be empty")

        if not self.bot_token:
            _logger.error("Discord bot token not configured")
            return []

        symbol = ticker.upper().strip()
        messages: List[Dict[str, Any]] = []

        try:
            # Get financial channels
            channels = await self._get_financial_channels()

            if not channels:
                _logger.warning("No financial channels found for Discord monitoring")
                return []

            messages_per_channel = max(1, limit // len(channels))

            for channel in channels:
                if len(messages) >= limit:
                    break

                channel_id = channel['id']

                try:
                    # Build API parameters
                    params = {
                        'limit': min(100, messages_per_channel)  # Discord API limit is 100 per request
                    }

                    # Add time filter if provided
                    if since_ts:
                        # Discord uses snowflake IDs which encode timestamps
                        # Convert timestamp to approximate snowflake
                        discord_epoch = 1420070400000  # Discord epoch in milliseconds
                        timestamp_ms = since_ts * 1000
                        snowflake = (timestamp_ms - discord_epoch) << 22
                        params['after'] = str(snowflake)

                    url = f"{self.base_url}/channels/{channel_id}/messages"
                    channel_messages = await self._get_with_retry(url, params=params, channel_id=channel_id)

                    if not channel_messages:
                        continue

                    for msg in channel_messages:
                        try:
                            # Validate required fields
                            if not msg.get("id") or not msg.get("content"):
                                continue

                            content = msg.get("content", "")

                            # Check if message mentions the ticker
                            if not self._message_mentions_ticker(content, symbol):
                                continue

                            # Extract user information
                            author = msg.get("author", {})

                            # Skip bot messages
                            if author.get("bot", False):
                                continue

                            normalized_msg = {
                                "id": str(msg.get("id")),
                                "body": content,
                                "created_at": msg.get("timestamp"),
                                "user": {
                                    "username": author.get("username", ""),
                                    "id": str(author.get("id", "")),
                                    "followers": 0,  # Discord doesn't have followers concept
                                    "discriminator": author.get("discriminator", ""),
                                    "avatar": author.get("avatar", "")
                                },
                                "likes": 0,  # Discord doesn't have likes
                                "replies": 0,  # Would need additional API calls to count
                                "retweets": 0,  # Discord doesn't have retweets
                                "reactions": len(msg.get("reactions", [])),
                                "channel": {
                                    "id": channel_id,
                                    "name": channel.get("name", ""),
                                    "guild_id": channel.get("guild_id", "")
                                },
                                "provider": "discord"
                            }
                            messages.append(normalized_msg)

                            if len(messages) >= limit:
                                break

                        except (ValueError, TypeError) as e:
                            _logger.debug("Error processing Discord message for ticker %s: %s", symbol, e)
                            continue

                    # Polite delay between channel requests
                    await asyncio.sleep(self.rate_limit_delay)

                except Exception as e:
                    _logger.warning("Failed to fetch messages from channel %s: %s", channel_id, e)
                    continue

            _logger.debug("Fetched %d Discord messages for ticker %s from %d channels",
                         len(messages), symbol, len(channels))
            return messages

        except Exception as e:
            _logger.error("Failed to fetch Discord messages for ticker %s: %s", symbol, e)
            self._update_health_failure(e)
            raise

    async def fetch_summary(self, ticker: str, since_ts: Optional[int] = None) -> Dict[str, Any]:
        """
        Fetch aggregated sentiment summary for a ticker from Discord.

        Args:
            ticker: Stock ticker symbol
            since_ts: Unix timestamp to fetch data since

        Returns:
            Dictionary containing sentiment metrics and counts
        """
        if not ticker or not ticker.strip():
            raise ValueError("Ticker cannot be empty")

        try:
            messages = await self.fetch_messages(ticker, since_ts=since_ts, limit=200)
            mentions = len(messages)
            bullish = 0
            bearish = 0
            neutral = 0

            total_reactions = 0
            unique_users = set()
            channel_distribution: Dict[str, int] = {}

            for msg in messages:
                try:
                    body = (msg.get("body") or "").lower()
                    if not body:
                        neutral += 1
                        continue

                    # Track user engagement
                    user_id = msg.get("user", {}).get("id")
                    if user_id:
                        unique_users.add(user_id)

                    # Track reactions
                    reactions = msg.get("reactions", 0)
                    total_reactions += reactions

                    # Track channel distribution
                    channel_name = msg.get("channel", {}).get("name", "unknown")
                    channel_distribution[channel_name] = channel_distribution.get(channel_name, 0) + 1

                    # Sentiment analysis with unified analyzer
                    result = self._analyzer.analyze_sentiment(body)
                    if result.score > 0.1:
                        bullish += 1
                    elif result.score < -0.1:
                        bearish += 1
                    else:
                        neutral += 1

                except Exception as e:
                    _logger.debug("Error processing Discord message sentiment for ticker %s: %s", ticker, e)
                    neutral += 1
                    continue

            # Calculate sentiment score (-1 to 1)
            if mentions > 0:
                score = (bullish - bearish) / mentions
            else:
                score = 0.0

            # Ensure score is within bounds
            score = max(-1.0, min(1.0, score))

            # Calculate additional metrics
            avg_reactions = total_reactions / mentions if mentions > 0 else 0
            unique_user_count = len(unique_users)

            # Get top channels
            top_channels = sorted(channel_distribution.items(), key=lambda x: x[1], reverse=True)[:5]

            summary = {
                "mentions": mentions,
                "bullish": bullish,
                "bearish": bearish,
                "neutral": neutral,
                "sentiment_score": float(score),
                "total_reactions": int(total_reactions),
                "avg_reactions": float(avg_reactions),
                "unique_users": unique_user_count,
                "channel_distribution": dict(top_channels),
                "provider": "discord",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            _logger.debug("Generated Discord summary for ticker %s: %d mentions, score %.3f, %d users",
                         ticker, mentions, score, unique_user_count)
            return summary

        except Exception as e:
            _logger.error("Failed to fetch Discord summary for ticker %s: %s", ticker, e)
            self._update_health_failure(e)
            raise

    async def close(self) -> None:
        """Clean up adapter resources."""
        try:
            if self._session and not self._provided_session:
                await self._session.close()
                self._session = None
            _logger.debug("Discord adapter closed successfully")
        except Exception as e:
            _logger.warning("Error closing Discord adapter: %s", e)