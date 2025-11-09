# src/common/sentiments/adapters/async_twitter.py
"""
Async Twitter/X adapter for sentiment data collection.

Provides:
- async fetch_messages(ticker, since_ts=None, limit=200)
- async fetch_summary(ticker, since_ts=None)

Features:
- Twitter API v2 integration
- Tweet sentiment analysis and engagement metrics
- Hashtag and mention tracking capabilities
- Rate limit handling and authentication
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

_logger = setup_logger(__name__)


class AsyncTwitterAdapter(BaseSentimentAdapter):
    """
    Async Twitter/X sentiment adapter using Twitter API v2.

    Supports tweet collection, sentiment analysis, and engagement metrics
    with proper rate limiting and authentication handling.
    """

    def __init__(self, name: str = "twitter", session: Optional[aiohttp.ClientSession] = None,
                 concurrency: int = 3, rate_limit_delay: float = 1.0, max_retries: int = 3,
                 bearer_token: Optional[str] = None):
        super().__init__(name, concurrency, rate_limit_delay)
        self._provided_session = session is not None
        self._session = session
        self.max_retries = max_retries
        self._consecutive_failures = 0

        # Twitter API configuration
        self.bearer_token = bearer_token or os.getenv('TWITTER_BEARER_TOKEN')
        self.base_url = "https://api.twitter.com/2"

        # Rate limiting - Twitter API v2 limits
        self.search_rate_limit = 300  # requests per 15 minutes
        self.search_window = 15 * 60  # 15 minutes in seconds
        self._search_requests = []

        # Tweet fields to request
        self.tweet_fields = [
            'id', 'text', 'created_at', 'author_id', 'public_metrics',
            'context_annotations', 'entities', 'referenced_tweets'
        ]
        self.user_fields = ['id', 'username', 'name', 'public_metrics', 'verified']

        if not self.bearer_token:
            _logger.warning("Twitter bearer token not provided - adapter will not function")

    def _get_headers(self) -> Dict[str, str]:
        """Get authentication headers for Twitter API."""
        if not self.bearer_token:
            raise ValueError("Twitter bearer token not configured")

        return {
            'Authorization': f'Bearer {self.bearer_token}',
            'Content-Type': 'application/json'
        }

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits for search requests."""
        current_time = time.time()

        # Remove old requests outside the window
        self._search_requests = [
            req_time for req_time in self._search_requests
            if current_time - req_time < self.search_window
        ]

        return len(self._search_requests) < self.search_rate_limit

    def _record_request(self) -> None:
        """Record a new API request for rate limiting."""
        self._search_requests.append(time.time())

    async def _get_with_retry(self, url: str, params: Optional[dict] = None, timeout: int = 30) -> Optional[dict]:
        """Make HTTP request with exponential backoff retry logic."""
        if not self.bearer_token:
            _logger.error("Twitter bearer token not configured")
            return None

        if not self._session:
            self._session = aiohttp.ClientSession()

        last_exception = None

        for attempt in range(self.max_retries + 1):
            # Check rate limits before making request
            if not self._check_rate_limit():
                wait_time = self.search_window - (time.time() - min(self._search_requests))
                _logger.warning("Twitter rate limit reached, waiting %.1f seconds", wait_time)
                await asyncio.sleep(wait_time)

            async with self.semaphore:
                try:
                    start_time = time.time()
                    headers = self._get_headers()

                    async with self._session.get(url, params=params, headers=headers, timeout=timeout) as resp:
                        response_time_ms = (time.time() - start_time) * 1000
                        self._record_request()

                        if resp.status == 429:
                            # Rate limited - check headers for reset time
                            reset_time = resp.headers.get('x-rate-limit-reset')
                            if reset_time:
                                wait_time = int(reset_time) - int(time.time())
                                wait_time = max(wait_time, 60)  # Wait at least 1 minute
                            else:
                                wait_time = self.rate_limit_delay * (2 ** attempt)

                            _logger.warning("Twitter 429 rate limit (attempt %d/%d) - sleeping %ds",
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
                            _logger.error("Twitter authentication failed - check bearer token")
                            raise aiohttp.ClientResponseError(
                                request_info=resp.request_info,
                                history=resp.history,
                                status=resp.status,
                                message="Authentication failed"
                            )

                        if resp.status >= 500:
                            # Server error - retry with backoff
                            if attempt < self.max_retries:
                                backoff_delay = self.rate_limit_delay * (2 ** attempt)
                                _logger.warning("Twitter server error %d (attempt %d/%d) - retrying in %.2fs",
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
                        _logger.warning("Twitter timeout (attempt %d/%d) - retrying in %.2fs",
                                      attempt + 1, self.max_retries + 1, backoff_delay)
                        await asyncio.sleep(backoff_delay)
                        continue

                except (aiohttp.ClientError, aiohttp.ClientResponseError) as e:
                    last_exception = e
                    if attempt < self.max_retries and not isinstance(e, aiohttp.ClientResponseError) or (
                        isinstance(e, aiohttp.ClientResponseError) and e.status >= 500
                    ):
                        backoff_delay = self.rate_limit_delay * (2 ** attempt)
                        _logger.warning("Twitter client error (attempt %d/%d) - retrying in %.2fs: %s",
                                      attempt + 1, self.max_retries + 1, backoff_delay, e)
                        await asyncio.sleep(backoff_delay)
                        continue
                    else:
                        # Client error that shouldn't be retried (4xx except 429)
                        break

                except Exception as e:
                    last_exception = e
                    _logger.debug("Twitter unexpected error (attempt %d/%d): %s",
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
            _logger.error("Twitter request failed after %d attempts: %s %s",
                         self.max_retries + 1, url, last_exception)

        return None

    def _build_search_query(self, ticker: str) -> str:
        """Build Twitter search query for a ticker."""
        symbol = ticker.upper().strip()

        # Build query with various ticker formats and exclude retweets
        query_parts = [
            f'${symbol}',  # Cashtag
            f'#{symbol}',  # Hashtag
            f'"{symbol}"',  # Exact match in quotes
        ]

        # Combine with OR and exclude retweets
        query = f"({' OR '.join(query_parts)}) -is:retweet lang:en"

        return query

    async def fetch_messages(self, ticker: str, since_ts: Optional[int] = None, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Fetch individual tweets for a ticker from Twitter API v2.

        Args:
            ticker: Stock ticker symbol
            since_ts: Unix timestamp to fetch messages since
            limit: Maximum number of tweets to fetch (max 100 per request)

        Returns:
            List of normalized tweet dictionaries
        """
        if not ticker or not ticker.strip():
            raise ValueError("Ticker cannot be empty")

        if not self.bearer_token:
            _logger.error("Twitter bearer token not configured")
            return []

        symbol = ticker.upper().strip()
        tweets: List[Dict[str, Any]] = []

        try:
            # Build search query
            query = self._build_search_query(symbol)

            # Build API parameters
            params = {
                'query': query,
                'max_results': min(100, limit),  # Twitter API v2 limit is 100 per request
                'tweet.fields': ','.join(self.tweet_fields),
                'user.fields': ','.join(self.user_fields),
                'expansions': 'author_id'
            }

            # Add time filter if provided
            if since_ts:
                start_time = datetime.fromtimestamp(since_ts).isoformat() + 'Z'
                params['start_time'] = start_time

            url = f"{self.base_url}/tweets/search/recent"
            payload = await self._get_with_retry(url, params=params)

            if not payload:
                _logger.warning("No payload received for ticker %s", symbol)
                return []

            tweet_data = payload.get("data", [])
            users_data = payload.get("includes", {}).get("users", [])

            # Create user lookup for efficiency
            users_lookup = {user['id']: user for user in users_data}

            if not tweet_data:
                _logger.debug("No tweets found for ticker %s", symbol)
                return []

            for tweet in tweet_data:
                try:
                    # Skip None tweets
                    if tweet is None:
                        continue

                    # Validate required fields
                    if not tweet.get("id"):
                        _logger.debug("Skipping tweet without ID for ticker %s", symbol)
                        continue

                    author_id = tweet.get("author_id")
                    user_info = users_lookup.get(author_id, {})

                    # Extract engagement metrics
                    metrics = tweet.get("public_metrics", {})

                    # Extract hashtags and mentions
                    entities = tweet.get("entities", {})
                    hashtags = [tag.get("tag", "") for tag in entities.get("hashtags", [])]
                    mentions = [mention.get("username", "") for mention in entities.get("mentions", [])]

                    normalized_tweet = {
                        "id": str(tweet.get("id")),
                        "body": str(tweet.get("text", "")),
                        "created_at": tweet.get("created_at"),
                        "user": {
                            "username": user_info.get("username", ""),
                            "id": str(author_id or ""),
                            "followers": int(user_info.get("public_metrics", {}).get("followers_count", 0)),
                            "verified": bool(user_info.get("verified", False)),
                            "name": user_info.get("name", "")
                        },
                        "likes": int(metrics.get("like_count", 0)),
                        "replies": int(metrics.get("reply_count", 0)),
                        "retweets": int(metrics.get("retweet_count", 0)),
                        "quotes": int(metrics.get("quote_count", 0)),
                        "hashtags": hashtags,
                        "mentions": mentions,
                        "provider": "twitter"
                    }
                    tweets.append(normalized_tweet)

                    if len(tweets) >= limit:
                        break

                except (ValueError, TypeError) as e:
                    _logger.debug("Error processing tweet for ticker %s: %s", symbol, e)
                    continue

            # Polite delay between requests
            await asyncio.sleep(self.rate_limit_delay)

            _logger.debug("Fetched %d tweets for ticker %s", len(tweets), symbol)
            return tweets

        except Exception as e:
            _logger.error("Failed to fetch tweets for ticker %s: %s", symbol, e)
            self._update_health_failure(e)
            raise

    async def fetch_summary(self, ticker: str, since_ts: Optional[int] = None) -> Dict[str, Any]:
        """
        Fetch aggregated sentiment summary for a ticker from Twitter.

        Args:
            ticker: Stock ticker symbol
            since_ts: Unix timestamp to fetch data since

        Returns:
            Dictionary containing sentiment metrics and counts
        """
        if not ticker or not ticker.strip():
            raise ValueError("Ticker cannot be empty")

        try:
            tweets = await self.fetch_messages(ticker, since_ts=since_ts, limit=200)
            mentions = len(tweets)
            bullish = 0
            bearish = 0
            neutral = 0

            total_engagement = 0
            total_followers = 0
            verified_tweets = 0
            hashtag_counts: Dict[str, int] = {}

            # Define sentiment keywords for financial context
            bullish_keywords = (
                "bull", "bullish", "moon", "to the moon", "diamond hands", "buy", "long",
                "🚀", "rocket", "hold", "hodl", "pump", "rally", "breakout", "calls",
                "green", "up", "rise", "surge", "gain", "profit", "target", "support"
            )
            bearish_keywords = (
                "bear", "bearish", "short", "sell", "dump", "crash", "fall", "drop",
                "puts", "red", "down", "decline", "loss", "resistance", "breakdown",
                "correction", "bubble", "overvalued", "panic", "fear"
            )

            for tweet in tweets:
                try:
                    body = (tweet.get("body") or "").lower()
                    if not body:
                        neutral += 1
                        continue

                    # Calculate engagement score
                    likes = tweet.get("likes", 0)
                    retweets = tweet.get("retweets", 0)
                    replies = tweet.get("replies", 0)
                    quotes = tweet.get("quotes", 0)
                    engagement = likes + (retweets * 2) + (replies * 1.5) + (quotes * 1.5)
                    total_engagement += engagement

                    # Track user metrics
                    user_info = tweet.get("user", {})
                    followers = user_info.get("followers", 0)
                    total_followers += followers

                    if user_info.get("verified", False):
                        verified_tweets += 1

                    # Count hashtags
                    for hashtag in tweet.get("hashtags", []):
                        if hashtag:
                            hashtag_counts[hashtag.lower()] = hashtag_counts.get(hashtag.lower(), 0) + 1

                    # Sentiment analysis with keyword matching
                    has_bullish = any(keyword in body for keyword in bullish_keywords)
                    has_bearish = any(keyword in body for keyword in bearish_keywords)

                    if has_bullish and not has_bearish:
                        bullish += 1
                    elif has_bearish and not has_bullish:
                        bearish += 1
                    else:
                        neutral += 1

                except Exception as e:
                    _logger.debug("Error processing tweet sentiment for ticker %s: %s", ticker, e)
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
            avg_engagement = total_engagement / mentions if mentions > 0 else 0
            avg_followers = total_followers / mentions if mentions > 0 else 0
            verified_ratio = verified_tweets / mentions if mentions > 0 else 0

            # Get top hashtags
            top_hashtags = sorted(hashtag_counts.items(), key=lambda x: x[1], reverse=True)[:5]

            summary = {
                "mentions": mentions,
                "bullish": bullish,
                "bearish": bearish,
                "neutral": neutral,
                "sentiment_score": float(score),
                "total_engagement": int(total_engagement),
                "avg_engagement": float(avg_engagement),
                "avg_followers": float(avg_followers),
                "verified_tweets": verified_tweets,
                "verified_ratio": float(verified_ratio),
                "top_hashtags": dict(top_hashtags),
                "provider": "twitter",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            _logger.debug("Generated Twitter summary for ticker %s: %d mentions, score %.3f, engagement %.1f",
                         ticker, mentions, score, avg_engagement)
            return summary

        except Exception as e:
            _logger.error("Failed to fetch Twitter summary for ticker %s: %s", ticker, e)
            self._update_health_failure(e)
            raise

    async def close(self) -> None:
        """Clean up adapter resources."""
        try:
            if self._session and not self._provided_session:
                await self._session.close()
                self._session = None
            _logger.debug("Twitter adapter closed successfully")
        except Exception as e:
            _logger.warning("Error closing Twitter adapter: %s", e)