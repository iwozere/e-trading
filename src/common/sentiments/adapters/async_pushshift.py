# src/common/sentiments/adapters/async_pushshift.py
"""
Async Pushshift adapter for Reddit (submissions & comments) using aiohttp.

Provides:
- async fetch_submissions(ticker, since_ts, limit)
- async fetch_comments(ticker, since_ts, limit)
- async fetch_mentions_summary(ticker, since_ts)

Notes:
- Pushshift endpoint: https://api.pushshift.io/reddit/search/submission (and /comment)
- Keep requests small and respect rate limits.
"""
import asyncio
import aiohttp
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
BASE = "https://api.pushshift.io/reddit/search"

class AsyncPushshiftAdapter(BaseSentimentAdapter):
    def __init__(self, name: str = "reddit", session: Optional[aiohttp.ClientSession] = None,
                 concurrency: int = 5, rate_limit_delay: float = 0.5, max_retries: int = 3):
        super().__init__(name, concurrency, rate_limit_delay)
        self._provided_session = session is not None
        self._session = session
        self.max_retries = max_retries
        self._consecutive_failures = 0

    async def _get_with_retry(self, endpoint: str, params: Dict, timeout: int = 15) -> List[Dict]:
        """Make HTTP request with exponential backoff retry logic."""
        url = f"{BASE}/{endpoint}"
        if not self._session:
            self._session = aiohttp.ClientSession()

        last_exception = None

        for attempt in range(self.max_retries + 1):
            async with self.semaphore:
                try:
                    start_time = time.time()

                    async with self._session.get(url, params=params, timeout=timeout) as resp:
                        response_time_ms = (time.time() - start_time) * 1000

                        if resp.status == 429:
                            # Rate limited - use exponential backoff
                            backoff_delay = self.rate_limit_delay * (2 ** attempt)
                            _logger.warning("Pushshift 429 rate limit (attempt %d/%d) - sleeping %.2fs",
                                          attempt + 1, self.max_retries + 1, backoff_delay)
                            await asyncio.sleep(backoff_delay)

                            if attempt < self.max_retries:
                                continue
                            else:
                                raise aiohttp.ClientResponseError(
                                    request_info=resp.request_info,
                                    history=resp.history,
                                    status=resp.status,
                                    message="Rate limit exceeded after retries"
                                )

                        if resp.status >= 500:
                            # Server error - retry with backoff
                            if attempt < self.max_retries:
                                backoff_delay = self.rate_limit_delay * (2 ** attempt)
                                _logger.warning("Pushshift server error %d (attempt %d/%d) - retrying in %.2fs",
                                              resp.status, attempt + 1, self.max_retries + 1, backoff_delay)
                                await asyncio.sleep(backoff_delay)
                                continue

                        resp.raise_for_status()
                        data = await resp.json()

                        # Success - reset failure counter and update health
                        self._consecutive_failures = 0
                        self._update_health_success(response_time_ms)

                        return data.get("data", [])

                except asyncio.TimeoutError as e:
                    last_exception = e
                    if attempt < self.max_retries:
                        backoff_delay = self.rate_limit_delay * (2 ** attempt)
                        _logger.warning("Pushshift timeout (attempt %d/%d) - retrying in %.2fs",
                                      attempt + 1, self.max_retries + 1, backoff_delay)
                        await asyncio.sleep(backoff_delay)
                        continue

                except (aiohttp.ClientError, aiohttp.ClientResponseError) as e:
                    last_exception = e
                    if attempt < self.max_retries and not isinstance(e, aiohttp.ClientResponseError) or (
                        isinstance(e, aiohttp.ClientResponseError) and e.status >= 500
                    ):
                        backoff_delay = self.rate_limit_delay * (2 ** attempt)
                        _logger.warning("Pushshift client error (attempt %d/%d) - retrying in %.2fs: %s",
                                      attempt + 1, self.max_retries + 1, backoff_delay, e)
                        await asyncio.sleep(backoff_delay)
                        continue
                    else:
                        # Client error that shouldn't be retried (4xx except 429)
                        break

                except Exception as e:
                    last_exception = e
                    _logger.debug("Pushshift unexpected error (attempt %d/%d): %s",
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
            _logger.error("Pushshift request failed after %d attempts: %s %s",
                         self.max_retries + 1, url, last_exception)

        return []

    async def fetch_submissions(self, ticker: str, since_ts: Optional[int], limit: int = 100) -> List[Dict]:
        """Fetch Reddit submissions mentioning the ticker."""
        if not ticker or not ticker.strip():
            raise ValueError("Ticker cannot be empty")

        ticker = ticker.upper().strip()
        q = f"{ticker} OR ${ticker}"
        params = {"q": q, "size": min(limit, 500)}

        if since_ts:
            params["after"] = since_ts

        try:
            res = await self._get_with_retry("submission", params)
            await asyncio.sleep(self.rate_limit_delay)
            _logger.debug("Fetched %d submissions for ticker %s", len(res), ticker)
            return res
        except Exception as e:
            _logger.error("Failed to fetch submissions for ticker %s: %s", ticker, e)
            self._update_health_failure(e)
            raise

    async def fetch_comments(self, ticker: str, since_ts: Optional[int], limit: int = 100) -> List[Dict]:
        """Fetch Reddit comments mentioning the ticker."""
        if not ticker or not ticker.strip():
            raise ValueError("Ticker cannot be empty")

        ticker = ticker.upper().strip()
        q = f"{ticker} OR ${ticker}"
        params = {"q": q, "size": min(limit, 500)}

        if since_ts:
            params["after"] = since_ts

        try:
            res = await self._get_with_retry("comment", params)
            await asyncio.sleep(self.rate_limit_delay)
            _logger.debug("Fetched %d comments for ticker %s", len(res), ticker)
            return res
        except Exception as e:
            _logger.error("Failed to fetch comments for ticker %s: %s", ticker, e)
            self._update_health_failure(e)
            raise

    async def fetch_mentions_summary(self, ticker: str, since_ts: Optional[int]) -> Dict[str, Any]:
        """
        Fetch aggregated sentiment summary for a ticker from Reddit.

        Args:
            ticker: Stock ticker symbol
            since_ts: Unix timestamp to fetch data since

        Returns:
            Dictionary containing sentiment metrics and counts
        """
        if not ticker or not ticker.strip():
            raise ValueError("Ticker cannot be empty")

        try:
            subs = await self.fetch_submissions(ticker, since_ts, limit=200)
            comms = await self.fetch_comments(ticker, since_ts, limit=500)

            mentions = len(subs) + len(comms)
            pos = 0
            neg = 0
            neutral = 0

            # Define sentiment keywords
            positive_keywords = ("moon", "rocket", "diamond", "buy", "long", "🚀", "hold", "bull", "up")
            negative_keywords = ("short", "sell", "dump", "bankrupt", "crash", "bear", "down", "fall")

            # Process submissions
            for s in subs:
                try:
                    text = ((s.get("title") or "") + " " + (s.get("selftext") or "")).lower()
                    if not text.strip():
                        neutral += 1
                        continue

                    has_positive = any(keyword in text for keyword in positive_keywords)
                    has_negative = any(keyword in text for keyword in negative_keywords)

                    if has_positive and not has_negative:
                        pos += 1
                    elif has_negative and not has_positive:
                        neg += 1
                    else:
                        neutral += 1

                except Exception as e:
                    _logger.debug("Error processing submission sentiment for ticker %s: %s", ticker, e)
                    neutral += 1
                    continue

            # Process comments
            for c in comms:
                try:
                    text = (c.get("body") or "").lower()
                    if not text.strip():
                        neutral += 1
                        continue

                    has_positive = any(keyword in text for keyword in positive_keywords)
                    has_negative = any(keyword in text for keyword in negative_keywords)

                    if has_positive and not has_negative:
                        pos += 1
                    elif has_negative and not has_positive:
                        neg += 1
                    else:
                        neutral += 1

                except Exception as e:
                    _logger.debug("Error processing comment sentiment for ticker %s: %s", ticker, e)
                    neutral += 1
                    continue

            # Calculate unique authors
            unique_authors = len({
                (m.get("author") or m.get("author_fullname") or "unknown")
                for m in (subs + comms)
                if (m.get("author") or m.get("author_fullname"))
            })

            # Calculate sentiment score (-1 to 1)
            if mentions > 0:
                score = (pos - neg) / mentions
            else:
                score = 0.0

            # Ensure score is within bounds
            score = max(-1.0, min(1.0, score))

            summary = {
                "mentions": mentions,
                "pos": pos,
                "neg": neg,
                "neutral": neutral,
                "sentiment_score": float(score),
                "unique_authors": unique_authors,
                "provider": "reddit",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            _logger.debug("Generated summary for ticker %s: %d mentions, %d unique authors, score %.3f",
                         ticker, mentions, unique_authors, score)
            return summary

        except Exception as e:
            _logger.error("Failed to fetch mentions summary for ticker %s: %s", ticker, e)
            self._update_health_failure(e)
            raise

    async def fetch_messages(self, ticker: str, since_ts: Optional[int] = None, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Fetch individual messages for a ticker from Reddit (submissions + comments).

        Args:
            ticker: Stock ticker symbol
            since_ts: Unix timestamp to fetch messages since
            limit: Maximum number of messages to fetch

        Returns:
            List of normalized message dictionaries
        """
        if not ticker or not ticker.strip():
            raise ValueError("Ticker cannot be empty")

        try:
            # Split limit between submissions and comments
            sub_limit = min(limit // 3, 100)  # Fewer submissions, more comments typically
            comm_limit = limit - sub_limit

            subs = await self.fetch_submissions(ticker, since_ts, sub_limit)
            comms = await self.fetch_comments(ticker, since_ts, comm_limit)

            messages: List[Dict[str, Any]] = []

            # Normalize submissions
            for s in subs:
                try:
                    msg = {
                        "id": str(s.get("id", "")),
                        "body": ((s.get("title") or "") + " " + (s.get("selftext") or "")).strip(),
                        "created_at": s.get("created_utc"),
                        "user": {
                            "username": s.get("author", ""),
                            "id": s.get("author_fullname", ""),
                            "followers": 0,  # Not available in Pushshift
                        },
                        "likes": int(s.get("score", 0)),
                        "replies": int(s.get("num_comments", 0)),
                        "retweets": 0,  # Not applicable to Reddit
                        "provider": "reddit",
                        "type": "submission"
                    }
                    messages.append(msg)
                except (ValueError, TypeError) as e:
                    _logger.debug("Error processing submission for ticker %s: %s", ticker, e)
                    continue

            # Normalize comments
            for c in comms:
                try:
                    msg = {
                        "id": str(c.get("id", "")),
                        "body": str(c.get("body", "")),
                        "created_at": c.get("created_utc"),
                        "user": {
                            "username": c.get("author", ""),
                            "id": c.get("author_fullname", ""),
                            "followers": 0,  # Not available in Pushshift
                        },
                        "likes": int(c.get("score", 0)),
                        "replies": 0,  # Not easily available in Pushshift
                        "retweets": 0,  # Not applicable to Reddit
                        "provider": "reddit",
                        "type": "comment"
                    }
                    messages.append(msg)
                except (ValueError, TypeError) as e:
                    _logger.debug("Error processing comment for ticker %s: %s", ticker, e)
                    continue

            # Sort by creation time if available
            messages.sort(key=lambda x: x.get("created_at", 0), reverse=True)

            _logger.debug("Fetched %d total messages for ticker %s", len(messages), ticker)
            return messages[:limit]  # Ensure we don't exceed limit

        except Exception as e:
            _logger.error("Failed to fetch messages for ticker %s: %s", ticker, e)
            self._update_health_failure(e)
            raise

    async def fetch_summary(self, ticker: str, since_ts: Optional[int] = None) -> Dict[str, Any]:
        """
        Fetch aggregated sentiment summary for a ticker from Reddit.

        This is an alias for fetch_mentions_summary to maintain interface compatibility.

        Args:
            ticker: Stock ticker symbol
            since_ts: Unix timestamp to fetch data since

        Returns:
            Dictionary containing sentiment metrics and counts
        """
        return await self.fetch_mentions_summary(ticker, since_ts)

    async def close(self) -> None:
        """Clean up adapter resources."""
        try:
            if self._session and not self._provided_session:
                await self._session.close()
                self._session = None
            _logger.debug("Reddit adapter closed successfully")
        except Exception as e:
            _logger.warning("Error closing Reddit adapter: %s", e)
