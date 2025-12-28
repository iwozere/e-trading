# src/common/sentiments/adapters/async_stocktwits.py
"""
Async Stocktwits adapter using aiohttp.

Provides:
- async fetch_messages(ticker, since_ts=None, limit=200)
- async fetch_summary(ticker, since_ts=None)

Notes:
- Keep modest concurrency to avoid hitting public API limits.
- Returns normalized dicts similar to sync adapter.
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
from src.common.sentiments.processing.heuristic_analyzer import HeuristicSentimentAnalyzer

_logger = setup_logger(__name__)
BASE = "https://api.stocktwits.com/api/2"

class AsyncStocktwitsAdapter(BaseSentimentAdapter):
    def __init__(self, name: str = "stocktwits", session: Optional[aiohttp.ClientSession] = None,
                 concurrency: int = 5, rate_limit_delay: float = 0.5, max_retries: int = 3):
        super().__init__(name, concurrency, rate_limit_delay)
        self._provided_session = session is not None
        self._session = session
        self.max_retries = max_retries
        self._consecutive_failures = 0
        self._analyzer = HeuristicSentimentAnalyzer()

    async def _get_with_retry(self, path: str, params: Optional[dict] = None, timeout: int = 10) -> Optional[dict]:
        """Make HTTP request with exponential backoff retry logic."""
        url = f"{BASE}{path}"
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
                            _logger.warning("Stocktwits 429 rate limit (attempt %d/%d) - sleeping %.2fs",
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
                                _logger.warning("Stocktwits server error %d (attempt %d/%d) - retrying in %.2fs",
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
                        _logger.warning("Stocktwits timeout (attempt %d/%d) - retrying in %.2fs",
                                      attempt + 1, self.max_retries + 1, backoff_delay)
                        await asyncio.sleep(backoff_delay)
                        continue

                except (aiohttp.ClientError, aiohttp.ClientResponseError) as e:
                    last_exception = e
                    if attempt < self.max_retries and not isinstance(e, aiohttp.ClientResponseError) or (
                        isinstance(e, aiohttp.ClientResponseError) and e.status >= 500
                    ):
                        backoff_delay = self.rate_limit_delay * (2 ** attempt)
                        _logger.warning("Stocktwits client error (attempt %d/%d) - retrying in %.2fs: %s",
                                      attempt + 1, self.max_retries + 1, backoff_delay, e)
                        await asyncio.sleep(backoff_delay)
                        continue
                    else:
                        # Client error that shouldn't be retried (4xx except 429)
                        break

                except Exception as e:
                    last_exception = e
                    _logger.debug("Stocktwits unexpected error (attempt %d/%d): %s",
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
            _logger.error("Stocktwits request failed after %d attempts: %s %s",
                         self.max_retries + 1, url, last_exception)

        return None

    async def fetch_messages(self, ticker: str, since_ts: Optional[int] = None, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Fetch individual messages for a ticker from StockTwits.

        Args:
            ticker: Stock ticker symbol
            since_ts: Unix timestamp to fetch messages since (not directly supported by StockTwits API)
            limit: Maximum number of messages to fetch

        Returns:
            List of normalized message dictionaries
        """
        if not ticker or not ticker.strip():
            raise ValueError("Ticker cannot be empty")

        symbol = ticker.upper().strip()
        msgs: List[Dict[str, Any]] = []
        path = f"/streams/symbol/{symbol}.json"
        params = {"limit": min(30, limit)}  # StockTwits API limit is 30 per request

        try:
            payload = await self._get_with_retry(path, params=params)
            if not payload:
                _logger.warning("No payload received for ticker %s", symbol)
                return []

            page = payload.get("messages", [])
            if not page:
                _logger.debug("No messages found for ticker %s", symbol)
                return []

            for m in page:
                try:
                    # Validate required fields
                    if not m.get("id"):
                        _logger.debug("Skipping message without ID for ticker %s", symbol)
                        continue

                    msg = {
                        "id": str(m.get("id")),
                        "body": str(m.get("body", "")),
                        "created_at": m.get("created_at"),
                        "user": {
                            "username": m.get("user", {}).get("username", ""),
                            "id": str(m.get("user", {}).get("id", "")),
                            "followers": int(m.get("user", {}).get("followers", 0)),
                        },
                        "likes": int(m.get("likes", 0)),
                        "replies": int(m.get("replies", 0)),
                        "retweets": 0,  # StockTwits doesn't have retweets, normalize to 0
                        "provider": "stocktwits"
                    }
                    msgs.append(msg)

                    if len(msgs) >= limit:
                        break

                except (ValueError, TypeError) as e:
                    _logger.debug("Error processing message for ticker %s: %s", symbol, e)
                    continue

            # Polite delay between requests
            await asyncio.sleep(self.rate_limit_delay)

            _logger.debug("Fetched %d messages for ticker %s", len(msgs), symbol)
            return msgs

        except Exception as e:
            _logger.error("Failed to fetch messages for ticker %s: %s", symbol, e)
            self._update_health_failure(e)
            raise

    async def fetch_summary(self, ticker: str, since_ts: Optional[int] = None) -> Dict[str, Any]:
        """
        Fetch aggregated sentiment summary for a ticker from StockTwits.

        Args:
            ticker: Stock ticker symbol
            since_ts: Unix timestamp to fetch data since (not directly supported by StockTwits API)

        Returns:
            Dictionary containing sentiment metrics and counts
        """
        if not ticker or not ticker.strip():
            raise ValueError("Ticker cannot be empty")

        try:
            msgs = await self.fetch_messages(ticker, since_ts=since_ts, limit=200)
            mentions = len(msgs)
            bullish = 0
            bearish = 0
            neutral = 0

            for m in msgs:
                try:
                    body = (m.get("body") or "").lower()
                    if not body:
                        neutral += 1
                        continue

                    result = self._analyzer.analyze_sentiment(body)
                    if result.score > 0.1:
                        bullish += 1
                    elif result.score < -0.1:
                        bearish += 1
                    else:
                        neutral += 1

                except Exception as e:
                    _logger.debug("Error processing message sentiment for ticker %s: %s", ticker, e)
                    neutral += 1
                    continue

            # Calculate sentiment score (-1 to 1)
            if mentions > 0:
                score = (bullish - bearish) / mentions
            else:
                score = 0.0

            # Ensure score is within bounds
            score = max(-1.0, min(1.0, score))

            summary = {
                "mentions": mentions,
                "bullish": bullish,
                "bearish": bearish,
                "neutral": neutral,
                "sentiment_score": float(score),
                "provider": "stocktwits",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            _logger.debug("Generated summary for ticker %s: %d mentions, score %.3f", ticker, mentions, score)
            return summary

        except Exception as e:
            _logger.error("Failed to fetch summary for ticker %s: %s", ticker, e)
            self._update_health_failure(e)
            raise

    async def close(self) -> None:
        """Clean up adapter resources."""
        try:
            if self._session and not self._provided_session:
                await self._session.close()
                self._session = None
            _logger.debug("StockTwits adapter closed successfully")
        except Exception as e:
            _logger.warning("Error closing StockTwits adapter: %s", e)
