# src/common/sentiments/adapters/async_reddit.py
"""
Async Reddit adapter using direct Reddit API (OAuth2).

Provides:
- async fetch_messages(ticker, since_ts=None, limit=200)
- async fetch_summary(ticker, since_ts=None)

Features:
- OAuth2 token management (auto-refresh)
- Configurable subreddits from JSON
- Retry mechanism with exponential backoff
- Ticker mention searching
"""
import asyncio
import aiohttp
import json
import time
import os
from typing import List, Dict, Optional, Any
from pathlib import Path
import sys
from datetime import datetime, timezone

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.common.sentiments.adapters.base_adapter import BaseSentimentAdapter
from src.common.sentiments.processing.heuristic_analyzer import HeuristicSentimentAnalyzer
import config.donotshare.donotshare as secrets

_logger = setup_logger(__name__)

REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
REDDIT_OAUTH_BASE = "https://oauth.reddit.com"
CONFIG_PATH = PROJECT_ROOT / "config" / "sentiments" / "subreddits.json"

class AsyncRedditAdapter(BaseSentimentAdapter):
    def __init__(self, name: str = "reddit_direct", session: Optional[aiohttp.ClientSession] = None,
                 concurrency: int = 5, rate_limit_delay: float = 1.0, max_retries: int = 3):
        super().__init__(name, concurrency, rate_limit_delay)
        self._provided_session = session is not None
        self._session = session
        self.max_retries = max_retries
        self._token: Optional[str] = None
        self._token_expiry: float = 0
        self._token_lock = asyncio.Lock()
        self._analyzer = HeuristicSentimentAnalyzer()
        self._subreddits: List[str] = self._analyzer.get_subreddits()

        # API Credentials from config
        self.client_id = secrets.REDDIT_API_KEY
        self.client_secret = secrets.REDDIT_API_SECRET
        self.user_agent = secrets.REDDIT_USER_AGENT or "e-trading:sentiment:v1.0.0 (by /u/akoss)"

        # Identify common placeholder strings
        placeholders = ["YOUR_CLIENT_ID", "YOUR_CLIENT_SECRET", "YOUR_API_KEY", "YOUR_API_SECRET"]
        has_placeholders = any(p in str(self.client_id) or p in str(self.client_secret) for p in placeholders)

        self.enabled = bool(self.client_id and self.client_secret and not has_placeholders)
        if not self.enabled:
            _logger.warning("Reddit credentials missing or placeholder detected - adapter will be inactive.")


    async def _ensure_token(self):
        """Ensure we have a valid OAuth2 token."""
        if not self.enabled:
            return

        if self._token and time.time() < self._token_expiry - 60:
            return

        async with self._token_lock:
            # Re-check after acquiring lock
            if self._token and time.time() < self._token_expiry - 60:
                return

        if not self._session:
            self._session = aiohttp.ClientSession()

        _logger.info("Fetching new Reddit OAuth token")
        auth = aiohttp.BasicAuth(self.client_id, self.client_secret)
        data = {"grant_type": "client_credentials"}
        headers = {"User-Agent": self.user_agent}

        try:
            async with self._session.post(REDDIT_TOKEN_URL, auth=auth, data=data, headers=headers) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    _logger.error("Failed to fetch Reddit token (Status %d): %s", resp.status, error_text)
                    resp.raise_for_status()
                payload = await resp.json()
                self._token = payload["access_token"]
                # Expires in usually 3600 seconds
                self._token_expiry = time.time() + payload.get("expires_in", 3600)
                _logger.debug("Successfully acquired Reddit token, expires in %d seconds", payload.get("expires_in", 3600))
        except Exception as e:
            _logger.error("Failed to fetch Reddit token: %s", e)
            self._update_health_failure(e)
            raise

    async def _request_with_retry(self, method: str, path: str, params: Optional[Dict] = None,
                                data: Optional[Dict] = None) -> Optional[Dict]:
        """Make an authenticated request to Reddit API with retry logic."""
        await self._ensure_token()

        url = f"{REDDIT_OAUTH_BASE}{path}"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "User-Agent": self.user_agent
        }

        last_exception = None
        for attempt in range(self.max_retries + 1):
            async with self.semaphore:
                try:
                    start_time = time.time()
                    async with self._session.request(method, url, params=params, data=data, headers=headers) as resp:
                        response_time_ms = (time.time() - start_time) * 1000

                        if resp.status == 429:
                            # Rate limited
                            retry_after = int(resp.headers.get("Retry-After", self.rate_limit_delay * (2 ** attempt)))
                            _logger.warning("Reddit 429 rate limit (attempt %d/%d) - sleeping %ds",
                                          attempt + 1, self.max_retries + 1, retry_after)
                            await asyncio.sleep(retry_after)
                            continue

                        if resp.status >= 500:
                            if attempt < self.max_retries:
                                backoff_delay = self.rate_limit_delay * (2 ** attempt)
                                _logger.warning("Reddit server error %d (attempt %d/%d) - retrying in %.2fs",
                                              resp.status, attempt + 1, self.max_retries + 1, backoff_delay)
                                await asyncio.sleep(backoff_delay)
                                continue

                        resp.raise_for_status()
                        payload = await resp.json()
                        self._update_health_success(response_time_ms)
                        return payload

                except Exception as e:
                    last_exception = e
                    if attempt < self.max_retries:
                        backoff_delay = self.rate_limit_delay * (2 ** attempt)
                        _logger.debug("Reddit API error %s (attempt %d/%d) - retrying in %.2fs",
                                    e, attempt + 1, self.max_retries + 1, backoff_delay)
                        await asyncio.sleep(backoff_delay)
                        continue
                    break

        if last_exception:
            _logger.error("Reddit request failed after %d attempts: %s", self.max_retries + 1, last_exception)
            self._update_health_failure(last_exception)

        return None

    async def fetch_messages(self, ticker: str, since_ts: Optional[int] = None, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Fetch individual messsages (submissions and comments) for a ticker from Reddit.
        """
        if not ticker or not self.enabled:
            return []

        ticker = ticker.upper().strip()
        query = f'"{ticker}" OR "${ticker}"'

        # We'll search across subreddits
        # Note: Reddit search is limited. Alternatively, we could stream from specific subreddits.
        # For a "fetch" approach, search is more appropriate.

        all_messages: List[Dict[str, Any]] = []

        # Split limit between search results
        # Reddit search can return both submissions and comments (if searching comments specifically)
        # But standard search is mostly submissions.

        # 1. Search submissions
        search_params = {
            "q": query,
            "sort": "new",
            "limit": min(limit, 100),
            "t": "day" if not since_ts else "all"
        }

        # If since_ts is provided, we might need to filter manually as 'after' in search is for 'fullname'

        try:
            # Search across all specified subreddits or just 'all'?
            # 'all' is broader. User requested "config/sentiments/subreddits.json"
            # We can search per subreddit or use 'sr:wsb OR sr:stocks ...'

            sr_query = " OR ".join([f"subreddit:{sr}" for sr in self._subreddits])
            full_query = f"({query}) AND ({sr_query})"
            search_params["q"] = full_query

            payload = await self._request_with_retry("GET", "/search.json", params=search_params)

            if payload and "data" in payload and "children" in payload["data"]:
                for child in payload["data"]["children"]:
                    data = child["data"]

                    # Filter by timestamp
                    created_utc = data.get("created_utc")
                    if since_ts and created_utc and created_utc < since_ts:
                        continue

                    msg = {
                        "id": data.get("name"), # Fullname
                        "body": (data.get("title", "") + " " + data.get("selftext", "")).strip(),
                        "created_at": created_utc,
                        "user": {
                            "username": data.get("author", ""),
                            "id": data.get("author_fullname", ""),
                            "followers": 0,
                        },
                        "likes": int(data.get("score", 0)),
                        "replies": int(data.get("num_comments", 0)),
                        "retweets": 0,
                        "provider": self.name,
                        "type": "submission",
                        "url": f"https://reddit.com{data.get('permalink', '')}"
                    }
                    all_messages.append(msg)

            # Sort by creation time
            all_messages.sort(key=lambda x: x.get("created_at", 0), reverse=True)
            return all_messages[:limit]

        except Exception as e:
            _logger.error("Error fetching Reddit messages for %s: %s", ticker, e)
            return []

    async def fetch_summary(self, ticker: str, since_ts: Optional[int] = None) -> Dict[str, Any]:
        """
        Fetch aggregated sentiment summary for a ticker from Reddit.
        """
        try:
            msgs = await self.fetch_messages(ticker, since_ts=since_ts, limit=200)
            mentions = len(msgs)

            # Unified sentiment analysis
            pos = 0
            neg = 0
            neutral = 0
            total_score = 0.0

            for m in msgs:
                result = self._analyzer.analyze_sentiment(m["body"])
                total_score += result.score
                if result.score > 0.1:
                    pos += 1
                elif result.score < -0.1:
                    neg += 1
                else:
                    neutral += 1

            avg_score = total_score / mentions if mentions > 0 else 0.0

            return {
                "mentions": mentions,
                "pos": pos,
                "neg": neg,
                "neutral": neutral,
                "sentiment_score": float(avg_score),
                "unique_authors": len({m["user"]["id"] for m in msgs if m["user"]["id"]}),
                "provider": self.name,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            _logger.error("Error fetching Reddit summary for %s: %s", ticker, e)
            return {
                "mentions": 0,
                "sentiment_score": 0.0,
                "provider": self.name,
                "error": str(e)
            }

    async def close(self) -> None:
        """Clean up adapter resources."""
        if self._session and not self._provided_session:
            await self._session.close()
            self._session = None
