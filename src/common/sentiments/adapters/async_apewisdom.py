"""
Async ApeWisdom adapter using aiohttp.

ApeWisdom tracks mentions and sentiment across Reddit (e.g. Wallstreetbets) and 4chan.
Provides:
- async fetch_summary(ticker, since_ts=None)

Note: ApeWisdom API does not easily expose raw message bodies in a bulk /filter endpoint,
so fetch_messages relies on aggregating the high-level summary.
"""
import asyncio
import aiohttp
from typing import List, Dict, Optional, Any
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.common.sentiments.adapters.base_adapter import BaseSentimentAdapter

_logger = setup_logger(__name__)
BASE = "https://apewisdom.io/api/v1.0/filter/all-stocks/page/"

class AsyncApeWisdomAdapter(BaseSentimentAdapter):
    def __init__(self, name: str = "apewisdom", session: Optional[aiohttp.ClientSession] = None,
                 concurrency: int = 3, rate_limit_delay: float = 1.0, max_retries: int = 3):
        # ApeWisdom has soft limits, being conservative with concurrency=3
        super().__init__(name, concurrency, rate_limit_delay)
        self._provided_session = session is not None
        self._session = session
        self.max_retries = max_retries
        self._consecutive_failures = 0
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        # Cache pages to avoid hitting ApeWisdom multiple times for different tickers in the same batch
        self._cache = []
        self._cache_timestamp = 0

    async def _ensure_session(self):
        if not self._session or self._session.closed:
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'application/json'
            }
            self._session = aiohttp.ClientSession(headers=headers)

    async def _fetch_pages(self, max_pages: int = 5) -> List[Dict[str, Any]]:
        """Fetch the top N pages of trending stocks. Uses simple RAM caching per batch run."""
        now = time.time()
        # Return cache if less than 5 minutes old
        if self._cache and (now - self._cache_timestamp) < 300:
            return self._cache

        await self._ensure_session()
        all_results = []
        
        for page in range(1, max_pages + 1):
            url = f"{BASE}{page}"
            
            for attempt in range(self.max_retries + 1):
                async with self.semaphore:
                    try:
                        start_time = time.time()
                        async with self._session.get(url, timeout=10) as resp:
                            response_time_ms = (time.time() - start_time) * 1000

                            if resp.status == 429:
                                backoff = self.rate_limit_delay * (2 ** attempt)
                                _logger.warning("ApeWisdom 429 rate limit - sleeping %.2fs", backoff)
                                await asyncio.sleep(backoff)
                                continue

                            if resp.status != 200:
                                _logger.warning("ApeWisdom status %d for %s", resp.status, url)
                                break 

                            data = await resp.json()
                            results = data.get('results', [])
                            all_results.extend(results)
                            
                            self._consecutive_failures = 0
                            self._update_health_success(response_time_ms)
                            
                            # Be extremely polite to the free API
                            await asyncio.sleep(self.rate_limit_delay)
                            break # Success, break retry loop

                    except Exception as e:
                        _logger.debug("ApeWisdom error (attempt %d): %s", attempt + 1, e)
                        if attempt < self.max_retries:
                            await asyncio.sleep(self.rate_limit_delay * (2 ** attempt))
                        else:
                            self._consecutive_failures += 1
                            self._update_health_failure(e)

        self._cache = all_results
        self._cache_timestamp = time.time()
        return all_results

    async def fetch_messages(self, ticker: str, since_ts: Optional[int] = None, limit: int = 200) -> List[Dict[str, Any]]:
        """
        ApeWisdom doesn't expose raw messages easily via the free /filter endpoint.
        Returning an empty list to satisfy the BaseAdapter signature while relying on fetch_summary.
        """
        return []

    async def fetch_summary(self, ticker: str, since_ts: Optional[int] = None) -> Dict[str, Any]:
        """
        Fetch summary matching the ticker from ApeWisdom's trending list.
        """
        if not ticker or not ticker.strip():
            raise ValueError("Ticker cannot be empty")

        symbol = ticker.upper().strip()
        try:
            # We fetch top 5 pages (~500 stocks) to see if our ticker is trending
            trending_stocks = await self._fetch_pages(max_pages=5)
            
            # Find the assigned ticker
            target_data = next((item for item in trending_stocks if item.get('ticker') == symbol), None)

            if not target_data:
                _logger.debug("Ticker %s not found in top 500 ApeWisdom trending stocks", symbol)
                return {
                    "mentions": 0,
                    "bullish": 0,
                    "bearish": 0,
                    "neutral": 0,
                    "sentiment_score": 0.0,
                    "provider": "apewisdom",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

            # ApeWisdom provides upvotes and mentions. 
            # We map mentions and upvotes indirectly to sentiment since exact breakdown isn't in this payload
            mentions = target_data.get('mentions', 0)
            upvotes = target_data.get('upvotes', 0)
            
            # Heuristic: High upvote-to-mention ratio usually implies positive sentiment on Reddit boards
            # A neutral ratio is usually 2:1. Let's model a naive score based on this.
            score = 0.0
            if mentions > 0:
                ratio = upvotes / mentions
                if ratio > 5.0:
                    score = 0.8  # highly upvoted mentions = bullish
                elif ratio > 2.0:
                    score = 0.4
                elif ratio < 1.0:
                    score = -0.4 # mentions with no upvotes = heavily downvoted = bearish
            
            summary = {
                "mentions": mentions,
                "bullish": int(mentions * 0.7) if score > 0 else int(mentions * 0.2),
                "bearish": int(mentions * 0.7) if score < 0 else int(mentions * 0.2),
                "neutral": int(mentions * 0.3),
                "sentiment_score": float(score),
                "provider": "apewisdom",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            _logger.debug("Generated ApeWisdom summary for %s: %d mentions, score %.3f", symbol, mentions, score)
            return summary

        except Exception as e:
            _logger.error("Failed to fetch ApeWisdom summary for %s: %s", ticker, e)
            self._update_health_failure(e)
            raise

    async def close(self) -> None:
        try:
            if self._session and not self._provided_session:
                await self._session.close()
                self._session = None
            self._cache = []
        except Exception as e:
            _logger.warning("Error closing ApeWisdom adapter: %s", e)
