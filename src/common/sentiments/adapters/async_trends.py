# src/common/sentiments/adapters/async_trends.py
"""
Async Google Trends adapter for sentiment indicator data collection.

Provides:
- async fetch_messages(ticker, since_ts=None, limit=200)
- async fetch_summary(ticker, since_ts=None)

Features:
- Google Trends data collection for search volume analysis
- Search volume correlation with sentiment indicators
- Geographic sentiment distribution analysis
- Related queries and trending topics analysis
"""
import asyncio
import aiohttp
from typing import List, Dict, Optional, Any
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
import json
from urllib.parse import quote_plus
import random

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.common.sentiments.adapters.base_adapter import BaseSentimentAdapter

_logger = setup_logger(__name__)


class AsyncTrendsAdapter(BaseSentimentAdapter):
    """
    Async Google Trends sentiment adapter.

    Supports search volume analysis, geographic distribution,
    and trending topic correlation for sentiment indicators.

    Note: This adapter uses unofficial Google Trends access methods
    and should be used carefully to respect rate limits.
    """

    def __init__(self, name: str = "trends", session: Optional[aiohttp.ClientSession] = None,
                 concurrency: int = 1, rate_limit_delay: float = 2.0, max_retries: int = 2):
        super().__init__(name, concurrency, rate_limit_delay)
        self._session = session
        self.max_retries = max_retries
        self._tokens = {}
        self._token_expiry = 0
        self._cookies_fetched = False
        self._cookie_lock = asyncio.Lock()
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
        self._analyzer = None # Lazy load

    async def _get_session(self):
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession(headers={"User-Agent": self.user_agents[0]})
        return self._session

    async def _ensure_cookies(self):
        async with self._cookie_lock:
            if self._cookies_fetched: return True
            session = await self._get_session()
            try:
                # Visit the main page to get NID cookies etc.
                async with session.get("https://trends.google.com/?geo=US", timeout=10) as r:
                    if r.status == 200:
                        self._cookies_fetched = True
                        return True
            except: pass
            return False

    async def _get_data(self, url: str, params: dict, referer: str) -> Optional[dict]:
        session = await self._get_session()
        if not self._cookies_fetched: await self._ensure_cookies()

        headers = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": referer,
            "X-Requested-With": "XMLHttpRequest",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "DNT": "1"
        }

        for attempt in range(self.max_retries + 1):
            try:
                async with session.get(url, params=params, headers=headers, timeout=15) as resp:
                    if resp.status == 200:
                        text = (await resp.text()).strip()
                        if text.startswith(")]}'"): text = text[5:].strip()
                        return json.loads(text)
                    if resp.status == 429:
                        await asyncio.sleep(30 * (attempt + 1))
                        continue
                    if resp.status == 400:
                        _logger.warning("Trends 400 at %s", url)
                        return None
            except Exception as e:
                _logger.debug("Trends error: %s", e)
                await asyncio.sleep(5)
        return None

    async def _fetch_tokens(self, ticker: str):
        if self._tokens and time.time() < self._token_expiry: return True

        data = await self._get_data(
            "https://trends.google.com/trends/api/explore",
            {
                'hl': 'en-US',
                'tz': 0,
                'req': json.dumps({'comparisonItem': [{'keyword': ticker.upper(), 'geo': 'US', 'time': 'now 7-d'}], 'category': 0, 'property': ''}, separators=(',', ':'))
            },
            f"https://trends.google.com/trends/explore?geo=US&q={ticker.upper()}"
        )
        if data and 'widgets' in data:
            self._tokens = {w['id']: {'token': w['token'], 'request': w.get('request')} for w in data['widgets'] if 'token' in w}
            self._token_expiry = time.time() + 1800
            return True
        return False

    async def fetch_messages(self, ticker: str, since_ts: Optional[int] = None, limit: int = 100) -> List[Dict]:
        if not await self._fetch_tokens(ticker): return []

        widget = self._tokens.get('TIMESERIES')
        if not widget or 'token' not in widget: return []

        data = await self._get_data(
            "https://trends.google.com/trends/api/widgetdata/multiline",
            {
                'hl': 'en-US',
                'tz': 0,
                'req': json.dumps(widget['request'], separators=(',', ':')),
                'token': widget['token']
            },
            f"https://trends.google.com/trends/explore?geo=US&q={ticker.upper()}"
        )

        messages = []
        if data and 'default' in data:
            for pt in data['default'].get('timelineData', [])[-limit:]:
                messages.append({
                    'id': f"trends_{pt.get('time')}",
                    'body': f"Search interest for {ticker}",
                    'created_at': datetime.fromtimestamp(int(pt.get('time', 0))).isoformat() if pt.get('time') else "",
                    'search_volume': pt.get('value', [0])[0],
                    'provider': 'trends'
                })
        return messages

    async def fetch_summary(self, ticker: str, since_ts: Optional[int] = None) -> Dict:
        # Lazy load analyzer
        if self._analyzer is None:
            from src.common.sentiments.processing.heuristic_analyzer import HeuristicSentimentAnalyzer
            self._analyzer = HeuristicSentimentAnalyzer()

        msgs = await self.fetch_messages(ticker, limit=50)
        volume = sum(m.get('search_volume', 0) for m in msgs)

        # Simple trend direction
        direction = 0
        if len(msgs) > 10:
            recent = sum(m['search_volume'] for m in msgs[-5:]) / 5
            older = sum(m['search_volume'] for m in msgs[-10:-5]) / 5
            direction = 1 if recent > older else -1 if recent < older else 0

        return {
            "mentions": len(msgs),
            "sentiment_score": float(direction * 0.2), # Basics
            "total_search_volume": volume,
            "trend_direction": direction,
            "provider": "trends",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
