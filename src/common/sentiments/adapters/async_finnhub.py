# src/common/sentiments/adapters/async_finnhub.py
"""
Async Finnhub Sentiment adapter.

Provides:
- async fetch_summary(ticker, since_ts=None)
- async fetch_messages(ticker, since_ts=None, limit=200) (as proxy to news)

Uses Finnhub's dedicated /news-sentiment and /stock/social-sentiment endpoints.
"""
import asyncio
import aiohttp
import os
import sys
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.common.sentiments.adapters.base_adapter import BaseSentimentAdapter
from src.common.sentiments.processing.heuristic_analyzer import HeuristicSentimentAnalyzer
from src.data.downloader.finnhub_data_downloader import FinnhubDataDownloader

_logger = setup_logger(__name__)

class AsyncFinnhubSentimentAdapter(BaseSentimentAdapter):
    """
    Adapter for Finnhub's proprietary sentiment analytics.

    This differs from AsyncNewsAdapter by using Finnhub's calculated
    sentiment scores rather than just raw news articles.
    """

    def __init__(self, name: str = "finnhub", session: Optional[aiohttp.ClientSession] = None,
                 concurrency: int = 2, rate_limit_delay: float = 1.0, max_retries: int = 3,
                 api_key: Optional[str] = None):
        super().__init__(name, concurrency, rate_limit_delay)
        self.api_key = api_key or os.getenv('FINNHUB_API_KEY')
        if not self.api_key:
            _logger.warning("Finnhub API key not found for sentiment adapter")

        self.downloader = FinnhubDataDownloader(api_key=self.api_key)
        self.max_retries = max_retries
        self._session = session
        self._provided_session = session is not None
        self._analyzer = HeuristicSentimentAnalyzer()

    async def fetch_messages(self, ticker: str, since_ts: Optional[int] = None, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Fetch news articles from Finnhub as individual messages.
        Used primarily for local analysis or HuggingFace enrichment.
        """
        if not ticker:
            return []

        symbol = ticker.upper().strip()
        try:
            # Calculate date range
            to_date = datetime.now().strftime('%Y-%m-%d')
            if since_ts:
                from_date = datetime.fromtimestamp(since_ts).strftime('%Y-%m-%d')
            else:
                from_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

            raw_news = await self.downloader.get_company_news(symbol, from_date, to_date)
            if not raw_news:
                return []

            msgs = []
            for article in raw_news[:limit]:
                msgs.append({
                    "id": f"finnhub_{article.get('id', '')}",
                    "body": f"{article.get('headline', '')} {article.get('summary', '')}",
                    "created_at": datetime.fromtimestamp(article.get('datetime', 0)).isoformat(),
                    "url": article.get('url', ''),
                    "user": {"username": article.get('source', 'finnhub')},
                    "provider": "finnhub"
                })
            return msgs
        except Exception as e:
            _logger.error("Error fetching Finnhub news messages for %s: %s", symbol, e)
            return []

    async def _fetch_summary_fallback(self, symbol: str, since_ts: Optional[int] = None) -> Dict[str, Any]:
        """Fallback to raw news analysis if premium sentiment endpoints are blocked."""
        try:
            msgs = await self.fetch_messages(symbol, since_ts)
            if not msgs:
                return {
                    "mentions": 0,
                    "sentiment_score": 0.0,
                    "provider": "finnhub",
                    "status": "no_data",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

            bullish = 0
            bearish = 0
            neutral = 0
            mentions = len(msgs)

            for m in msgs:
                body = m.get("body", "")
                result = self._analyzer.analyze_sentiment(body)
                if result.score > 0.1:
                    bullish += 1
                elif result.score < -0.1:
                    bearish += 1
                else:
                    neutral += 1

            sentiment_score = (bullish - bearish) / mentions if mentions > 0 else 0.0

            return {
                "mentions": mentions,
                "sentiment_score": float(sentiment_score),
                "bullish": bullish,
                "bearish": bearish,
                "neutral": neutral,
                "provider": "finnhub",
                "status": "fallback_news",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            _logger.error("Error in Finnhub fallback analysis for %s: %s", symbol, e)
            return {"mentions": 0, "sentiment_score": 0.0, "error": str(e), "provider": "finnhub", "timestamp": datetime.now(timezone.utc).isoformat()}

    async def fetch_summary(self, ticker: str, since_ts: Optional[int] = None) -> Dict[str, Any]:
        """
        Fetch aggregated sentiment summary from Finnhub.
        Attempts premium endpoints first, then falls back to news analysis.
        """
        if not ticker:
            raise ValueError("Ticker cannot be empty")

        symbol = ticker.upper().strip()

        try:
            # Parallel fetch news sentiment and social sentiment (Premium endpoints)
            news_task = self.downloader.get_news_sentiment(symbol)
            social_task = self.downloader.get_social_sentiment(symbol)

            news_data, social_data = await asyncio.gather(news_task, social_task)

            # If both failed (likely 403), use fallback
            if news_data is None and social_data is None:
                _logger.info("Finnhub premium sentiment endpoints unavailable for %s. Using news fallback.", symbol)
                return await self._fetch_summary_fallback(symbol, since_ts)

            mentions = 0
            sentiment_score = 0.0
            sources = {}

            # news_data is a SentimentData object if success
            if news_data:
                sources['news'] = {
                    'sentiment_score': news_data.sentiment_score,
                    'bullish_percent': news_data.bullish_score,
                    'bearish_percent': news_data.bearish_score,
                    'buzz_ratio': news_data.buzz_ratio,
                    'article_count': news_data.article_count
                }
                mentions += news_data.article_count or 0
                sentiment_score += (news_data.sentiment_score or 0.0) * 0.6

            # social_data is a SentimentData object if success
            if social_data:
                reddit = social_data.reddit_data or {}
                twitter = social_data.twitter_data or {}

                sources['social'] = {
                    'sentiment_score': social_data.sentiment_score,
                    'reddit_mentions': reddit.get('mentions', 0),
                    'twitter_mentions': twitter.get('mentions', 0)
                }

                social_mentions = reddit.get('mentions', 0) + twitter.get('mentions', 0)
                mentions += social_mentions
                sentiment_score += (social_data.sentiment_score or 0.0) * 0.4

            # Final score calculation (weighted average if both exist)
            if news_data and social_data:
                final_score = sentiment_score
            elif news_data:
                final_score = news_data.sentiment_score or 0.0
            elif social_data:
                final_score = social_data.sentiment_score or 0.0
            else:
                final_score = 0.0

            return {
                "mentions": mentions,
                "sentiment_score": float(final_score),
                "bullish": 1 if final_score > 0.1 else 0, # Rough mapping for basic metrics
                "bearish": 1 if final_score < -0.1 else 0,
                "neutral": 1 if -0.1 <= final_score <= 0.1 else 0,
                "provider": "finnhub",
                "status": "premium_api",
                "sources": sources,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            _logger.error("Failed in fetch_summary for %s: %s", symbol, e)
            self._update_health_failure(e)
            return await self._fetch_summary_fallback(symbol, since_ts)

    async def close(self) -> None:
        """Clean up resources if needed."""
        # FinnhubDataDownloader currently creates its own sessions if not provided
        # so there's not much to close here unless we pass a session in.
        pass
