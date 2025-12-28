# src/common/sentiments/adapters/async_news.py
"""
Async News sentiment adapter for financial news data collection.

Provides:
- async fetch_messages(ticker, since_ts=None, limit=200)
- async fetch_summary(ticker, since_ts=None)

Features:
- Multiple news API support (Finnhub, Alpha Vantage, NewsAPI)
- Article sentiment analysis and summarization
- Source credibility weighting and bias detection
- Rate limit handling for various news APIs
"""
import asyncio
import aiohttp
import os
from typing import List, Dict, Optional, Any
from pathlib import Path
import sys
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.common.sentiments.adapters.base_adapter import BaseSentimentAdapter
from src.common.sentiments.processing.heuristic_analyzer import HeuristicSentimentAnalyzer

_logger = setup_logger(__name__)

from src.data.downloader.finnhub_data_downloader import FinnhubDataDownloader
from src.data.downloader.alpha_vantage_data_downloader import AlphaVantageDataDownloader
from src.data.downloader.newsapi_data_downloader import NewsAPIDataDownloader


class AsyncNewsAdapter(BaseSentimentAdapter):
    """
    Async News sentiment adapter supporting multiple news APIs.

    Supports article collection from financial news sources,
    sentiment analysis, and source credibility weighting.
    """

    def __init__(self, name: str = "news", session: Optional[aiohttp.ClientSession] = None,
                 concurrency: int = 3, rate_limit_delay: float = 1.0, max_retries: int = 3,
                 finnhub_token: Optional[str] = None, alpha_vantage_token: Optional[str] = None,
                 newsapi_token: Optional[str] = None):
        super().__init__(name, concurrency, rate_limit_delay)
        self._provided_session = session is not None
        self._session = session
        self.max_retries = max_retries
        self._consecutive_failures = 0
        self._analyzer = HeuristicSentimentAnalyzer()

        # API tokens
        self.finnhub_token = finnhub_token or os.getenv('FINNHUB_API_KEY')
        self.alpha_vantage_token = alpha_vantage_token or os.getenv('ALPHA_VANTAGE_API_KEY')
        self.newsapi_token = newsapi_token or os.getenv('NEWSAPI_API_KEY')

        # Rate limiting per API
        self.finnhub_rate_limit = 60  # requests per minute
        self.alpha_vantage_rate_limit = 5  # requests per minute

        self._finnhub_requests = []
        self._alpha_vantage_requests = []

        # Initialize downloaders
        self.finnhub_downloader = FinnhubDataDownloader(api_key=self.finnhub_token)
        self.av_downloader = AlphaVantageDataDownloader(api_key=self.alpha_vantage_token)
        self.newsapi_downloader = NewsAPIDataDownloader(api_key=self.newsapi_token)


        # Bias detection keywords
        self.bias_indicators = self._analyzer.config.get("bias_indicators", {})

    def _get_source_credibility(self, url: str) -> float:
        """Get credibility score for a news source."""
        return self._analyzer.get_credibility(url)

    def _detect_bias(self, title: str, content: str) -> Dict[str, bool]:
        """Detect potential bias indicators in article content."""
        return self._analyzer.analyze_bias(f"{title} {content}")

    def _check_rate_limit(self, api: str) -> bool:
        """Check if we're within rate limits for specific API."""
        current_time = time.time()

        if api == 'finnhub':
            self._finnhub_requests = [
                req_time for req_time in self._finnhub_requests
                if current_time - req_time < 60
            ]
            return len(self._finnhub_requests) < self.finnhub_rate_limit

        elif api == 'alpha_vantage':
            self._alpha_vantage_requests = [
                req_time for req_time in self._alpha_vantage_requests
                if current_time - req_time < 60
            ]
            return len(self._alpha_vantage_requests) < self.alpha_vantage_rate_limit

        return True

    def _record_request(self, api: str) -> None:
        """Record a new API request for rate limiting."""
        current_time = time.time()

        if api == 'finnhub':
            self._finnhub_requests.append(current_time)
        elif api == 'alpha_vantage':
            self._alpha_vantage_requests.append(current_time)

    async def _get_with_retry(self, url: str, params: Optional[dict] = None,
                             api: str = 'generic', timeout: int = 30) -> Optional[dict]:
        """Make HTTP request with exponential backoff retry logic."""
        if not self._session:
            self._session = aiohttp.ClientSession()

        last_exception = None

        for attempt in range(self.max_retries + 1):
            # Check rate limits before making request
            if not self._check_rate_limit(api):
                wait_time = 60 if api != 'newsapi' else 3600  # Wait 1 hour for NewsAPI
                _logger.warning("%s rate limit reached, waiting %d seconds", api, wait_time)
                await asyncio.sleep(min(wait_time, 60))  # Cap wait time for this attempt

            async with self.semaphore:
                try:
                    start_time = time.time()

                    async with self._session.get(url, params=params, timeout=timeout) as resp:
                        response_time_ms = (time.time() - start_time) * 1000
                        self._record_request(api)

                        if resp.status == 429:
                            # Rate limited
                            backoff_delay = self.rate_limit_delay * (2 ** attempt)
                            _logger.warning("%s 429 rate limit (attempt %d/%d) - sleeping %.2fs",
                                          api, attempt + 1, self.max_retries + 1, backoff_delay)
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

                        if resp.status == 401:
                            _logger.error("%s authentication failed - check API key", api)
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
                                _logger.warning("%s server error %d (attempt %d/%d) - retrying in %.2fs",
                                              api, resp.status, attempt + 1, self.max_retries + 1, backoff_delay)
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
                        _logger.warning("%s timeout (attempt %d/%d) - retrying in %.2fs",
                                      api, attempt + 1, self.max_retries + 1, backoff_delay)
                        await asyncio.sleep(backoff_delay)
                        continue

                except (aiohttp.ClientError, aiohttp.ClientResponseError) as e:
                    last_exception = e
                    if attempt < self.max_retries and not isinstance(e, aiohttp.ClientResponseError) or (
                        isinstance(e, aiohttp.ClientResponseError) and e.status >= 500
                    ):
                        backoff_delay = self.rate_limit_delay * (2 ** attempt)
                        _logger.warning("%s client error (attempt %d/%d) - retrying in %.2fs: %s",
                                      api, attempt + 1, self.max_retries + 1, backoff_delay, e)
                        await asyncio.sleep(backoff_delay)
                        continue
                    else:
                        # Client error that shouldn't be retried (4xx except 429)
                        break

                except Exception as e:
                    last_exception = e
                    _logger.debug("%s unexpected error (attempt %d/%d): %s",
                                api, attempt + 1, self.max_retries + 1, e)
                    if attempt < self.max_retries:
                        backoff_delay = self.rate_limit_delay * (2 ** attempt)
                        await asyncio.sleep(backoff_delay)
                        continue
                    break

        # All retries failed
        self._consecutive_failures += 1
        if last_exception:
            self._update_health_failure(last_exception)
            _logger.error("%s request failed after %d attempts: %s %s",
                         api, self.max_retries + 1, url, last_exception)

        return None

    async def _fetch_finnhub_news(self, ticker: str, since_ts: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch news from Finnhub API via downloader."""
        if not self.finnhub_token:
            return []

        try:
            # Calculate date range
            to_date = datetime.now().strftime('%Y-%m-%d')
            if since_ts:
                from_date = datetime.fromtimestamp(since_ts).strftime('%Y-%m-%d')
            else:
                from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

            data = await self.finnhub_downloader.get_company_news(ticker, from_date, to_date)
            if not data:
                return []

            articles = []
            for article in data[:limit]:
                try:
                    articles.append({
                        'id': f"finnhub_{article.get('id', '')}",
                        'title': article.get('headline', ''),
                        'content': article.get('summary', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', 'finnhub'),
                        'published_at': datetime.fromtimestamp(article.get('datetime', 0)).isoformat(),
                        'credibility': self._get_source_credibility(article.get('url', '')),
                        'provider': 'finnhub'
                    })
                except Exception as e:
                    _logger.debug("Error processing Finnhub article: %s", e)
                    continue

            return articles

        except Exception as e:
            _logger.warning("Failed to fetch Finnhub news for %s: %s", ticker, e)
            return []

    async def _fetch_alpha_vantage_news(self, ticker: str, since_ts: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch news from Alpha Vantage API via downloader."""
        if not self.alpha_vantage_token:
            return []

        try:
            time_from = None
            if since_ts:
                time_from = datetime.fromtimestamp(since_ts).strftime('%Y%m%dT%H%M')

            data = await self.av_downloader.get_news_articles(ticker, time_from=time_from, limit=limit)
            if not data:
                return []

            articles = []
            for article in data:
                try:
                    # Extract ticker-specific sentiment if available
                    ticker_sentiment = None
                    if 'ticker_sentiment' in article:
                        for ts in article['ticker_sentiment']:
                            if ts.get('ticker', '').upper() == ticker.upper():
                                ticker_sentiment = ts
                                break

                    articles.append({
                        'id': f"av_{hash(article.get('url', ''))}",
                        'title': article.get('title', ''),
                        'content': article.get('summary', ''),
                        'url': article.get('url', ''),
                        'source': ', '.join(article.get('authors', [])) or 'alpha_vantage',
                        'published_at': article.get('time_published', ''),
                        'credibility': self._get_source_credibility(article.get('url', '')),
                        'overall_sentiment': article.get('overall_sentiment_label', ''),
                        'overall_sentiment_score': float(article.get('overall_sentiment_score', 0)),
                        'ticker_sentiment': ticker_sentiment,
                        'provider': 'alpha_vantage'
                    })
                except Exception as e:
                    _logger.debug("Error processing Alpha Vantage article: %s", e)
                    continue

            return articles

        except Exception as e:
            _logger.warning("Failed to fetch Alpha Vantage news for %s: %s", ticker, e)
            return []

    async def _fetch_newsapi_news(self, ticker: str, since_ts: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch news from NewsAPI."""
        if not self.newsapi_token:
            return []

        try:
            # Build search query
            query = f'"{ticker}" OR "${ticker.upper()}" OR "#{ticker.upper()}"'

            # Add time filter if provided
            from_date = None
            if since_ts:
                from_date = datetime.fromtimestamp(since_ts).strftime('%Y-%m-%d')

            data = await self.newsapi_downloader.get_everything(
                query=query,
                from_date=from_date,
                page_size=limit
            )
            if not data:
                return []

            articles = []
            for article in data[:limit]:
                try:
                    articles.append({
                        'id': f"newsapi_{hash(article.get('url', ''))}",
                        'title': article.get('title', ''),
                        'content': article.get('description', '') or article.get('content', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', {}).get('name', 'newsapi'),
                        'published_at': article.get('publishedAt', ''),
                        'credibility': self._get_source_credibility(article.get('url', '')),
                        'author': article.get('author', ''),
                        'provider': 'newsapi'
                    })
                except Exception as e:
                    _logger.debug("Error processing NewsAPI article: %s", e)
                    continue

            return articles

        except Exception as e:
            _logger.warning("Failed to fetch NewsAPI news for %s: %s", ticker, e)
            return []

    async def fetch_messages(self, ticker: str, since_ts: Optional[int] = None, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Fetch individual news articles for a ticker.

        Args:
            ticker: Stock ticker symbol
            since_ts: Unix timestamp to fetch articles since
            limit: Maximum number of articles to fetch

        Returns:
            List of normalized article dictionaries
        """
        if not ticker or not ticker.strip():
            raise ValueError("Ticker cannot be empty")

        symbol = ticker.upper().strip()
        all_articles: List[Dict[str, Any]] = []

        try:
            # Fetch from all available APIs concurrently
            tasks = []
            articles_per_source = max(1, limit // 3)  # Distribute across 3 sources

            if self.finnhub_token:
                tasks.append(self._fetch_finnhub_news(symbol, since_ts, articles_per_source))

            if self.alpha_vantage_token:
                tasks.append(self._fetch_alpha_vantage_news(symbol, since_ts, articles_per_source))

            if self.newsapi_token:
                tasks.append(self._fetch_newsapi_news(symbol, since_ts, articles_per_source))

            if not tasks:
                _logger.warning("No news API tokens configured")
                return []

            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Combine results
            for result in results:
                if isinstance(result, list):
                    all_articles.extend(result)
                elif isinstance(result, Exception):
                    _logger.warning("News API task failed: %s", result)

            # Remove duplicates based on URL
            seen_urls = set()
            unique_articles = []

            for article in all_articles:
                url = article.get('url', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_articles.append(article)

            # Sort by published date (newest first) and limit
            unique_articles.sort(key=lambda x: x.get('published_at', ''), reverse=True)
            final_articles = unique_articles[:limit]

            # Add bias detection
            for article in final_articles:
                title = article.get('title', '')
                content = article.get('content', '')
                article['bias_indicators'] = self._detect_bias(title, content)

            _logger.debug("Fetched %d news articles for ticker %s from %d sources",
                         len(final_articles), symbol, len(tasks))
            return final_articles

        except Exception as e:
            _logger.error("Failed to fetch news articles for ticker %s: %s", symbol, e)
            self._update_health_failure(e)
            raise

    async def fetch_summary(self, ticker: str, since_ts: Optional[int] = None) -> Dict[str, Any]:
        """
        Fetch aggregated sentiment summary for a ticker from news sources.

        Args:
            ticker: Stock ticker symbol
            since_ts: Unix timestamp to fetch data since

        Returns:
            Dictionary containing sentiment metrics and counts
        """
        if not ticker or not ticker.strip():
            raise ValueError("Ticker cannot be empty")

        try:
            articles = await self.fetch_messages(ticker, since_ts=since_ts, limit=100)
            total_articles = len(articles)
            bullish = 0
            bearish = 0
            neutral = 0

            total_credibility = 0.0
            source_distribution: Dict[str, int] = {}
            bias_counts: Dict[str, int] = {}

            # Define sentiment keywords for financial news
            bullish_keywords = (
                "surge", "rally", "gain", "rise", "up", "positive", "growth", "profit",
                "beat", "exceed", "outperform", "strong", "robust", "bullish", "upgrade",
                "buy", "target", "optimistic", "confident", "breakthrough", "success"
            )
            bearish_keywords = (
                "fall", "drop", "decline", "down", "negative", "loss", "miss", "weak",
                "underperform", "bearish", "downgrade", "sell", "concern", "worry",
                "risk", "challenge", "struggle", "disappointing", "cut", "reduce"
            )

            for article in articles:
                try:
                    title = (article.get('title', '') or '').lower()
                    content = (article.get('content', '') or '').lower()
                    text = f"{title} {content}"

                    if not text.strip():
                        neutral += 1
                        continue

                    # Track credibility
                    credibility = article.get('credibility', 0.5)
                    total_credibility += credibility

                    # Track source distribution
                    source = article.get('source', 'unknown')
                    source_distribution[source] = source_distribution.get(source, 0) + 1

                    # Track bias indicators
                    bias_indicators = article.get('bias_indicators', {})
                    for bias_type, detected in bias_indicators.items():
                        if detected:
                            bias_counts[bias_type] = bias_counts.get(bias_type, 0) + 1

                    # Use Alpha Vantage sentiment if available
                    if article.get('provider') == 'alpha_vantage':
                        av_sentiment = article.get('overall_sentiment', '').lower()
                        if av_sentiment == 'bullish':
                            bullish += 1
                        elif av_sentiment == 'bearish':
                            bearish += 1
                        else:
                            neutral += 1
                        continue

                    # Keyword-based sentiment analysis
                    has_bullish = any(keyword in text for keyword in bullish_keywords)
                    has_bearish = any(keyword in text for keyword in bearish_keywords)

                    if has_bullish and not has_bearish:
                        bullish += 1
                    elif has_bearish and not has_bullish:
                        bearish += 1
                    else:
                        neutral += 1

                except Exception as e:
                    _logger.debug("Error processing article sentiment for ticker %s: %s", ticker, e)
                    neutral += 1
                    continue

            # Calculate sentiment score (-1 to 1)
            if total_articles > 0:
                score = (bullish - bearish) / total_articles
            else:
                score = 0.0

            # Ensure score is within bounds
            score = max(-1.0, min(1.0, score))

            # Calculate additional metrics
            avg_credibility = total_credibility / total_articles if total_articles > 0 else 0.0

            # Get top sources
            top_sources = sorted(source_distribution.items(), key=lambda x: x[1], reverse=True)[:5]

            summary = {
                "mentions": total_articles,
                "bullish": bullish,
                "bearish": bearish,
                "neutral": neutral,
                "sentiment_score": float(score),
                "avg_credibility": float(avg_credibility),
                "source_distribution": dict(top_sources),
                "bias_indicators": bias_counts,
                "provider": "news",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            _logger.debug("Generated news summary for ticker %s: %d articles, score %.3f, credibility %.2f",
                         ticker, total_articles, score, avg_credibility)
            return summary

        except Exception as e:
            _logger.error("Failed to fetch news summary for ticker %s: %s", ticker, e)
            self._update_health_failure(e)
            raise

    async def close(self) -> None:
        """Clean up adapter resources."""
        try:
            if self._session and not self._provided_session:
                await self._session.close()
                self._session = None
            _logger.debug("News adapter closed successfully")
        except Exception as e:
            _logger.warning("Error closing News adapter: %s", e)