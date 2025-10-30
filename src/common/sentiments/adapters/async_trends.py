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
import os
from typing import List, Dict, Optional, Any, Set
from pathlib import Path
import sys
import time
from datetime import datetime, timedelta, timezone
import json
import re
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
                 concurrency: int = 1, rate_limit_delay: float = 2.0, max_retries: int = 3,
                 proxy_list: Optional[List[str]] = None):
        super().__init__(name, concurrency, rate_limit_delay)
        self._provided_session = session is not None
        self._session = session
        self.max_retries = max_retries
        self._consecutive_failures = 0

        # Google Trends configuration
        self.trends_base = "https://trends.google.com/trends/api"
        self.proxy_list = proxy_list or []

        # Rate limiting - be very conservative with Google Trends
        self.requests_per_hour = 30
        self._request_times = []

        # Geographic regions for analysis
        self.regions = {
            'US': 'United States',
            'GB': 'United Kingdom',
            'CA': 'Canada',
            'AU': 'Australia',
            'DE': 'Germany',
            'JP': 'Japan',
            'CN': 'China',
            'IN': 'India'
        }

        # Sentiment-related search terms to correlate with ticker
        self.sentiment_terms = {
            'bullish': [['buy'], ['invest'], ['bull', 'market'], ['stock', 'price', 'target']],
            'bearish': [['sell'], ['short'], ['bear', 'market'], ['crash']],
            'neutral': [['analysis'], ['financial', 'news'], ['earnings', 'report']]
        }

        # User agents for rotation
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]

    def _get_headers(self) -> Dict[str, str]:
        """Get headers with random user agent for Google Trends requests."""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        current_time = time.time()

        # Remove old requests outside the 1-hour window
        self._request_times = [
            req_time for req_time in self._request_times
            if current_time - req_time < 3600
        ]

        return len(self._request_times) < self.requests_per_hour

    def _record_request(self) -> None:
        """Record a new API request for rate limiting."""
        self._request_times.append(time.time())

    async def _get_with_retry(self, url: str, params: Optional[dict] = None,
                             timeout: int = 30) -> Optional[dict]:
        """Make HTTP request with exponential backoff retry logic."""
        if not self._session:
            connector = None
            if self.proxy_list:
                # Use proxy if available
                proxy = random.choice(self.proxy_list)
                connector = aiohttp.TCPConnector()

            self._session = aiohttp.ClientSession(connector=connector)

        last_exception = None

        for attempt in range(self.max_retries + 1):
            # Check rate limits before making request
            if not self._check_rate_limit():
                wait_time = 3600 - (time.time() - min(self._request_times))
                _logger.warning("Google Trends rate limit reached, waiting %.1f seconds", min(wait_time, 300))
                await asyncio.sleep(min(wait_time, 300))  # Cap wait time

            async with self.semaphore:
                try:
                    start_time = time.time()
                    headers = self._get_headers()

                    # Add random delay to appear more human-like
                    await asyncio.sleep(random.uniform(1.0, 3.0))

                    proxy = random.choice(self.proxy_list) if self.proxy_list else None

                    async with self._session.get(url, params=params, headers=headers,
                                               timeout=timeout, proxy=proxy) as resp:
                        response_time_ms = (time.time() - start_time) * 1000
                        self._record_request()

                        if resp.status == 429:
                            # Rate limited
                            backoff_delay = self.rate_limit_delay * (2 ** attempt) + random.uniform(5, 15)
                            _logger.warning("Google Trends 429 rate limit (attempt %d/%d) - sleeping %.2fs",
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

                        if resp.status == 403:
                            _logger.warning("Google Trends access forbidden - may be blocked")
                            return None

                        if resp.status >= 500:
                            # Server error - retry with backoff
                            if attempt < self.max_retries:
                                backoff_delay = self.rate_limit_delay * (2 ** attempt) + random.uniform(5, 15)
                                _logger.warning("Google Trends server error %d (attempt %d/%d) - retrying in %.2fs",
                                              resp.status, attempt + 1, self.max_retries + 1, backoff_delay)
                                await asyncio.sleep(backoff_delay)
                                continue

                        if resp.status != 200:
                            _logger.warning("Google Trends returned status %d", resp.status)
                            return None

                        # Google Trends API returns JSONP, need to clean it
                        text = await resp.text()

                        # Remove JSONP callback wrapper
                        if text.startswith(')]}\''):
                            text = text[4:]

                        try:
                            data = json.loads(text)
                        except json.JSONDecodeError:
                            _logger.warning("Failed to parse Google Trends response")
                            return None

                        # Success - reset failure counter and update health
                        self._consecutive_failures = 0
                        self._update_health_success(response_time_ms)

                        return data

                except asyncio.TimeoutError as e:
                    last_exception = e
                    if attempt < self.max_retries:
                        backoff_delay = self.rate_limit_delay * (2 ** attempt) + random.uniform(5, 15)
                        _logger.warning("Google Trends timeout (attempt %d/%d) - retrying in %.2fs",
                                      attempt + 1, self.max_retries + 1, backoff_delay)
                        await asyncio.sleep(backoff_delay)
                        continue

                except (aiohttp.ClientError, aiohttp.ClientResponseError) as e:
                    last_exception = e
                    if attempt < self.max_retries and not isinstance(e, aiohttp.ClientResponseError) or (
                        isinstance(e, aiohttp.ClientResponseError) and e.status >= 500
                    ):
                        backoff_delay = self.rate_limit_delay * (2 ** attempt) + random.uniform(5, 15)
                        _logger.warning("Google Trends client error (attempt %d/%d) - retrying in %.2fs: %s",
                                      attempt + 1, self.max_retries + 1, backoff_delay, e)
                        await asyncio.sleep(backoff_delay)
                        continue
                    else:
                        # Client error that shouldn't be retried
                        break

                except Exception as e:
                    last_exception = e
                    _logger.debug("Google Trends unexpected error (attempt %d/%d): %s",
                                attempt + 1, self.max_retries + 1, e)
                    if attempt < self.max_retries:
                        backoff_delay = self.rate_limit_delay * (2 ** attempt) + random.uniform(5, 15)
                        await asyncio.sleep(backoff_delay)
                        continue
                    break

        # All retries failed
        self._consecutive_failures += 1
        if last_exception:
            self._update_health_failure(last_exception)
            _logger.error("Google Trends request failed after %d attempts: %s %s",
                         self.max_retries + 1, url, last_exception)

        return None

    def _build_trends_query(self, ticker: str, timeframe: str = 'today 7-d') -> str:
        """Build Google Trends query parameters."""
        # Create comparison query with ticker and related terms
        terms = [
            ticker.upper(),
            f"{ticker.upper()} stock",
            f"{ticker.upper()} price"
        ]

        # Encode terms for URL
        encoded_terms = [quote_plus(term) for term in terms]
        query = ','.join(encoded_terms)

        return query

    async def _fetch_interest_over_time(self, ticker: str, timeframe: str = 'today 7-d',
                                      region: str = 'US') -> Optional[Dict[str, Any]]:
        """Fetch interest over time data from Google Trends."""
        try:
            query = self._build_trends_query(ticker, timeframe)

            # Build request URL (this is a simplified approach)
            url = f"{self.trends_base}/widgetdata/multiline"
            params = {
                'hl': 'en-US',
                'tz': '-480',
                'req': json.dumps({
                    'comparisonItem': [
                        {
                            'keyword': ticker.upper(),
                            'geo': region,
                            'time': timeframe
                        }
                    ],
                    'category': 0,
                    'property': ''
                }),
                'token': '12345'  # Placeholder token
            }

            data = await self._get_with_retry(url, params=params)
            return data

        except Exception as e:
            _logger.warning("Failed to fetch Google Trends interest over time for %s: %s", ticker, e)
            return None

    async def _fetch_related_queries(self, ticker: str, region: str = 'US') -> Optional[Dict[str, Any]]:
        """Fetch related queries from Google Trends."""
        try:
            # Build request URL for related queries
            url = f"{self.trends_base}/widgetdata/relatedsearches"
            params = {
                'hl': 'en-US',
                'tz': '-480',
                'req': json.dumps({
                    'restriction': {
                        'keyword': ticker.upper(),
                        'geo': region,
                        'time': 'today 7-d'
                    },
                    'keywordType': 'QUERY',
                    'metric': ['TOP', 'RISING'],
                    'trendinessSettings': {
                        'compareTime': 'today 7-d'
                    }
                }),
                'token': '12345'  # Placeholder token
            }

            data = await self._get_with_retry(url, params=params)
            return data

        except Exception as e:
            _logger.warning("Failed to fetch Google Trends related queries for %s: %s", ticker, e)
            return None

    def _analyze_search_sentiment(self, queries: List[str]) -> Dict[str, int]:
        """Analyze sentiment of search queries."""
        sentiment_counts = {'bullish': 0, 'bearish': 0, 'neutral': 0}

        for query in queries:
            query_lower = query.lower()

            # Check for bullish indicators
            bullish_found = any(
                all(term in query_lower for term in term_group)
                for term_group in self.sentiment_terms['bullish']
            )

            # Check for bearish indicators
            bearish_found = any(
                all(term in query_lower for term in term_group)
                for term_group in self.sentiment_terms['bearish']
            )

            if bullish_found and not bearish_found:
                sentiment_counts['bullish'] += 1
            elif bearish_found and not bullish_found:
                sentiment_counts['bearish'] += 1
            else:
                sentiment_counts['neutral'] += 1

        return sentiment_counts

    async def fetch_messages(self, ticker: str, since_ts: Optional[int] = None, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Fetch Google Trends data points as individual 'messages'.

        Args:
            ticker: Stock ticker symbol
            since_ts: Unix timestamp to fetch data since
            limit: Maximum number of data points to fetch

        Returns:
            List of normalized trend data dictionaries
        """
        if not ticker or not ticker.strip():
            raise ValueError("Ticker cannot be empty")

        symbol = ticker.upper().strip()
        trend_data: List[Dict[str, Any]] = []

        try:
            # Determine timeframe based on since_ts
            if since_ts:
                days_ago = (time.time() - since_ts) / 86400
                if days_ago <= 1:
                    timeframe = 'now 1-d'
                elif days_ago <= 7:
                    timeframe = 'today 7-d'
                elif days_ago <= 30:
                    timeframe = 'today 1-m'
                else:
                    timeframe = 'today 3-m'
            else:
                timeframe = 'today 7-d'

            # Fetch interest over time data for multiple regions
            tasks = []
            for region_code in list(self.regions.keys())[:3]:  # Limit to 3 regions to avoid rate limits
                tasks.append(self._fetch_interest_over_time(symbol, timeframe, region_code))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(results):
                if isinstance(result, dict) and result:
                    region_code = list(self.regions.keys())[i]
                    region_name = self.regions[region_code]

                    # Extract timeline data (this is simplified - actual Google Trends API structure may vary)
                    timeline = result.get('default', {}).get('timelineData', [])

                    for point in timeline[:limit]:
                        try:
                            trend_point = {
                                'id': f"trends_{region_code}_{point.get('time', '')}",
                                'body': f"Search interest for {symbol} in {region_name}",
                                'created_at': datetime.fromtimestamp(point.get('time', 0)).isoformat(),
                                'user': {
                                    'username': f'trends_{region_code}',
                                    'id': region_code,
                                    'followers': 0,
                                    'region': region_name
                                },
                                'likes': 0,
                                'replies': 0,
                                'retweets': 0,
                                'search_volume': point.get('value', [0])[0] if point.get('value') else 0,
                                'region': region_code,
                                'timeframe': timeframe,
                                'provider': 'trends'
                            }
                            trend_data.append(trend_point)

                        except Exception as e:
                            _logger.debug("Error processing trends data point: %s", e)
                            continue

            # Sort by timestamp
            trend_data.sort(key=lambda x: x.get('created_at', ''), reverse=True)

            _logger.debug("Fetched %d Google Trends data points for ticker %s", len(trend_data), symbol)
            return trend_data[:limit]

        except Exception as e:
            _logger.error("Failed to fetch Google Trends data for ticker %s: %s", symbol, e)
            self._update_health_failure(e)
            raise

    async def fetch_summary(self, ticker: str, since_ts: Optional[int] = None) -> Dict[str, Any]:
        """
        Fetch aggregated sentiment summary based on Google Trends data.

        Args:
            ticker: Stock ticker symbol
            since_ts: Unix timestamp to fetch data since

        Returns:
            Dictionary containing trend-based sentiment metrics
        """
        if not ticker or not ticker.strip():
            raise ValueError("Ticker cannot be empty")

        try:
            # Fetch trends data and related queries
            trend_data = await self.fetch_messages(ticker, since_ts=since_ts, limit=100)

            # Fetch related queries for sentiment analysis
            related_queries_data = await self._fetch_related_queries(ticker)

            # Calculate basic metrics
            total_data_points = len(trend_data)
            total_search_volume = sum(point.get('search_volume', 0) for point in trend_data)
            avg_search_volume = total_search_volume / total_data_points if total_data_points > 0 else 0

            # Calculate trend direction (rising/falling interest)
            if len(trend_data) >= 2:
                recent_volume = sum(point.get('search_volume', 0) for point in trend_data[:len(trend_data)//2])
                older_volume = sum(point.get('search_volume', 0) for point in trend_data[len(trend_data)//2:])

                recent_avg = recent_volume / (len(trend_data)//2) if len(trend_data)//2 > 0 else 0
                older_avg = older_volume / (len(trend_data) - len(trend_data)//2) if (len(trend_data) - len(trend_data)//2) > 0 else 0

                trend_direction = 1 if recent_avg > older_avg else -1 if recent_avg < older_avg else 0
            else:
                trend_direction = 0

            # Analyze related queries for sentiment
            related_queries = []
            sentiment_analysis = {'bullish': 0, 'bearish': 0, 'neutral': 0}

            if related_queries_data:
                # Extract queries (structure may vary)
                top_queries = related_queries_data.get('default', {}).get('rankedList', [])
                for query_list in top_queries:
                    for query_item in query_list.get('rankedKeyword', []):
                        query_text = query_item.get('query', '')
                        if query_text:
                            related_queries.append(query_text)

                # Analyze sentiment of related queries
                sentiment_analysis = self._analyze_search_sentiment(related_queries)

            # Calculate sentiment score based on trend direction and query sentiment
            total_sentiment_queries = sum(sentiment_analysis.values())
            if total_sentiment_queries > 0:
                query_sentiment_score = (sentiment_analysis['bullish'] - sentiment_analysis['bearish']) / total_sentiment_queries
            else:
                query_sentiment_score = 0

            # Combine trend direction with query sentiment
            # Trend direction weight: 0.3, Query sentiment weight: 0.7
            combined_sentiment_score = (trend_direction * 0.3) + (query_sentiment_score * 0.7)

            # Normalize to -1 to 1 range
            combined_sentiment_score = max(-1.0, min(1.0, combined_sentiment_score))

            # Regional distribution
            regional_data = {}
            for point in trend_data:
                region = point.get('region', 'unknown')
                if region not in regional_data:
                    regional_data[region] = {'count': 0, 'total_volume': 0}
                regional_data[region]['count'] += 1
                regional_data[region]['total_volume'] += point.get('search_volume', 0)

            summary = {
                "mentions": total_data_points,
                "bullish": sentiment_analysis['bullish'],
                "bearish": sentiment_analysis['bearish'],
                "neutral": sentiment_analysis['neutral'],
                "sentiment_score": float(combined_sentiment_score),
                "total_search_volume": int(total_search_volume),
                "avg_search_volume": float(avg_search_volume),
                "trend_direction": trend_direction,
                "related_queries": related_queries[:10],  # Top 10 related queries
                "regional_distribution": regional_data,
                "provider": "trends",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            _logger.debug("Generated Google Trends summary for ticker %s: %d data points, score %.3f, volume %.1f",
                         ticker, total_data_points, combined_sentiment_score, avg_search_volume)
            return summary

        except Exception as e:
            _logger.error("Failed to fetch Google Trends summary for ticker %s: %s", ticker, e)
            self._update_health_failure(e)
            raise

    async def close(self) -> None:
        """Clean up adapter resources."""
        try:
            if self._session and not self._provided_session:
                await self._session.close()
                self._session = None
            _logger.debug("Google Trends adapter closed successfully")
        except Exception as e:
            _logger.warning("Error closing Google Trends adapter: %s", e)