from typing import List, Optional, Dict, Any
from datetime import datetime
import aiohttp
import asyncio
import os
from src.notification.logger import setup_logger
from src.data.downloader.base_data_downloader import BaseDataDownloader

_logger = setup_logger(__name__)

"""
Data downloader implementation for NewsAPI, focusing on financial news data collection.

Main Features:
- Download news articles for any ticker or search query
- Handles NewsAPI-specific authentication and endpoints
- Built-in rate limit handling for NewsAPI (1000 requests/day for free tier)
"""

class NewsAPIDataDownloader(BaseDataDownloader):
    """
    A class to fetch news articles from NewsAPI.

    Rate Limits: 1,000 requests per day (free tier)
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        # Get API key from parameter or config
        self.api_key = api_key or self._get_config_value('NEWSAPI_API_KEY', 'NEWSAPI_API_KEY')
        self.base_url = "https://newsapi.org/v2"

        if not self.api_key:
            _logger.warning("NewsAPI API key is missing. Some news features may be disabled.")

    def get_supported_intervals(self) -> List[str]:
        """NewsAPI doesn't support OHLCV intervals."""
        return []

    def get_ohlcv(self, symbol: str, interval: str, start_date: datetime, end_date: datetime, **kwargs):
        """OHLCV data is not supported by NewsAPI."""
        import pandas as pd
        return pd.DataFrame()

    async def get_everything(self, query: str, from_date: Optional[str] = None,
                             sort_by: str = 'publishedAt', language: str = 'en',
                             page_size: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch news articles using the /everything endpoint (async).

        Args:
            query: Search query
            from_date: Optional start date (YYYY-MM-DD or ISO format)
            sort_by: Sorting criteria ('relevancy', 'popularity', 'publishedAt')
            language: Article language ('en', 'de', etc.)
            page_size: Number of articles to return (max 100)

        Returns:
            List of news articles
        """
        if not self.api_key:
            return []

        try:
            url = f"{self.base_url}/everything"
            params = {
                'q': query,
                'language': language,
                'sortBy': sort_by,
                'pageSize': min(page_size, 100),
                'apiKey': self.api_key
            }

            if from_date:
                params['from'] = from_date

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 429:
                        _logger.warning("NewsAPI rate limit exceeded")
                        return []

                    if response.status != 200:
                        error_data = await response.json()
                        _logger.warning("NewsAPI error %s: %s", response.status, error_data.get('message', 'Unknown error'))
                        return []

                    data = await response.json()
                    return data.get('articles', []) if isinstance(data, dict) else []

        except Exception as e:
            _logger.error("Error fetching news from NewsAPI for query '%s': %s", query, e)
            return []
