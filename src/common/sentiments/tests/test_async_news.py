"""
Unit tests for AsyncNewsAdapter.

Tests cover:
- Multiple news API integration (Finnhub, Alpha Vantage, NewsAPI)
- Article fetching and deduplication
- Source credibility weighting and bias detection
- Rate limiting across different APIs
- Sentiment analysis and aggregation
"""
import pytest
import pytest_asyncio
import asyncio
import aiohttp
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from pathlib import Path
import sys
import time

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.common.sentiments.adapters.async_news import AsyncNewsAdapter
from src.common.sentiments.adapters.base_adapter import AdapterStatus


class TestAsyncNewsAdapter:
    """Test suite for AsyncNewsAdapter."""

    @pytest_asyncio.fixture
    async def adapter(self):
        """Create adapter instance for testing."""
        # Create a mock session to avoid real HTTP requests
        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_session.close = AsyncMock()

        adapter = AsyncNewsAdapter(
            finnhub_token="test_finnhub",
            alpha_vantage_token="test_av",
            newsapi_token="test_newsapi",
            concurrency=2,
            rate_limit_delay=0.01,
            max_retries=1,
            session=mock_session
        )
        yield adapter
        await adapter.close()

    @pytest.fixture
    def sample_finnhub_response(self):
        """Sample Finnhub news API response."""
        return [
            {
                "id": 123456,
                "headline": "AAPL Reports Strong Q4 Earnings",
                "summary": "Apple Inc reported better than expected earnings with strong iPhone sales driving growth.",
                "url": "https://reuters.com/article/aapl-earnings",
                "source": "Reuters",
                "datetime": 1672531200  # 2023-01-01 00:00:00
            },
            {
                "id": 123457,
                "headline": "AAPL Stock Faces Headwinds",
                "summary": "Apple stock may face challenges due to supply chain concerns and market volatility.",
                "url": "https://bloomberg.com/article/aapl-challenges",
                "source": "Bloomberg",
                "datetime": 1672527600  # 2022-12-31 23:00:00
            }
        ]

    @pytest.fixture
    def sample_alpha_vantage_response(self):
        """Sample Alpha Vantage news API response."""
        return {
            "feed": [
                {
                    "title": "Apple Stock Surges on Innovation News",
                    "summary": "Apple's latest product announcements drive investor optimism and stock price gains.",
                    "url": "https://cnbc.com/article/apple-innovation",
                    "authors": ["John Smith", "Jane Doe"],
                    "time_published": "20230101T120000",
                    "overall_sentiment_label": "Bullish",
                    "overall_sentiment_score": 0.75,
                    "ticker_sentiment": [
                        {
                            "ticker": "AAPL",
                            "relevance_score": 0.9,
                            "ticker_sentiment_score": 0.8,
                            "ticker_sentiment_label": "Bullish"
                        }
                    ]
                }
            ]
        }

    @pytest.fixture
    def sample_newsapi_response(self):
        """Sample NewsAPI response."""
        return {
            "articles": [
                {
                    "title": "Apple Announces New Product Line",
                    "description": "Apple unveils innovative products that could boost revenue significantly.",
                    "url": "https://techcrunch.com/article/apple-products",
                    "source": {"name": "TechCrunch"},
                    "author": "Tech Reporter",
                    "publishedAt": "2023-01-01T12:00:00Z"
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_initialization_with_tokens(self):
        """Test adapter initialization with API tokens."""
        adapter = AsyncNewsAdapter(
            finnhub_token="fh_token",
            alpha_vantage_token="av_token",
            newsapi_token="na_token"
        )

        try:
            assert adapter.finnhub_token == "fh_token"
            assert adapter.alpha_vantage_token == "av_token"
            assert adapter.newsapi_token == "na_token"
            assert adapter.finnhub_base == "https://finnhub.io/api/v1"
            assert adapter.alpha_vantage_base == "https://www.alphavantage.co/query"
            assert adapter.newsapi_base == "https://newsapi.org/v2"
        finally:
            await adapter.close()

    def test_get_source_credibility(self, adapter):
        """Test source credibility scoring."""
        test_cases = [
            ("https://reuters.com/article", 0.95),
            ("https://www.bloomberg.com/news", 0.95),
            ("https://wsj.com/articles", 0.95),
            ("https://cnbc.com/news", 0.90),
            ("https://yahoo.com/finance", 0.80),
            ("https://unknown-source.com/news", 0.50),  # Default
            ("invalid-url", 0.50),  # Invalid URL
        ]

        for url, expected_score in test_cases:
            score = adapter._get_source_credibility(url)
            assert score == expected_score

    def test_detect_bias(self, adapter):
        """Test bias detection in article content."""
        test_cases = [
            # (title, content, expected_bias)
            ("Sponsored: Amazing Stock Pick", "This is a paid promotion", {"promotional": True, "emotional": True}),
            ("Stock Could Surge", "This might be speculation", {"speculative": True}),
            ("Shocking Market News", "Incredible gains today", {"emotional": True}),
            ("Regular Market Update", "Standard earnings report", {}),  # No bias
            ("Mixed Content", "This could be amazing sponsored content",
             {"promotional": True, "speculative": True, "emotional": True}),
        ]

        for title, content, expected_bias in test_cases:
            bias = adapter._detect_bias(title, content)

            for bias_type in ["promotional", "speculative", "emotional"]:
                expected_value = expected_bias.get(bias_type, False)
                assert bias[bias_type] == expected_value, f"Failed for {bias_type} in '{title}': expected {expected_value}, got {bias[bias_type]}"

    def test_rate_limit_checking(self, adapter):
        """Test rate limit checking for different APIs."""
        import time
        current_time = time.time()

        # Test Finnhub rate limit
        assert adapter._check_rate_limit('finnhub') is True

        # Simulate many requests
        adapter._finnhub_requests = [current_time - i for i in range(60)]
        assert adapter._check_rate_limit('finnhub') is False

        # Test Alpha Vantage rate limit
        assert adapter._check_rate_limit('alpha_vantage') is True
        adapter._alpha_vantage_requests = [current_time - i for i in range(5)]
        assert adapter._check_rate_limit('alpha_vantage') is False

        # Test NewsAPI rate limit (daily)
        assert adapter._check_rate_limit('newsapi') is True
        adapter._newsapi_requests = [current_time - i for i in range(1000)]
        assert adapter._check_rate_limit('newsapi') is False

    def test_record_request(self, adapter):
        """Test request recording for different APIs."""
        initial_fh = len(adapter._finnhub_requests)
        initial_av = len(adapter._alpha_vantage_requests)
        initial_na = len(adapter._newsapi_requests)

        adapter._record_request('finnhub')
        adapter._record_request('alpha_vantage')
        adapter._record_request('newsapi')

        assert len(adapter._finnhub_requests) == initial_fh + 1
        assert len(adapter._alpha_vantage_requests) == initial_av + 1
        assert len(adapter._newsapi_requests) == initial_na + 1

    @pytest.mark.asyncio
    async def test_fetch_finnhub_news_success(self, adapter, sample_finnhub_response):
        """Test successful Finnhub news fetching."""
        with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = sample_finnhub_response

            articles = await adapter._fetch_finnhub_news("AAPL", limit=10)

            assert len(articles) == 2

            # Check first article
            article1 = articles[0]
            assert article1["id"] == "finnhub_123456"
            assert article1["title"] == "AAPL Reports Strong Q4 Earnings"
            assert "better than expected earnings" in article1["content"]
            assert article1["provider"] == "finnhub"
            assert article1["credibility"] == 0.95  # Reuters credibility

    @pytest.mark.asyncio
    async def test_fetch_finnhub_news_no_token(self):
        """Test Finnhub fetching without token."""
        adapter = AsyncNewsAdapter(finnhub_token=None)

        try:
            articles = await adapter._fetch_finnhub_news("AAPL")
            assert articles == []
        finally:
            await adapter.close()

    @pytest.mark.asyncio
    async def test_fetch_alpha_vantage_news_success(self, adapter, sample_alpha_vantage_response):
        """Test successful Alpha Vantage news fetching."""
        with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = sample_alpha_vantage_response

            articles = await adapter._fetch_alpha_vantage_news("AAPL", limit=10)

            assert len(articles) == 1

            article = articles[0]
            assert "Apple Stock Surges" in article["title"]
            assert article["provider"] == "alpha_vantage"
            assert article["overall_sentiment"] == "Bullish"
            assert article["overall_sentiment_score"] == 0.75
            assert article["ticker_sentiment"] is not None

    @pytest.mark.asyncio
    async def test_fetch_newsapi_news_success(self, adapter, sample_newsapi_response):
        """Test successful NewsAPI fetching."""
        with patch.object(adapter, '_get_with_retry', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = sample_newsapi_response

            articles = await adapter._fetch_newsapi_news("AAPL", limit=10)

            assert len(articles) == 1

            article = articles[0]
            assert "Apple Announces New Product Line" in article["title"]
            assert article["provider"] == "newsapi"
            assert article["source"] == "TechCrunch"
            assert article["author"] == "Tech Reporter"

    @pytest.mark.asyncio
    async def test_fetch_messages_success(self, adapter, sample_finnhub_response,
                                        sample_alpha_vantage_response, sample_newsapi_response):
        """Test successful message fetching from all sources."""
        with patch.object(adapter, '_fetch_finnhub_news', new_callable=AsyncMock) as mock_fh:
            with patch.object(adapter, '_fetch_alpha_vantage_news', new_callable=AsyncMock) as mock_av:
                with patch.object(adapter, '_fetch_newsapi_news', new_callable=AsyncMock) as mock_na:
                    # Mock responses from each API
                    mock_fh.return_value = [{"id": "fh1", "url": "https://reuters.com/1", "title": "FH Article"}]
                    mock_av.return_value = [{"id": "av1", "url": "https://cnbc.com/1", "title": "AV Article"}]
                    mock_na.return_value = [{"id": "na1", "url": "https://techcrunch.com/1", "title": "NA Article"}]

                    articles = await adapter.fetch_messages("AAPL", limit=30)

                    # Should get articles from all 3 sources
                    assert len(articles) == 3

                    # Verify each API was called
                    mock_fh.assert_called_once()
                    mock_av.assert_called_once()
                    mock_na.assert_called_once()

                    # Check that bias detection was added
                    for article in articles:
                        assert "bias_indicators" in article

    @pytest.mark.asyncio
    async def test_fetch_messages_deduplication(self, adapter):
        """Test article deduplication by URL."""
        duplicate_articles = [
            {"id": "1", "url": "https://example.com/article", "title": "Article 1"},
            {"id": "2", "url": "https://example.com/article", "title": "Article 2"},  # Duplicate URL
            {"id": "3", "url": "https://example.com/different", "title": "Article 3"},
        ]

        with patch.object(adapter, '_fetch_finnhub_news', new_callable=AsyncMock) as mock_fh:
            with patch.object(adapter, '_fetch_alpha_vantage_news', new_callable=AsyncMock) as mock_av:
                with patch.object(adapter, '_fetch_newsapi_news', new_callable=AsyncMock) as mock_na:
                    mock_fh.return_value = duplicate_articles
                    mock_av.return_value = []
                    mock_na.return_value = []

                    articles = await adapter.fetch_messages("AAPL")

                    # Should deduplicate and return only 2 unique URLs
                    assert len(articles) == 2
                    urls = [article["url"] for article in articles]
                    assert len(set(urls)) == 2  # All URLs should be unique

    @pytest.mark.asyncio
    async def test_fetch_messages_no_tokens(self):
        """Test fetch messages when no API tokens are configured."""
        adapter = AsyncNewsAdapter()  # No tokens

        try:
            articles = await adapter.fetch_messages("AAPL")
            assert articles == []
        finally:
            await adapter.close()

    @pytest.mark.asyncio
    async def test_fetch_messages_invalid_ticker(self, adapter):
        """Test error handling for invalid ticker."""
        with pytest.raises(ValueError, match="Ticker cannot be empty"):
            await adapter.fetch_messages("")

        with pytest.raises(ValueError, match="Ticker cannot be empty"):
            await adapter.fetch_messages("   ")

    @pytest.mark.asyncio
    async def test_fetch_messages_with_since_ts(self, adapter):
        """Test fetch messages with since_ts parameter."""
        with patch.object(adapter, '_fetch_finnhub_news', new_callable=AsyncMock) as mock_fh:
            with patch.object(adapter, '_fetch_alpha_vantage_news', new_callable=AsyncMock) as mock_av:
                with patch.object(adapter, '_fetch_newsapi_news', new_callable=AsyncMock) as mock_na:
                    mock_fh.return_value = []
                    mock_av.return_value = []
                    mock_na.return_value = []

                    since_ts = 1672531200
                    await adapter.fetch_messages("AAPL", since_ts=since_ts)

                    # Verify since_ts was passed to each API (limit is divided by 3 sources)
                    expected_limit = 200 // 3  # Default limit divided by 3 sources
                    mock_fh.assert_called_with("AAPL", since_ts, expected_limit)
                    mock_av.assert_called_with("AAPL", since_ts, expected_limit)
                    mock_na.assert_called_with("AAPL", since_ts, expected_limit)

    @pytest.mark.asyncio
    async def test_fetch_summary_success(self, adapter):
        """Test successful summary generation."""
        # Create articles with clear sentiment patterns
        articles = [
            {"title": "AAPL surge rally strong", "content": "Positive growth outlook",
             "credibility": 0.9, "source": "Reuters", "bias_indicators": {"promotional": False}},
            {"title": "AAPL decline fall weak", "content": "Bearish market conditions",
             "credibility": 0.8, "source": "Bloomberg", "bias_indicators": {"promotional": False}},
            {"title": "AAPL analysis report", "content": "Neutral market assessment",
             "credibility": 0.7, "source": "CNBC", "bias_indicators": {"promotional": False}},
            {"title": "AAPL bullish upgrade buy", "content": "Strong recommendation",
             "credibility": 0.95, "source": "WSJ", "bias_indicators": {"promotional": True}},
        ]

        with patch.object(adapter, 'fetch_messages', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = articles

            summary = await adapter.fetch_summary("AAPL")

            assert summary["mentions"] == 4
            assert summary["bullish"] == 2  # Two bullish articles
            assert summary["bearish"] == 1   # One bearish article
            assert summary["neutral"] == 1   # One neutral article
            assert summary["sentiment_score"] == 0.25  # (2-1)/4 = 0.25
            assert summary["provider"] == "news"
            assert abs(summary["avg_credibility"] - 0.8625) < 0.001  # (0.9+0.8+0.7+0.95)/4 with floating point tolerance
            assert "timestamp" in summary

            # Check source distribution
            assert "source_distribution" in summary
            assert len(summary["source_distribution"]) > 0

            # Check bias indicators
            assert "bias_indicators" in summary
            assert summary["bias_indicators"]["promotional"] == 1

    @pytest.mark.asyncio
    async def test_fetch_summary_alpha_vantage_sentiment(self, adapter):
        """Test summary generation using Alpha Vantage sentiment labels."""
        articles = [
            {"provider": "alpha_vantage", "overall_sentiment": "Bullish",
             "title": "Test", "content": "", "credibility": 0.8, "source": "AV", "bias_indicators": {}},
            {"provider": "alpha_vantage", "overall_sentiment": "Bearish",
             "title": "Test", "content": "", "credibility": 0.8, "source": "AV", "bias_indicators": {}},
            {"provider": "alpha_vantage", "overall_sentiment": "Neutral",
             "title": "Test", "content": "", "credibility": 0.8, "source": "AV", "bias_indicators": {}},
        ]

        with patch.object(adapter, 'fetch_messages', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = articles

            summary = await adapter.fetch_summary("AAPL")

            # Should use Alpha Vantage sentiment labels directly
            assert summary["bullish"] == 1
            assert summary["bearish"] == 1
            assert summary["neutral"] == 1

    @pytest.mark.asyncio
    async def test_fetch_summary_no_articles(self, adapter):
        """Test summary generation with no articles."""
        with patch.object(adapter, 'fetch_messages', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = []

            summary = await adapter.fetch_summary("AAPL")

            assert summary["mentions"] == 0
            assert summary["bullish"] == 0
            assert summary["bearish"] == 0
            assert summary["neutral"] == 0
            assert summary["sentiment_score"] == 0.0
            assert summary["avg_credibility"] == 0.0

    @pytest.mark.asyncio
    async def test_fetch_summary_sentiment_keywords(self, adapter):
        """Test sentiment analysis with financial news keywords."""
        test_cases = [
            # (title, content, expected_sentiment)
            ("AAPL surge rally", "Strong growth profit", "bullish"),
            ("AAPL decline crash", "Weak performance loss", "bearish"),
            ("AAPL analysis report", "Standard assessment", "neutral"),
            ("AAPL upgrade buy", "Positive outlook target", "bullish"),
            ("AAPL downgrade sell", "Negative concerns risk", "bearish"),
        ]

        for title, content, expected_sentiment in test_cases:
            articles = [{"title": title, "content": content, "credibility": 0.8,
                       "source": "test", "bias_indicators": {}, "provider": "test"}]

            with patch.object(adapter, 'fetch_messages', new_callable=AsyncMock) as mock_fetch:
                mock_fetch.return_value = articles

                summary = await adapter.fetch_summary("TEST")

                if expected_sentiment == "bullish":
                    assert summary["bullish"] == 1 and summary["bearish"] == 0
                elif expected_sentiment == "bearish":
                    assert summary["bearish"] == 1 and summary["bullish"] == 0
                else:  # neutral
                    assert summary["neutral"] == 1

    @pytest.mark.asyncio
    async def test_get_with_retry_success(self, adapter):
        """Test successful HTTP request with retry logic."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"test": "data"})
        mock_response.raise_for_status = Mock()
        mock_response.request_info = Mock()
        mock_response.history = []

        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        adapter._session = mock_session

        result = await adapter._get_with_retry("https://test.com", {"param": "value"}, api="finnhub")

        assert result == {"test": "data"}
        assert adapter._health_info.status == AdapterStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_get_with_retry_authentication_error(self, adapter):
        """Test handling of authentication errors (401)."""
        mock_response = Mock()
        mock_response.status = 401
        mock_response.request_info = Mock()
        mock_response.history = []

        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        adapter._session = mock_session

        with pytest.raises(aiohttp.ClientResponseError, match="Authentication failed"):
            await adapter._get_with_retry("https://test.com", api="finnhub")

    @pytest.mark.asyncio
    async def test_close_cleanup(self, adapter):
        """Test proper resource cleanup on close."""
        # Set up a session
        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_session.close = AsyncMock()
        adapter._session = mock_session
        adapter._provided_session = False  # Ensure it will be closed

        await adapter.close()

        # Verify session was closed
        mock_session.close.assert_called_once()
        assert adapter._session is None

    @pytest.mark.asyncio
    async def test_api_error_handling(self, adapter):
        """Test handling of API errors during fetching."""
        with patch.object(adapter, '_fetch_finnhub_news', new_callable=AsyncMock) as mock_fh:
            with patch.object(adapter, '_fetch_alpha_vantage_news', new_callable=AsyncMock) as mock_av:
                with patch.object(adapter, '_fetch_newsapi_news', new_callable=AsyncMock) as mock_na:
                    # Simulate API errors
                    mock_fh.side_effect = Exception("Finnhub API error")
                    mock_av.return_value = [{"id": "av1", "url": "https://test.com", "title": "AV Article"}]
                    mock_na.side_effect = Exception("NewsAPI error")

                    # Should still return articles from working API
                    articles = await adapter.fetch_messages("AAPL")

                    assert len(articles) == 1
                    assert articles[0]["id"] == "av1"

    @pytest.mark.asyncio
    async def test_article_sorting_by_date(self, adapter):
        """Test that articles are sorted by published date."""
        articles = [
            {"id": "old", "url": "https://old.com", "published_at": "2023-01-01T10:00:00Z"},
            {"id": "new", "url": "https://new.com", "published_at": "2023-01-01T12:00:00Z"},
            {"id": "middle", "url": "https://middle.com", "published_at": "2023-01-01T11:00:00Z"},
        ]

        with patch.object(adapter, '_fetch_finnhub_news', new_callable=AsyncMock) as mock_fh:
            with patch.object(adapter, '_fetch_alpha_vantage_news', new_callable=AsyncMock) as mock_av:
                with patch.object(adapter, '_fetch_newsapi_news', new_callable=AsyncMock) as mock_na:
                    mock_fh.return_value = articles
                    mock_av.return_value = []
                    mock_na.return_value = []

                    result = await adapter.fetch_messages("AAPL")

                    # Should be sorted by published_at in descending order (newest first)
                    assert result[0]["id"] == "new"
                    assert result[1]["id"] == "middle"
                    assert result[2]["id"] == "old"