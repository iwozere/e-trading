"""
Tests for the common business logic module.

Tests the fundamentals retrieval and normalization functionality.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import unittest
from unittest.mock import patch, MagicMock
from src.common.fundamentals import get_fundamentals, normalize_fundamentals
from src.model.telegram_bot import Fundamentals


class TestCommonFundamentals(unittest.TestCase):
    """Test the common fundamentals functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_symbol = "AAPL"
        self.mock_yf_data = {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "current_price": 150.0,
            "market_cap": 2500000000000,
            "pe_ratio": 25.0,
            "forward_pe": 24.0,
            "dividend_yield": 0.5,
            "earnings_per_share": 6.0,
            "price_to_book": 15.0,
            "return_on_equity": 0.15,
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "data_source": "Yahoo Finance",
            "last_updated": "2023-01-01 12:00:00"
        }
        self.mock_av_data = {
            "Symbol": "AAPL",
            "Name": "Apple Inc.",
            "MarketCapitalization": 2500000000000,
            "PERatio": 25.0,
            "ForwardPE": 24.0,
            "DividendYield": 0.5,
            "EPS": 6.0,
            "PriceToBookRatio": 15.0,
            "ReturnOnEquityTTM": 0.15,
            "Sector": "Technology",
            "Industry": "Consumer Electronics",
            "data_source": "Alpha Vantage",
            "last_updated": "2023-01-01 12:00:00"
        }
        self.mock_fh_data = {
            "ticker": "AAPL",
            "current_price": 150.0,
            "market_cap": 2500000000000,
            "pe_ratio": 25.0,
            "earnings_per_share": 6.0,
            "revenue": 400000000000,
            "data_source": "Finnhub",
            "last_updated": "2023-01-01 12:00:00"
        }

    def test_normalize_fundamentals_single_provider(self):
        """Test normalization with a single provider."""
        sources = {"yf": self.mock_yf_data}
        result = normalize_fundamentals(sources)

        self.assertIsInstance(result, Fundamentals)
        self.assertEqual(result.ticker, "AAPL")
        self.assertEqual(result.company_name, "Apple Inc.")
        self.assertEqual(result.current_price, 150.0)
        self.assertEqual(result.pe_ratio, 25.0)
        self.assertEqual(result.sources["ticker"], "yf")
        self.assertEqual(result.sources["company_name"], "yf")

    def test_normalize_fundamentals_multiple_providers(self):
        """Test normalization with multiple providers, checking priority."""
        sources = {
            "av": self.mock_av_data,
            "fh": self.mock_fh_data,
            "yf": self.mock_yf_data
        }
        result = normalize_fundamentals(sources)

        # Should use yf data first (highest priority)
        self.assertEqual(result.ticker, "AAPL")
        self.assertEqual(result.company_name, "Apple Inc.")
        self.assertEqual(result.current_price, 150.0)
        self.assertEqual(result.sources["ticker"], "yf")
        self.assertEqual(result.sources["company_name"], "yf")

    def test_normalize_fundamentals_fallback(self):
        """Test normalization when higher priority providers don't have data."""
        # yf missing some fields, av has them
        yf_data = {"ticker": "AAPL", "current_price": 150.0}
        av_data = {"Symbol": "AAPL", "Name": "Apple Inc.", "MarketCapitalization": 2500000000000}

        sources = {"yf": yf_data, "av": av_data}
        result = normalize_fundamentals(sources)

        self.assertEqual(result.ticker, "AAPL")  # from yf
        self.assertEqual(result.company_name, "Apple Inc.")  # from av
        self.assertEqual(result.market_cap, 2500000000000)  # from av
        self.assertEqual(result.sources["ticker"], "yf")
        self.assertEqual(result.sources["company_name"], "av")
        self.assertEqual(result.sources["market_cap"], "av")

    def test_normalize_fundamentals_empty_sources(self):
        """Test normalization with empty sources."""
        result = normalize_fundamentals({})

        self.assertIsInstance(result, Fundamentals)
        self.assertIsNone(result.ticker)
        self.assertIsNone(result.company_name)
        self.assertEqual(result.sources, {})

    def test_normalize_fundamentals_none_values(self):
        """Test normalization with None values."""
        sources = {
            "yf": {"ticker": None, "company_name": "", "current_price": 0.0}
        }
        result = normalize_fundamentals(sources)

        self.assertIsNone(result.ticker)
        self.assertIsNone(result.company_name)
        self.assertEqual(result.current_price, 0.0)

    def test_normalize_fundamentals_alpha_vantage_fields(self):
        """Test normalization with Alpha Vantage specific field names."""
        sources = {"av": self.mock_av_data}
        result = normalize_fundamentals(sources)

        self.assertEqual(result.ticker, "AAPL")
        self.assertEqual(result.company_name, "Apple Inc.")
        self.assertEqual(result.market_cap, 2500000000000)
        self.assertEqual(result.pe_ratio, 25.0)
        self.assertEqual(result.sources["ticker"], "av")
        self.assertEqual(result.sources["company_name"], "av")

    def test_normalize_fundamentals_finnhub_fields(self):
        """Test normalization with Finnhub data."""
        sources = {"fh": self.mock_fh_data}
        result = normalize_fundamentals(sources)

        self.assertEqual(result.ticker, "AAPL")
        self.assertEqual(result.current_price, 150.0)
        self.assertEqual(result.market_cap, 2500000000000)
        self.assertEqual(result.revenue, 400000000000)
        self.assertEqual(result.sources["ticker"], "fh")
        self.assertEqual(result.sources["current_price"], "fh")

    @patch('src.common.fundamentals.DataDownloaderFactory')
    def test_get_fundamentals_single_provider(self, mock_factory):
        """Test get_fundamentals with a single provider."""
        # Mock the downloader
        mock_downloader = MagicMock()
        mock_downloader.get_fundamentals.return_value = self.mock_yf_data
        mock_factory.create_downloader.return_value = mock_downloader

        result = get_fundamentals(self.test_symbol, provider="yf")

        self.assertIsInstance(result, Fundamentals)
        self.assertEqual(result.ticker, "AAPL")
        self.assertEqual(result.company_name, "Apple Inc.")
        mock_factory.create_downloader.assert_called_once_with("yf")

    @patch('src.common.fundamentals.DataDownloaderFactory')
    def test_get_fundamentals_single_provider_fundamentals_object(self, mock_factory):
        """Test get_fundamentals when downloader returns Fundamentals object."""
        # Mock the downloader returning a Fundamentals object
        mock_fundamentals = Fundamentals(
            ticker="AAPL",
            company_name="Apple Inc.",
            current_price=150.0,
            market_cap=2500000000000,
            pe_ratio=25.0,
            forward_pe=24.0,
            dividend_yield=0.5,
            earnings_per_share=6.0
        )

        mock_downloader = MagicMock()
        mock_downloader.get_fundamentals.return_value = mock_fundamentals
        mock_factory.create_downloader.return_value = mock_downloader

        result = get_fundamentals(self.test_symbol, provider="yf")

        self.assertIsInstance(result, Fundamentals)
        self.assertEqual(result.ticker, "AAPL")
        self.assertEqual(result.company_name, "Apple Inc.")
        # Should return the original object, not normalize it
        self.assertIs(result, mock_fundamentals)

    @patch('src.common.fundamentals.DataDownloaderFactory')
    def test_get_fundamentals_unknown_provider(self, mock_factory):
        """Test get_fundamentals with unknown provider."""
        mock_factory.create_downloader.return_value = None

        with self.assertRaises(ValueError) as context:
            get_fundamentals(self.test_symbol, provider="unknown")

        self.assertIn("Unknown or unsupported provider", str(context.exception))

    @patch('src.common.fundamentals.DataDownloaderFactory')
    def test_get_fundamentals_no_provider_try_all(self, mock_factory):
        """Test get_fundamentals without provider, trying all providers."""
        # Mock successful yf downloader
        mock_yf_downloader = MagicMock()
        mock_yf_downloader.get_fundamentals.return_value = self.mock_yf_data

        # Mock failed av downloader (raises exception)
        mock_av_downloader = MagicMock()
        mock_av_downloader.get_fundamentals.side_effect = Exception("API error")

        # Mock successful fh downloader
        mock_fh_downloader = MagicMock()
        mock_fh_downloader.get_fundamentals.return_value = self.mock_fh_data

        # Mock factory to return different downloaders for different calls
        mock_factory.create_downloader.side_effect = [
            mock_yf_downloader,  # yf
            mock_av_downloader,  # av (will fail)
            mock_fh_downloader,  # fh
            None,               # td (no downloader)
            None                # pg (no downloader)
        ]

        result = get_fundamentals(self.test_symbol)

        self.assertIsInstance(result, Fundamentals)
        self.assertEqual(result.ticker, "AAPL")
        self.assertEqual(result.company_name, "Apple Inc.")
        # Should use yf data (highest priority)
        self.assertEqual(result.sources["ticker"], "yf")
        self.assertEqual(result.sources["company_name"], "yf")

    @patch('src.common.fundamentals.DataDownloaderFactory')
    def test_get_fundamentals_no_provider_all_fail(self, mock_factory):
        """Test get_fundamentals when all providers fail."""
        # Mock all downloaders to fail
        mock_factory.create_downloader.return_value = None

        result = get_fundamentals(self.test_symbol)

        self.assertIsInstance(result, Fundamentals)
        # Should return empty Fundamentals object
        self.assertIsNone(result.ticker)
        self.assertIsNone(result.company_name)
        self.assertEqual(result.sources, {})

    def test_normalize_fundamentals_field_priority(self):
        """Test that field priority is respected across providers."""
        # yf has ticker and company_name
        # av has Symbol and Name (should be used as fallback)
        # fh has ticker (should not be used due to priority)
        sources = {
            "fh": {"ticker": "AAPL_FH", "current_price": 155.0},
            "av": {"Symbol": "AAPL_AV", "Name": "Apple Inc. AV"},
            "yf": {"ticker": "AAPL_YF", "company_name": "Apple Inc. YF"}
        }

        result = normalize_fundamentals(sources)

        # Should use yf data first
        self.assertEqual(result.ticker, "AAPL_YF")
        self.assertEqual(result.company_name, "Apple Inc. YF")
        self.assertEqual(result.sources["ticker"], "yf")
        self.assertEqual(result.sources["company_name"], "yf")

    def test_normalize_fundamentals_mixed_data_types(self):
        """Test normalization with mixed data types (strings, numbers)."""
        sources = {
            "yf": {
                "ticker": "AAPL",
                "current_price": 150.0,
                "market_cap": "2500000000000",  # string
                "pe_ratio": 25.0,
                "dividend_yield": "0.5"  # string
            }
        }

        result = normalize_fundamentals(sources)

        self.assertEqual(result.ticker, "AAPL")
        self.assertEqual(result.current_price, 150.0)
        self.assertEqual(result.market_cap, "2500000000000")  # should preserve string
        self.assertEqual(result.pe_ratio, 25.0)
        self.assertEqual(result.dividend_yield, "0.5")  # should preserve string

    def test_normalize_fundamentals_comprehensive_fields(self):
        """Test normalization with all possible fields."""
        sources = {
            "yf": {
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "current_price": 150.0,
                "market_cap": 2500000000000,
                "pe_ratio": 25.0,
                "forward_pe": 24.0,
                "dividend_yield": 0.5,
                "earnings_per_share": 6.0,
                "price_to_book": 15.0,
                "return_on_equity": 0.15,
                "return_on_assets": 0.10,
                "debt_to_equity": 0.5,
                "current_ratio": 1.5,
                "quick_ratio": 1.2,
                "revenue": 400000000000,
                "revenue_growth": 0.08,
                "net_income": 100000000000,
                "net_income_growth": 0.12,
                "free_cash_flow": 80000000000,
                "operating_margin": 0.25,
                "profit_margin": 0.20,
                "beta": 1.2,
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "country": "US",
                "exchange": "NASDAQ",
                "currency": "USD",
                "shares_outstanding": 16000000000,
                "float_shares": 15000000000,
                "short_ratio": 2.0,
                "payout_ratio": 0.25,
                "peg_ratio": 1.5,
                "price_to_sales": 5.0,
                "enterprise_value": 2600000000000,
                "enterprise_value_to_ebitda": 20.0,
                "data_source": "Yahoo Finance",
                "last_updated": "2023-01-01 12:00:00"
            }
        }

        result = normalize_fundamentals(sources)

        # Test all fields are populated
        self.assertEqual(result.ticker, "AAPL")
        self.assertEqual(result.company_name, "Apple Inc.")
        self.assertEqual(result.current_price, 150.0)
        self.assertEqual(result.market_cap, 2500000000000)
        self.assertEqual(result.pe_ratio, 25.0)
        self.assertEqual(result.forward_pe, 24.0)
        self.assertEqual(result.dividend_yield, 0.5)
        self.assertEqual(result.earnings_per_share, 6.0)
        self.assertEqual(result.price_to_book, 15.0)
        self.assertEqual(result.return_on_equity, 0.15)
        self.assertEqual(result.return_on_assets, 0.10)
        self.assertEqual(result.debt_to_equity, 0.5)
        self.assertEqual(result.current_ratio, 1.5)
        self.assertEqual(result.quick_ratio, 1.2)
        self.assertEqual(result.revenue, 400000000000)
        self.assertEqual(result.revenue_growth, 0.08)
        self.assertEqual(result.net_income, 100000000000)
        self.assertEqual(result.net_income_growth, 0.12)
        self.assertEqual(result.free_cash_flow, 80000000000)
        self.assertEqual(result.operating_margin, 0.25)
        self.assertEqual(result.profit_margin, 0.20)
        self.assertEqual(result.beta, 1.2)
        self.assertEqual(result.sector, "Technology")
        self.assertEqual(result.industry, "Consumer Electronics")
        self.assertEqual(result.country, "US")
        self.assertEqual(result.exchange, "NASDAQ")
        self.assertEqual(result.currency, "USD")
        self.assertEqual(result.shares_outstanding, 16000000000)
        self.assertEqual(result.float_shares, 15000000000)
        self.assertEqual(result.short_ratio, 2.0)
        self.assertEqual(result.payout_ratio, 0.25)
        self.assertEqual(result.peg_ratio, 1.5)
        self.assertEqual(result.price_to_sales, 5.0)
        self.assertEqual(result.enterprise_value, 2600000000000)
        self.assertEqual(result.enterprise_value_to_ebitda, 20.0)
        self.assertEqual(result.data_source, "Yahoo Finance")
        self.assertEqual(result.last_updated, "2023-01-01 12:00:00")

        # Test that all fields have sources recorded
        for field_name in result.sources:
            self.assertEqual(result.sources[field_name], "yf")


if __name__ == '__main__':
    unittest.main()
