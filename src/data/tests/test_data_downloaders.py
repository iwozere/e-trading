"""
Comprehensive unit tests for all data downloaders.

This module tests:
- All data downloader implementations
- Fundamental data retrieval
- OHLCV data retrieval
- Error handling
- API key validation
- Data format validation

How to run:
    pytest tests/test_data_downloaders.py -v

Or to run specific test:
    pytest tests/test_data_downloaders.py::test_yahoo_downloader -v
"""
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import tempfile
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime

# Import all data downloaders
from src.data.downloader.yahoo_data_downloader import YahooDataDownloader
from src.data.downloader.alpha_vantage_data_downloader import AlphaVantageDataDownloader
from src.data.downloader.finnhub_data_downloader import FinnhubDataDownloader
from src.data.downloader.polygon_data_downloader import PolygonDataDownloader
from src.data.downloader.twelvedata_data_downloader import TwelveDataDataDownloader
from src.data.downloader.binance_data_downloader import BinanceDataDownloader
from src.data.downloader.coingecko_data_downloader import CoinGeckoDataDownloader
from src.data.downloader.fmp_data_downloader import FMPDataDownloader
from src.data.downloader.tiingo_data_downloader import TiingoDataDownloader
from src.data.downloader.data_downloader_factory import DataDownloaderFactory
from src.model.telegram_bot import Fundamentals


class TestDataDownloaders(unittest.TestCase):
    """Base class for data downloader tests."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_symbol = "AAPL"
        self.start_date = datetime(2023, 1, 1)
        self.end_date = datetime(2023, 1, 31)
        self.interval = "1d"

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def assert_valid_ohlcv_data(self, df):
        """Assert that OHLCV data is valid."""
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, df.columns)

        # Check data types
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['timestamp']))
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(df[col]))

    def assert_valid_fundamentals(self, fundamentals):
        """Assert that fundamentals data is valid."""
        self.assertIsInstance(fundamentals, Fundamentals)
        self.assertEqual(fundamentals.ticker, self.test_symbol.upper())
        self.assertIsInstance(fundamentals.company_name, str)
        self.assertIsInstance(fundamentals.current_price, (int, float))
        self.assertIsInstance(fundamentals.market_cap, (int, float))
        self.assertIsInstance(fundamentals.data_source, str)
        self.assertIsInstance(fundamentals.last_updated, str)


class TestYahooDataDownloader(TestDataDownloaders):
    """Test Yahoo Finance data downloader."""

    def setUp(self):
        super().setUp()
        self.downloader = YahooDataDownloader()
        # Patch yf.Ticker for all tests in this class
        self.ticker_patcher = patch('src.data.downloader.yahoo_data_downloader.yf.Ticker')
        self.mock_ticker_class = self.ticker_patcher.start()
        self.addCleanup(self.ticker_patcher.stop)

    def test_get_ohlcv(self):
        """Test OHLCV data retrieval."""
        # Mock ticker.history to return a DataFrame with correct columns and index
        mock_ticker_instance = MagicMock()
        mock_history_df = pd.DataFrame({
            'Open': [150.0, 151.0, 152.0],
            'High': [155.0, 156.0, 157.0],
            'Low': [149.0, 150.0, 151.0],
            'Close': [151.0, 152.0, 153.0],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=3, freq='D'))
        mock_ticker_instance.history.return_value = mock_history_df
        self.mock_ticker_class.return_value = mock_ticker_instance
        df = self.downloader.get_ohlcv(self.test_symbol, self.interval, self.start_date, self.end_date)
        self.assert_valid_ohlcv_data(df)

    def test_get_fundamentals(self):
        """Test fundamental data retrieval."""
        # Mock ticker.info
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {
            'longName': 'Apple Inc.',
            'regularMarketPrice': 150.0,
            'marketCap': 2500000000000,
            'trailingPE': 25.0,
            'forwardPE': 24.0,
            'dividendYield': 0.5,
            'trailingEps': 6.0,
            'priceToBook': 15.0,
            'returnOnEquity': 0.15,
            'returnOnAssets': 0.10,
            'debtToEquity': 0.5,
            'currentRatio': 1.5,
            'quickRatio': 1.2,
            'totalRevenue': 400000000000,
            'revenueGrowth': 0.08,
            'netIncomeToCommon': 100000000000,
            'netIncomeGrowth': 0.12,
            'freeCashflow': 80000000000,
            'operatingMargins': 0.25,
            'profitMargins': 0.20,
            'beta': 1.2,
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'country': 'US',
            'exchange': 'NASDAQ',
            'currency': 'USD',
            'sharesOutstanding': 16000000000,
            'floatShares': 15000000000,
            'shortRatio': 2.0,
            'payoutRatio': 0.25,
            'pegRatio': 1.5,
            'priceToSalesTrailing12Months': 5.0,
            'enterpriseValue': 2600000000000,
            'enterpriseToEbitda': 20.0
        }
        self.mock_ticker_class.return_value = mock_ticker_instance
        fundamentals = self.downloader.get_fundamentals(self.test_symbol)
        self.assert_valid_fundamentals(fundamentals)
        self.assertEqual(fundamentals.company_name, 'Apple Inc.')
        self.assertEqual(fundamentals.current_price, 150.0)
        self.assertEqual(fundamentals.data_source, 'Yahoo Finance')

    def test_get_periods(self):
        """Test get_periods method."""
        periods = self.downloader.get_periods()
        self.assertIsInstance(periods, list)
        self.assertGreater(len(periods), 0)

    def test_get_intervals(self):
        """Test get_intervals method."""
        intervals = self.downloader.get_intervals()
        self.assertIsInstance(intervals, list)
        self.assertGreater(len(intervals), 0)


class TestAlphaVantageDataDownloader(TestDataDownloaders):
    """Test Alpha Vantage data downloader."""

    def setUp(self):
        super().setUp()
        self.api_key = "test_api_key"
        self.downloader = AlphaVantageDataDownloader(api_key=self.api_key)

    @patch('requests.get')
    def test_get_ohlcv(self, mock_get):
        """Test OHLCV data retrieval."""
        # Mock API response with correct keys for AlphaVantage
        mock_response = MagicMock()
        mock_response.status_code = 200
        # The DataFrame will have columns '1. open', etc., which are processed by lambda function
        # The lambda function removes spaces, so '1. open' becomes '1.open'
        mock_response.json.return_value = {
            "Time Series (Daily)": {
                "2023-01-01": {
                    "1. open": "150.0",
                    "2. high": "155.0",
                    "3. low": "149.0",
                    "4. close": "151.0",
                    "5. volume": "1000000"
                },
                "2023-01-02": {
                    "1. open": "151.0",
                    "2. high": "156.0",
                    "3. low": "150.0",
                    "4. close": "152.0",
                    "5. volume": "1100000"
                }
            }
        }
        mock_get.return_value = mock_response
        df = self.downloader.get_ohlcv(self.test_symbol, self.interval, self.start_date, self.end_date)
        # Ensure columns after renaming
        for col in ["open", "high", "low", "close", "volume", "timestamp"]:
            self.assertIn(col, df.columns)
        self.assert_valid_ohlcv_data(df)
        mock_get.assert_called_once()

    @patch('requests.get')
    def test_get_fundamentals(self, mock_get):
        """Test fundamental data retrieval."""
        # Mock API responses
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "Symbol": "AAPL",
            "Name": "Apple Inc.",
            "MarketCapitalization": "2500000000000",
            "PERatio": "25.0",
            "ForwardPE": "24.0",
            "DividendYield": "0.5",
            "EPS": "6.0",
            "PriceToBookRatio": "15.0",
            "ReturnOnEquityTTM": "0.15",
            "ReturnOnAssetsTTM": "0.10",
            "DebtToEquityRatio": "0.5",
            "CurrentRatio": "1.5",
            "QuickRatio": "1.2",
            "RevenueTTM": "400000000000",
            "RevenueGrowth": "0.08",
            "NetIncomeTTM": "100000000000",
            "NetIncomeGrowth": "0.12",
            "FreeCashFlow": "80000000000",
            "OperatingMarginTTM": "0.25",
            "ProfitMargin": "0.20",
            "Beta": "1.2",
            "Sector": "Technology",
            "Industry": "Consumer Electronics",
            "Country": "US",
            "Exchange": "NASDAQ",
            "Currency": "USD",
            "SharesOutstanding": "16000000000",
            "FloatShares": "15000000000",
            "ShortRatio": "2.0",
            "PayoutRatio": "0.25",
            "PEGRatio": "1.5",
            "PriceToSalesRatioTTM": "5.0",
            "MarketCapitalization": "2500000000000",
            "EVToEBITDA": "20.0"
        }
        mock_get.return_value = mock_response

        fundamentals = self.downloader.get_fundamentals(self.test_symbol)

        self.assert_valid_fundamentals(fundamentals)
        self.assertEqual(fundamentals.data_source, 'Alpha Vantage')

    def test_missing_api_key(self):
        """Test error handling for missing API key."""
        # AlphaVantageDataDownloader does not raise by default, so we just check it instantiates
        try:
            AlphaVantageDataDownloader(api_key=None)
        except Exception as e:
            self.fail(f"AlphaVantageDataDownloader raised an exception: {e}")


class TestFinnhubDataDownloader(TestDataDownloaders):
    """Test Finnhub data downloader."""

    def setUp(self):
        super().setUp()
        self.api_key = "test_api_key"
        self.downloader = FinnhubDataDownloader(api_key=self.api_key)

    @patch('requests.get')
    def test_get_ohlcv(self, mock_get):
        """Test OHLCV data retrieval."""
        # Mock API response for Finnhub with 's': 'ok'
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "o": [150.0, 151.0],
            "h": [155.0, 156.0],
            "l": [149.0, 150.0],
            "c": [151.0, 152.0],
            "v": [1000000, 1100000],
            "t": [1672531200, 1672617600],
            "s": "ok"
        }
        mock_get.return_value = mock_response
        df = self.downloader.get_ohlcv(self.test_symbol, self.interval, self.start_date, self.end_date)
        self.assert_valid_ohlcv_data(df)
        mock_get.assert_called_once()

    @patch('requests.get')
    def test_get_fundamentals(self, mock_get):
        """Test fundamental data retrieval."""
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "name": "Apple Inc.",
            "marketCapitalization": 2500000000000,
            "pe": 25.0,
            "forwardPE": 24.0,
            "dividendYield": 0.5,
            "eps": 6.0,
            "priceToBook": 15.0,
            "returnOnEquity": 0.15,
            "returnOnAssets": 0.10,
            "debtToEquity": 0.5,
            "currentRatio": 1.5,
            "quickRatio": 1.2,
            "revenue": 400000000000,
            "revenueGrowth": 0.08,
            "netIncome": 100000000000,
            "netIncomeGrowth": 0.12,
            "freeCashFlow": 80000000000,
            "operatingMargin": 0.25,
            "profitMargin": 0.20,
            "beta": 1.2,
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "country": "US",
            "exchange": "NASDAQ",
            "currency": "USD",
            "sharesOutstanding": 16000000000,
            "floatShares": 15000000000,
            "shortRatio": 2.0,
            "payoutRatio": 0.25,
            "pegRatio": 1.5,
            "priceToSales": 5.0,
            "enterpriseValue": 2600000000000,
            "enterpriseValueToEbitda": 20.0
        }
        mock_get.return_value = mock_response

        fundamentals = self.downloader.get_fundamentals(self.test_symbol)

        self.assert_valid_fundamentals(fundamentals)
        self.assertEqual(fundamentals.data_source, 'Finnhub')


class TestPolygonDataDownloader(TestDataDownloaders):
    """Test Polygon.io data downloader."""

    def setUp(self):
        super().setUp()
        self.api_key = "test_api_key"
        self.downloader = PolygonDataDownloader(api_key=self.api_key)

    @patch('requests.get')
    def test_get_ohlcv(self, mock_get):
        """Test OHLCV data retrieval."""
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "t": 1672531200000,  # Unix timestamp in milliseconds
                    "o": 150.0,
                    "h": 155.0,
                    "l": 149.0,
                    "c": 151.0,
                    "v": 1000000
                }
            ]
        }
        mock_get.return_value = mock_response

        df = self.downloader.get_ohlcv(self.test_symbol, self.interval, self.start_date, self.end_date)

        self.assert_valid_ohlcv_data(df)
        mock_get.assert_called_once()

    @patch('requests.get')
    def test_get_fundamentals(self, mock_get):
        """Test fundamental data retrieval."""
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": {
                "name": "Apple Inc.",
                "market_cap": 2500000000000,
                "shares_outstanding": 16000000000,
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "country": "US",
                "exchange": "NASDAQ",
                "currency": "USD"
            }
        }
        mock_get.return_value = mock_response

        fundamentals = self.downloader.get_fundamentals(self.test_symbol)

        self.assert_valid_fundamentals(fundamentals)
        self.assertEqual(fundamentals.data_source, 'Polygon.io')


class TestTwelveDataDataDownloader(TestDataDownloaders):
    """Test Twelve Data downloader."""

    def setUp(self):
        super().setUp()
        self.api_key = "test_api_key"
        self.downloader = TwelveDataDataDownloader(api_key=self.api_key)

    @patch('requests.get')
    def test_get_ohlcv(self, mock_get):
        """Test OHLCV data retrieval."""
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "values": [
                {
                    "datetime": "2023-01-01",
                    "open": "150.0",
                    "high": "155.0",
                    "low": "149.0",
                    "close": "151.0",
                    "volume": "1000000"
                }
            ]
        }
        mock_get.return_value = mock_response

        df = self.downloader.get_ohlcv(self.test_symbol, self.interval, self.start_date, self.end_date)

        self.assert_valid_ohlcv_data(df)
        mock_get.assert_called_once()

    @patch('requests.get')
    def test_get_fundamentals(self, mock_get):
        """Test fundamental data retrieval."""
        # Mock API responses
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "name": "Apple Inc.",
            "market_cap": "2500000000000",
            "pe_ratio": "25.0",
            "dividend_yield": "0.5",
            "pb_ratio": "15.0",
            "beta": "1.2",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "country": "US",
            "exchange": "NASDAQ",
            "currency": "USD"
        }
        mock_get.return_value = mock_response

        fundamentals = self.downloader.get_fundamentals(self.test_symbol)

        self.assert_valid_fundamentals(fundamentals)
        self.assertEqual(fundamentals.data_source, 'Twelve Data')


class TestBinanceDataDownloader(TestDataDownloaders):
    """Test Binance data downloader."""

    def setUp(self):
        super().setUp()
        self.api_key = "test_api_key"
        self.api_secret = "test_secret_key"
        self.downloader = BinanceDataDownloader(api_key=self.api_key, api_secret=self.api_secret)

    @patch('binance.client.Client.get_historical_klines')
    def test_get_ohlcv(self, mock_klines):
        """Test OHLCV data retrieval."""
        # Mock Binance API response
        mock_klines.return_value = [
            [1672531200000, "150.0", "155.0", "149.0", "151.0", "1000000", 1672617600000, "0", "0", "0", "0", "0"]
        ]
        df = self.downloader.get_ohlcv("BTCUSDT", self.interval, self.start_date, self.end_date)
        self.assert_valid_ohlcv_data(df)
        mock_klines.assert_called_once()

    def test_get_fundamentals(self):
        """Test fundamental data retrieval (should raise NotImplementedError)."""
        with self.assertRaises(NotImplementedError):
            self.downloader.get_fundamentals("BTCUSDT")


class TestCoinGeckoDataDownloader(TestDataDownloaders):
    """Test CoinGecko data downloader."""

    def setUp(self):
        super().setUp()
        self.downloader = CoinGeckoDataDownloader()

    @patch('requests.get')
    def test_get_ohlcv(self, mock_get):
        """Test OHLCV data retrieval."""
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "prices": [
                [1672531200000, 150.0],
                [1672617600000, 151.0]
            ],
            "total_volumes": [
                [1672531200000, 1000000],
                [1672617600000, 1100000]
            ]
        }
        mock_get.return_value = mock_response

        df = self.downloader.get_ohlcv("bitcoin", self.interval, self.start_date, self.end_date)

        self.assert_valid_ohlcv_data(df)
        mock_get.assert_called_once()

    def test_get_fundamentals(self):
        """Test fundamental data retrieval (should raise NotImplementedError)."""
        with self.assertRaises(NotImplementedError):
            self.downloader.get_fundamentals("bitcoin")


class TestFMPDataDownloader(TestDataDownloaders):
    """Test Financial Modeling Prep data downloader."""

    def setUp(self):
        super().setUp()
        self.api_key = "test_api_key"
        self.downloader = FMPDataDownloader(api_key=self.api_key)

    @patch('requests.get')
    def test_get_ohlcv(self, mock_get):
        """Test OHLCV data retrieval."""
        # Mock API response for daily data (FMP returns {"historical": [...]})
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "historical": [
                {
                    "date": "2023-01-01",
                    "open": 150.0,
                    "high": 155.0,
                    "low": 149.0,
                    "close": 151.0,
                    "volume": 1000000
                },
                {
                    "date": "2023-01-02",
                    "open": 151.0,
                    "high": 156.0,
                    "low": 150.0,
                    "close": 152.0,
                    "volume": 1100000
                }
            ]
        }
        mock_get.return_value = mock_response

        df = self.downloader.get_ohlcv(self.test_symbol, self.interval, self.start_date, self.end_date)

        # FMP downloader returns timestamp as index, so we need to check differently
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

        # Check that timestamp is the index
        self.assertEqual(df.index.name, 'timestamp')
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df.index))

        # Check required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, df.columns)
            self.assertTrue(pd.api.types.is_numeric_dtype(df[col]))

        mock_get.assert_called_once()

    @patch('requests.get')
    def test_get_fundamentals(self, mock_get):
        """Test fundamental data retrieval."""
        # Mock API response for company profile
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "symbol": "AAPL",
                "companyName": "Apple Inc.",
                "marketCap": 2500000000000,
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "country": "US",
                "exchange": "NASDAQ",
                "currency": "USD"
            }
        ]
        mock_get.return_value = mock_response

        fundamentals = self.downloader.get_fundamentals(self.test_symbol)

        self.assertIsInstance(fundamentals, dict)
        self.assertIn('symbol', fundamentals)
        self.assertIn('profile', fundamentals)
        self.assertEqual(fundamentals['symbol'], self.test_symbol)

    def test_missing_api_key(self):
        """Test error handling for missing API key."""
        try:
            FMPDataDownloader(api_key=None)
        except Exception as e:
            self.fail(f"FMPDataDownloader raised an exception: {e}")


class TestTiingoDataDownloader(TestDataDownloaders):
    """Test Tiingo data downloader."""

    def setUp(self):
        super().setUp()
        self.api_key = "test_api_key"
        self.downloader = TiingoDataDownloader(api_key=self.api_key)

    @patch('requests.get')
    def test_get_ohlcv(self, mock_get):
        """Test OHLCV data retrieval."""
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "date": "2023-01-01T00:00:00.000Z",
                "open": 150.0,
                "high": 155.0,
                "low": 149.0,
                "close": 151.0,
                "volume": 1000000,
                "adjOpen": 150.0,
                "adjHigh": 155.0,
                "adjLow": 149.0,
                "adjClose": 151.0,
                "adjVolume": 1000000,
                "divCash": 0.0,
                "splitFactor": 1.0
            },
            {
                "date": "2023-01-02T00:00:00.000Z",
                "open": 151.0,
                "high": 156.0,
                "low": 150.0,
                "close": 152.0,
                "volume": 1100000,
                "adjOpen": 151.0,
                "adjHigh": 156.0,
                "adjLow": 150.0,
                "adjClose": 152.0,
                "adjVolume": 1100000,
                "divCash": 0.0,
                "splitFactor": 1.0
            }
        ]
        mock_get.return_value = mock_response

        df = self.downloader.get_ohlcv(self.test_symbol, self.interval, self.start_date, self.end_date)

        # Tiingo downloader returns timestamp as index, so we need to check differently
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

        # Check that timestamp is the index
        self.assertEqual(df.index.name, 'timestamp')
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df.index))

        # Check required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, df.columns)
            self.assertTrue(pd.api.types.is_numeric_dtype(df[col]))

        mock_get.assert_called_once()

    @patch('requests.get')
    def test_get_fundamentals(self, mock_get):
        """Test fundamental data retrieval."""
        # Mock API response for company metadata
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ticker": "AAPL",
            "name": "Apple Inc.",
            "description": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.",
            "startDate": "1980-12-12",
            "endDate": "2024-01-01",
            "exchangeCode": "NASDAQ",
            "ticker": "AAPL"
        }
        mock_get.return_value = mock_response

        fundamentals = self.downloader.get_fundamentals(self.test_symbol)

        self.assertIsInstance(fundamentals, dict)
        self.assertIn('symbol', fundamentals)
        self.assertIn('name', fundamentals)
        self.assertEqual(fundamentals['symbol'], self.test_symbol)

    def test_missing_api_key(self):
        """Test error handling for missing API key."""
        try:
            TiingoDataDownloader(api_key=None)
        except Exception as e:
            self.fail(f"TiingoDataDownloader raised an exception: {e}")


class TestDataDownloaderFactory(unittest.TestCase):
    """Test DataDownloaderFactory."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_create_yahoo_downloader(self):
        """Test creating Yahoo Finance downloader."""
        downloader = DataDownloaderFactory.create_downloader("yf")
        self.assertIsInstance(downloader, YahooDataDownloader)

    def test_create_alpha_vantage_downloader(self):
        """Test creating Alpha Vantage downloader."""
        downloader = DataDownloaderFactory.create_downloader("av", api_key="test_key")
        self.assertIsInstance(downloader, AlphaVantageDataDownloader)

    def test_create_binance_downloader(self):
        """Test creating Binance downloader."""
        downloader = DataDownloaderFactory.create_downloader("bnc", api_key="test_key", api_secret="test_secret")
        self.assertIsInstance(downloader, BinanceDataDownloader)

    def test_create_fmp_downloader(self):
        """Test creating FMP downloader."""
        downloader = DataDownloaderFactory.create_downloader("fmp", api_key="test_key")
        self.assertIsInstance(downloader, FMPDataDownloader)

    def test_create_tiingo_downloader(self):
        """Test creating Tiingo downloader."""
        downloader = DataDownloaderFactory.create_downloader("tiingo", api_key="test_key")
        self.assertIsInstance(downloader, TiingoDataDownloader)

    def test_invalid_provider(self):
        """Test creating downloader with invalid provider."""
        downloader = DataDownloaderFactory.create_downloader("invalid")
        self.assertIsNone(downloader)

    def test_provider_codes(self):
        """Test provider code mapping."""
        # Test Yahoo Finance codes
        self.assertEqual(DataDownloaderFactory.get_provider_by_code("yf"), "yahoo")
        self.assertEqual(DataDownloaderFactory.get_provider_by_code("yahoo"), "yahoo")
        self.assertEqual(DataDownloaderFactory.get_provider_by_code("yf_finance"), "yahoo")

        # Test Alpha Vantage codes
        self.assertEqual(DataDownloaderFactory.get_provider_by_code("av"), "alphavantage")
        self.assertEqual(DataDownloaderFactory.get_provider_by_code("alphavantage"), "alphavantage")

        # Test FMP codes
        self.assertEqual(DataDownloaderFactory.get_provider_by_code("fmp"), "fmp")
        self.assertEqual(DataDownloaderFactory.get_provider_by_code("financialmodelingprep"), "fmp")

        # Test Tiingo codes
        self.assertEqual(DataDownloaderFactory.get_provider_by_code("tiingo"), "tiingo")

        # Test invalid code
        self.assertIsNone(DataDownloaderFactory.get_provider_by_code("invalid"))

    def test_get_supported_providers(self):
        """Test getting supported providers."""
        providers = DataDownloaderFactory.get_supported_providers()
        self.assertIsInstance(providers, list)
        self.assertGreater(len(providers), 0)
        self.assertIn("yf", providers)
        self.assertIn("av", providers)
        self.assertIn("bnc", providers)
        self.assertIn("fmp", providers)
        self.assertIn("tiingo", providers)

    def test_get_provider_info(self):
        """Test getting provider information."""
        info = DataDownloaderFactory.get_provider_info()
        self.assertIsInstance(info, dict)
        self.assertIn("yahoo", info)
        self.assertIn("alphavantage", info)
        self.assertIn("binance", info)
        self.assertIn("fmp", info)
        self.assertIn("tiingo", info)


if __name__ == '__main__':
    unittest.main()
