"""
Comprehensive unit tests for all live data feeds.

This module tests:
- All live feed implementations
- Connection handling
- Data streaming
- Error handling and reconnection
- Backtrader integration
- WebSocket and polling mechanisms

How to run:
    pytest tests/test_live_feeds.py -v

Or to run specific test:
    pytest tests/test_live_feeds.py::TestBinanceLiveFeed::test_connection -v
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import tempfile
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

# Import all live feeds
from src.data.feed.binance_live_feed import BinanceLiveDataFeed
from src.data.feed.yahoo_live_feed import YahooLiveDataFeed
from src.data.feed.ibkr_live_feed import IBKRLiveDataFeed
from src.data.feed.coingecko_live_feed import CoinGeckoLiveDataFeed
from src.data.feed.data_feed_factory import DataFeedFactory
from src.data.feed.base_live_data_feed import BaseLiveDataFeed


class TestLiveFeeds(unittest.TestCase):
    """Base class for live feed tests."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_symbol = "AAPL"
        self.interval = "1d"
        self.lookback_bars = 10

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def assert_valid_ohlcv_data(self, df):
        """Assert that OHLCV data is valid."""
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, df.columns)

        # Check data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(df[col]))

    def assert_valid_status(self, status):
        """Assert that status dictionary is valid."""
        self.assertIsInstance(status, dict)
        required_keys = ['symbol', 'interval', 'is_connected', 'data_source']
        for key in required_keys:
            self.assertIn(key, status)


class TestBinanceLiveFeed(TestLiveFeeds):
    """Test Binance live data feed."""

    def setUp(self):
        super().setUp()
        self.api_key = "test_api_key"
        self.api_secret = "test_api_secret"

    @patch('binance.client.Client.get_historical_klines')
    def test_historical_data_loading(self, mock_klines):
        """Test historical data loading."""
        # Mock historical klines
        mock_klines.return_value = [
            [1672531200000, "150.0", "155.0", "149.0", "151.0", "1000000", 1672617600000, "0", "0", "0", "0", "0"],
            [1672617600000, "151.0", "156.0", "150.0", "152.0", "1100000", 1672704000000, "0", "0", "0", "0", "0"]
        ]

        feed = BinanceLiveDataFeed(
            symbol="BTCUSDT",
            interval=self.interval,
            api_key=self.api_key,
            api_secret=self.api_secret,
            lookback_bars=self.lookback_bars
        )

        # Check that historical data was loaded
        self.assertIsInstance(feed.df, pd.DataFrame)
        self.assertGreater(len(feed.df), 0)
        self.assert_valid_ohlcv_data(feed.df)

    @patch('websockets.connect')
    @patch('binance.client.Client.get_historical_klines')
    def test_websocket_connection(self, mock_klines, mock_websockets_connect):
        """Test WebSocket connection."""
        # Mock historical data
        mock_klines.return_value = [
            [1672531200000, "150.0", "155.0", "149.0", "151.0", "1000000", 1672617600000, "0", "0", "0", "0", "0"]
        ]

        # Mock WebSocket connection
        mock_ws = MagicMock()
        mock_websockets_connect.return_value.__aenter__.return_value = mock_ws

        feed = BinanceLiveDataFeed(
            symbol="BTCUSDT",
            interval=self.interval,
            api_key=self.api_key,
            api_secret=self.api_secret,
            lookback_bars=self.lookback_bars
        )

        # Test connection
        connected = feed._connect_realtime()
        self.assertTrue(connected)
        mock_websockets_connect.assert_called_once()

    @patch('binance.client.Client.get_historical_klines')
    def test_interval_conversion(self, mock_klines):
        """Test interval conversion."""
        mock_klines.return_value = []

        feed = BinanceLiveDataFeed(
            symbol="BTCUSDT",
            interval="1h",
            api_key=self.api_key,
            api_secret=self.api_secret
        )

        self.assertEqual(feed.binance_interval, "1h")

    @patch('binance.client.Client.get_historical_klines')
    def test_get_status(self, mock_klines):
        """Test status retrieval."""
        mock_klines.return_value = []

        feed = BinanceLiveDataFeed(
            symbol="BTCUSDT",
            interval=self.interval,
            api_key=self.api_key,
            api_secret=self.api_secret
        )

        status = feed.get_status()
        self.assert_valid_status(status)
        self.assertEqual(status['symbol'], "BTCUSDT")
        self.assertEqual(status['interval'], self.interval)
        self.assertEqual(status['data_source'], "Binance")


class TestYahooLiveFeed(TestLiveFeeds):
    """Test Yahoo Finance live data feed."""

    @patch('yfinance.download')
    def test_historical_data_loading(self, mock_download):
        """Test historical data loading."""
        # Mock yfinance response
        mock_data = pd.DataFrame({
            'Open': [150.0, 151.0, 152.0],
            'High': [155.0, 156.0, 157.0],
            'Low': [149.0, 150.0, 151.0],
            'Close': [151.0, 152.0, 153.0],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=3, freq='D'))

        mock_download.return_value = mock_data

        feed = YahooLiveDataFeed(
            symbol="BTCUSDT",
            interval=self.interval,
            lookback_bars=self.lookback_bars
        )

        # Check that historical data was loaded
        self.assertIsInstance(feed.df, pd.DataFrame)
        self.assertGreater(len(feed.df), 0)
        self.assert_valid_ohlcv_data(feed.df)

    @patch('yfinance.download')
    @patch('yfinance.Ticker')
    def test_polling_mechanism(self, mock_ticker, mock_download):
        """Test polling mechanism."""
        # Mock historical data
        mock_data = pd.DataFrame({
            'Open': [150.0],
            'High': [155.0],
            'Low': [149.0],
            'Close': [151.0],
            'Volume': [1000000]
        }, index=pd.date_range('2023-01-01', periods=1, freq='D'))
        mock_download.return_value = mock_data

        # Mock ticker for polling
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {
            'regularMarketPrice': 150.0,
            'regularMarketVolume': 1000000
        }
        # Mock the history method to return data with a future timestamp
        mock_ticker_instance.history.return_value = pd.DataFrame({
            'Open': [150.0],
            'High': [155.0],
            'Low': [149.0],
            'Close': [151.0],
            'Volume': [1000000]
        }, index=pd.date_range('2025-09-07', periods=1, freq='D'))  # Future date to ensure new data
        mock_ticker.return_value = mock_ticker_instance

        feed = YahooLiveDataFeed(
            symbol="BTCUSDT",
            interval=self.interval,
            polling_interval=1  # Short interval for testing
        )

        # Initialize the ticker by connecting
        feed._connect_realtime()

        # Test polling
        latest_data = feed._get_latest_data()
        self.assertIsInstance(latest_data, pd.DataFrame)

    @patch('yfinance.download')
    def test_get_status(self, mock_download):
        """Test status retrieval."""
        mock_data = pd.DataFrame({
            'Open': [150.0],
            'High': [155.0],
            'Low': [149.0],
            'Close': [151.0],
            'Volume': [1000000]
        }, index=pd.date_range('2023-01-01', periods=1, freq='D'))
        mock_download.return_value = mock_data

        feed = YahooLiveDataFeed(
            symbol="BTCUSDT",
            interval=self.interval
        )

        status = feed.get_status()
        self.assert_valid_status(status)
        self.assertEqual(status['symbol'], "BTCUSDT")
        self.assertEqual(status['interval'], self.interval)
        self.assertEqual(status['data_source'], "Yahoo Finance")


class TestIBKRLiveFeed(TestLiveFeeds):
    """Test IBKR live data feed."""

    def setUp(self):
        super().setUp()
        self.host = "127.0.0.1"
        self.port = 7497
        self.client_id = 1

    @patch('ibapi.client.EClient.connect')
    @patch('ib_insync.ib.IB.reqHistoricalData')
    @patch('ib_insync.ib.IB.reqContractDetails')
    @patch('ib_insync.ib.IB.connect')
    def test_connection(self, mock_ib_connect, mock_req_contract, mock_req_historical, mock_connect):
        """Test IBKR connection."""
        # Mock connection
        mock_connect.return_value = None
        mock_ib_connect.return_value = None
        # Mock contract details
        mock_req_contract.return_value = True
        # Mock historical data with some sample data
        from ib_insync import BarData
        from datetime import datetime
        mock_bars = [
            BarData(
                date=datetime.now(),
                open=150.0,
                high=155.0,
                low=149.0,
                close=154.0,
                volume=1000000,
                barCount=1,
                average=152.0
            )
        ]
        mock_req_historical.return_value = mock_bars

        feed = IBKRLiveDataFeed(
            symbol=self.test_symbol,
            interval=self.interval,
            host=self.host,
            port=self.port,
            client_id=self.client_id,
            lookback_bars=self.lookback_bars
        )

        # Test connection
        connected = feed._connect_realtime()
        self.assertTrue(connected)
        mock_connect.assert_called_once_with(self.host, self.port, self.client_id)

    @patch('ibapi.client.EClient.connect')
    @patch('ib_insync.ib.IB.reqHistoricalData')
    @patch('ib_insync.ib.IB.reqContractDetails')
    @patch('ib_insync.ib.IB.connect')
    def test_get_status(self, mock_ib_connect, mock_req_contract, mock_req_historical, mock_connect):
        """Test status retrieval."""
        mock_connect.return_value = None
        mock_ib_connect.return_value = None
        # Mock contract details
        mock_req_contract.return_value = True
        # Mock historical data with some sample data
        from ib_insync import BarData
        from datetime import datetime
        mock_bars = [
            BarData(
                date=datetime.now(),
                open=150.0,
                high=155.0,
                low=149.0,
                close=154.0,
                volume=1000000,
                barCount=1,
                average=152.0
            )
        ]
        mock_req_historical.return_value = mock_bars

        feed = IBKRLiveDataFeed(
            symbol=self.test_symbol,
            interval=self.interval,
            host=self.host,
            port=self.port,
            client_id=self.client_id
        )

        status = feed.get_status()
        self.assert_valid_status(status)
        self.assertEqual(status['symbol'], self.test_symbol)
        self.assertEqual(status['interval'], self.interval)
        self.assertEqual(status['data_source'], "IBKR")


class TestCoinGeckoLiveFeed(TestLiveFeeds):
    """Test CoinGecko live data feed."""

    @patch('requests.get')
    def test_historical_data_loading(self, mock_get):
        """Test historical data loading."""
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

        feed = CoinGeckoLiveDataFeed(
            symbol="bitcoin",
            interval=self.interval,
            lookback_bars=self.lookback_bars
        )

        # Check that historical data was loaded
        self.assertIsInstance(feed.df, pd.DataFrame)
        self.assertGreater(len(feed.df), 0)
        self.assert_valid_ohlcv_data(feed.df)

    @patch('requests.get')
    def test_polling_mechanism(self, mock_get):
        """Test polling mechanism."""
        # Mock historical data
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "prices": [[1672531200000, 150.0]],
            "total_volumes": [[1672531200000, 1000000]]
        }
        mock_get.return_value = mock_response

        feed = CoinGeckoLiveDataFeed(
            symbol="bitcoin",
            interval=self.interval,
            polling_interval=1  # Short interval for testing
        )

        # Mock current price API call
        mock_response.json.return_value = {
            "bitcoin": {
                "usd": 150.0,
                "usd_24h_vol": 24000000,
                "last_updated_at": 1672531200
            }
        }

        # Test polling
        latest_data = feed._get_latest_data()
        self.assertIsInstance(latest_data, pd.DataFrame)

    @patch('requests.get')
    def test_rate_limiting(self, mock_get):
        """Test rate limiting mechanism."""
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "prices": [[1672531200000, 150.0]],
            "total_volumes": [[1672531200000, 1000000]]
        }
        mock_get.return_value = mock_response

        feed = CoinGeckoLiveDataFeed(
            symbol="bitcoin",
            interval=self.interval
        )

        # Test rate limiting
        self.assertEqual(feed.max_calls_per_minute, 50)
        self.assertIsInstance(feed.call_times, list)

    @patch('requests.get')
    def test_get_status(self, mock_get):
        """Test status retrieval."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "prices": [[1672531200000, 150.0]],
            "total_volumes": [[1672531200000, 1000000]]
        }
        mock_get.return_value = mock_response

        feed = CoinGeckoLiveDataFeed(
            symbol="bitcoin",
            interval=self.interval
        )

        status = feed.get_status()
        self.assert_valid_status(status)
        self.assertEqual(status['symbol'], "bitcoin")
        self.assertEqual(status['interval'], self.interval)
        self.assertEqual(status['data_source'], "CoinGecko")


class TestDataFeedFactory(unittest.TestCase):
    """Test DataFeedFactory."""

    def test_create_binance_feed(self):
        """Test creating Binance feed."""
        config = {
            "data_source": "binance",
            "symbol": "BTCUSDT",
            "interval": "1d",
            "api_key": "test_key",
            "api_secret": "test_secret"
        }

        with patch('binance.client.Client.get_historical_klines') as mock_klines:
            mock_klines.return_value = []
            feed = DataFeedFactory.create_data_feed(config)

            self.assertIsInstance(feed, BinanceLiveDataFeed)
            self.assertEqual(feed.symbol, "BTCUSDT")

    def test_create_yahoo_feed(self):
        """Test creating Yahoo feed."""
        config = {
            "data_source": "yahoo",
            "symbol": "AAPL",
            "interval": "1d"
        }

        with patch('yfinance.download') as mock_download:
            mock_download.return_value = pd.DataFrame()
            feed = DataFeedFactory.create_data_feed(config)

            self.assertIsInstance(feed, YahooLiveDataFeed)
            self.assertEqual(feed.symbol, "AAPL")

    def test_create_ibkr_feed(self):
        """Test creating IBKR feed."""
        config = {
            "data_source": "ibkr",
            "symbol": "AAPL",
            "interval": "1d",
            "host": "127.0.0.1",
            "port": 7497,
            "client_id": 1
        }

        with patch('ibapi.client.EClient.connect'), \
             patch('ib_insync.ib.IB.reqContractDetails') as mock_req_contract, \
             patch('ib_insync.ib.IB.reqHistoricalData') as mock_req_historical, \
             patch('ib_insync.ib.IB.sleep'):
            # Mock the contract details response
            mock_req_contract.return_value = True
            # Mock historical data response with some sample data
            from ib_insync import BarData
            from datetime import datetime
            mock_bars = [
                BarData(
                    date=datetime.now(),
                    open=150.0,
                    high=155.0,
                    low=149.0,
                    close=154.0,
                    volume=1000000,
                    barCount=1,
                    average=152.0
                )
            ]
            mock_req_historical.return_value = mock_bars

            feed = DataFeedFactory.create_data_feed(config)

            self.assertIsInstance(feed, IBKRLiveDataFeed)
            self.assertEqual(feed.symbol, "AAPL")

    def test_create_coingecko_feed(self):
        """Test creating CoinGecko feed."""
        config = {
            "data_source": "coingecko",
            "symbol": "bitcoin",
            "interval": "1d"
        }

        # Mock the historical data loading to avoid network calls
        mock_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1d'),
            'open': [100.0] * 10,
            'high': [110.0] * 10,
            'low': [90.0] * 10,
            'close': [105.0] * 10,
            'volume': [1000000] * 10
        })
        mock_df = mock_df.set_index('timestamp')

        with patch.object(CoinGeckoLiveDataFeed, '_load_historical_data', return_value=mock_df):
            feed = DataFeedFactory.create_data_feed(config)

            self.assertIsInstance(feed, CoinGeckoLiveDataFeed)
            self.assertEqual(feed.symbol, "bitcoin")

    def test_invalid_data_source(self):
        """Test creating feed with invalid data source."""
        config = {
            "data_source": "invalid",
            "symbol": "AAPL",
            "interval": "1d"
        }

        feed = DataFeedFactory.create_data_feed(config)
        self.assertIsNone(feed)

    def test_get_supported_sources(self):
        """Test getting supported sources."""
        sources = DataFeedFactory.get_supported_sources()
        self.assertIsInstance(sources, list)
        self.assertIn("binance", sources)
        self.assertIn("yahoo", sources)
        self.assertIn("ibkr", sources)
        self.assertIn("coingecko", sources)

    def test_get_source_info(self):
        """Test getting source information."""
        info = DataFeedFactory.get_source_info()
        self.assertIsInstance(info, dict)
        self.assertIn("binance", info)
        self.assertIn("yahoo", info)
        self.assertIn("ibkr", info)
        self.assertIn("coingecko", info)


class TestBaseLiveDataFeed(unittest.TestCase):
    """Test BaseLiveDataFeed functionality."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_abstract_methods(self):
        """Test that abstract methods are properly defined."""
        # Test that BaseLiveDataFeed cannot be instantiated directly
        with self.assertRaises(TypeError):
            BaseLiveDataFeed("AAPL", "1d")

    def test_common_functionality(self):
        """Test common functionality through a concrete implementation."""
        with patch('yfinance.download') as mock_download:
            mock_download.return_value = pd.DataFrame({
                'Open': [150.0],
                'High': [155.0],
                'Low': [149.0],
                'Close': [151.0],
                'Volume': [1000000]
            }, index=pd.date_range('2023-01-01', periods=1, freq='D'))

            feed = YahooLiveDataFeed(
                symbol="BTCUSDT",
                interval="1d",
                lookback_bars=1
            )

            # Test common properties
            self.assertEqual(feed.symbol, "BTCUSDT")
            self.assertEqual(feed.interval, "1d")
            self.assertEqual(feed.lookback_bars, 1)
            self.assertFalse(feed.is_connected)
            self.assertFalse(feed.should_stop)


if __name__ == '__main__':
    unittest.main()
