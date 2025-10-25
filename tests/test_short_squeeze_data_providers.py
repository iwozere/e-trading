#!/usr/bin/env python3
"""
Short Squeeze Data Provider Integration Tests
--------------------------------------------

This module tests the short squeeze extensions to FMP and Finnhub data downloaders.
Tests include mock API responses, rate limiting, and error handling scenarios.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.data.downloader.fmp_data_downloader import FMPDataDownloader
from src.data.downloader.finnhub_data_downloader import FinnhubDataDownloader

_logger = setup_logger(__name__)


class TestFMPShortSqueezeExtensions(unittest.TestCase):
    """Test FMP data downloader short squeeze extensions."""

    def setUp(self):
        """Set up test fixtures."""
        self.downloader = FMPDataDownloader(api_key="test_api_key", rate_limit_delay=0.01)

    @patch('src.data.downloader.fmp_data_downloader.requests.get')
    def test_get_short_interest_data_success(self, mock_get):
        """Test successful short interest data retrieval."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [
            {
                'symbol': 'AAPL',
                'date': '2024-01-15',
                'shortInterest': 50000000,
                'sharesOutstanding': 15000000000,
                'shortInterestRatio': 3.33
            }
        ]
        mock_get.return_value = mock_response

        # Test the method
        result = self.downloader.get_short_interest_data('AAPL')

        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['shortInterest'], 50000000)
        self.assertEqual(result['shortInterestRatio'], 3.33)

        # Verify API call
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertIn('short-interest/AAPL', args[0])
        self.assertEqual(kwargs['params']['apikey'], 'test_api_key')

    @patch('src.data.downloader.fmp_data_downloader.requests.get')
    def test_get_short_interest_data_no_data(self, mock_get):
        """Test short interest data retrieval with no data."""
        # Mock empty response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        # Test the method
        result = self.downloader.get_short_interest_data('INVALID')

        # Assertions
        self.assertIsNone(result)

    @patch('src.data.downloader.fmp_data_downloader.requests.get')
    def test_get_float_shares_data_success(self, mock_get):
        """Test successful float shares data retrieval."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [
            {
                'symbol': 'AAPL',
                'companyName': 'Apple Inc.',
                'sharesOutstanding': 15000000000,
                'floatShares': 14500000000,
                'mktCap': 3000000000000,
                'lastUpdated': '2024-01-15T10:00:00Z'
            }
        ]
        mock_get.return_value = mock_response

        # Test the method
        result = self.downloader.get_float_shares_data('AAPL')

        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['floatShares'], 14500000000)
        self.assertEqual(result['sharesOutstanding'], 15000000000)

    @patch('src.data.downloader.fmp_data_downloader.requests.get')
    def test_load_universe_from_screener_success(self, mock_get):
        """Test successful universe loading from screener."""
        # Mock successful screener response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [
            {'symbol': 'AAPL', 'companyName': 'Apple Inc.', 'marketCap': 3000000000000},
            {'symbol': 'MSFT', 'companyName': 'Microsoft Corp.', 'marketCap': 2500000000000},
            {'symbol': 'GOOGL', 'companyName': 'Alphabet Inc.', 'marketCap': 1800000000000}
        ]
        mock_get.return_value = mock_response

        # Test the method
        result = self.downloader.load_universe_from_screener()

        # Assertions
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertIn('AAPL', result)
        self.assertIn('MSFT', result)
        self.assertIn('GOOGL', result)

    @patch('src.data.downloader.fmp_data_downloader.requests.get')
    def test_get_short_squeeze_batch_data_success(self, mock_get):
        """Test successful batch data retrieval."""
        # Mock responses for different endpoints
        def mock_response_side_effect(url, **kwargs):
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None

            if 'profile' in url:
                mock_response.json.return_value = [
                    {
                        'symbol': 'AAPL',
                        'companyName': 'Apple Inc.',
                        'mktCap': 3000000000000,
                        'sharesOutstanding': 15000000000,
                        'floatShares': 14500000000
                    }
                ]
            elif 'short-interest' in url:
                mock_response.json.return_value = [
                    {
                        'symbol': 'AAPL',
                        'shortInterest': 50000000,
                        'shortInterestRatio': 3.33
                    }
                ]
            elif 'key-metrics' in url:
                mock_response.json.return_value = [
                    {
                        'symbol': 'AAPL',
                        'marketCapitalization': 3000000000000,
                        'peRatio': 25.5
                    }
                ]
            else:
                mock_response.json.return_value = {}

            return mock_response

        mock_get.side_effect = mock_response_side_effect

        # Test the method
        result = self.downloader.get_short_squeeze_batch_data(['AAPL'])

        # Assertions
        self.assertIsInstance(result, dict)
        self.assertIn('AAPL', result)
        self.assertIn('profile', result['AAPL'])
        self.assertIn('shortInterest', result['AAPL'])
        self.assertIn('metrics', result['AAPL'])

    @patch('src.data.downloader.fmp_data_downloader.requests.get')
    def test_api_error_handling(self, mock_get):
        """Test API error handling."""
        # Mock API error
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_get.return_value = mock_response

        # Test the method
        result = self.downloader.get_short_interest_data('AAPL')

        # Assertions
        self.assertIsNone(result)

    @patch('src.data.downloader.fmp_data_downloader.time.sleep')
    def test_rate_limiting(self, mock_sleep):
        """Test rate limiting implementation."""
        with patch('src.data.downloader.fmp_data_downloader.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = []
            mock_get.return_value = mock_response

            # Test the method
            self.downloader.get_short_interest_data('AAPL')

            # Verify rate limiting delay was called
            mock_sleep.assert_called_with(self.downloader.rate_limit_delay)


class TestFinnhubShortSqueezeExtensions(unittest.TestCase):
    """Test Finnhub data downloader short squeeze extensions."""

    def setUp(self):
        """Set up test fixtures."""
        self.downloader = FinnhubDataDownloader(api_key="test_api_key")

    @patch('src.data.downloader.finnhub_data_downloader.requests.get')
    def test_get_sentiment_data_success(self, mock_get):
        """Test successful sentiment data retrieval."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'sentiment': {
                'bearishPercent': 0.3,
                'bullishPercent': 0.7
            },
            'buzz': {
                'articlesInLastWeek': 150,
                'weeklyAverage': 100
            },
            'companyNewsScore': 0.8,
            'sectorAverageBullishPercent': 0.6,
            'sectorAverageNewsScore': 0.7
        }
        mock_get.return_value = mock_response

        # Test the method
        result = self.downloader.get_sentiment_data('AAPL')

        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['bullish_percent'], 0.7)
        self.assertEqual(result['bearish_percent'], 0.3)
        self.assertEqual(result['buzz_articles_in_last_week'], 150)

    @patch('src.data.downloader.finnhub_data_downloader.requests.get')
    def test_get_options_data_success(self, mock_get):
        """Test successful options data retrieval."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': [
                {'type': 'call', 'volume': 1000, 'openInterest': 5000},
                {'type': 'call', 'volume': 800, 'openInterest': 3000},
                {'type': 'put', 'volume': 500, 'openInterest': 2000},
                {'type': 'put', 'volume': 300, 'openInterest': 1500}
            ]
        }
        mock_get.return_value = mock_response

        # Test the method
        result = self.downloader.get_options_data('AAPL')

        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['total_call_volume'], 1800)
        self.assertEqual(result['total_put_volume'], 800)
        self.assertEqual(result['total_call_open_interest'], 8000)
        self.assertEqual(result['total_put_open_interest'], 3500)
        self.assertAlmostEqual(result['call_put_volume_ratio'], 2.25)
        self.assertAlmostEqual(result['call_put_oi_ratio'], 8000/3500, places=2)

    @patch('src.data.downloader.finnhub_data_downloader.requests.get')
    def test_get_borrow_rates_data_success(self, mock_get):
        """Test successful borrow rates data retrieval."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'fee': 0.05,
            'available': 1000000,
            'feeRate': 5.0
        }
        mock_get.return_value = mock_response

        # Test the method
        result = self.downloader.get_borrow_rates_data('AAPL')

        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['borrow_fee_rate'], 0.05)
        self.assertEqual(result['available_shares'], 1000000)
        self.assertEqual(result['fee_rate_percentage'], 5.0)

    def test_calculate_call_put_ratio_success(self):
        """Test call/put ratio calculation."""
        # Test data
        options_data = {
            'symbol': 'AAPL',
            'total_call_volume': 1800,
            'total_put_volume': 800
        }

        # Test the method
        result = self.downloader.calculate_call_put_ratio(options_data)

        # Assertions
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 2.25)

    def test_calculate_call_put_ratio_zero_puts(self):
        """Test call/put ratio calculation with zero put volume."""
        # Test data
        options_data = {
            'symbol': 'AAPL',
            'total_call_volume': 1800,
            'total_put_volume': 0
        }

        # Test the method
        result = self.downloader.calculate_call_put_ratio(options_data)

        # Assertions
        self.assertIsNone(result)

    @patch('src.data.downloader.finnhub_data_downloader.requests.get')
    def test_aggregate_24h_sentiment_success(self, mock_get):
        """Test 24-hour sentiment aggregation."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'sentiment': {
                'bearishPercent': 0.3,
                'bullishPercent': 0.7
            },
            'buzz': {
                'articlesInLastWeek': 150,
                'weeklyAverage': 100
            },
            'companyNewsScore': 0.8,
            'sectorAverageBullishPercent': 0.6,
            'sectorAverageNewsScore': 0.7
        }
        mock_get.return_value = mock_response

        # Test the method
        result = self.downloader.aggregate_24h_sentiment('AAPL')

        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertAlmostEqual(result['sentiment_score_24h'], 0.4)  # (0.7 - 0.3) / (0.7 + 0.3)
        self.assertEqual(result['bullish_percent_24h'], 0.7)
        self.assertEqual(result['bearish_percent_24h'], 0.3)
        self.assertAlmostEqual(result['buzz_intensity'], 1.5)  # 150 / 100

    @patch('src.data.downloader.finnhub_data_downloader.requests.get')
    def test_rate_limit_handling(self, mock_get):
        """Test rate limit error handling."""
        # Mock rate limit response
        mock_response = Mock()
        mock_response.status_code = 429
        mock_get.return_value = mock_response

        # Test the method
        result = self.downloader.get_sentiment_data('AAPL')

        # Assertions
        self.assertIsNone(result)

    @patch('src.data.downloader.finnhub_data_downloader.requests.get')
    def test_api_error_handling(self, mock_get):
        """Test API error handling."""
        # Mock API error
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        # Test the method
        result = self.downloader.get_sentiment_data('AAPL')

        # Assertions
        self.assertIsNone(result)

    @patch('src.data.downloader.finnhub_data_downloader.time.sleep')
    @patch('src.data.downloader.finnhub_data_downloader.requests.get')
    def test_batch_data_rate_limiting(self, mock_get, mock_sleep):
        """Test batch data retrieval with rate limiting."""
        # Mock successful responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_get.return_value = mock_response

        # Test the method with multiple tickers
        result = self.downloader.get_short_squeeze_batch_data(['AAPL', 'MSFT'])

        # Verify rate limiting delays were called
        self.assertEqual(mock_sleep.call_count, 2)  # One for each ticker
        for call in mock_sleep.call_args_list:
            self.assertEqual(call[0][0], 1.1)  # 1.1 second delay


class TestDataProviderIntegration(unittest.TestCase):
    """Test integration between data providers."""

    def setUp(self):
        """Set up test fixtures."""
        self.fmp_downloader = FMPDataDownloader(api_key="test_fmp_key", rate_limit_delay=0.01)
        self.finnhub_downloader = FinnhubDataDownloader(api_key="test_finnhub_key")

    def test_combined_data_retrieval(self):
        """Test combining data from both providers."""
        # Test FMP data retrieval separately
        with patch('src.data.downloader.fmp_data_downloader.requests.get') as mock_fmp_get:
            mock_fmp_response = Mock()
            mock_fmp_response.raise_for_status.return_value = None
            mock_fmp_response.json.return_value = [
                {
                    'symbol': 'AAPL',
                    'shortInterest': 50000000,
                    'shortInterestRatio': 3.33
                }
            ]
            mock_fmp_get.return_value = mock_fmp_response

            fmp_data = self.fmp_downloader.get_short_interest_data('AAPL')
            self.assertIsNotNone(fmp_data)
            self.assertEqual(fmp_data['symbol'], 'AAPL')

        # Test Finnhub data retrieval separately
        with patch('src.data.downloader.finnhub_data_downloader.requests.get') as mock_finnhub_get:
            mock_finnhub_response = Mock()
            mock_finnhub_response.status_code = 200
            mock_finnhub_response.json.return_value = {
                'sentiment': {
                    'bearishPercent': 0.3,
                    'bullishPercent': 0.7
                },
                'buzz': {
                    'articlesInLastWeek': 150,
                    'weeklyAverage': 100
                },
                'companyNewsScore': 0.8,
                'sectorAverageBullishPercent': 0.6,
                'sectorAverageNewsScore': 0.7
            }
            mock_finnhub_get.return_value = mock_finnhub_response

            finnhub_data = self.finnhub_downloader.get_sentiment_data('AAPL')
            self.assertIsNotNone(finnhub_data)
            self.assertEqual(finnhub_data['symbol'], 'AAPL')

    def test_error_resilience(self):
        """Test that one provider failure doesn't affect the other."""
        with patch('src.data.downloader.fmp_data_downloader.requests.get') as mock_fmp_get, \
             patch('src.data.downloader.finnhub_data_downloader.requests.get') as mock_finnhub_get:

            # Mock FMP failure
            mock_fmp_response = Mock()
            mock_fmp_response.raise_for_status.side_effect = Exception("FMP API Error")
            mock_fmp_get.return_value = mock_fmp_response

            # Mock Finnhub success
            mock_finnhub_response = Mock()
            mock_finnhub_response.status_code = 200
            mock_finnhub_response.json.return_value = {
                'sentiment': {
                    'bearishPercent': 0.3,
                    'bullishPercent': 0.7
                },
                'buzz': {
                    'articlesInLastWeek': 150,
                    'weeklyAverage': 100
                },
                'companyNewsScore': 0.8,
                'sectorAverageBullishPercent': 0.6,
                'sectorAverageNewsScore': 0.7
            }
            mock_finnhub_get.return_value = mock_finnhub_response

            # Get data from both providers
            fmp_data = self.fmp_downloader.get_short_interest_data('AAPL')
            finnhub_data = self.finnhub_downloader.get_sentiment_data('AAPL')

            # Assertions
            self.assertIsNone(fmp_data)  # FMP failed
            self.assertIsNotNone(finnhub_data)  # Finnhub succeeded


def run_tests():
    """Run all data provider integration tests."""
    print("🚀 Starting Short Squeeze Data Provider Integration Tests")
    print("=" * 60)

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_classes = [
        TestFMPShortSqueezeExtensions,
        TestFinnhubShortSqueezeExtensions,
        TestDataProviderIntegration
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print("\n" + "=" * 60)
    print(f"📊 Test Results:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("🎉 All tests passed! Data provider extensions are working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)