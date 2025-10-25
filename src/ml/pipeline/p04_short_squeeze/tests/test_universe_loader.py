"""
Unit tests for Universe Loader module.

Tests the universe loading functionality including filtering, caching, and validation.
"""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from datetime import datetime, timedelta

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p04_short_squeeze.core.universe_loader import UniverseLoader, create_universe_loader
from src.ml.pipeline.p04_short_squeeze.config.data_classes import UniverseConfig
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class TestUniverseLoader(unittest.TestCase):
    """Test cases for Universe Loader."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_fmp_downloader = Mock()
        self.universe_config = UniverseConfig(
            min_market_cap=100_000_000,
            max_market_cap=10_000_000_000,
            min_avg_volume=200_000,
            exchanges=['NYSE', 'NASDAQ']
        )

        # Create temporary cache directory
        self.temp_dir = tempfile.mkdtemp()

        # Create universe loader with mocked downloader
        self.universe_loader = UniverseLoader(self.mock_fmp_downloader, self.universe_config)
        # Override cache directory for testing
        self.universe_loader._cache_dir = Path(self.temp_dir) / "universe"
        self.universe_loader._cache_dir.mkdir(parents=True, exist_ok=True)

    def test_load_universe_from_screener_success(self):
        """Test successful universe loading from screener."""
        # Mock screener response
        mock_tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        self.mock_fmp_downloader.load_universe_from_screener.return_value = mock_tickers

        # Load universe
        result = self.universe_loader.load_universe()

        # Assertions
        self.assertEqual(len(result), 5)
        self.assertIn('AAPL', result)
        self.assertIn('GOOGL', result)
        self.mock_fmp_downloader.load_universe_from_screener.assert_called_once()

    def test_load_universe_empty_response(self):
        """Test universe loading with empty screener response."""
        # Mock empty response
        self.mock_fmp_downloader.load_universe_from_screener.return_value = []

        # Load universe
        result = self.universe_loader.load_universe()

        # Assertions
        self.assertEqual(len(result), 0)
        self.mock_fmp_downloader.load_universe_from_screener.assert_called_once()

    def test_load_universe_with_cache(self):
        """Test universe loading from cache."""
        # Create cache file
        cache_data = {
            'tickers': ['AAPL', 'GOOGL', 'MSFT'],
            'created_at': datetime.now().isoformat(),
            'config': {
                'min_market_cap': self.universe_config.min_market_cap,
                'max_market_cap': self.universe_config.max_market_cap,
                'min_avg_volume': self.universe_config.min_avg_volume,
                'exchanges': self.universe_config.exchanges
            }
        }

        cache_file = self.universe_loader._cache_dir / "universe_cache.json"
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)

        # Load universe (should use cache)
        result = self.universe_loader.load_universe()

        # Assertions
        self.assertEqual(len(result), 3)
        self.assertIn('AAPL', result)
        # Should not call screener since cache is used
        self.mock_fmp_downloader.load_universe_from_screener.assert_not_called()

    def test_load_universe_expired_cache(self):
        """Test universe loading with expired cache."""
        # Create expired cache file
        old_time = datetime.now() - timedelta(hours=25)  # Older than 24 hours
        cache_data = {
            'tickers': ['OLD1', 'OLD2'],
            'created_at': old_time.isoformat(),
            'config': {}
        }

        cache_file = self.universe_loader._cache_dir / "universe_cache.json"
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)

        # Set file modification time to old time
        import os
        os.utime(cache_file, (old_time.timestamp(), old_time.timestamp()))

        # Mock fresh screener response
        mock_tickers = ['NEW1', 'NEW2', 'NEW3']
        self.mock_fmp_downloader.load_universe_from_screener.return_value = mock_tickers

        # Load universe (should ignore expired cache)
        result = self.universe_loader.load_universe()

        # Assertions
        self.assertEqual(len(result), 3)
        self.assertIn('NEW1', result)
        self.assertNotIn('OLD1', result)
        self.mock_fmp_downloader.load_universe_from_screener.assert_called_once()

    def test_is_valid_ticker(self):
        """Test ticker validation logic."""
        # Valid tickers
        self.assertTrue(self.universe_loader._is_valid_ticker('AAPL'))
        self.assertTrue(self.universe_loader._is_valid_ticker('GOOGL'))
        self.assertTrue(self.universe_loader._is_valid_ticker('MSFT'))
        self.assertTrue(self.universe_loader._is_valid_ticker('A'))

        # Invalid tickers
        self.assertFalse(self.universe_loader._is_valid_ticker(''))
        self.assertFalse(self.universe_loader._is_valid_ticker('   '))
        self.assertFalse(self.universe_loader._is_valid_ticker('TOOLONG'))
        self.assertFalse(self.universe_loader._is_valid_ticker('TEST.PR'))
        self.assertFalse(self.universe_loader._is_valid_ticker('TEST-A'))
        self.assertFalse(self.universe_loader._is_valid_ticker('TESTWARR'))

    def test_filter_by_market_cap(self):
        """Test market cap filtering."""
        # Mock market cap data
        def mock_get_market_cap_data(ticker):
            market_caps = {
                'AAPL': {'marketCap': 2_000_000_000},  # $2B - within range
                'SMALL': {'marketCap': 50_000_000},    # $50M - below min
                'LARGE': {'marketCap': 15_000_000_000}, # $15B - above max
                'GOOD': {'marketCap': 500_000_000}     # $500M - within range
            }
            return market_caps.get(ticker)

        self.mock_fmp_downloader.get_market_cap_data.side_effect = mock_get_market_cap_data

        # Test filtering
        tickers = ['AAPL', 'SMALL', 'LARGE', 'GOOD']
        result = self.universe_loader.filter_by_market_cap(
            tickers, 100_000_000, 10_000_000_000
        )

        # Assertions
        self.assertEqual(len(result), 2)
        self.assertIn('AAPL', result)
        self.assertIn('GOOD', result)
        self.assertNotIn('SMALL', result)
        self.assertNotIn('LARGE', result)

    def test_filter_by_volume(self):
        """Test volume filtering."""
        import pandas as pd

        # Mock volume data
        def mock_get_ohlcv(ticker, interval, start_date, end_date):
            volumes = {
                'HIGH_VOL': pd.DataFrame({
                    'volume': [300_000, 350_000, 400_000, 320_000, 380_000] * 6  # 30 days
                }),
                'LOW_VOL': pd.DataFrame({
                    'volume': [50_000, 60_000, 45_000, 55_000, 48_000] * 6  # 30 days
                })
            }
            return volumes.get(ticker)

        self.mock_fmp_downloader.get_ohlcv.side_effect = mock_get_ohlcv

        # Test filtering
        tickers = ['HIGH_VOL', 'LOW_VOL']
        result = self.universe_loader.filter_by_volume(tickers, 200_000)

        # Assertions
        self.assertEqual(len(result), 1)
        self.assertIn('HIGH_VOL', result)
        self.assertNotIn('LOW_VOL', result)

    def test_cache_operations(self):
        """Test cache save and load operations."""
        # Test saving to cache
        test_tickers = ['AAPL', 'GOOGL', 'MSFT']
        self.universe_loader._save_to_cache(test_tickers)

        # Test loading from cache
        loaded_tickers = self.universe_loader._load_from_cache()
        self.assertEqual(loaded_tickers, test_tickers)

        # Test cache info
        cache_info = self.universe_loader.get_cache_info()
        self.assertTrue(cache_info['exists'])
        self.assertEqual(cache_info['size'], 3)
        self.assertFalse(cache_info['expired'])

        # Test clearing cache
        self.universe_loader.clear_cache()
        cache_info_after_clear = self.universe_loader.get_cache_info()
        self.assertFalse(cache_info_after_clear['exists'])

    def test_factory_function(self):
        """Test the factory function."""
        loader = create_universe_loader(self.mock_fmp_downloader, self.universe_config)

        self.assertIsInstance(loader, UniverseLoader)
        self.assertEqual(loader.fmp_downloader, self.mock_fmp_downloader)
        self.assertEqual(loader.config, self.universe_config)

    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test screener failure
        self.mock_fmp_downloader.load_universe_from_screener.side_effect = Exception("API Error")

        result = self.universe_loader.load_universe()
        self.assertEqual(len(result), 0)

        # Test market cap filter with API errors
        self.mock_fmp_downloader.get_market_cap_data.side_effect = Exception("API Error")

        tickers = ['AAPL', 'GOOGL']
        result = self.universe_loader.filter_by_market_cap(tickers, 100_000_000, 10_000_000_000)
        # Should return empty list when all API calls fail
        self.assertEqual(len(result), 0)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()