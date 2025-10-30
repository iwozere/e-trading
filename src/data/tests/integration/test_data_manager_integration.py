#!/usr/bin/env python3
"""
Integration Tests for DataManager

This module contains comprehensive integration tests for the new DataManager architecture,
testing the complete data retrieval flow including caching, provider selection, and data validation.

Test Coverage:
- Cache hit/miss scenarios
- Provider selection and failover
- Data validation and quality scoring
- Live feed integration
- Error handling and retries
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_manager import DataManager, ProviderSelector
from src.data.cache.unified_cache import UnifiedCache
from src.data.downloader.binance_data_downloader import BinanceDataDownloader
from src.data.downloader.yahoo_data_downloader import YahooDataDownloader
from src.data.feed.base_live_data_feed import BaseLiveDataFeed
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class TestDataManagerIntegration:
    """Integration tests for DataManager architecture."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def data_manager(self, temp_cache_dir):
        """Create a DataManager instance with temporary cache."""
        return DataManager(cache_dir=temp_cache_dir)

    @pytest.fixture
    def mock_downloader(self):
        """Create a mock downloader for testing."""
        downloader = Mock()
        downloader.get_ohlcv.return_value = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [101.0, 102.0, 103.0],
            'low': [99.0, 100.0, 101.0],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1h'))
        return downloader

    def test_cache_miss_then_hit(self, data_manager, mock_downloader):
        """Test that first request is a cache miss, second is a cache hit."""
        symbol = "BTCUSDT"
        timeframe = "1h"
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 1, 3)  # End at 3 AM to include the 3 data points

        # Mock the provider selector to return our mock downloader
        with patch.object(data_manager.provider_selector, 'get_provider_with_failover') as mock_failover:
            mock_failover.return_value = ['test_provider']

            with patch.dict(data_manager.provider_selector.downloaders, {
                'test_provider': mock_downloader
            }):
                # First request - should be cache miss
                result1 = data_manager.get_ohlcv(symbol, timeframe, start_date, end_date)

                # Verify downloader was called
                mock_downloader.get_ohlcv.assert_called_once()

                # Second request - should be cache hit
                result2 = data_manager.get_ohlcv(symbol, timeframe, start_date, end_date)


                # Verify downloader was not called again
                assert mock_downloader.get_ohlcv.call_count == 1

                # Verify results are identical (ignore frequency differences due to caching)
                pd.testing.assert_frame_equal(result1, result2, check_freq=False)

    def test_provider_selection_crypto(self, data_manager):
        """Test that crypto symbols select appropriate providers."""
        # Test crypto symbol
        downloader = data_manager.provider_selector.get_best_downloader("BTCUSDT", "1h")
        assert downloader is not None
        assert hasattr(downloader, 'get_ohlcv')

    def test_provider_selection_stock(self, data_manager):
        """Test that stock symbols select appropriate providers."""
        # Test stock symbol
        downloader = data_manager.provider_selector.get_best_downloader("AAPL", "1d")
        assert downloader is not None
        assert hasattr(downloader, 'get_ohlcv')

    def test_provider_failover(self, data_manager):
        """Test provider failover mechanism."""
        # Create mock downloaders
        primary_downloader = Mock()
        primary_downloader.get_ohlcv.side_effect = Exception("API Error")

        fallback_downloader = Mock()
        fallback_downloader.get_ohlcv.return_value = pd.DataFrame({
            'open': [100.0], 'high': [101.0], 'low': [99.0],
            'close': [100.5], 'volume': [1000]
        }, index=pd.date_range('2024-01-01', periods=1, freq='1h'))

        # Mock provider selector to return failover chain
        with patch.object(data_manager.provider_selector, 'get_provider_with_failover') as mock_failover:
            mock_failover.return_value = ['primary', 'fallback']

            # Mock the downloaders dictionary
            with patch.dict(data_manager.provider_selector.downloaders, {
                'primary': primary_downloader,
                'fallback': fallback_downloader
            }):
                result = data_manager.get_ohlcv("TEST", "1h", datetime(2024, 1, 1), datetime(2024, 1, 2))

                # Verify primary was tried first
                primary_downloader.get_ohlcv.assert_called_once()

                # Verify fallback was used
                fallback_downloader.get_ohlcv.assert_called_once()

                # Verify result is from fallback
                assert result is not None
                assert len(result) == 1

    def test_data_validation(self, data_manager, mock_downloader):
        """Test that data validation is performed."""
        # Create invalid data
        invalid_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [99.0, 100.0, 101.0],  # High < Open (invalid)
            'low': [99.0, 100.0, 101.0],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1h'))

        mock_downloader.get_ohlcv.return_value = invalid_data

        with patch.object(data_manager.provider_selector, 'get_provider_with_failover') as mock_failover:
            mock_failover.return_value = ['test_provider']

            with patch.dict(data_manager.provider_selector.downloaders, {
                'test_provider': mock_downloader
            }):
                # Should still return data but log validation warnings
                result = data_manager.get_ohlcv("TEST", "1h", datetime(2024, 1, 1), datetime(2024, 1, 2))

                assert result is not None
                assert len(result) == 3

    def test_live_feed_integration(self, data_manager, mock_downloader):
        """Test that live feeds use DataManager for historical data."""
        with patch.object(data_manager.provider_selector, 'get_provider_with_failover') as mock_failover:
            mock_failover.return_value = ['test_provider']

            with patch.dict(data_manager.provider_selector.downloaders, {
                'test_provider': mock_downloader
            }):
                # Create a mock live feed
                class TestLiveFeed(BaseLiveDataFeed):
                    def _connect_realtime(self):
                        return True

                    def _disconnect_realtime(self):
                        pass

                    def _fetch_realtime_data(self):
                        return None

                # Test that live feed can be created with DataManager
                feed = TestLiveFeed(
                    symbol="BTCUSDT",
                    interval="1h",
                    lookback_bars=10,
                    data_manager=data_manager
                )

                assert feed.data_manager == data_manager
                assert feed.symbol == "BTCUSDT"
                assert feed.interval == "1h"

    def test_error_handling(self, data_manager):
        """Test error handling when all providers fail."""
        # Create a downloader that always fails
        failing_downloader = Mock()
        failing_downloader.get_ohlcv.side_effect = Exception("All providers failed")

        with patch.object(data_manager.provider_selector, 'get_provider_with_failover') as mock_failover:
            mock_failover.return_value = ['failing_provider']

            with patch.dict(data_manager.provider_selector.downloaders, {
                'failing_provider': failing_downloader
            }):
                # Should raise RuntimeError when all providers fail
                with pytest.raises(RuntimeError, match="All providers failed"):
                    data_manager.get_ohlcv("TEST", "1h", datetime(2024, 1, 1), datetime(2024, 1, 2))

    def test_cache_structure(self, data_manager, mock_downloader, temp_cache_dir):
        """Test that cache files are created with correct structure."""
        symbol = "BTCUSDT"
        timeframe = "1h"
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 2)

        with patch.object(data_manager.provider_selector, 'get_provider_with_failover') as mock_failover:
            mock_failover.return_value = ['test_provider']

            with patch.dict(data_manager.provider_selector.downloaders, {
                'test_provider': mock_downloader
            }):
                # Make a request
                data_manager.get_ohlcv(symbol, timeframe, start_date, end_date)

                # Check cache structure (new structure: ohlcv/symbol/timeframe)
                cache_path = Path(temp_cache_dir) / "ohlcv" / symbol / timeframe
                assert cache_path.exists()

                # Check for data file
                data_files = list(cache_path.glob("*.csv.gz"))
                assert len(data_files) > 0

                # Check for metadata file
                metadata_files = list(cache_path.glob("*.metadata.json"))
                assert len(metadata_files) > 0

    def test_provider_selector_configuration(self, data_manager):
        """Test that ProviderSelector uses configuration correctly."""
        # Test symbol classification
        crypto_info = data_manager.provider_selector.get_ticker_info("BTCUSDT")
        assert crypto_info['symbol_type'] == 'crypto'

        stock_info = data_manager.provider_selector.get_ticker_info("AAPL")
        assert stock_info['symbol_type'] == 'stock'

        # Test provider config generation
        crypto_config = data_manager.provider_selector.get_data_provider_config("BTCUSDT", "1h")
        assert 'provider' in crypto_config

        stock_config = data_manager.provider_selector.get_data_provider_config("AAPL", "1d")
        assert 'provider' in stock_config


class TestProviderSelectorIntegration:
    """Integration tests for ProviderSelector."""

    def test_symbol_classification_rules(self):
        """Test that symbol classification rules work correctly."""
        selector = ProviderSelector()

        # Test crypto symbols
        assert selector._classify_symbol("BTCUSDT") == "crypto"
        assert selector._classify_symbol("ETHUSDT") == "crypto"
        assert selector._classify_symbol("ADAUSDT") == "crypto"

        # Test stock symbols
        assert selector._classify_symbol("AAPL") == "stock"
        assert selector._classify_symbol("MSFT") == "stock"
        assert selector._classify_symbol("GOOGL") == "stock"

        # Test stock with exchange suffix
        assert selector._classify_symbol("AAPL.NASDAQ") == "stock"
        assert selector._classify_symbol("MSFT.NYSE") == "stock"

    def test_provider_selection_rules(self):
        """Test that provider selection follows configured rules."""
        selector = ProviderSelector()

        # Test crypto provider selection
        crypto_provider = selector.get_best_provider("BTCUSDT", "1h")
        assert crypto_provider is not None

        # Test stock provider selection
        stock_provider = selector.get_best_provider("AAPL", "1d")
        assert stock_provider is not None

    def test_ticker_validation(self):
        """Test comprehensive ticker validation."""
        selector = ProviderSelector()

        # Test valid tickers
        btc_validation = selector.validate_ticker("BTCUSDT")
        assert btc_validation['valid'] == True
        assert btc_validation['symbol_type'] == 'crypto'

        aapl_validation = selector.validate_ticker("AAPL")
        assert aapl_validation['valid'] == True
        assert aapl_validation['symbol_type'] == 'stock'

        # Test invalid ticker
        invalid_validation = selector.validate_ticker("INVALID")
        assert invalid_validation['valid'] == False


def run_integration_tests():
    """Run all integration tests."""
    print("Running DataManager Integration Tests...")
    print("=" * 50)

    # Run pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--color=yes"
    ])


if __name__ == "__main__":
    run_integration_tests()
