#!/usr/bin/env python3
"""
Test Enhanced Fundamentals Implementation

This test validates the enhanced DataManager fundamentals integration.
"""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.data.data_manager import DataManager


class TestEnhancedFundamentals(unittest.TestCase):
    """Test enhanced fundamentals functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.data_manager = DataManager()

    def test_normalize_symbol(self):
        """Test symbol normalization."""
        # Test valid symbols
        self.assertEqual(self.data_manager._normalize_symbol("aapl"), "AAPL")
        self.assertEqual(self.data_manager._normalize_symbol("  GOOGL  "), "GOOGL")
        self.assertEqual(self.data_manager._normalize_symbol("BRK.B"), "BRK-B")

        # Test invalid symbols
        self.assertEqual(self.data_manager._normalize_symbol(""), "")
        self.assertEqual(self.data_manager._normalize_symbol(None), "")
        self.assertEqual(self.data_manager._normalize_symbol("INVALID@SYMBOL"), "")

    def test_normalize_fundamentals_data(self):
        """Test fundamentals data normalization."""
        # Test dictionary input
        dict_data = {"pe_ratio": 15.5, "market_cap": 1000000}
        result = self.data_manager._normalize_fundamentals_data(dict_data)
        self.assertEqual(result, dict_data)

        # Test object with __dict__
        class MockFundamentals:
            def __init__(self):
                self.pe_ratio = 15.5
                self.market_cap = 1000000

        obj_data = MockFundamentals()
        result = self.data_manager._normalize_fundamentals_data(obj_data)
        self.assertEqual(result, {"pe_ratio": 15.5, "market_cap": 1000000})

        # Test None input
        result = self.data_manager._normalize_fundamentals_data(None)
        self.assertIsNone(result)

    def test_validate_combined_fundamentals(self):
        """Test combined fundamentals validation."""
        # Test valid data
        valid_data = {"pe_ratio": 15.5, "market_cap": 1000000}
        self.assertTrue(self.data_manager._validate_combined_fundamentals(valid_data))

        # Test empty data
        self.assertFalse(self.data_manager._validate_combined_fundamentals({}))
        self.assertFalse(self.data_manager._validate_combined_fundamentals(None))

    @patch('src.data.data_manager.get_fundamentals_cache')
    @patch('src.data.data_manager.get_fundamentals_combiner')
    def test_get_fundamentals_with_cache_hit(self, mock_combiner, mock_cache):
        """Test get_fundamentals with cache hit."""
        # Setup mocks
        mock_cache_instance = Mock()
        mock_cache.return_value = mock_cache_instance

        mock_cached_data = {"pe_ratio": 15.5, "market_cap": 1000000}
        mock_cache_instance.find_latest_json.return_value = Mock(
            provider="yfinance",
            timestamp=datetime.now(),
            file_path="/path/to/cache"
        )
        mock_cache_instance.read_json.return_value = mock_cached_data

        # Test cache hit
        result = self.data_manager.get_fundamentals("AAPL")

        # Verify cache was checked
        mock_cache_instance.find_latest_json.assert_called_once()
        mock_cache_instance.read_json.assert_called_once()

        # Verify result
        self.assertEqual(result, mock_cached_data)

    @patch('src.data.data_manager.get_fundamentals_cache')
    @patch('src.data.data_manager.get_fundamentals_combiner')
    def test_get_fundamentals_invalid_symbol(self, mock_combiner, mock_cache):
        """Test get_fundamentals with invalid symbol."""
        result = self.data_manager.get_fundamentals("")
        self.assertEqual(result, {})

        result = self.data_manager.get_fundamentals("INVALID@SYMBOL")
        self.assertEqual(result, {})

    def test_select_fundamentals_providers_with_requested(self):
        """Test provider selection with requested providers."""
        # Mock combiner
        mock_combiner = Mock()

        # Mock provider selector with available downloaders
        self.data_manager.provider_selector.downloaders = {
            'yfinance': Mock(),
            'fmp': Mock(),
            'alpha_vantage': Mock()
        }

        # Add get_fundamentals method to mocks
        for downloader in self.data_manager.provider_selector.downloaders.values():
            downloader.get_fundamentals = Mock()

        # Test with valid requested providers
        result = self.data_manager._select_fundamentals_providers(
            "AAPL", ["yfinance", "fmp"], "ratios", mock_combiner
        )
        self.assertEqual(result, ["yfinance", "fmp"])

        # Test with invalid requested providers
        result = self.data_manager._select_fundamentals_providers(
            "AAPL", ["invalid_provider"], "ratios", mock_combiner
        )
        # Should fall back to configuration-based selection
        mock_combiner.get_provider_sequence.assert_called_with("ratios")


if __name__ == '__main__':
    unittest.main()