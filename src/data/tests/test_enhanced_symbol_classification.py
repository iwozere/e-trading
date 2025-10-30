#!/usr/bin/env python3
"""
Test Enhanced Symbol Classification for Fundamentals

This test validates the enhanced symbol classification functionality.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.data.data_manager import DataManager


class TestEnhancedSymbolClassification(unittest.TestCase):
    """Test enhanced symbol classification for fundamentals."""

    def setUp(self):
        """Set up test fixtures."""
        self.data_manager = DataManager()
        self.provider_selector = self.data_manager.provider_selector

    def test_us_stock_classification(self):
        """Test US stock symbol classification."""
        # Test major US stocks
        result = self.provider_selector.classify_symbol_for_fundamentals("AAPL")
        self.assertEqual(result['symbol_type'], 'stock')
        self.assertEqual(result['country'], 'US')
        self.assertEqual(result['market'], 'US')
        self.assertEqual(result['currency'], 'USD')
        self.assertEqual(result['fundamentals_support'], 'full')
        self.assertFalse(result['international'])

        result = self.provider_selector.classify_symbol_for_fundamentals("GOOGL")
        self.assertEqual(result['symbol_type'], 'stock')
        self.assertEqual(result['exchange'], 'NASDAQ')

        result = self.provider_selector.classify_symbol_for_fundamentals("JPM")
        self.assertEqual(result['symbol_type'], 'stock')
        self.assertEqual(result['exchange'], 'NYSE')

    def test_us_etf_classification(self):
        """Test US ETF symbol classification."""
        result = self.provider_selector.classify_symbol_for_fundamentals("SPY")
        self.assertEqual(result['symbol_type'], 'etf')
        self.assertEqual(result['country'], 'US')
        self.assertEqual(result['fundamentals_support'], 'full')

        result = self.provider_selector.classify_symbol_for_fundamentals("QQQ")
        self.assertEqual(result['symbol_type'], 'etf')

        result = self.provider_selector.classify_symbol_for_fundamentals("VANGUARD_ETF")
        self.assertEqual(result['symbol_type'], 'etf')

    def test_international_stock_classification(self):
        """Test international stock symbol classification."""
        # UK stocks
        result = self.provider_selector.classify_symbol_for_fundamentals("ASML.AS")
        self.assertEqual(result['symbol_type'], 'stock')
        self.assertEqual(result['country'], 'NL')
        self.assertEqual(result['market'], 'EU')
        self.assertEqual(result['exchange'], 'AMS')
        self.assertEqual(result['currency'], 'EUR')
        self.assertEqual(result['fundamentals_support'], 'limited')
        self.assertTrue(result['international'])

        # London Stock Exchange
        result = self.provider_selector.classify_symbol_for_fundamentals("VODAFONE.L")
        self.assertEqual(result['country'], 'GB')
        self.assertEqual(result['market'], 'UK')
        self.assertEqual(result['exchange'], 'LSE')
        self.assertEqual(result['currency'], 'GBP')

        # German stocks
        result = self.provider_selector.classify_symbol_for_fundamentals("SAP.DE")
        self.assertEqual(result['country'], 'DE')
        self.assertEqual(result['market'], 'EU')
        self.assertEqual(result['exchange'], 'XETRA')

    def test_crypto_classification(self):
        """Test cryptocurrency symbol classification."""
        # Test USDT pairs
        result = self.provider_selector.classify_symbol_for_fundamentals("BTCUSDT")
        self.assertEqual(result['symbol_type'], 'crypto')
        self.assertEqual(result['fundamentals_support'], 'none')

        result = self.provider_selector.classify_symbol_for_fundamentals("ETHUSDT")
        self.assertEqual(result['symbol_type'], 'crypto')
        self.assertEqual(result['fundamentals_support'], 'none')

        # Test USDC pairs (important for stablecoin trading)
        result = self.provider_selector.classify_symbol_for_fundamentals("BTCUSDC")
        self.assertEqual(result['symbol_type'], 'crypto')
        self.assertEqual(result['fundamentals_support'], 'none')

        result = self.provider_selector.classify_symbol_for_fundamentals("ETHUSDC")
        self.assertEqual(result['symbol_type'], 'crypto')
        self.assertEqual(result['fundamentals_support'], 'none')

        # Test other stablecoin pairs
        result = self.provider_selector.classify_symbol_for_fundamentals("BTCBUSD")
        self.assertEqual(result['symbol_type'], 'crypto')
        self.assertEqual(result['fundamentals_support'], 'none')

        # Test stablecoin-to-stablecoin pairs
        result = self.provider_selector.classify_symbol_for_fundamentals("USDCUSDT")
        self.assertEqual(result['symbol_type'], 'crypto')
        self.assertEqual(result['fundamentals_support'], 'none')

    def test_reit_classification(self):
        """Test REIT symbol classification."""
        result = self.provider_selector.classify_symbol_for_fundamentals("REALTY_REIT")
        self.assertEqual(result['symbol_type'], 'reit')
        self.assertEqual(result['fundamentals_support'], 'full')

    def test_exchange_detection(self):
        """Test exchange detection from symbol suffixes."""
        # Test various exchange suffixes
        test_cases = [
            ("ASML.AS", "AMS", "NL", "EU"),
            ("VODAFONE.L", "LSE", "GB", "UK"),
            ("SAP.DE", "XETRA", "DE", "EU"),
            ("LVMH.PA", "EPA", "FR", "EU"),
            ("TOYOTA.T", "TSE", "JP", "ASIA"),
            ("TENCENT.HK", "HKEX", "HK", "ASIA"),
        ]

        for symbol, expected_exchange, expected_country, expected_market in test_cases:
            result = self.provider_selector.classify_symbol_for_fundamentals(symbol)
            self.assertEqual(result['exchange'], expected_exchange, f"Failed for {symbol}")
            self.assertEqual(result['country'], expected_country, f"Failed for {symbol}")
            self.assertEqual(result['market'], expected_market, f"Failed for {symbol}")

    def test_fundamentals_support_assessment(self):
        """Test fundamentals support level assessment."""
        # US stocks should have full support
        result = self.provider_selector.classify_symbol_for_fundamentals("AAPL")
        self.assertEqual(result['fundamentals_support'], 'full')

        # International stocks should have limited support
        result = self.provider_selector.classify_symbol_for_fundamentals("ASML.AS")
        self.assertEqual(result['fundamentals_support'], 'limited')

        # Crypto should have no support (correctly returns 'none')
        result = self.provider_selector.classify_symbol_for_fundamentals("BTCUSDT")
        self.assertEqual(result['fundamentals_support'], 'none')

    def test_symbol_normalization(self):
        """Test symbol normalization in classification."""
        result = self.provider_selector.classify_symbol_for_fundamentals("  aapl  ")
        self.assertEqual(result['normalized'], 'AAPL')
        self.assertEqual(result['symbol'], '  aapl  ')  # Original preserved

        result = self.provider_selector.classify_symbol_for_fundamentals("asml.as")
        self.assertEqual(result['normalized'], 'ASML')  # Suffix removed in normalization

    def test_error_handling(self):
        """Test error handling in symbol classification."""
        # Test with None input
        result = self.provider_selector.classify_symbol_for_fundamentals(None)
        self.assertEqual(result['symbol_type'], 'unknown')
        self.assertEqual(result['fundamentals_support'], 'none')

        # Test with empty string
        result = self.provider_selector.classify_symbol_for_fundamentals("")
        self.assertEqual(result['symbol_type'], 'unknown')
        self.assertEqual(result['fundamentals_support'], 'none')

    def test_us_exchange_determination(self):
        """Test US exchange determination logic."""
        # Test known NASDAQ symbols
        nasdaq_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        for symbol in nasdaq_symbols:
            result = self.provider_selector.classify_symbol_for_fundamentals(symbol)
            self.assertEqual(result['exchange'], 'NASDAQ', f"Failed for {symbol}")

        # Test known NYSE symbols
        nyse_symbols = ['JPM', 'BAC', 'WFC', 'JNJ', 'PG']
        for symbol in nyse_symbols:
            result = self.provider_selector.classify_symbol_for_fundamentals(symbol)
            self.assertEqual(result['exchange'], 'NYSE', f"Failed for {symbol}")


if __name__ == '__main__':
    unittest.main()