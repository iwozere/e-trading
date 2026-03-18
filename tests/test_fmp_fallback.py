import unittest
from unittest.mock import MagicMock, patch
from src.data.data_manager import DataManager
from src.error_handling.exceptions import RateLimitException

class TestFMPFallback(unittest.TestCase):
    def setUp(self):
        self.data_manager = DataManager()
        self.symbol = "AAPL"

    @patch('src.data.downloader.fmp_data_downloader.FMPDataDownloader.get_company_profile')
    @patch('src.data.downloader.finnhub_data_downloader.FinnhubDataDownloader.get_company_profile')
    def test_fmp_rate_limit_fallback(self, mock_finnhub, mock_fmp):
        """Test that DataManager falls back when FMP hits rate limit."""
        # Setup FMP to return 429 (via raising RateLimitException)
        mock_fmp.side_effect = RateLimitException(url="https://financialmodelingprep.com/stable/profile", retry_after=1)
        
        # Setup Finnhub to return valid data
        mock_finnhub.return_value = {
            "name": "Apple Inc.",
            "finnhubIndustry": "Technology",
            "marketCapitalization": 3000000,
            "shareOutstanding": 15000,
            "currency": "USD"
        }

        # Request fundamentals explicitly asking for FMP first (if possible) or just verify sequence
        # Based on our updated fundamentals.json, finnhub is now first.
        # To test FALLBACK to FMP or FROM FMP, we need to know the sequence.
        
        # Let's force a sequence where FMP is tried first for the test
        providers = ["fmp", "finnhub"]
        
        # We need to mock _fetch_with_timeout or similar because get_fundamentals calls it
        # Actually, get_fundamentals calls _fetch_fundamentals_from_providers
        
        result = self.data_manager.get_fundamentals(self.symbol, providers=providers, force_refresh=True)

        # Verify FMP was called
        mock_fmp.assert_called()
        
        # Verify Finnhub was called as fallback
        mock_finnhub.assert_called()
        
        # Verify result contains Finnhub data
        self.assertEqual(result['company_name'], "Apple Inc.")

if __name__ == '__main__':
    unittest.main()
