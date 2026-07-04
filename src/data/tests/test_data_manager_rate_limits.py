import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Ensure the src directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.data.data_manager import DataManager, RateLimitException


class TestDataManagerRateLimits(unittest.TestCase):
    def setUp(self):
        self.dm = DataManager()
        # Reset cooldowns and blacklists
        self.dm._provider_cooldowns = {}
        self.dm._rate_limited_providers = set()

    def test_provider_cooldown_logic(self):
        """Test that _is_provider_on_cooldown correctly identifies expired and active cooldowns."""
        provider = "test_provider"

        # 1. No cooldown should be False
        self.assertFalse(self.dm._is_provider_on_cooldown(provider))

        # 2. Active cooldown should be True
        self.dm._set_provider_cooldown(provider, 30)
        self.assertTrue(self.dm._is_provider_on_cooldown(provider))

        # 3. Expired cooldown should be False and removed from dict
        # Mocking datetime to fast-forward
        with patch("src.data.data_manager.datetime") as mock_datetime:
            # Set fixed time for setting cooldown
            now = datetime(2026, 3, 19, 12, 0, 0)
            mock_datetime.now.return_value = now

            self.dm._set_provider_cooldown(provider, 10)
            self.assertTrue(self.dm._is_provider_on_cooldown(provider))

            # Fast-forward 20 seconds
            mock_datetime.now.return_value = now + timedelta(seconds=20)
            self.assertFalse(self.dm._is_provider_on_cooldown(provider))
            self.assertNotIn(provider, self.dm._provider_cooldowns)

    def test_rate_limit_sets_cooldown(self):
        """Test that RateLimitException in data fetch sets a temporary cooldown."""
        symbol = "AAPL"
        providers = ["yahoo"]

        # Mock the downloader and _fetch_with_timeout to raise RateLimitException
        self.dm._validate_single_provider_availability = MagicMock(return_value=True)
        self.dm.provider_selector.downloaders = {"yahoo": MagicMock()}

        with patch.object(self.dm, "_fetch_with_timeout") as mock_fetch:
            # RateLimitException takes url as first argument
            mock_fetch.side_effect = RateLimitException("https://api.yahoo.com/fundamentals", retry_after=60)

            # Should not raise exception, but return empty data and set cooldown
            result = self.dm._fetch_fundamentals_from_providers(symbol, providers)

            self.assertEqual(result, {})
            self.assertTrue(self.dm._is_provider_on_cooldown("yahoo"))
            # Initial cooldown for a single failure should be 60s (based on our implementation)
            exp_time = self.dm._provider_cooldowns["yahoo"]
            self.assertLessEqual((exp_time - datetime.now()).total_seconds(), 60)

    def test_filter_compatible_providers_respects_cooldown(self):
        """Test that _filter_compatible_providers skips providers on cooldown."""
        sequence = ["yahoo", "fmp"]
        symbol_classification = {
            "symbol": "AAPL",
            "symbol_type": "stock",
            "market": "US",
            "exchange": "NASDAQ",
            "country": "US",
            "international": False,
            "fundamentals_support": "full",
        }

        # Initialize downloaders in mock
        self.dm.provider_selector._initialize_downloader = MagicMock(return_value=True)
        self.dm.provider_selector.downloaders = {
            "yahoo": MagicMock(spec=["get_fundamentals"]),
            "fmp": MagicMock(spec=["get_fundamentals"]),
        }

        # Mock compatibility check to always return True for both
        self.dm._check_provider_symbol_compatibility = MagicMock(return_value={"compatible": True, "reason": "OK"})

        # 1. Initially both should be returned
        compatible = self.dm._filter_compatible_providers(sequence, symbol_classification)
        self.assertIn("yahoo", compatible)
        self.assertIn("fmp", compatible)

        # 2. Put yahoo on cooldown
        self.dm._set_provider_cooldown("yahoo", 60)
        compatible_after = self.dm._filter_compatible_providers(sequence, symbol_classification)
        self.assertNotIn("yahoo", compatible_after)
        self.assertIn("fmp", compatible_after)


if __name__ == "__main__":
    unittest.main()
