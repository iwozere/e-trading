import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_manager import DataManager, RateLimitException  # noqa: E402


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

    def test_rate_limit_skips_to_retry_pass(self):
        """A RateLimitException gives up on the provider at once: no retries, no wait, no cooldown.

        Since 168f2c0 the caller collects rate-limited tickers and re-attempts them together after a global
        cooldown, so this method must neither sleep nor block the provider for the remaining tickers.
        """
        symbol = "AAPL"
        providers = ["yahoo"]

        self.dm._validate_single_provider_availability = MagicMock(return_value=True)
        self.dm.provider_selector.downloaders = {"yahoo": MagicMock()}

        with patch.object(self.dm, "_fetch_with_timeout") as mock_fetch, patch.object(
            self.dm, "_sleep_with_jitter"
        ) as mock_sleep:
            # RateLimitException takes url as first argument
            mock_fetch.side_effect = RateLimitException("https://api.yahoo.com/fundamentals", retry_after=60)

            result = self.dm._fetch_fundamentals_from_providers(symbol, providers)

            self.assertEqual(result, {})
            self.assertEqual(mock_fetch.call_count, 1)
            mock_sleep.assert_not_called()
            self.assertFalse(self.dm._is_provider_on_cooldown("yahoo"))

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
