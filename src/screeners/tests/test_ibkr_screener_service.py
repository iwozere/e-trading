"""
Unit tests for IBKRScreenerService.

Tests focus on de-duplication logic, signal fingerprinting, path resolution,
and injectable dependencies — without requiring live IBKR or database access.
"""

import unittest
from pathlib import Path
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

from src.screeners.discovery.base import IDiscoveryProvider
from src.screeners.ibkr_screener_service import IBKRScreenerService

# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------


class _FakeDiscovery(IDiscoveryProvider):
    """Returns a fixed list of symbols."""

    def __init__(self, symbols: List[str]):
        self._symbols = symbols

    async def get_symbols(self) -> List[str]:
        return list(self._symbols)


class _FakeDownloader:
    """Returns an empty DataFrame for every request."""

    def get_ohlcv(self, symbol: str, interval: str, start_date: Any, end_date: Any) -> Any:
        import pandas as pd

        return pd.DataFrame()


class _FakeStrategy:
    pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSignalFingerprint(unittest.TestCase):
    """Tests for _signal_fingerprint()."""

    def _make_service(self) -> IBKRScreenerService:
        service = IBKRScreenerService(
            strategy_class=_FakeStrategy,  # type: ignore[arg-type]  # lightweight test double
            strategy_config={},
            discovery_providers=[],
            downloader=_FakeDownloader(),  # type: ignore[arg-type]
        )
        return service

    def test_same_content_same_fingerprint(self):
        """Identical signals must produce the same fingerprint."""
        service = self._make_service()
        signal = {"symbol": "AAPL", "signal": "buy", "price": 150.0, "indicators": {"rsi": 30.1}}
        fp1 = service._signal_fingerprint(signal)
        fp2 = service._signal_fingerprint(signal)
        self.assertEqual(fp1, fp2)

    def test_volatile_keys_excluded(self):
        """timestamp / scan_time differences must NOT change the fingerprint."""
        service = self._make_service()
        sig_a = {"symbol": "AAPL", "signal": "buy", "price": 150.0, "timestamp": "2026-01-01T00:00:00"}
        sig_b = {"symbol": "AAPL", "signal": "buy", "price": 150.0, "timestamp": "2026-06-01T12:00:00"}
        self.assertEqual(service._signal_fingerprint(sig_a), service._signal_fingerprint(sig_b))

    def test_price_change_changes_fingerprint(self):
        """A changed price must yield a different fingerprint."""
        service = self._make_service()
        sig_a = {"symbol": "AAPL", "signal": "buy", "price": 150.0}
        sig_b = {"symbol": "AAPL", "signal": "buy", "price": 151.0}
        self.assertNotEqual(service._signal_fingerprint(sig_a), service._signal_fingerprint(sig_b))

    def test_signal_direction_change_changes_fingerprint(self):
        """buy → sell must yield a different fingerprint."""
        service = self._make_service()
        sig_buy = {"symbol": "TSLA", "signal": "buy", "price": 200.0}
        sig_sell = {"symbol": "TSLA", "signal": "sell", "price": 200.0}
        self.assertNotEqual(service._signal_fingerprint(sig_buy), service._signal_fingerprint(sig_sell))


class TestSignalDeduplication(unittest.IsolatedAsyncioTestCase):
    """Tests for the in-memory de-duplication in run_once()."""

    def _make_service(self, symbols: List[str]) -> IBKRScreenerService:
        service = IBKRScreenerService(
            strategy_class=_FakeStrategy,  # type: ignore[arg-type]  # lightweight test double
            strategy_config={},
            discovery_providers=[_FakeDiscovery(symbols)],
            downloader=_FakeDownloader(),  # type: ignore[arg-type]
        )
        return service

    async def test_duplicate_signal_suppressed_on_second_scan(self):
        """The notifier must not be called for a signal with an unchanged fingerprint."""
        service = self._make_service(["AAPL"])
        notifier = MagicMock()
        notifier.notify_signals = AsyncMock()
        service.notifier = notifier

        static_result = {"symbol": "AAPL", "signal": "buy", "price": 150.0}

        with patch.object(service, "_process_symbol", new=AsyncMock(return_value=static_result)):
            await service.run_once()  # first scan — must notify
            await service.run_once()  # second scan — identical signal, must suppress

        # Signals were notified only once (in the first scan)
        self.assertEqual(notifier.notify_signals.call_count, 1)
        sent = notifier.notify_signals.call_args[0][0]
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["symbol"], "AAPL")

    async def test_changed_signal_notified_again(self):
        """When the price changes the signal must be re-broadcast."""
        service = self._make_service(["AAPL"])
        notifier = MagicMock()
        notifier.notify_signals = AsyncMock()
        service.notifier = notifier

        results = iter(
            [
                {"symbol": "AAPL", "signal": "buy", "price": 150.0},
                {"symbol": "AAPL", "signal": "buy", "price": 155.0},  # price changed
            ]
        )

        with patch.object(service, "_process_symbol", new=AsyncMock(side_effect=lambda s: next(results))):
            await service.run_once()
            await service.run_once()

        self.assertEqual(notifier.notify_signals.call_count, 2)


class TestResultsDir(unittest.TestCase):
    """Tests for project-root-relative results_dir."""

    def test_results_dir_is_absolute(self):
        """results_dir must be an absolute path regardless of cwd."""
        service = IBKRScreenerService(
            strategy_class=_FakeStrategy,  # type: ignore[arg-type]  # lightweight test double
            strategy_config={},
            discovery_providers=[],
            downloader=_FakeDownloader(),  # type: ignore[arg-type]
        )
        self.assertTrue(service.results_dir.is_absolute())

    def test_results_dir_ends_with_expected_suffix(self):
        """results_dir must end with …/results/screeners/ibkr."""
        service = IBKRScreenerService(
            strategy_class=_FakeStrategy,  # type: ignore[arg-type]  # lightweight test double
            strategy_config={},
            discovery_providers=[],
            downloader=_FakeDownloader(),  # type: ignore[arg-type]
        )
        self.assertTrue(str(service.results_dir).endswith(str(Path("results") / "screeners" / "ibkr")))


class TestDownloaderInjection(unittest.TestCase):
    """Tests for injectable IBKRDownloader (Issue 12)."""

    def test_custom_downloader_is_used(self):
        """When a downloader is passed it must be stored as-is."""
        custom_dl = _FakeDownloader()
        service = IBKRScreenerService(
            strategy_class=_FakeStrategy,  # type: ignore[arg-type]  # lightweight test double
            strategy_config={},
            discovery_providers=[],
            downloader=custom_dl,  # type: ignore[arg-type]
        )
        self.assertIs(service.downloader, custom_dl)

    def test_default_downloader_created_when_none(self):
        """When downloader=None an IBKRDownloader instance must be created."""
        with patch("src.screeners.ibkr_screener_service.IBKRDownloader") as MockDL:
            MockDL.return_value = MagicMock()
            service = IBKRScreenerService(
                strategy_class=_FakeStrategy,  # type: ignore[arg-type]  # lightweight test double
                strategy_config={},
                discovery_providers=[],
            )
            MockDL.assert_called_once()
            self.assertIs(service.downloader, MockDL.return_value)


if __name__ == "__main__":
    unittest.main()
