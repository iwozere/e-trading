"""Tests for the partial ticker.info retry and the shared rate-limit backoff."""

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from yfinance.exceptions import YFRateLimitError

from src.data.downloader import yahoo_data_downloader as ydd

FULL = {"maxAge": 86400, "regularMarketPrice": 1.5, "sector": "Tech", "floatShares": 1000}
PRICE_ONLY = {"regularMarketPrice": 1.5, "quoteType": "EQUITY"}
DEAD = {"trailingPegRatio": None}


def _tickers(*infos):
    """Return a yf.Ticker side_effect yielding objects whose .info is each dict in turn."""
    return [MagicMock(info=i) for i in infos]


class TestPartialInfoRetry(unittest.TestCase):
    """Price-only info (quoteSummary 401) is refetched once; dead symbols are not."""

    def test_is_partial_info(self) -> None:
        self.assertTrue(ydd._is_partial_info(PRICE_ONLY))
        self.assertFalse(ydd._is_partial_info(FULL))
        self.assertFalse(ydd._is_partial_info(DEAD))
        self.assertFalse(ydd._is_partial_info({}))
        self.assertFalse(ydd._is_partial_info(None))

    @patch.object(ydd.time, "sleep", new=MagicMock())
    @patch.object(ydd.yf, "Ticker")
    def test_price_only_info_is_retried(self, ticker: MagicMock) -> None:
        ticker.side_effect = _tickers(PRICE_ONLY, FULL)

        self.assertEqual(ydd._fetch_ticker_info("ABC"), FULL)
        self.assertEqual(ticker.call_count, 2)

    @patch.object(ydd.time, "sleep", new=MagicMock())
    @patch.object(ydd.yf, "Ticker")
    def test_full_and_dead_info_are_not_retried(self, ticker: MagicMock) -> None:
        for info in (FULL, DEAD):
            ticker.reset_mock()
            ticker.side_effect = _tickers(info)
            self.assertEqual(ydd._fetch_ticker_info("ABC"), info)
            self.assertEqual(ticker.call_count, 1)

    @patch.object(ydd.time, "sleep", new=MagicMock())
    @patch.object(ydd.yf, "Ticker")
    def test_keeps_first_result_when_retry_is_worse(self, ticker: MagicMock) -> None:
        ticker.side_effect = _tickers(PRICE_ONLY, {})

        self.assertEqual(ydd._fetch_ticker_info("ABC"), PRICE_ONLY)

    @patch.object(ydd.time, "sleep", new=MagicMock())
    @patch.object(ydd.yf, "Ticker")
    def test_get_fundamentals_uses_retried_info(self, ticker: MagicMock) -> None:
        ticker.side_effect = _tickers(PRICE_ONLY, FULL)

        fundamentals = ydd.YahooDataDownloader().get_fundamentals("abc")

        self.assertIsNotNone(fundamentals)
        assert fundamentals is not None
        self.assertEqual(fundamentals.sector, "Tech")
        self.assertEqual(fundamentals.float_shares, 1000)


class TestSharedBackoff(unittest.TestCase):
    """A rate limit hit by one thread pauses all threads."""

    def test_wait_blocks_other_threads_until_pause_ends(self) -> None:
        backoff = ydd._SharedBackoff()
        backoff.extend(0.2)
        waited = []

        def worker() -> None:
            start = time.monotonic()
            backoff.wait()
            waited.append(time.monotonic() - start)

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(waited), 3)
        self.assertTrue(all(w >= 0.15 for w in waited), waited)

    def test_extend_never_shortens_pause(self) -> None:
        backoff = ydd._SharedBackoff()
        backoff.extend(0.2)
        backoff.extend(0.0)
        start = time.monotonic()
        backoff.wait()
        self.assertGreaterEqual(time.monotonic() - start, 0.15)

    def test_options_rate_limit_sets_shared_pause(self) -> None:
        backoff = MagicMock()
        ticker = MagicMock()
        type(ticker).options = property(MagicMock(side_effect=[YFRateLimitError(), ()]))

        with patch.object(ydd, "_yahoo_rate_limit_backoff", backoff), patch.object(ydd.yf, "Ticker", return_value=ticker):
            result = ydd.YahooDataDownloader().get_options_chain_full("ABC")

        self.assertTrue(result.empty)
        backoff.extend.assert_called_once_with(10.0)
        self.assertEqual(backoff.wait.call_count, 2)


if __name__ == "__main__":
    unittest.main()
