"""Tests for the process-wide curl_cffi session handed to yfinance.

yfinance keeps one YfData singleton whose session is swapped on every
yf.Ticker(..., session=s) call, so per-thread sessions raced and caused
401 "Invalid Crumb" errors. Every thread must get the same session object.
"""

import unittest
from concurrent.futures import ThreadPoolExecutor

from src.data.downloader import yahoo_data_downloader


class TestYahooSharedSession(unittest.TestCase):
    """Ensure _get_shared_session returns one session across all threads."""

    def test_same_session_across_threads(self) -> None:
        with ThreadPoolExecutor(max_workers=8) as pool:
            sessions = list(pool.map(lambda _: yahoo_data_downloader._get_shared_session(), range(32)))

        self.assertEqual(len({id(s) for s in sessions}), 1)
        self.assertIs(sessions[0], yahoo_data_downloader._get_shared_session())


if __name__ == "__main__":
    unittest.main()
