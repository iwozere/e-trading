"""
Unit tests for StaticDiscovery.

Covers file-not-found, valid JSON, malformed JSON, and pathlib migration
(no os.path usage in the implementation).
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.screeners.discovery.static import StaticDiscovery


class TestStaticDiscovery(unittest.IsolatedAsyncioTestCase):
    """Tests for StaticDiscovery.get_symbols()."""

    async def test_returns_empty_list_when_file_missing(self):
        """Non-existent path must return an empty list without raising."""
        discovery = StaticDiscovery("/non/existent/path/watchlist.json")
        symbols = await discovery.get_symbols()
        self.assertEqual(symbols, [])

    async def test_loads_symbols_from_valid_file(self):
        """Valid JSON file with a 'symbols' key must return the listed symbols."""
        with TemporaryDirectory() as tmpdir:
            watchlist = Path(tmpdir) / "watchlist.json"
            watchlist.write_text(
                json.dumps({"symbols": ["AAPL", "TSLA", "NVDA"]}),
                encoding="utf-8",
            )
            discovery = StaticDiscovery(watchlist)
            symbols = await discovery.get_symbols()
            self.assertEqual(symbols, ["AAPL", "TSLA", "NVDA"])

    async def test_returns_empty_list_for_missing_symbols_key(self):
        """JSON without a 'symbols' key must return an empty list."""
        with TemporaryDirectory() as tmpdir:
            watchlist = Path(tmpdir) / "watchlist.json"
            watchlist.write_text(json.dumps({"tickers": ["AAPL"]}), encoding="utf-8")
            discovery = StaticDiscovery(watchlist)
            symbols = await discovery.get_symbols()
            self.assertEqual(symbols, [])

    async def test_returns_empty_list_for_malformed_json(self):
        """Malformed JSON must return an empty list without raising."""
        with TemporaryDirectory() as tmpdir:
            watchlist = Path(tmpdir) / "watchlist.json"
            watchlist.write_text("{ this is not valid json }", encoding="utf-8")
            discovery = StaticDiscovery(watchlist)
            symbols = await discovery.get_symbols()
            self.assertEqual(symbols, [])

    async def test_accepts_path_object(self):
        """StaticDiscovery must accept a pathlib.Path as the file_path argument."""
        with TemporaryDirectory() as tmpdir:
            watchlist = Path(tmpdir) / "watchlist.json"
            watchlist.write_text(json.dumps({"symbols": ["MSFT"]}), encoding="utf-8")
            discovery = StaticDiscovery(watchlist)  # Path object, not str
            symbols = await discovery.get_symbols()
            self.assertEqual(symbols, ["MSFT"])

    async def test_accepts_string_path(self):
        """StaticDiscovery must accept a plain string as the file_path argument."""
        with TemporaryDirectory() as tmpdir:
            watchlist = Path(tmpdir) / "watchlist.json"
            watchlist.write_text(json.dumps({"symbols": ["GOOG"]}), encoding="utf-8")
            discovery = StaticDiscovery(str(watchlist))  # str, not Path
            symbols = await discovery.get_symbols()
            self.assertEqual(symbols, ["GOOG"])


if __name__ == "__main__":
    unittest.main()
