"""Tests for UniverseLoader."""

from pathlib import Path
from unittest.mock import patch
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytest

from src.ml.pipeline.p05_ai_selector.stages.universe_loader import UniverseLoader
from src.ml.pipeline.p05_ai_selector.config import CRYPTO_TICKERS


def _mock_russell_df() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "TSLA"],
        "name": ["Apple", "Microsoft", "Tesla"],
        "sector": ["Technology", "Technology", "Consumer Cyclical"],
        "industry": ["Consumer Electronics", "Software", "Auto"],
        "exchange": ["NASDAQ", "NASDAQ", "NASDAQ"],
    })


class TestUniverseLoader:
    def test_load_returns_list(self):
        """load() returns a non-empty list of strings."""
        with patch("src.ml.pipeline.p05_ai_selector.stages.universe_loader.Russell3000Downloader") as MockDl:
            MockDl.return_value.load.return_value = _mock_russell_df()
            result = UniverseLoader().load()

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(t, str) for t in result)

    def test_crypto_tickers_included(self):
        """All CRYPTO_TICKERS appear in the result."""
        with patch("src.ml.pipeline.p05_ai_selector.stages.universe_loader.Russell3000Downloader") as MockDl:
            MockDl.return_value.load.return_value = _mock_russell_df()
            result = UniverseLoader().load()

        for ticker in CRYPTO_TICKERS:
            assert ticker in result, f"{ticker} missing from universe"

    def test_no_duplicates(self):
        """Result contains no duplicate tickers."""
        russell_df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "BTC-USD"],  # BTC-USD also in crypto list
            "name": ["Apple", "Microsoft", "Bitcoin"],
            "sector": ["Tech", "Tech", "Crypto"],
            "industry": ["", "", ""],
            "exchange": ["NASDAQ", "NASDAQ", ""],
        })
        with patch("src.ml.pipeline.p05_ai_selector.stages.universe_loader.Russell3000Downloader") as MockDl:
            MockDl.return_value.load.return_value = russell_df
            result = UniverseLoader().load()

        assert len(result) == len(set(result))
