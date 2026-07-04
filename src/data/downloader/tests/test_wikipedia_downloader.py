"""
Tests for WikipediaDownloader.
"""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytest

from src.data.downloader.wikipedia_downloader import WikipediaDownloader


@pytest.fixture()
def downloader(tmp_path):
    """Return a WikipediaDownloader with a temp cache dir."""
    dl = WikipediaDownloader()
    dl._cache_dir = tmp_path / "index_changes"
    return dl


def test_download_index_changes(downloader):
    """Verify Wikipedia parsing, merging, cleaning, and caching."""
    # Mock S&P 500 HTML table
    # S&P table uses MultiIndex columns for Added/Removed Tickers
    sp_columns = pd.MultiIndex.from_tuples(
        [
            ("Effective Date", "Effective Date"),
            ("Added", "Ticker"),
            ("Added", "Security"),
            ("Removed", "Ticker"),
            ("Removed", "Security"),
            ("Reason", "Reason"),
        ]
    )
    sp_data = [
        ["June 30, 2026", "KKR", "KKR & Co.", "CMA", "Comerica", "Market cap change"],
    ]
    df_sp = pd.DataFrame(sp_data, columns=sp_columns)

    # Mock Nasdaq-100 HTML table
    nd_columns = pd.MultiIndex.from_tuples(
        [
            ("Date", "Date"),
            ("Added", "Ticker"),
            ("Added", "Security"),
            ("Removed", "Ticker"),
            ("Removed", "Security"),
            ("Reason", "Reason"),
        ]
    )
    nd_data = [
        ["June 22, 2026", "LRCX", "Lam Research", "", "", "Quarterly reconstitution"],
    ]
    df_nd = pd.DataFrame(nd_data, columns=nd_columns)

    # We patch requests.get to return fake responses
    # The first requests.get (S&P 500) will succeed, and the second (Nasdaq-100) will succeed
    response_sp = MagicMock()
    response_sp.text = "SP500 Table"
    response_sp.ok = True

    response_nd = MagicMock()
    response_nd.text = "NASDAQ100 Table"
    response_nd.ok = True

    with (
        patch("requests.get", side_effect=[response_sp, response_nd]),
        patch(
            "pandas.read_html",
            side_effect=[
                [None, df_sp],  # First tables list for S&P 500
                [None, None, None, None, None, None, df_nd],  # Tables list for Nasdaq-100 (Table 6)
            ],
        ),
    ):
        result = downloader.download_index_changes(date(2026, 7, 3))

    # Assert cache file was created
    expected_cache = downloader._cache_dir / "2026-07-03.csv.gz"
    assert expected_cache.exists()

    # Read back the cached file
    cached_df = pd.read_csv(expected_cache)

    assert len(cached_df) == 2
    assert list(cached_df.columns) == [
        "date",
        "index_name",
        "Added_Ticker",
        "Added_Security",
        "Removed_Ticker",
        "Removed_Security",
        "Reason",
    ]
    assert cached_df.loc[0, "date"] == "2026-06-30"
    assert cached_df.loc[0, "index_name"] == "SP500"
    assert cached_df.loc[0, "Added_Ticker"] == "KKR"
    assert cached_df.loc[1, "date"] == "2026-06-22"
    assert cached_df.loc[1, "index_name"] == "NASDAQ100"
    assert cached_df.loc[1, "Added_Ticker"] == "LRCX"
