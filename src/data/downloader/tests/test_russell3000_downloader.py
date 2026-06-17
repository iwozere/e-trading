"""
Tests for Russell3000Downloader.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytest

from src.data.downloader.russell3000_downloader import Russell3000Downloader, _REQUIRED_COLUMNS


@pytest.fixture()
def downloader(tmp_path):
    """Return a downloader with a temp cache dir and no real API key."""
    dl = Russell3000Downloader(api_key="fake-key")
    dl._cache_dir = tmp_path / "universe"
    dl._cache_file = dl._cache_dir / "russell3000.csv.gz"
    dl._meta_file = dl._cache_dir / "russell3000_meta.json"
    dl._static_csv_path = dl._cache_dir / "russell3000_static.csv"
    return dl


def _make_sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "GOOGL"],
        "name": ["Apple Inc.", "Microsoft Corporation", "Alphabet Inc."],
        "sector": ["Technology", "Technology", "Communication Services"],
        "industry": ["Consumer Electronics", "Software-Infrastructure", "Internet Content & Information"],
        "exchange": ["NASDAQ", "NASDAQ", "NASDAQ"],
    })


def _write_fresh_cache(downloader, df: pd.DataFrame) -> None:
    downloader._cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(downloader._cache_file, index=False, compression="gzip")
    meta = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "source_used": "fmp",
        "row_count": len(df),
    }
    downloader._meta_file.write_text(json.dumps(meta))


class TestLoad:
    def test_load_returns_dataframe(self, downloader):
        """Fresh cache hit: returns DataFrame with correct columns; no HTTP call made."""
        df = _make_sample_df()
        _write_fresh_cache(downloader, df)

        with patch("src.data.downloader.russell3000_downloader.requests.get") as mock_get:
            result = downloader.load()

        mock_get.assert_not_called()
        assert isinstance(result, pd.DataFrame)
        assert set(_REQUIRED_COLUMNS).issubset(result.columns)
        assert len(result) == 3

    def test_stale_cache_triggers_refresh(self, downloader):
        """Cache older than 90 days triggers a call to _fetch_from_fmp."""
        df = _make_sample_df()
        downloader._cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(downloader._cache_file, index=False, compression="gzip")
        old_date = (datetime.now(timezone.utc) - timedelta(days=91)).isoformat()
        downloader._meta_file.write_text(json.dumps({"last_updated": old_date, "source_used": "fmp", "row_count": 3}))

        fmp_df = _make_sample_df()
        with patch.object(downloader, "_fetch_from_fmp", return_value=fmp_df) as mock_fmp:
            result = downloader.load()

        mock_fmp.assert_called_once()
        assert len(result) == 3

    def test_fmp_failure_falls_back_to_static(self, downloader):
        """When _fetch_from_fmp returns None, load() falls back to static CSV."""
        with patch.object(downloader, "_fetch_from_fmp", return_value=None):
            with patch.object(downloader, "_load_static_fallback", return_value=_make_sample_df()) as mock_static:
                result = downloader.load(force=True)

        mock_static.assert_called_once()
        assert downloader.last_source_used == "static"
        assert isinstance(result, pd.DataFrame)


class TestStaticFallback:
    def test_static_fallback_columns(self, downloader):
        """Static CSV seeded in DATA_CACHE_DIR/universe/ loads correctly."""
        downloader._cache_dir.mkdir(parents=True, exist_ok=True)
        _make_sample_df().to_csv(downloader._static_csv_path, index=False)
        df = downloader._load_static_fallback()
        assert set(_REQUIRED_COLUMNS).issubset(df.columns)
        assert len(df) > 0

    def test_static_fallback_missing_raises(self, downloader):
        """Missing static CSV raises FileNotFoundError with helpful message."""
        with pytest.raises(FileNotFoundError, match="russell3000_static.csv"):
            downloader._load_static_fallback()


class TestIsStale:
    def test_is_stale_no_cache(self, downloader):
        """No cache file present → is_stale() returns True."""
        assert downloader.is_stale() is True

    def test_is_stale_fresh_cache(self, downloader):
        """Fresh cache with recent last_updated → is_stale() returns False."""
        df = _make_sample_df()
        _write_fresh_cache(downloader, df)
        assert downloader.is_stale() is False

    def test_is_stale_old_cache(self, downloader):
        """Cache with last_updated > 90 days ago → is_stale() returns True."""
        df = _make_sample_df()
        downloader._cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(downloader._cache_file, index=False, compression="gzip")
        old_ts = (datetime.now(timezone.utc) - timedelta(days=95)).isoformat()
        downloader._meta_file.write_text(json.dumps({"last_updated": old_ts}))
        assert downloader.is_stale() is True
