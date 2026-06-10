"""
Tests for Phase 6.2 — TRF cache TTL validation in trf_downloader.py.
"""

import sys
import time
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestIsCacheFresh:

    def test_returns_false_for_nonexistent_file(self, tmp_path):
        from src.ml.pipeline.shared.trf_downloader import _is_cache_fresh
        assert _is_cache_fresh(tmp_path / "missing.csv") is False

    def test_fresh_file_returns_true(self, tmp_path):
        from src.ml.pipeline.shared.trf_downloader import _is_cache_fresh
        f = tmp_path / "trf.csv"
        f.write_text("ticker,short_volume,total_volume\n")
        assert _is_cache_fresh(f, max_age_days=1) is True

    def test_stale_file_returns_false(self, tmp_path):
        from src.ml.pipeline.shared.trf_downloader import _is_cache_fresh
        f = tmp_path / "trf.csv"
        f.write_text("ticker,short_volume,total_volume\n")
        # Simulate file written 2 days ago by patching stat().st_mtime
        two_days_ago = datetime.now().timestamp() - 2 * 86400
        with patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value = MagicMock(st_mtime=two_days_ago)
            result = _is_cache_fresh(f, max_age_days=1)
        assert result is False

    def test_just_under_boundary_is_fresh(self, tmp_path):
        from src.ml.pipeline.shared.trf_downloader import _is_cache_fresh
        f = tmp_path / "trf.csv"
        f.write_text("x")
        # 23 hours old → well within max_age_days=1 → fresh
        twenty_three_hours_ago = datetime.now().timestamp() - 23 * 3600
        with patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value = MagicMock(st_mtime=twenty_three_hours_ago)
            result = _is_cache_fresh(f, max_age_days=1)
        assert result is True


class TestDownloadTrfCacheBehavior:

    def test_fresh_cache_skips_download(self, tmp_path, monkeypatch):
        """download_trf should return immediately when cache is fresh."""
        from src.ml.pipeline.shared import trf_downloader as mod

        trf_date = datetime(2024, 1, 15)
        date_str = trf_date.strftime("%Y-%m-%d")
        # Point the module's output path into tmp_path
        trf_dir = tmp_path / "results" / "trf_data" / date_str
        trf_dir.mkdir(parents=True)
        output_file = trf_dir / "trf.csv"
        output_file.write_text("ticker,short_volume,total_volume\n")

        monkeypatch.chdir(tmp_path)

        mock_dl = MagicMock()
        with patch.object(mod, "_is_cache_fresh", return_value=True), \
             patch.object(mod, "FinraDataDownloader", return_value=mock_dl):
            result = mod.download_trf(trf_date, force_download=False)
            mock_dl.run.assert_not_called()

    def test_stale_cache_triggers_download(self, tmp_path):
        """download_trf should re-download when cache is stale."""
        from src.ml.pipeline.shared import trf_downloader as mod

        trf_date = datetime(2024, 1, 10)
        date_str = trf_date.strftime("%Y-%m-%d")
        output_file = tmp_path / date_str / "trf.csv"
        output_file.parent.mkdir(parents=True)
        output_file.write_text("ticker,short_volume,total_volume\n")

        mock_downloader = MagicMock()

        with patch.object(mod, "_is_cache_fresh", return_value=False), \
             patch.object(mod, "FinraDataDownloader", return_value=mock_downloader):
            try:
                mod.download_trf(trf_date, force_download=False)
            except Exception:
                pass  # network call may fail in CI; we just verify the downloader was called
            mock_downloader.run.assert_called_once()
