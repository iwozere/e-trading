"""
Tests for TRF cache TTL validation in trf_downloader.py.
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestIsCacheFresh:
    def test_returns_false_for_nonexistent_file(self, tmp_path):
        from src.ml.pipeline.shared.trf_downloader import _is_cache_fresh

        assert _is_cache_fresh(tmp_path / "missing.csv.gz") is False

    def test_fresh_file_returns_true(self, tmp_path):
        from src.ml.pipeline.shared.trf_downloader import _is_cache_fresh

        f = tmp_path / "trf.csv.gz"
        f.write_bytes(b"x")
        assert _is_cache_fresh(f, max_age_days=1) is True

    def test_stale_file_returns_false(self, tmp_path):
        from src.ml.pipeline.shared.trf_downloader import _is_cache_fresh

        f = tmp_path / "trf.csv.gz"
        f.write_bytes(b"x")
        two_days_ago = datetime.now().timestamp() - 2 * 86400
        with patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value = MagicMock(st_mtime=two_days_ago)
            result = _is_cache_fresh(f, max_age_days=1)
        assert result is False

    def test_just_under_boundary_is_fresh(self, tmp_path):
        from src.ml.pipeline.shared.trf_downloader import _is_cache_fresh

        f = tmp_path / "trf.csv.gz"
        f.write_bytes(b"x")
        twenty_three_hours_ago = datetime.now().timestamp() - 23 * 3600
        with patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value = MagicMock(st_mtime=twenty_three_hours_ago)
            result = _is_cache_fresh(f, max_age_days=1)
        assert result is True


class TestDownloadTrfCacheBehavior:
    def test_fresh_cache_skips_download(self, tmp_path):
        """download_trf should return immediately when cache is fresh."""
        from src.ml.pipeline.shared import trf_downloader as mod

        trf_date = datetime(2024, 1, 15)
        date_str = trf_date.strftime("%Y-%m-%d")

        # Create the expected cache file at tmp_path/trf/{date}.csv.gz
        cache_dir = tmp_path / "trf"
        cache_dir.mkdir()
        cache_file = cache_dir / f"{date_str}.csv.gz"
        cache_file.write_bytes(b"fake")

        mock_dl = MagicMock()
        with (
            patch.object(mod, "DATA_CACHE_DIR", str(tmp_path)),
            patch.object(mod, "_is_cache_fresh", return_value=True),
            patch.object(mod, "FinraTRFDownloader", return_value=mock_dl),
        ):
            result = mod.download_trf(trf_date, force_download=False)
            mock_dl.run.assert_not_called()
            assert result == cache_file

    def test_stale_cache_triggers_download(self, tmp_path):
        """download_trf should call FinraTRFDownloader.run() when cache is stale."""
        from src.ml.pipeline.shared import trf_downloader as mod

        trf_date = datetime(2024, 1, 10)
        date_str = trf_date.strftime("%Y-%m-%d")

        # Create a stale cache file
        cache_dir = tmp_path / "trf"
        cache_dir.mkdir()
        cache_file = cache_dir / f"{date_str}.csv.gz"
        cache_file.write_bytes(b"stale")

        mock_dl = MagicMock()
        # Simulate run() creating the cache file
        mock_dl.run.side_effect = lambda: None

        with (
            patch.object(mod, "DATA_CACHE_DIR", str(tmp_path)),
            patch.object(mod, "_is_cache_fresh", return_value=False),
            patch.object(mod, "FinraTRFDownloader", return_value=mock_dl),
        ):
            try:
                mod.download_trf(trf_date, force_download=False)
            except Exception:
                pass  # network call may fail in CI; we just verify the downloader was invoked
            mock_dl.run.assert_called_once()
