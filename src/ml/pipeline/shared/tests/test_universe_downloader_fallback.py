"""
Tests for the FTP retry / HTTPS fallback in universe_downloader.py.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.shared import universe_downloader as ud

NASDAQ_TEXT = (
    "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
    "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\n"
    "ZZZT|Test Co|Q|Y|N|100|N|N\n"
)
OTHER_TEXT = "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\nA|Agilent|N|A|N|100|N|A\n"
TEXTS = {"nasdaqlisted.txt": NASDAQ_TEXT, "otherlisted.txt": OTHER_TEXT}


@pytest.fixture
def downloader(tmp_path: Path) -> ud.NasdaqUniverseDownloader:
    """Downloader caching into a temp dir."""
    return ud.NasdaqUniverseDownloader(results_dir=tmp_path)


def test_ftp_success_skips_https(downloader: ud.NasdaqUniverseDownloader) -> None:
    with (
        patch.object(ud.NasdaqUniverseDownloader, "_fetch_via_ftp", return_value=TEXTS),
        patch.object(ud.NasdaqUniverseDownloader, "_fetch_via_https") as https,
    ):
        assert downloader._fetch_symbol_files() == TEXTS
    https.assert_not_called()


def test_ftp_timeout_falls_back_to_https(downloader: ud.NasdaqUniverseDownloader) -> None:
    with (
        patch.object(ud.NasdaqUniverseDownloader, "_fetch_via_ftp", side_effect=TimeoutError("timed out")),
        patch.object(ud.NasdaqUniverseDownloader, "_fetch_via_https", return_value=TEXTS),
        patch.object(ud.time, "sleep") as sleep,
    ):
        assert downloader._fetch_symbol_files() == TEXTS
    sleep.assert_not_called()


def test_retries_after_both_transports_fail(downloader: ud.NasdaqUniverseDownloader) -> None:
    with (
        patch.object(ud.NasdaqUniverseDownloader, "_fetch_via_ftp", side_effect=[TimeoutError("t"), TimeoutError("t"), TEXTS]),
        patch.object(ud.NasdaqUniverseDownloader, "_fetch_via_https", side_effect=ConnectionError("down")),
        patch.object(ud.time, "sleep") as sleep,
    ):
        assert downloader._fetch_symbol_files() == TEXTS
    assert sleep.call_count == 2


def test_raises_after_all_attempts_fail(downloader: ud.NasdaqUniverseDownloader) -> None:
    with (
        patch.object(ud.NasdaqUniverseDownloader, "_fetch_via_ftp", side_effect=TimeoutError("t")),
        patch.object(ud.NasdaqUniverseDownloader, "_fetch_via_https", side_effect=ConnectionError("down")),
        patch.object(ud.time, "sleep"),
        pytest.raises(RuntimeError),
    ):
        downloader._fetch_symbol_files()


def test_download_uses_fallback_texts(downloader: ud.NasdaqUniverseDownloader) -> None:
    with patch.object(ud.NasdaqUniverseDownloader, "_fetch_symbol_files", return_value=TEXTS):
        tickers = downloader._download_from_nasdaq_trader()
    assert "AAPL" in tickers
    assert "ZZZT" not in tickers  # test issue excluded


def test_download_includes_otherlisted_tickers(downloader: ud.NasdaqUniverseDownloader) -> None:
    with patch.object(ud.NasdaqUniverseDownloader, "_fetch_symbol_files", return_value=TEXTS):
        tickers = downloader._download_from_nasdaq_trader()
    assert "A" in tickers  # NYSE ticker from otherlisted.txt ("ACT Symbol" column)


def test_total_failure_returns_empty_universe(downloader: ud.NasdaqUniverseDownloader) -> None:
    with patch.object(ud.NasdaqUniverseDownloader, "_fetch_symbol_files", side_effect=RuntimeError("down")):
        assert downloader._download_from_nasdaq_trader() == []
