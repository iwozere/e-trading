"""
Tests for FinraTRFDownloader's OAuth token acquisition retry behavior.

2026-09-08 incident (monitoring.txt): the scheduled "FINRA TRF Daily Download"
job and two internal EMPS3 calls, all within ~10s of each other and sharing
the same FINRA_API_CLIENT, each got 400 Bad Request from FINRA's
client_credentials endpoint; an isolated call 2+ hours later with the same
credentials succeeded immediately. `_get_access_token` now retries a failed
token request a few times with backoff before giving up.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import requests

from src.data.downloader.finra_trf_downloader import FinraTRFDownloader


def _response(status_code: int, json_body: dict | None = None) -> MagicMock:
    """Build a fake `requests.Response` good enough for `raise_for_status`/`.json()`."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body or {}
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(f"{status_code} error", response=resp)
    else:
        resp.raise_for_status.return_value = None
    return resp


@patch("src.data.downloader.finra_trf_downloader.time.sleep")
@patch("src.data.downloader.finra_trf_downloader.requests.post")
def test_get_access_token_retries_and_succeeds(mock_post: MagicMock, mock_sleep: MagicMock) -> None:
    """A 400 followed by a success (the observed FINRA behavior) must not raise."""
    mock_post.side_effect = [
        _response(400),
        _response(200, {"access_token": "tok-123", "expires_in": 1800}),
    ]

    downloader = FinraTRFDownloader(date="2026-09-07", fetch_yfinance_data=False)
    token = downloader._get_access_token()

    assert token == "tok-123"
    assert mock_post.call_count == 2
    mock_sleep.assert_called_once()  # backoff between attempt 1 and 2, none after success


@patch("src.data.downloader.finra_trf_downloader.time.sleep")
@patch("src.data.downloader.finra_trf_downloader.requests.post")
def test_get_access_token_raises_after_exhausting_retries(mock_post: MagicMock, mock_sleep: MagicMock) -> None:
    """A persistently failing endpoint still raises, after the full retry budget."""
    mock_post.side_effect = [_response(400), _response(400), _response(400)]

    downloader = FinraTRFDownloader(date="2026-09-07", fetch_yfinance_data=False)
    with pytest.raises(requests.HTTPError):
        downloader._get_access_token()

    assert mock_post.call_count == downloader._TOKEN_REQUEST_MAX_ATTEMPTS
    assert mock_sleep.call_count == downloader._TOKEN_REQUEST_MAX_ATTEMPTS - 1


@patch("src.data.downloader.finra_trf_downloader.requests.post")
def test_get_access_token_uses_cache_within_expiry(mock_post: MagicMock) -> None:
    """A still-valid cached token is reused without another network call."""
    mock_post.return_value = _response(200, {"access_token": "tok-abc", "expires_in": 1800})

    downloader = FinraTRFDownloader(date="2026-09-07", fetch_yfinance_data=False)
    first = downloader._get_access_token()
    second = downloader._get_access_token()

    assert first == second == "tok-abc"
    mock_post.assert_called_once()
