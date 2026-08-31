"""
Tests for ingest/http_retry.py — the retryable-vs-non-retryable distinction
that the original per-client implementations got wrong (see
docs/implementation-plan.md §4.1).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.http_retry import get_with_retry


@pytest.fixture(autouse=True)
def _no_real_sleep():
    with patch("src.ml.pipeline.p22_biotech_ma.ingest.http_retry.time.sleep"):
        yield


def _mock_response(status_code):
    resp = MagicMock()
    resp.status_code = status_code
    return resp


def test_success_returns_immediately_without_retry():
    client = MagicMock()
    client.get.return_value = _mock_response(200)

    resp = get_with_retry(client, "https://example.com", max_attempts=5)

    assert resp is not None
    assert resp.status_code == 200
    assert client.get.call_count == 1


def test_400_is_returned_immediately_not_retried():
    """
    The bug this module exists to fix: a non-retryable 4xx must not be
    retried 5 times with backoff — it must come straight back so the caller
    can decide (fail fast, or treat a specific code like 404 specially).
    """
    client = MagicMock()
    client.get.return_value = _mock_response(400)

    resp = get_with_retry(client, "https://example.com", max_attempts=5)

    assert resp is not None
    assert resp.status_code == 400
    assert client.get.call_count == 1


def test_404_is_returned_immediately_not_retried():
    client = MagicMock()
    client.get.return_value = _mock_response(404)

    resp = get_with_retry(client, "https://example.com", max_attempts=5)

    assert resp is not None
    assert resp.status_code == 404
    assert client.get.call_count == 1


def test_429_is_retried_until_success():
    client = MagicMock()
    client.get.side_effect = [_mock_response(429), _mock_response(429), _mock_response(200)]

    resp = get_with_retry(client, "https://example.com", max_attempts=5)

    assert resp is not None
    assert resp.status_code == 200
    assert client.get.call_count == 3


def test_500_is_retried_and_gives_up_after_max_attempts():
    client = MagicMock()
    client.get.return_value = _mock_response(503)

    resp = get_with_retry(client, "https://example.com", max_attempts=3)

    assert resp is None
    assert client.get.call_count == 3


def test_rate_limiter_acquire_called_before_each_attempt():
    client = MagicMock()
    client.get.side_effect = [_mock_response(429), _mock_response(200)]
    limiter = MagicMock()

    get_with_retry(client, "https://example.com", rate_limiter=limiter, max_attempts=5)

    assert limiter.acquire.call_count == 2


def test_transport_error_is_retried():
    import httpx

    client = MagicMock()
    client.get.side_effect = [httpx.ConnectError("boom"), _mock_response(200)]

    resp = get_with_retry(client, "https://example.com", max_attempts=5)

    assert resp is not None
    assert resp.status_code == 200
