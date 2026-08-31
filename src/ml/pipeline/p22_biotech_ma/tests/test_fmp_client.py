"""Tests for ingest/fmp_client.py — mocked HTTP, no network calls."""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.fmp_client import FMPClient


@pytest.fixture(autouse=True)
def _no_real_sleep():
    # Status-500 test cases below exercise get_with_retry's real backoff loop —
    # skip the actual sleep so the suite doesn't take ~30s per 5xx case.
    with patch("src.ml.pipeline.p22_biotech_ma.ingest.http_retry.time.sleep"):
        yield


def _mock_response(json_data, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    return resp


def _client() -> FMPClient:
    # api_key passed explicitly so tests don't depend on FMP_API_KEY being set in the environment.
    return FMPClient(api_key="test-key")


def test_fetch_historical_price_full_returns_raw_json_verbatim():
    client = _client()
    raw_payload = [{"symbol": "MRNA", "date": "2024-01-02", "close": 100.0, "vwap": 99.5, "changePercent": 1.2}]
    client._client.get = MagicMock(return_value=_mock_response(raw_payload))

    result = client.fetch_historical_price_full("MRNA", date(2024, 1, 1), date(2024, 1, 31))

    assert result == raw_payload  # every field preserved, not just OHLCV


def test_fetch_historical_price_full_returns_none_on_unexpected_shape():
    client = _client()
    client._client.get = MagicMock(return_value=_mock_response({"unexpected": "dict, not a list"}))

    assert client.fetch_historical_price_full("MRNA", date(2024, 1, 1), date(2024, 1, 31)) is None


def test_fetch_historical_price_full_returns_none_on_404():
    client = _client()
    client._client.get = MagicMock(return_value=_mock_response({}, status_code=404))

    assert client.fetch_historical_price_full("DELISTEDX", date(2010, 1, 1), date(2010, 12, 31)) is None


def test_fetch_historical_price_full_returns_none_on_402():
    client = _client()
    client._client.get = MagicMock(return_value=_mock_response({}, status_code=402))

    assert client.fetch_historical_price_full("MRNA", date(2024, 1, 1), date(2024, 1, 31)) is None


def test_fetch_historical_price_full_returns_none_on_error_status():
    client = _client()
    client._client.get = MagicMock(return_value=_mock_response({}, status_code=500))

    assert client.fetch_historical_price_full("MRNA", date(2024, 1, 1), date(2024, 1, 31)) is None


def test_search_company_by_name_returns_list():
    client = _client()
    client._client.get = MagicMock(return_value=_mock_response([{"symbol": "XYZ", "name": "XYZ Corp"}]))

    results = client.search_company_by_name("XYZ Corp")

    assert results == [{"symbol": "XYZ", "name": "XYZ Corp"}]


def test_search_company_by_name_empty_on_non_list_response():
    """If the endpoint shape has changed (unverified, per module docstring), fail safe to empty."""
    client = _client()
    client._client.get = MagicMock(return_value=_mock_response({"error": "unexpected shape"}))

    assert client.search_company_by_name("XYZ Corp") == []


def test_search_company_by_name_empty_on_error_status():
    client = _client()
    client._client.get = MagicMock(return_value=_mock_response({}, status_code=500))

    assert client.search_company_by_name("XYZ Corp") == []
