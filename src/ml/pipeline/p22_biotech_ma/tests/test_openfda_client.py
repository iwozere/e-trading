"""Tests for ingest/openfda_client.py — mocked HTTP, no network calls."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.openfda_client import OpenFDAClient


def _mock_response(json_data, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


def test_fetch_applications_single_page():
    client = OpenFDAClient()
    client._client.get = MagicMock(
        return_value=_mock_response({"results": [{"application_number": "NDA001"}]})
    )

    apps = client.fetch_applications_for_sponsor("Example Therapeutics Inc")

    assert len(apps) == 1


def test_fetch_applications_paginates_until_short_page():
    client = OpenFDAClient()
    full_page = [{"application_number": f"NDA{i:03d}"} for i in range(100)]
    short_page = [{"application_number": "NDA999"}]
    client._client.get = MagicMock(
        side_effect=[
            _mock_response({"results": full_page}),
            _mock_response({"results": short_page}),
        ]
    )

    apps = client.fetch_applications_for_sponsor("Big Pharma Inc")

    assert len(apps) == 101
    assert client._client.get.call_count == 2


def test_404_treated_as_no_results_not_error():
    client = OpenFDAClient()
    client._client.get = MagicMock(return_value=_mock_response({}, status_code=404))

    apps = client.fetch_applications_for_sponsor("Nonexistent Sponsor")

    assert apps == []
