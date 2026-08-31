"""Tests for ingest/clinicaltrials_client.py — mocked HTTP, no network calls."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.clinicaltrials_client import ClinicalTrialsClient


def _mock_response(json_data, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


def test_fetch_studies_for_sponsor_single_page():
    client = ClinicalTrialsClient()
    client._client.get = MagicMock(
        return_value=_mock_response({"studies": [{"nctId": "NCT001"}, {"nctId": "NCT002"}]})
    )

    studies = client.fetch_studies_for_sponsor("Example Therapeutics Inc")

    assert len(studies) == 2
    assert client._client.get.call_count == 1


def test_fetch_studies_for_sponsor_paginates():
    client = ClinicalTrialsClient()
    client._client.get = MagicMock(
        side_effect=[
            _mock_response({"studies": [{"nctId": "NCT001"}], "nextPageToken": "TOKEN2"}),
            _mock_response({"studies": [{"nctId": "NCT002"}]}),
        ]
    )

    studies = client.fetch_studies_for_sponsor("Example Therapeutics Inc")

    assert len(studies) == 2
    assert client._client.get.call_count == 2


def test_fetch_studies_returns_empty_on_no_results():
    client = ClinicalTrialsClient()
    client._client.get = MagicMock(return_value=_mock_response({"studies": []}))

    studies = client.fetch_studies_for_sponsor("Nonexistent Sponsor")

    assert studies == []


def test_fetch_study_version_history():
    client = ClinicalTrialsClient()
    client._client.get = MagicMock(
        return_value=_mock_response({"changes": [{"version": 0}, {"version": 1}]})
    )

    history = client.fetch_study_version_history("NCT001")

    assert len(history) == 2


def test_fetch_study_version_history_returns_empty_on_404():
    client = ClinicalTrialsClient()
    client._client.get = MagicMock(return_value=_mock_response({}, status_code=404))

    history = client.fetch_study_version_history("NCT_NOT_FOUND")

    assert history == []


def test_context_manager_closes_client():
    close_mock = MagicMock()
    with ClinicalTrialsClient() as client:
        client._client.close = close_mock
    close_mock.assert_called_once()
