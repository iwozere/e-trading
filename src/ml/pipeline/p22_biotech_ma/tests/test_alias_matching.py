"""Tests for ingest/alias_matching.py (spec §3.3). No live DB — repo is a MagicMock."""

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.alias_matching import (
    FUZZY_MATCH_THRESHOLD,
    extract_ctgov_sponsor_names,
    extract_openfda_sponsor_names,
    match_alias,
    resolve_aliases,
)

_KNOWN = {1: "Acme Therapeutics Inc", 2: "Beta Pharmaceuticals Corp"}
_KNOWN_FROM = datetime(2024, 3, 1, tzinfo=timezone.utc)


def test_match_alias_deterministic_on_normalized_equality():
    result = match_alias("ACME THERAPEUTICS, INC.", _KNOWN)
    assert result.match_type == "deterministic"
    assert result.company_id == 1
    assert result.score == 100.0


def test_match_alias_fuzzy_match_above_threshold():
    # A typo close enough to clear the token-set ratio >= 88 threshold, but not
    # an exact normalized match, so this exercises the fuzzy path specifically.
    result = match_alias("Acme Therapuetics Inc", _KNOWN)
    assert result.match_type == "fuzzy"
    assert result.company_id == 1


def test_match_alias_no_match_below_threshold():
    result = match_alias("Completely Unrelated Biotech Company XYZ", _KNOWN)
    assert result.match_type == "none"
    assert result.company_id is None


def test_match_alias_fuzzy_threshold_is_88():
    assert FUZZY_MATCH_THRESHOLD == 88


def test_resolve_aliases_writes_deterministic_match_as_verified_alias():
    repo = MagicMock()
    counts = resolve_aliases([("Acme Therapeutics Inc", _KNOWN_FROM)], _KNOWN, repo, source="clinicaltrials")

    repo.add_company_alias.assert_called_once_with(
        company_id=1,
        alias="Acme Therapeutics Inc",
        source="clinicaltrials",
        is_verified=True,
        known_from=_KNOWN_FROM,
    )
    repo.add_review_item.assert_not_called()
    assert counts["deterministic"] == 1


def test_resolve_aliases_routes_fuzzy_match_to_review_queue_not_alias_table():
    repo = MagicMock()
    counts = resolve_aliases([("Acme Therapuetics Inc", _KNOWN_FROM)], _KNOWN, repo, source="openfda")

    # Never auto-accepted (spec §3.3) — no direct alias write for a fuzzy match.
    repo.add_company_alias.assert_not_called()
    repo.add_review_item.assert_called_once()
    payload = repo.add_review_item.call_args.kwargs["payload"]
    assert payload["reason"] == "fuzzy_alias_candidate"
    assert payload["source"] == "openfda"
    assert payload["known_from"] == _KNOWN_FROM.isoformat()
    assert counts["fuzzy_flagged"] == 1


def test_resolve_aliases_logs_unresolved_without_any_write():
    repo = MagicMock()
    counts = resolve_aliases([("Totally Unrelated Corp XYZ", _KNOWN_FROM)], _KNOWN, repo, source="openfda")

    repo.add_company_alias.assert_not_called()
    repo.add_review_item.assert_not_called()
    assert counts["unresolved"] == 1


def test_extract_ctgov_sponsor_names_pulls_lead_sponsor_name():
    studies = [
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT00000001"},
                "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Acme Therapeutics Inc", "class": "INDUSTRY"}},
            }
        },
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT00000002"},
                "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Beta Pharmaceuticals Corp", "class": "INDUSTRY"}},
            }
        },
    ]
    assert extract_ctgov_sponsor_names(studies) == ["Acme Therapeutics Inc", "Beta Pharmaceuticals Corp"]


def test_extract_ctgov_sponsor_names_skips_missing_lead_sponsor():
    studies = [{"protocolSection": {"identificationModule": {"nctId": "NCT00000003"}}}]
    assert extract_ctgov_sponsor_names(studies) == []


def test_extract_openfda_sponsor_names_pulls_sponsor_name():
    applications = [{"sponsor_name": "ACME THERAPEUTICS"}, {"sponsor_name": "BETA PHARMACEUTICALS"}]
    assert extract_openfda_sponsor_names(applications) == ["ACME THERAPEUTICS", "BETA PHARMACEUTICALS"]


def test_extract_openfda_sponsor_names_skips_missing_sponsor_name():
    applications = [{"application_number": "NDA000000"}]
    assert extract_openfda_sponsor_names(applications) == []
