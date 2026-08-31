"""Tests for ingest/trial_normalization.py. No live DB — repo is a MagicMock."""

import sys
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.trial_normalization import (
    extract_trial_record,
    extract_trial_records,
    write_trial_records,
)

_KNOWN_FROM = datetime(2026, 8, 30, tzinfo=timezone.utc)

# Trimmed to the fields CLINICALTRIALS_FIELDS actually fetches, shape
# live-verified 2026-08-30 against a real CT.gov study (NCT05668741).
_STUDY = {
    "protocolSection": {
        "identificationModule": {"nctId": "NCT05668741", "briefTitle": "A Phase 1/2 Study of VX-522"},
        "statusModule": {
            "overallStatus": "ACTIVE_NOT_RECRUITING",
            "primaryCompletionDateStruct": {"date": "2026-04-21", "type": "ACTUAL"},
        },
        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Vertex Pharmaceuticals Incorporated"}},
        "conditionsModule": {"conditions": ["Cystic Fibrosis"]},
        "designModule": {
            "studyType": "INTERVENTIONAL",
            "phases": ["PHASE1", "PHASE2"],
            "designInfo": {"allocation": "NA", "interventionModel": "SEQUENTIAL"},
            "enrollmentInfo": {"count": 26, "type": "ACTUAL"},
        },
        "armsInterventionsModule": {
            "interventions": [
                {"type": "DRUG", "name": "VX-522 mRNA therapy"},
                {"type": "DRUG", "name": "IVA"},
            ]
        },
        "outcomesModule": {
            "primaryOutcomes": [
                {"measure": "Safety and Tolerability as Assessed by Number of Participants With AEs"}
            ]
        },
        "contactsLocationsModule": {
            "locations": [
                {"facility": "UAB", "country": "United States"},
                {"facility": "Stanford", "country": "United States"},
            ]
        },
    }
}


def test_extract_trial_record_basic_fields():
    record = extract_trial_record(_STUDY, _KNOWN_FROM)

    assert record is not None
    assert record.nct_id == "NCT05668741"
    assert record.phase == "PHASE1/PHASE2"
    assert record.status == "ACTIVE_NOT_RECRUITING"
    assert record.enrollment == 26
    assert record.primary_completion_date == date(2026, 4, 21)
    assert record.countries == ["United States"]  # deduped
    assert record.primary_endpoint_text == "Safety and Tolerability as Assessed by Number of Participants With AEs"
    assert record.known_from == _KNOWN_FROM


def test_extract_trial_record_allocation_na_means_is_randomized_is_none():
    """CT.gov's `NA` allocation is a real "not applicable" case, not a False (see module docstring)."""
    record = extract_trial_record(_STUDY, _KNOWN_FROM)
    assert record is not None
    assert record.is_randomized is None


def test_extract_trial_record_randomized_allocation_maps_true():
    study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000001"},
            "designModule": {"designInfo": {"allocation": "RANDOMIZED"}},
        }
    }
    record = extract_trial_record(study, _KNOWN_FROM)
    assert record is not None
    assert record.is_randomized is True


def test_extract_trial_record_non_randomized_allocation_maps_false():
    study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000002"},
            "designModule": {"designInfo": {"allocation": "NON_RANDOMIZED"}},
        }
    }
    record = extract_trial_record(study, _KNOWN_FROM)
    assert record is not None
    assert record.is_randomized is False


def test_extract_trial_record_always_none_fields_not_fabricated():
    """uses_biomarker_selection / has_active_comparator / endpoint_changed_midtrial are always None
    this pass — the data to fill them isn't fetched yet (see module docstring)."""
    record = extract_trial_record(_STUDY, _KNOWN_FROM)
    assert record is not None
    assert record.uses_biomarker_selection is None
    assert record.has_active_comparator is None
    assert record.endpoint_changed_midtrial is None


def test_extract_trial_record_missing_nct_id_returns_none():
    assert extract_trial_record({"protocolSection": {}}, _KNOWN_FROM) is None


def test_extract_trial_record_partial_date_treated_as_first_of_month():
    study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000003"},
            "statusModule": {"primaryCompletionDateStruct": {"date": "2027-06", "type": "ESTIMATED"}},
        }
    }
    record = extract_trial_record(study, _KNOWN_FROM)
    assert record is not None
    assert record.primary_completion_date == date(2027, 6, 1)


def test_extract_trial_record_unparseable_date_returns_none_not_raises():
    study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000004"},
            "statusModule": {"primaryCompletionDateStruct": {"date": "not-a-date"}},
        }
    }
    record = extract_trial_record(study, _KNOWN_FROM)
    assert record is not None
    assert record.primary_completion_date is None


def test_extract_trial_records_drops_studies_missing_nct_id():
    studies = [_STUDY, {"protocolSection": {}}]
    records = extract_trial_records(studies, _KNOWN_FROM)
    assert len(records) == 1
    assert records[0].nct_id == "NCT05668741"


def test_write_trial_records_calls_repo_with_asset_id_none_without_company_id():
    """Multi-intervention study (the _STUDY fixture), no company_id given: never linked."""
    repo = MagicMock()
    records = extract_trial_records([_STUDY], _KNOWN_FROM)

    count = write_trial_records(records, repo)

    assert count == 1
    repo.upsert_trial.assert_called_once()
    kwargs = repo.upsert_trial.call_args.kwargs
    assert kwargs["nct_id"] == "NCT05668741"
    assert kwargs["asset_id"] is None  # deliberately not linked — see module docstring
    assert kwargs["known_from"] == _KNOWN_FROM


def test_write_trial_records_asset_id_none_for_multi_intervention_even_with_company_id():
    """The _STUDY fixture has 2 DRUG interventions — must stay unlinked even when company_id IS given."""
    repo = MagicMock()
    records = extract_trial_records([_STUDY], _KNOWN_FROM)

    write_trial_records(records, repo, company_id=7)

    repo.upsert_asset.assert_not_called()
    kwargs = repo.upsert_trial.call_args.kwargs
    assert kwargs["asset_id"] is None


def test_write_trial_records_links_asset_for_single_intervention_trial():
    single_intervention_study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT09999999"},
            "armsInterventionsModule": {"interventions": [{"type": "DRUG", "name": "VX-522 mRNA therapy"}]},
            "conditionsModule": {"conditions": ["Cystic Fibrosis"]},
        }
    }
    repo = MagicMock()
    repo.get_asset_by_company_and_name.return_value = None
    repo.upsert_asset.return_value = 123
    records = extract_trial_records([single_intervention_study], _KNOWN_FROM)

    write_trial_records(records, repo, company_id=7)

    repo.upsert_asset.assert_called_once()
    assert repo.upsert_asset.call_args.kwargs["name"] == "VX-522 mRNA therapy"
    trial_kwargs = repo.upsert_trial.call_args.kwargs
    assert trial_kwargs["asset_id"] == 123
