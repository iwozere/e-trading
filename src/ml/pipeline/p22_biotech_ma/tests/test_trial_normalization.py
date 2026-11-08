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


def _single_intervention_study(nct_id="NCT09999999", lead="Vertex Pharmaceuticals Incorporated", interventions=None):
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct_id},
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": lead}},
            "armsInterventionsModule": {"interventions": interventions or [{"type": "DRUG", "name": "VX-522 mRNA therapy"}]},
            "conditionsModule": {"conditions": ["Cystic Fibrosis"]},
        }
    }


def test_extract_trial_record_reads_lead_sponsor():
    record = extract_trial_record(_STUDY, _KNOWN_FROM)
    assert record is not None
    assert record.lead_sponsor == "Vertex Pharmaceuticals Incorporated"


def test_extract_trial_record_reads_start_and_first_posted_dates():
    study = {"protocolSection": {
        "identificationModule": {"nctId": "NCT00000002"},
        "statusModule": {"startDateStruct": {"date": "2015-03", "type": "ACTUAL"},
                         "studyFirstPostDateStruct": {"date": "2015-01-12", "type": "ACTUAL"}},
    }}
    record = extract_trial_record(study, _KNOWN_FROM)
    assert record is not None
    assert (record.start_date, record.first_posted_date) == (date(2015, 3, 1), date(2015, 1, 12))
    vx522 = extract_trial_record(_STUDY, _KNOWN_FROM)
    assert vx522 is not None and vx522.first_posted_date is None  # not in the payload


def test_write_trial_records_passes_trial_dates_to_upsert():
    repo = MagicMock()
    study = _single_intervention_study()
    study["protocolSection"]["statusModule"] = {"startDateStruct": {"date": "2016-05-01"}}
    write_trial_records(extract_trial_records([study], _KNOWN_FROM), repo)
    kwargs = repo.upsert_trial.call_args.kwargs
    assert (kwargs["start_date"], kwargs["first_posted_date"]) == (date(2016, 5, 1), None)


def test_write_trial_records_without_company_writes_trial_and_never_links():
    repo = MagicMock()
    records = extract_trial_records([_single_intervention_study()], _KNOWN_FROM)

    count = write_trial_records(records, repo)

    assert count == 1
    kwargs = repo.upsert_trial.call_args.kwargs
    assert kwargs["nct_id"] == "NCT09999999"
    assert "asset_id" not in kwargs  # attribution lives in p22_trial_asset, not on the trial row
    assert kwargs["known_from"] == _KNOWN_FROM
    repo.upsert_asset.assert_not_called()
    repo.replace_trial_asset_links.assert_not_called()  # existing links left untouched


def test_write_trial_records_multi_intervention_not_linked_but_links_still_replaced():
    """The _STUDY fixture has 2 DRUG interventions: no asset, and the company's link set is replaced
    with an empty one (so a stale link from an earlier run disappears)."""
    repo = MagicMock()
    records = extract_trial_records([_STUDY], _KNOWN_FROM)

    write_trial_records(records, repo, company_id=7, company_name="VERTEX PHARMACEUTICALS INC / MA")

    repo.upsert_asset.assert_not_called()
    repo.replace_trial_asset_links.assert_called_once_with(7, [])


def test_write_trial_records_links_asset_for_lead_sponsored_single_intervention_trial():
    repo = MagicMock()
    repo.get_asset_by_company_and_name.return_value = None
    repo.upsert_asset.return_value = 123
    records = extract_trial_records([_single_intervention_study()], _KNOWN_FROM)

    write_trial_records(records, repo, company_id=7, company_name="VERTEX PHARMACEUTICALS INC / MA")

    repo.upsert_asset.assert_called_once()
    assert repo.upsert_asset.call_args.kwargs["name"] == "VX-522 mRNA therapy"
    assert repo.upsert_asset.call_args.kwargs["modality"] == "rna_therapeutic"  # threaded through, 2026-09-08
    repo.replace_trial_asset_links.assert_called_once_with(7, [("NCT09999999", 123)])


def test_write_trial_records_skips_trial_led_by_another_sponsor():
    """A CRO's or partner's hit list contains trials it only collaborates on — not its assets."""
    repo = MagicMock()
    records = extract_trial_records([_single_intervention_study(lead="Xencor, Inc.")], _KNOWN_FROM)

    write_trial_records(records, repo, company_id=9, company_name="ICON PLC /ADR/")

    repo.upsert_trial.assert_called_once()  # the trial itself is still stored
    repo.upsert_asset.assert_not_called()
    repo.replace_trial_asset_links.assert_called_once_with(9, [])


def test_write_trial_records_placebo_controlled_trial_is_single_intervention():
    repo = MagicMock()
    repo.get_asset_by_company_and_name.return_value = {"asset_id": 55}
    study = _single_intervention_study(lead="Alpine Immune Sciences, Inc.", interventions=[
        {"type": "DRUG", "name": "ALPN-303"}, {"type": "DRUG", "name": "Placebo"},
    ])
    records = extract_trial_records([study], _KNOWN_FROM)

    write_trial_records(records, repo, company_id=3, company_name="ALPINE IMMUNE SCIENCES, INC.")

    repo.replace_trial_asset_links.assert_called_once_with(3, [("NCT09999999", 55)])
