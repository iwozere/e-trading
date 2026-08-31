"""Tests for ingest/asset_normalization.py. No live DB — repo is a MagicMock."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.asset_normalization import (
    extract_conditions,
    extract_single_intervention_name,
    resolve_or_create_asset,
)

_SINGLE_INTERVENTION_STUDY = {
    "protocolSection": {
        "armsInterventionsModule": {"interventions": [{"type": "DRUG", "name": "VX-522 mRNA therapy"}]},
        "conditionsModule": {"conditions": ["Cystic Fibrosis"]},
    }
}

_MULTI_INTERVENTION_STUDY = {
    "protocolSection": {
        "armsInterventionsModule": {
            "interventions": [
                {"type": "DRUG", "name": "VX-522 mRNA therapy"},
                {"type": "DRUG", "name": "IVA"},
            ]
        },
    }
}

_NO_DRUG_INTERVENTION_STUDY = {
    "protocolSection": {
        "armsInterventionsModule": {"interventions": [{"type": "DEVICE", "name": "Insulin Pump"}]},
    }
}


def test_extract_single_intervention_name_single_drug():
    assert extract_single_intervention_name(_SINGLE_INTERVENTION_STUDY) == "VX-522 mRNA therapy"


def test_extract_single_intervention_name_none_when_multiple():
    assert extract_single_intervention_name(_MULTI_INTERVENTION_STUDY) is None


def test_extract_single_intervention_name_none_when_zero_drug_biological():
    assert extract_single_intervention_name(_NO_DRUG_INTERVENTION_STUDY) is None


def test_extract_single_intervention_name_biological_type_counts():
    study = {
        "protocolSection": {
            "armsInterventionsModule": {"interventions": [{"type": "BIOLOGICAL", "name": "Some mAb"}]},
        }
    }
    assert extract_single_intervention_name(study) == "Some mAb"


def test_extract_conditions_returns_list():
    assert extract_conditions(_SINGLE_INTERVENTION_STUDY) == ["Cystic Fibrosis"]


def test_extract_conditions_empty_when_missing():
    assert extract_conditions(_MULTI_INTERVENTION_STUDY) == []


def test_resolve_or_create_asset_creates_new_when_no_existing():
    repo = MagicMock()
    repo.get_asset_by_company_and_name.return_value = None
    repo.upsert_asset.return_value = 99

    asset_id = resolve_or_create_asset(
        company_id=7, intervention_name="VX-522 mRNA therapy", conditions=["Cystic Fibrosis"], repo=repo
    )

    assert asset_id == 99
    repo.upsert_asset.assert_called_once()
    kwargs = repo.upsert_asset.call_args.kwargs
    assert kwargs["company_id"] == 7
    assert kwargs["name"] == "VX-522 mRNA therapy"
    assert kwargs["therapeutic_area"] == "respiratory"  # cystic fibrosis -> respiratory
    assert kwargs["indication"] == "Cystic Fibrosis"
    assert kwargs["modality"] is None
    assert kwargs["target_protein"] is None
    assert kwargs["is_lead"] is None


def test_resolve_or_create_asset_reuses_existing():
    repo = MagicMock()
    repo.get_asset_by_company_and_name.return_value = {"asset_id": 42}

    asset_id = resolve_or_create_asset(
        company_id=7, intervention_name="VX-522 mRNA therapy", conditions=["Cystic Fibrosis"], repo=repo
    )

    assert asset_id == 42
    repo.upsert_asset.assert_not_called()


def test_resolve_or_create_asset_unclassified_conditions_still_writes():
    repo = MagicMock()
    repo.get_asset_by_company_and_name.return_value = None
    repo.upsert_asset.return_value = 1

    resolve_or_create_asset(company_id=7, intervention_name="Novel Drug X", conditions=[], repo=repo)

    kwargs = repo.upsert_asset.call_args.kwargs
    assert kwargs["therapeutic_area"] == "unclassified"
    assert kwargs["indication"] is None
