"""Tests for ingest/sponsor_names.py. Pure string logic — names are real SEC registrant / CT.gov sponsor strings."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.sponsor_names import (
    company_query_tokens,
    ctgov_query_name,
    sponsor_matches_company,
)


def test_ctgov_query_name_strips_legal_forms_keeps_descriptive_words():
    assert ctgov_query_name("MAIA BIOTECHNOLOGY, INC.") == "maia biotechnology"
    assert ctgov_query_name("CORBUS PHARMACEUTICALS HOLDINGS, INC.") == "corbus pharmaceuticals"
    assert ctgov_query_name("TONIX PHARMACEUTICALS HOLDING CORP.") == "tonix pharmaceuticals"
    assert ctgov_query_name("COMPASS THERAPEUTICS, INC.") == "compass therapeutics"  # not just "compass"


def test_ctgov_query_name_drops_edgar_tags_dotted_forms_and_single_letters():
    assert ctgov_query_name("SEATTLE GENETICS INC /WA") == "seattle genetics"
    assert ctgov_query_name("VERTEX PHARMACEUTICALS INC / MA") == "vertex pharmaceuticals"
    assert ctgov_query_name("ICON PLC /ADR/") == "icon"
    assert ctgov_query_name("MERUS N.V.") == "merus"
    assert ctgov_query_name("NOVO NORDISK A S") == "novo nordisk"
    assert ctgov_query_name("JOHNSON & JOHNSON") == "johnson johnson"


def test_ctgov_query_name_falls_back_when_only_legal_forms():
    assert ctgov_query_name("Holdings Inc.") == "holdings inc."


def test_company_query_tokens_keeps_order():
    assert company_query_tokens("X4 Pharmaceuticals, Inc") == ["x4", "pharmaceuticals"]


def test_sponsor_matches_company_with_and_without_legal_suffix():
    assert sponsor_matches_company("Spero Therapeutics, Inc.", "SPERO THERAPEUTICS, INC.")
    assert sponsor_matches_company("Maia Biotechnology", "MAIA BIOTECHNOLOGY, INC.")
    assert sponsor_matches_company("ADC Therapeutics S.A.", "ADC Therapeutics SA")
    assert sponsor_matches_company("Merck Sharp & Dohme LLC", "MERCK & CO. INC.")


def test_sponsor_matches_company_rejects_other_lead_sponsors():
    assert not sponsor_matches_company("Hartford Hospital", "SPERO THERAPEUTICS, INC.")
    assert not sponsor_matches_company("Xencor, Inc.", "ICON PLC /ADR/")
    assert not sponsor_matches_company("Compass Pathways", "COMPASS THERAPEUTICS, INC.")


def test_sponsor_matches_company_empty_inputs_never_match():
    assert not sponsor_matches_company("", "SPERO THERAPEUTICS, INC.")
    assert not sponsor_matches_company("Anything Inc", "Inc.")
