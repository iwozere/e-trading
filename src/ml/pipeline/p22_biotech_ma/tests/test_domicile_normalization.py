"""Tests for ingest/domicile_normalization.py (spec §4.5)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.domicile_normalization import extract_is_foreign_domiciled


def test_extract_is_foreign_domiciled_true_via_is_foreign_location_flag():
    """Real shape: AstraZeneca PLC (live-verified 2026-09-08)."""
    submissions = {"addresses": {"business": {"stateOrCountry": None, "isForeignLocation": 1}}}
    assert extract_is_foreign_domiciled(submissions) is True


def test_extract_is_foreign_domiciled_true_via_state_or_country_when_flag_missing():
    """Real shape: Novo Nordisk A/S (live-verified 2026-09-08) — isForeignLocation is null despite
    being clearly foreign; stateOrCountry="G7" (Denmark) must catch it independently."""
    submissions = {"addresses": {"business": {"stateOrCountry": "G7", "isForeignLocation": None}}}
    assert extract_is_foreign_domiciled(submissions) is True


def test_extract_is_foreign_domiciled_false_for_real_us_state():
    submissions = {"addresses": {"business": {"stateOrCountry": "DE", "isForeignLocation": None}}}
    assert extract_is_foreign_domiciled(submissions) is False


def test_extract_is_foreign_domiciled_false_for_us_territory():
    submissions = {"addresses": {"business": {"stateOrCountry": "PR", "isForeignLocation": None}}}
    assert extract_is_foreign_domiciled(submissions) is False


def test_extract_is_foreign_domiciled_none_when_no_address_data():
    assert extract_is_foreign_domiciled({}) is None
    assert extract_is_foreign_domiciled({"addresses": {}}) is None
    assert extract_is_foreign_domiciled({"addresses": {"business": {}}}) is None


def test_extract_is_foreign_domiciled_case_insensitive_state_code():
    submissions = {"addresses": {"business": {"stateOrCountry": "de", "isForeignLocation": None}}}
    assert extract_is_foreign_domiciled(submissions) is False
