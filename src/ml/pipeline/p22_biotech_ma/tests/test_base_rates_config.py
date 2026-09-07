"""Tests for ingest/base_rates_config.py."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.base_rates_config import BaseRates, load_base_rates


def test_load_base_rates_reads_real_repo_config():
    """Round-trips the real config/pipeline/p22_base_rates.yaml file."""
    rates = load_base_rates()
    assert isinstance(rates, BaseRates)
    assert 0.0 < rates.loa_from_phase_1_overall < 1.0
    assert rates.by_therapeutic_area["oncology_solid"] is not None
    assert rates.biomarker_selection_modifier == 2.0


def test_load_base_rates_raises_on_missing_keys(tmp_path):
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("loa_from_phase_1_overall: 0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required top-level keys"):
        load_base_rates(bad_file)


def test_load_base_rates_preserves_null_areas(tmp_path):
    f = tmp_path / "rates.yaml"
    f.write_text(
        "loa_from_phase_1_overall: 0.1\n"
        "by_therapeutic_area:\n  oncology_solid: 0.05\n  immunology: null\n"
        "modifiers:\n  biomarker_selection: 2.0\n",
        encoding="utf-8",
    )
    rates = load_base_rates(f)
    assert rates.by_therapeutic_area == {"oncology_solid": 0.05, "immunology": None}
