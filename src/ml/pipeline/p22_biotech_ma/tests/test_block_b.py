"""Tests for features/block_b.py (spec §4.2). Fake repo, no live DB — exercises both the real
computation path and the null path (spec §8.1: "including the null path")."""

import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.features.block_b import (
    asset_count_ph2plus,
    catalyst_window,
    lead_asset_poa,
    phase_max,
)
from src.ml.pipeline.p22_biotech_ma.features.context import FeatureContext
from src.ml.pipeline.p22_biotech_ma.ingest.base_rates_config import BaseRates

_AS_OF = date(2026, 9, 8)
_COMPANY_ID = 7


class _FakeRepo:
    def __init__(self, trials: List[Dict[str, Any]] | None = None):
        self._trials = trials or []

    def get_trials_for_company(self, company_id: int):
        del company_id
        return self._trials


def _ctx(trials=None) -> FeatureContext:
    return FeatureContext(as_of=_AS_OF, repo=_FakeRepo(trials))


def _trial(asset_id=1, ta="oncology_solid", phase="PHASE1", pcd=None, status="RECRUITING"):
    return {"asset_id": asset_id, "therapeutic_area": ta, "phase": phase,
            "primary_completion_date": pcd, "status": status}


# ---------------------------------------------------------------------
# phase_max
# ---------------------------------------------------------------------

def test_phase_max_none_when_no_trials():
    assert phase_max(_COMPANY_ID, _AS_OF, _ctx([])) is None


def test_phase_max_takes_highest_across_trials():
    trials = [_trial(phase="PHASE1"), _trial(phase="PHASE3"), _trial(phase="PHASE2")]
    assert phase_max(_COMPANY_ID, _AS_OF, _ctx(trials)) == 3.0  # PHASE3 rank


def test_phase_max_combined_phase_takes_highest_part():
    trials = [_trial(phase="PHASE2/PHASE3")]
    assert phase_max(_COMPANY_ID, _AS_OF, _ctx(trials)) == 3.0


def test_phase_max_ignores_unrecognized_phase():
    trials = [_trial(phase="NA"), _trial(phase="PHASE1")]
    assert phase_max(_COMPANY_ID, _AS_OF, _ctx(trials)) == 1.0


# ---------------------------------------------------------------------
# asset_count_ph2plus
# ---------------------------------------------------------------------

def test_asset_count_ph2plus_none_when_no_trials():
    assert asset_count_ph2plus(_COMPANY_ID, _AS_OF, _ctx([])) is None


def test_asset_count_ph2plus_zero_when_all_pre_phase2():
    trials = [_trial(asset_id=1, phase="PHASE1"), _trial(asset_id=2, phase="EARLY_PHASE1")]
    assert asset_count_ph2plus(_COMPANY_ID, _AS_OF, _ctx(trials)) == 0.0


def test_asset_count_ph2plus_counts_distinct_assets_not_trials():
    trials = [
        _trial(asset_id=1, phase="PHASE2"), _trial(asset_id=1, phase="PHASE3"),  # same asset, 2 trials
        _trial(asset_id=2, phase="PHASE1"),  # pre-ph2
        _trial(asset_id=3, phase="PHASE3"),
    ]
    assert asset_count_ph2plus(_COMPANY_ID, _AS_OF, _ctx(trials)) == 2.0


# ---------------------------------------------------------------------
# catalyst_window
# ---------------------------------------------------------------------

def test_catalyst_window_none_when_no_upcoming_dates():
    trials = [_trial(pcd=date(2020, 1, 1))]  # already past
    assert catalyst_window(_COMPANY_ID, _AS_OF, _ctx(trials)) is None


def test_catalyst_window_buckets_nearest_upcoming():
    trials = [_trial(pcd=date(2026, 10, 1)), _trial(pcd=date(2027, 6, 1))]  # ~23 days and ~9 months out
    assert catalyst_window(_COMPANY_ID, _AS_OF, _ctx(trials)) == 0.0  # nearest wins, <60 days


def test_catalyst_window_far_future_bucket():
    trials = [_trial(pcd=date(2029, 1, 1))]
    assert catalyst_window(_COMPANY_ID, _AS_OF, _ctx(trials)) == 4.0  # >730 days


def test_catalyst_window_mid_buckets():
    assert catalyst_window(_COMPANY_ID, _AS_OF, _ctx([_trial(pcd=date(2026, 12, 1))])) == 1.0  # 60-180
    assert catalyst_window(_COMPANY_ID, _AS_OF, _ctx([_trial(pcd=date(2027, 4, 1))])) == 2.0  # 180-365 (205d)
    assert catalyst_window(_COMPANY_ID, _AS_OF, _ctx([_trial(pcd=date(2028, 1, 1))])) == 3.0  # 365-730 (481d)


# ---------------------------------------------------------------------
# lead_asset_poa
# ---------------------------------------------------------------------

_FAKE_RATES = BaseRates(
    loa_from_phase_1_overall=0.096,
    by_therapeutic_area={"oncology_solid": 0.046, "immunology": None},
    biomarker_selection_modifier=2.0,
)


def test_lead_asset_poa_none_when_no_trials():
    with patch("src.ml.pipeline.p22_biotech_ma.features.block_b.load_base_rates", return_value=_FAKE_RATES):
        assert lead_asset_poa(_COMPANY_ID, _AS_OF, _ctx([])) is None


def test_lead_asset_poa_real_computation_for_phase1_lead_asset():
    trials = [_trial(ta="oncology_solid", phase="PHASE1")]
    with patch("src.ml.pipeline.p22_biotech_ma.features.block_b.load_base_rates", return_value=_FAKE_RATES):
        assert lead_asset_poa(_COMPANY_ID, _AS_OF, _ctx(trials)) == 0.046


def test_lead_asset_poa_none_when_lead_asset_past_phase1():
    # Lead asset (furthest-progressed) is at PHASE3 -> spec's LOA-from-Phase-1 figure doesn't apply.
    trials = [_trial(ta="oncology_solid", phase="PHASE3")]
    with patch("src.ml.pipeline.p22_biotech_ma.features.block_b.load_base_rates", return_value=_FAKE_RATES):
        assert lead_asset_poa(_COMPANY_ID, _AS_OF, _ctx(trials)) is None


def test_lead_asset_poa_falls_back_to_overall_when_ta_rate_is_null():
    trials = [_trial(ta="immunology", phase="PHASE1")]
    with patch("src.ml.pipeline.p22_biotech_ma.features.block_b.load_base_rates", return_value=_FAKE_RATES):
        assert lead_asset_poa(_COMPANY_ID, _AS_OF, _ctx(trials)) == 0.096


def test_lead_asset_poa_uses_furthest_progressed_asset_as_lead_proxy():
    trials = [
        _trial(asset_id=1, ta="oncology_solid", phase="EARLY_PHASE1"),
        _trial(asset_id=2, ta="immunology", phase="PHASE1"),  # more advanced -> the proxy lead
    ]
    with patch("src.ml.pipeline.p22_biotech_ma.features.block_b.load_base_rates", return_value=_FAKE_RATES):
        assert lead_asset_poa(_COMPANY_ID, _AS_OF, _ctx(trials)) == 0.096  # immunology's fallback rate
