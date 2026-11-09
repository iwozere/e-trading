"""Tests for `features/ta_demand.py` and the presence fallback in `scoring/fit.ta_overlap`."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.features.ta_demand import DealAreaHistory, lead_therapeutic_area, ta_demand_features
from src.ml.pipeline.p22_biotech_ma.scoring.fit import AcquirerProfile, TargetProfile, ta_overlap


def _trial(area: str, phase: str, posted: date = date(2015, 1, 1)) -> Dict[str, Any]:
    return {"therapeutic_area": area, "phase": phase, "asset_id": 1, "first_posted_date": posted}


def test_lead_area_is_furthest_phase_then_most_common_classified():
    assert lead_therapeutic_area([_trial("neurology", "PHASE1"), _trial("oncology_solid", "PHASE3")]) == "oncology_solid"
    trials = [_trial("unclassified", "PHASE3"), _trial("neurology", "PHASE1"), _trial("autoimmune", "PHASE1"), _trial("neurology", "PHASE2")]
    assert lead_therapeutic_area(trials) == "neurology"
    assert lead_therapeutic_area([_trial("unclassified", "PHASE2")]) is None
    assert lead_therapeutic_area([]) is None


class _Repo:
    def __init__(self) -> None:
        self.trials = {
            1: [_trial("oncology_solid", "PHASE2", date(2015, 1, 1))],
            2: [_trial("neurology", "PHASE2", date(2015, 1, 1))],
            3: [_trial("oncology_solid", "PHASE1", date(2019, 1, 1))],  # posted after its deal
        }
        self.calls = 0

    def get_trials_for_company(self, company_id: int, as_of: date) -> List[Dict[str, Any]]:
        self.calls += 1
        return [t for t in self.trials.get(company_id, []) if t["first_posted_date"] <= as_of]


def _deal(target_id: int, announced: date, deal_type: str = "strategic_acquisition") -> Dict[str, Any]:
    return {"target_id": target_id, "announcement_date": announced, "deal_type": deal_type}


def test_deal_heat_counts_same_area_deals_in_trailing_window_only():
    repo = _Repo()
    history = DealAreaHistory(repo, [
        _deal(1, date(2018, 6, 1)),
        _deal(2, date(2018, 7, 1)),                       # other area
        _deal(1, date(2016, 1, 1)),                       # older than 24 months
        _deal(1, date(2020, 3, 1)),                       # after as_of
        _deal(1, date(2018, 8, 1), "reverse_merger"),     # not a positive deal type
        _deal(3, date(2018, 9, 1)),                       # its only trial was posted later: area unknown then
    ])
    assert history.heat("oncology_solid", date(2019, 1, 1)) == 1.0
    assert history.heat("neurology", date(2019, 1, 1)) == 1.0
    assert history.heat(None, date(2019, 1, 1)) is None
    calls = repo.calls
    history.heat("oncology_solid", date(2019, 2, 1))
    assert repo.calls == calls  # areas are cached across snapshots


def test_ta_demand_features_count_acquirers_with_phase3_in_area():
    phase3 = {100: {"oncology_solid": 3, "neurology": 1}, 101: {"oncology_solid": 2}, 102: {"autoimmune": 4}}
    features = ta_demand_features("oncology_solid", phase3, date(2019, 1, 1), None)
    assert features == {"ta_lead_area_known": 1.0, "ta_acquirer_demand": 2.0, "ta_acquirer_phase3_assets": 5.0, "ta_deal_heat_24m": None,
                        "ta_acquirer_revenue_growth": None}


def test_ta_acquirer_revenue_growth_is_median_of_acquirers_present_in_area():
    phase3 = {100: {"oncology_solid": 3}, 101: {"oncology_solid": 2}, 102: {"oncology_solid": 1}, 103: {"autoimmune": 4}}
    growth = {100: 0.10, 101: -0.05, 102: None, 103: 0.50}  # 102 unknown, 103 is in another area
    features = ta_demand_features("oncology_solid", phase3, date(2019, 1, 1), None, growth)
    assert features["ta_acquirer_revenue_growth"] == 0.025
    assert ta_demand_features("neurology", phase3, date(2019, 1, 1), None, growth)["ta_acquirer_revenue_growth"] is None
    unknown = ta_demand_features(None, phase3, date(2019, 1, 1), None)
    assert unknown.pop("ta_lead_area_known") == 0.0
    assert set(unknown.values()) == {None}


def test_ta_overlap_prefers_gap_then_falls_back_to_presence():
    target = TargetProfile(company_id=1, lead_therapeutic_area="oncology_solid")
    with_gap = AcquirerProfile(company_id=2, name="A", bloc="us", pipeline_gap_by_ta={"oncology_solid": -1.0}, phase3_by_ta={"oncology_solid": 3})
    assert ta_overlap(with_gap, target) == 0.0
    present = AcquirerProfile(company_id=2, name="A", bloc="us", phase3_by_ta={"oncology_solid": 3})
    absent = AcquirerProfile(company_id=2, name="A", bloc="us", phase3_by_ta={"neurology": 3})
    assert ta_overlap(present, target) == 1.0
    assert ta_overlap(absent, target) == 0.0
    assert ta_overlap(AcquirerProfile(company_id=2, name="A", bloc="us"), target) is None
