"""Tests for features/block_a.py (spec §4.1). Fake repo, no live DB — exercises both the real
computation path (facts present) and the null path (spec §8.1: "including the null path")."""

import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import src.ml.pipeline.p22_biotech_ma.features.block_a as block_a
from src.ml.pipeline.p22_biotech_ma.features.block_a import (
    cash_capacity,
    dry_powder,
    equity_capacity,
    percentile_rank,
    pipeline_gap_by_ta,
    revenue_at_risk_3y,
    revenue_at_risk_5y,
)
from src.ml.pipeline.p22_biotech_ma.features.context import FeatureContext

_AS_OF = date(2026, 9, 1)
_COMPANY_ID = 7


class _FakeRepo:
    """Duck-typed P22Repo stand-in for Block A's reads."""

    def __init__(
        self,
        facts: Dict[str, float] | None = None,
        patents: List[Dict[str, Any]] | None = None,
        phase3_counts: Dict[str, int] | None = None,
    ):
        self._facts = facts or {}
        self._patents = patents or []
        self._phase3_counts = phase3_counts or {}

    def get_financial_facts_as_of(self, company_id: int, metric: str, as_of_date: date):
        del company_id, as_of_date
        if metric in self._facts:
            return [{"value": self._facts[metric]}]
        return []

    def get_patent_expiries_for_acquirer(self, acquirer_id: int):
        del acquirer_id
        return self._patents

    def count_phase3_assets_by_therapeutic_area(self, company_id: int) -> Dict[str, int]:
        del company_id
        return self._phase3_counts


def _ctx(facts=None, patents=None, phase3_counts=None) -> FeatureContext:
    return FeatureContext(as_of=_AS_OF, repo=_FakeRepo(facts, patents, phase3_counts))


def setup_function():
    # cash_capacity reads this module-level constant directly; reset between tests since some
    # tests need to set it to exercise the real-computation path.
    block_a.TARGET_LEVERAGE_RATIO = None


# ---------------------------------------------------------------------
# revenue_at_risk_3y / revenue_at_risk_5y
# ---------------------------------------------------------------------

def test_revenue_at_risk_none_when_total_ttm_revenue_missing():
    ctx = _ctx(facts={}, patents=[{"loe_date": date(2027, 1, 1), "ttm_revenue_usd": 100.0}])
    assert revenue_at_risk_3y(_COMPANY_ID, _AS_OF, ctx) is None


def test_revenue_at_risk_none_when_in_window_patent_has_unknown_revenue():
    ctx = _ctx(
        facts={"total_ttm_revenue": 1000.0},
        patents=[{"loe_date": date(2027, 1, 1), "ttm_revenue_usd": None}],
    )
    assert revenue_at_risk_3y(_COMPANY_ID, _AS_OF, ctx) is None


def test_revenue_at_risk_real_computation():
    ctx = _ctx(
        facts={"total_ttm_revenue": 1000.0},
        patents=[
            {"loe_date": date(2027, 1, 1), "ttm_revenue_usd": 100.0},  # within 36mo window
            {"loe_date": date(2035, 1, 1), "ttm_revenue_usd": 500.0},  # far outside any window
        ],
    )
    assert revenue_at_risk_3y(_COMPANY_ID, _AS_OF, ctx) == 0.1


def test_revenue_at_risk_5y_includes_wider_window():
    ctx = _ctx(
        facts={"total_ttm_revenue": 1000.0},
        patents=[
            {"loe_date": date(2027, 1, 1), "ttm_revenue_usd": 100.0},  # inside both windows
            {"loe_date": date(2031, 6, 1), "ttm_revenue_usd": 200.0},  # inside 5y, outside 3y
        ],
    )
    assert revenue_at_risk_3y(_COMPANY_ID, _AS_OF, ctx) == 0.1
    assert revenue_at_risk_5y(_COMPANY_ID, _AS_OF, ctx) == 0.3


def test_revenue_at_risk_excludes_already_expired_patents():
    ctx = _ctx(
        facts={"total_ttm_revenue": 1000.0},
        patents=[{"loe_date": date(2020, 1, 1), "ttm_revenue_usd": 900.0}],
    )
    assert revenue_at_risk_3y(_COMPANY_ID, _AS_OF, ctx) == 0.0


# ---------------------------------------------------------------------
# cash_capacity
# ---------------------------------------------------------------------

def test_cash_capacity_none_when_target_leverage_ratio_not_curated():
    ctx = _ctx(facts={"cash_and_equivalents": 100.0, "ebitda": 50.0})
    assert cash_capacity(_COMPANY_ID, _AS_OF, ctx) is None


def test_cash_capacity_none_when_ebitda_missing():
    block_a.TARGET_LEVERAGE_RATIO = 2.0
    ctx = _ctx(facts={"cash_and_equivalents": 100.0})
    assert cash_capacity(_COMPANY_ID, _AS_OF, ctx) is None


def test_cash_capacity_real_computation():
    block_a.TARGET_LEVERAGE_RATIO = 2.0
    ctx = _ctx(facts={
        "cash_and_equivalents": 100.0, "short_term_investments": 20.0,
        "ebitda": 50.0, "total_debt": 60.0,
    })
    # existing_net_debt = 60 - 100 - 20 = -60
    # capacity = (100 + 20) + (2.0 * 50 - (-60)) = 120 + 160 = 280
    assert cash_capacity(_COMPANY_ID, _AS_OF, ctx) == 280.0


def test_cash_capacity_floored_at_zero():
    block_a.TARGET_LEVERAGE_RATIO = 0.1
    ctx = _ctx(facts={"cash_and_equivalents": 10.0, "ebitda": 5.0, "total_debt": 500.0})
    # existing_net_debt = 500 - 10 - 0 = 490; capacity = 10 + (0.5 - 490) = -479.5 -> floored
    assert cash_capacity(_COMPANY_ID, _AS_OF, ctx) == 0.0


# ---------------------------------------------------------------------
# equity_capacity / dry_powder
# ---------------------------------------------------------------------

def test_equity_capacity_none_when_currency_quality_missing():
    ctx = _ctx(facts={"market_cap": 1_000_000.0})
    assert equity_capacity(_COMPANY_ID, _AS_OF, ctx) is None


def test_equity_capacity_real_computation():
    ctx = _ctx(facts={"market_cap": 1_000_000.0, "currency_quality": 0.8})
    assert equity_capacity(_COMPANY_ID, _AS_OF, ctx) == 1_000_000.0 * 0.15 * 0.8


def test_dry_powder_none_when_either_leg_missing():
    ctx = _ctx(facts={"market_cap": 1_000_000.0, "currency_quality": 0.8})  # cash_capacity blocked
    assert dry_powder(_COMPANY_ID, _AS_OF, ctx) is None


def test_dry_powder_real_computation():
    block_a.TARGET_LEVERAGE_RATIO = 2.0
    ctx = _ctx(facts={
        "cash_and_equivalents": 100.0, "ebitda": 50.0, "total_debt": 0.0,
        "market_cap": 1_000_000.0, "currency_quality": 0.8,
    })
    expected_cash_capacity = 100.0 + (2.0 * 50.0 - (0.0 - 100.0))
    expected_equity_capacity = 1_000_000.0 * 0.15 * 0.8
    assert dry_powder(_COMPANY_ID, _AS_OF, ctx) == expected_cash_capacity + expected_equity_capacity


# ---------------------------------------------------------------------
# percentile_rank
# ---------------------------------------------------------------------

def test_percentile_rank_none_when_no_peers():
    assert percentile_rank(10.0, []) is None


def test_percentile_rank_middle_of_distribution():
    assert percentile_rank(3.0, [1.0, 2.0, 3.0, 4.0, 5.0]) == 0.6


def test_percentile_rank_lowest_value():
    assert percentile_rank(1.0, [1.0, 2.0, 3.0]) == 1 / 3


# ---------------------------------------------------------------------
# pipeline_gap_by_ta
# ---------------------------------------------------------------------

def test_pipeline_gap_by_ta_empty_when_no_phase3_assets():
    ctx = _ctx(phase3_counts={})
    assert pipeline_gap_by_ta(_COMPANY_ID, _AS_OF, ctx) == {}


def test_pipeline_gap_by_ta_none_when_inputs_missing():
    ctx = _ctx(phase3_counts={"oncology_solid": 2})
    result = pipeline_gap_by_ta(_COMPANY_ID, _AS_OF, ctx)
    assert result == {"oncology_solid": None}


def test_pipeline_gap_by_ta_real_computation():
    ctx = _ctx(
        facts={"revenue_at_risk_ta:oncology_solid": 500.0},
        phase3_counts={"oncology_solid": 2},
    )
    result = pipeline_gap_by_ta(
        _COMPANY_ID, _AS_OF, ctx, assumed_peak_sales_by_ta={"oncology_solid": 100.0}
    )
    assert result == {"oncology_solid": 500.0 - 2 * 100.0}
