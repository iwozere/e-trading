"""Tests for features/block_c.py (spec §4.3). Fake repo, no live DB — exercises both the real
computation path (facts present) and the null path (spec §8.1: "including the null path")."""

import sys
from datetime import date
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.features.block_c import (
    atm_capacity_pct,
    cash_runway_months,
    dilution_risk,
    enterprise_value,
    ev_to_cash,
    size_band,
)
from src.ml.pipeline.p22_biotech_ma.features.context import FeatureContext

_AS_OF = date(2024, 6, 1)
_COMPANY_ID = 7


class _FakeRepo:
    """Duck-typed P22Repo stand-in: metric -> value, read through get_financial_facts_as_of."""

    def __init__(self, facts: Dict[str, float]):
        self._facts = facts

    def get_financial_facts_as_of(self, company_id: int, metric: str, as_of_date: date):
        del company_id, as_of_date
        if metric in self._facts:
            return [{"value": self._facts[metric]}]
        return []


def _ctx(facts: Dict[str, float]) -> FeatureContext:
    return FeatureContext(as_of=_AS_OF, repo=_FakeRepo(facts))


def test_enterprise_value_computes_from_market_cap_cash_debt():
    ctx = _ctx({"market_cap": 1_000.0, "cash_and_equivalents": 200.0, "short_term_investments": 50.0, "total_debt": 30.0})
    assert enterprise_value(_COMPANY_ID, _AS_OF, ctx) == 1_000.0 - (200.0 + 50.0) + 30.0


def test_enterprise_value_can_be_negative():
    """Negative EV is legitimate (spec §8.2) — must not be clamped to 0 or treated as missing."""
    ctx = _ctx({"market_cap": 50.0, "cash_and_equivalents": 200.0})
    ev = enterprise_value(_COMPANY_ID, _AS_OF, ctx)
    assert ev is not None
    assert ev == 50.0 - 200.0
    assert ev < 0


def test_enterprise_value_none_when_market_cap_missing():
    """market_cap is vendor-sourced and not wired yet (spec §2.4) — must propagate None, not 0."""
    ctx = _ctx({"cash_and_equivalents": 200.0})
    assert enterprise_value(_COMPANY_ID, _AS_OF, ctx) is None


def test_enterprise_value_none_when_cash_missing():
    ctx = _ctx({"market_cap": 1_000.0})
    assert enterprise_value(_COMPANY_ID, _AS_OF, ctx) is None


def test_enterprise_value_defaults_missing_debt_and_st_investments_to_zero():
    ctx = _ctx({"market_cap": 1_000.0, "cash_and_equivalents": 200.0})
    assert enterprise_value(_COMPANY_ID, _AS_OF, ctx) == 800.0


def test_cash_runway_months_converts_quarterly_burn_to_months():
    # quarterly_opex_burn is RAW SIGNED (negative = cash used) — see block_c.py docstring.
    ctx = _ctx({"cash_and_equivalents": 900.0, "quarterly_opex_burn": -100.0})
    # 900 / 100 = 9 quarters -> 27 months
    assert cash_runway_months(_COMPANY_ID, _AS_OF, ctx) == 27.0


def test_cash_runway_months_includes_short_term_investments():
    ctx = _ctx({"cash_and_equivalents": 500.0, "short_term_investments": 400.0, "quarterly_opex_burn": -100.0})
    assert cash_runway_months(_COMPANY_ID, _AS_OF, ctx) == 27.0


def test_cash_runway_months_none_when_burn_not_normalized_yet():
    """No quarterly_opex_burn history at all for this company — must be None, not a crash."""
    ctx = _ctx({"cash_and_equivalents": 900.0})
    assert cash_runway_months(_COMPANY_ID, _AS_OF, ctx) is None


def test_cash_runway_months_none_when_burn_zero():
    ctx = _ctx({"cash_and_equivalents": 900.0, "quarterly_opex_burn": 0.0})
    assert cash_runway_months(_COMPANY_ID, _AS_OF, ctx) is None


def test_cash_runway_months_none_when_operating_cash_flow_positive():
    """A positive average (net cash PROVIDED by operations, not used) means there's no burn to
    compute a runway against — not the same as missing data, but the same None result."""
    ctx = _ctx({"cash_and_equivalents": 900.0, "quarterly_opex_burn": 50.0})
    assert cash_runway_months(_COMPANY_ID, _AS_OF, ctx) is None


def test_ev_to_cash_flags_negative_ev():
    ctx = _ctx({"market_cap": 50.0, "cash_and_equivalents": 200.0})
    result = ev_to_cash(_COMPANY_ID, _AS_OF, ctx)
    assert result is not None
    assert result < 0


def test_ev_to_cash_none_when_ev_none():
    ctx = _ctx({})
    assert ev_to_cash(_COMPANY_ID, _AS_OF, ctx) is None


def test_ev_to_cash_none_when_cash_zero():
    ctx = _ctx({"market_cap": 1_000.0, "cash_and_equivalents": 0.0})
    assert ev_to_cash(_COMPANY_ID, _AS_OF, ctx) is None


def test_dilution_risk_true_when_runway_short_and_catalyst_inside_window():
    ctx = _ctx({
        "cash_and_equivalents": 300.0,
        "quarterly_opex_burn": -100.0,  # 9 months runway
        "catalyst_days_to_next": 60.0,
    })
    assert dilution_risk(_COMPANY_ID, _AS_OF, ctx) == 1.0


def test_dilution_risk_false_when_runway_long():
    ctx = _ctx({
        "cash_and_equivalents": 900.0,
        "quarterly_opex_burn": -30.0,  # 90 months runway
        "catalyst_days_to_next": 60.0,
    })
    assert dilution_risk(_COMPANY_ID, _AS_OF, ctx) == 0.0


def test_dilution_risk_none_when_catalyst_data_not_normalized_yet():
    ctx = _ctx({"cash_and_equivalents": 300.0, "quarterly_opex_burn": -100.0})
    assert dilution_risk(_COMPANY_ID, _AS_OF, ctx) is None


def test_atm_capacity_pct_none_when_shelf_not_extracted_yet():
    ctx = _ctx({"market_cap": 1_000.0})
    assert atm_capacity_pct(_COMPANY_ID, _AS_OF, ctx) is None


def test_atm_capacity_pct_computes_when_both_present():
    ctx = _ctx({"market_cap": 1_000.0, "atm_shelf_remaining": 100.0})
    assert atm_capacity_pct(_COMPANY_ID, _AS_OF, ctx) == 0.1


def test_size_band_buckets_correctly():
    cases: Tuple[Tuple[float, float], ...] = (
        (400e6, 0.0),
        (1e9, 1.0),
        (3e9, 2.0),
        (10e9, 3.0),
        (20e9, 4.0),
    )
    for market_cap, expected_band in cases:
        ctx = _ctx({"market_cap": market_cap, "cash_and_equivalents": 0.0})
        assert size_band(_COMPANY_ID, _AS_OF, ctx) == expected_band


def test_size_band_none_when_ev_none():
    ctx = _ctx({})
    assert size_band(_COMPANY_ID, _AS_OF, ctx) is None
