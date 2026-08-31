"""
P22 — Block C: Financial Screen (spec §4.3, M3).

Definitions match the spec table exactly. Every function reads through
`FeatureContext.get_latest_fact`/`get_trailing_average`, so `None` propagates
automatically wherever an underlying metric hasn't been normalized into
`p22_financial_fact` yet. As of 2026-08-30: `cash_runway_months` and
`dilution_risk`'s runway leg are real (their inputs — `cash_and_equivalents`,
`short_term_investments`, `quarterly_opex_burn` — are all normalized by
`ingest/financial_facts.py`); `enterprise_value`/`ev_to_cash`/`size_band`/
`atm_capacity_pct`, and `dilution_risk`'s catalyst leg, are still `None`
end-to-end — the former need `market_cap` (blocked on the market-data vendor
decision, spec §2.4/§2.0.6 — see `docs/Tasks.md`), the latter needs
CT.gov trial-completion-date extraction into a `catalyst_days_to_next` metric
(not built). These functions are correct and unit-tested against synthetic
fixtures now (spec §8.1: "every feature function with hand-constructed
fixtures, including the null path"); they start returning real values the
moment those upstream normalizers exist, with no change needed here.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.features.context import FeatureContext
from src.ml.pipeline.p22_biotech_ma.features.registry import register_feature

# EV band lower edges: <500M(0), 500M-2B(1), 2B-5B(2), 5B-15B(3), >15B(4).
_SIZE_BAND_EDGES = [500e6, 2e9, 5e9, 15e9]
SIZE_BAND_LABELS: List[str] = ["<500M", "500M-2B", "2B-5B", "5B-15B", ">15B"]

_AVG_DAYS_PER_MONTH = 30.44


@register_feature("block_c.enterprise_value")
def enterprise_value(company_id: int, as_of: date, ctx: FeatureContext) -> Optional[float]:
    """`market_cap - (cash + short_term_investments) + total_debt` (spec §4.3). May be negative
    (a legitimate result, spec §8.2) — market cap below net cash happens for real distressed names."""
    market_cap = ctx.get_latest_fact(company_id, "market_cap")
    cash = ctx.get_latest_fact(company_id, "cash_and_equivalents")
    if market_cap is None or cash is None:
        return None
    short_term_investments = ctx.get_latest_fact(company_id, "short_term_investments") or 0.0
    total_debt = ctx.get_latest_fact(company_id, "total_debt") or 0.0
    return market_cap - (cash + short_term_investments) + total_debt


@register_feature("block_c.cash_runway_months")
def cash_runway_months(company_id: int, as_of: date, ctx: FeatureContext) -> Optional[float]:
    """`(cash + short_term_investments) / trailing-4Q average quarterly operating burn` (spec §4.3),
    converted from quarters to months (x3). `None` if burn is zero/negative (no burn -> not
    meaningfully "runway-limited") or unknown, rather than dividing by zero or returning infinity.

    `quarterly_opex_burn` (`ingest/financial_facts.py`) stores the RAW SIGNED XBRL delta for
    `NetCashProvidedByUsedInOperatingActivities` — negative means cash used by operations. This
    function, not the normalizer, is where that sign gets turned into a burn magnitude (per that
    module's docstring: "this module does not reinterpret the sign... that's Block C's job"): a
    positive trailing average (net cash PROVIDED, not used) means there's no burn to compute a
    runway against, same as the missing-data case.
    """
    cash = ctx.get_latest_fact(company_id, "cash_and_equivalents")
    avg_operating_cash_flow = ctx.get_trailing_average(company_id, "quarterly_opex_burn", periods=4)
    if cash is None or avg_operating_cash_flow is None or avg_operating_cash_flow >= 0:
        return None
    quarterly_burn = -avg_operating_cash_flow  # flip cash-used-is-negative into a positive magnitude
    short_term_investments = ctx.get_latest_fact(company_id, "short_term_investments") or 0.0
    return (cash + short_term_investments) / quarterly_burn * 3.0


@register_feature("block_c.ev_to_cash")
def ev_to_cash(company_id: int, as_of: date, ctx: FeatureContext) -> Optional[float]:
    """`EV / cash` — flags negative-EV situations (spec §4.3)."""
    ev = enterprise_value(company_id, as_of, ctx)
    cash = ctx.get_latest_fact(company_id, "cash_and_equivalents")
    if ev is None or not cash:
        return None
    return ev / cash


@register_feature("block_c.dilution_risk")
def dilution_risk(company_id: int, as_of: date, ctx: FeatureContext) -> Optional[float]:
    """
    Boolean, encoded 1.0/0.0 per the shared feature-function signature
    (spec §4: `float | None`): runway < 12 months **and** a catalyst date
    inside the runway window (spec §4.3). `catalyst_days_to_next` isn't
    normalized into the store yet (needs CT.gov trial-completion-date
    extraction, M3+ work) — always `None` today.
    """
    runway_months = cash_runway_months(company_id, as_of, ctx)
    catalyst_days = ctx.get_latest_fact(company_id, "catalyst_days_to_next")
    if runway_months is None or catalyst_days is None:
        return None
    catalyst_inside_runway = catalyst_days <= runway_months * _AVG_DAYS_PER_MONTH
    return 1.0 if (runway_months < 12.0 and catalyst_inside_runway) else 0.0


@register_feature("block_c.atm_capacity_pct")
def atm_capacity_pct(company_id: int, as_of: date, ctx: FeatureContext) -> Optional[float]:
    """
    Remaining ATM shelf ÷ market cap, parsed from 424B5/10-Q (spec §4.3).
    `atm_shelf_remaining` isn't extracted into the store yet (needs 424B5/
    10-Q text parsing, not built) — always `None` today.
    """
    remaining_shelf = ctx.get_latest_fact(company_id, "atm_shelf_remaining")
    market_cap = ctx.get_latest_fact(company_id, "market_cap")
    if remaining_shelf is None or not market_cap:
        return None
    return remaining_shelf / market_cap


@register_feature("block_c.size_band")
def size_band(company_id: int, as_of: date, ctx: FeatureContext) -> Optional[float]:
    """
    Bucket EV into <500M/500M-2B/2B-5B/5B-15B/>15B (spec §4.3). Encoded as a
    float band index (0-4) per the shared `float | None` signature, not a
    string — map back to a label via `SIZE_BAND_LABELS[int(band)]`.
    """
    ev = enterprise_value(company_id, as_of, ctx)
    if ev is None:
        return None
    band = sum(1 for edge in _SIZE_BAND_EDGES if ev >= edge)
    return float(band)
