"""
Tests for P19 Layer 0 XBRL fact extraction — split-adjustment and de-cumulation
are the two non-negotiable correctness fixes (StructuralSignals.md §7 traps #1/#2,
requirements-v2.md).
"""

import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p19_penny_intraday.structural.xbrl_facts import (
    cagr,
    cash_and_burn,
    decumulate_quarterly,
    has_near_term_debt_maturity,
    proceeds_from_issuance,
    shares_outstanding_series,
    split_adjust,
)


def _shares_fact(*points, tag="EntityCommonStockSharesOutstanding", taxonomy="dei"):
    """points: (end_date_str, value, filed_date_str)"""
    return {
        "facts": {
            taxonomy: {
                tag: {
                    "units": {
                        "shares": [{"end": end, "val": val, "filed": filed} for end, val, filed in points]
                    }
                }
            }
        }
    }


def _duration_fact(*points, tag="NetCashProvidedByUsedInOperatingActivities"):
    """points: (start_date_str, end_date_str, value)"""
    return {
        "facts": {
            "us-gaap": {
                tag: {"units": {"USD": [{"start": s, "end": e, "val": v} for s, e, v in points]}}
            }
        }
    }


# ── shares_outstanding_series ───────────────────────────────────────────────


def test_shares_outstanding_series_extracts_and_sorts():
    cf = _shares_fact(("2026-03-31", 10_000_000, "2026-04-15"), ("2025-12-31", 8_000_000, "2026-01-20"))
    series = shares_outstanding_series(cf)
    assert series == [(date(2025, 12, 31), 8_000_000.0), (date(2026, 3, 31), 10_000_000.0)]


def test_shares_outstanding_series_dedups_to_latest_filed():
    # Same end date filed twice (restatement) -- keep the later filing.
    cf = _shares_fact(
        ("2026-03-31", 10_000_000, "2026-04-15"),
        ("2026-03-31", 10_500_000, "2026-06-01"),  # restated, filed later
    )
    series = shares_outstanding_series(cf)
    assert series == [(date(2026, 3, 31), 10_500_000.0)]


def test_shares_outstanding_series_empty_when_no_tag_resolves():
    assert shares_outstanding_series({"facts": {}}) == []


# ── split_adjust — the #1 trap ──────────────────────────────────────────────


def test_split_adjust_reverse_split_prevents_negative_cagr():
    """
    The exact failure mode StructuralSignals.md §7 #1 describes: a 1-for-10
    reverse split makes the unadjusted series look like collapsing share count.
    Post-adjustment, the CAGR must reflect the true dilution (positive), not
    the split artefact (negative).
    """
    # Pre-split: 100M shares. Reverse split 1-for-10 (ratio 0.1) on 2026-01-01.
    # Post-split: 12M shares (still net dilutive vs the 10M post-split-equivalent).
    unadjusted = [(date(2025, 6, 30), 100_000_000.0), (date(2026, 6, 30), 12_000_000.0)]
    splits = [(date(2026, 1, 1), 0.1)]

    adjusted = split_adjust(unadjusted, splits)
    # Pre-split point gets scaled down by the ratio to its post-split equivalent.
    assert adjusted[0] == (date(2025, 6, 30), 10_000_000.0)
    assert adjusted[1] == (date(2026, 6, 30), 12_000_000.0)

    unadjusted_cagr = cagr(unadjusted, lookback_quarters=8)
    adjusted_cagr = cagr(adjusted, lookback_quarters=8)
    assert unadjusted_cagr is not None and unadjusted_cagr < 0  # the artefact this fixes
    assert adjusted_cagr is not None and adjusted_cagr > 0  # true dilution, correctly positive


def test_split_adjust_forward_split_scales_up_history():
    series = [(date(2025, 1, 1), 5_000_000.0)]
    splits = [(date(2025, 6, 1), 2.0)]  # 2-for-1 forward split after this point
    adjusted = split_adjust(series, splits)
    assert adjusted == [(date(2025, 1, 1), 10_000_000.0)]


def test_split_adjust_no_splits_is_identity():
    series = [(date(2025, 1, 1), 5_000_000.0), (date(2025, 6, 1), 5_500_000.0)]
    assert split_adjust(series, []) == series


def test_split_adjust_only_applies_to_points_before_split_date():
    series = [(date(2026, 6, 1), 12_000_000.0)]  # after the split
    splits = [(date(2026, 1, 1), 0.1)]
    assert split_adjust(series, splits) == series  # unaffected -- already post-split


# ── decumulate_quarterly — the #2 trap ──────────────────────────────────────


def test_decumulate_quarterly_no_q4_spike_artefact():
    """
    Synthetic fiscal-year YTD series: Q1=100 (already discrete), Q2 YTD=250,
    Q3 YTD=390, Q4 YTD=400 (full year). Discrete quarters must be
    100, 150, 140, 10 -- NOT a spike at the fiscal-year-end entry.
    """
    entries = [
        (date(2026, 1, 1), date(2026, 3, 31), 100.0),  # Q1, ~90 days -> already discrete
        (date(2026, 1, 1), date(2026, 6, 30), 250.0),  # H1 YTD
        (date(2026, 1, 1), date(2026, 9, 30), 390.0),  # 9-month YTD
        (date(2026, 1, 1), date(2026, 12, 31), 400.0),  # FY YTD
    ]
    result = decumulate_quarterly(entries)
    values = [v for _, v in result]
    assert values == [100.0, 150.0, 140.0, 10.0]
    # The naive (un-decumulated) reading would show Q4 raw YTD (400) dwarfing
    # every discrete quarter -- confirm we did NOT reproduce that artefact.
    assert max(values) < 400.0


def test_decumulate_quarterly_all_already_discrete_passthrough():
    entries = [
        (date(2026, 1, 1), date(2026, 3, 31), 100.0),
        (date(2026, 4, 1), date(2026, 6, 30), 120.0),
    ]
    result = decumulate_quarterly(entries)
    assert result == [(date(2026, 3, 31), 100.0), (date(2026, 6, 30), 120.0)]


def test_decumulate_quarterly_empty_input():
    assert decumulate_quarterly([]) == []


# ── cash_and_burn ────────────────────────────────────────────────────────────


def test_cash_and_burn_runway_from_discrete_burn():
    cf = {
        "facts": {
            "us-gaap": {
                "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [{"end": "2026-06-30", "val": 3_000_000}]}},
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {
                        "USD": [
                            {"start": "2026-01-01", "end": "2026-03-31", "val": -500_000},
                            {"start": "2026-04-01", "end": "2026-06-30", "val": -500_000},
                        ]
                    }
                },
            }
        }
    }
    cash, burn, runway = cash_and_burn(cf, trailing_quarters=4)
    assert cash == 3_000_000
    assert burn == 500_000
    assert runway == 6.0


def test_cash_and_burn_positive_ocf_gives_null_burn_not_zero():
    """Positive operating cash flow every quarter -> burn is None (P5/P6 territory), not 0."""
    cf = {
        "facts": {
            "us-gaap": {
                "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [{"end": "2026-06-30", "val": 1_000_000}]}},
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {"USD": [{"start": "2026-01-01", "end": "2026-03-31", "val": 200_000}]}
                },
            }
        }
    }
    cash, burn, runway = cash_and_burn(cf)
    assert cash == 1_000_000
    assert burn is None
    assert runway is None


def test_cash_and_burn_missing_data_returns_nones():
    assert cash_and_burn({"facts": {}}) == (None, None, None)


# ── proceeds_from_issuance ───────────────────────────────────────────────────


def test_proceeds_from_issuance_decumulates():
    cf = _duration_fact(
        ("2026-01-01", "2026-03-31", 0.0),
        ("2026-01-01", "2026-06-30", 250_000.0),
        tag="ProceedsFromIssuanceOfCommonStock",
    )
    result = proceeds_from_issuance(cf)
    assert result == [(date(2026, 3, 31), 0.0), (date(2026, 6, 30), 250_000.0)]


# ── has_near_term_debt_maturity (P9) ─────────────────────────────────────────


def _instant_fact(end, val, tag="LongTermDebtCurrent"):
    return {"facts": {"us-gaap": {tag: {"units": {"USD": [{"end": end, "val": val, "filed": end}]}}}}}


def test_debt_maturity_positive_current_portion_fires_true():
    cf = _instant_fact("2026-06-30", 500_000.0)
    assert has_near_term_debt_maturity(cf) is True


def test_debt_maturity_zero_current_portion_is_false():
    cf = _instant_fact("2026-06-30", 0.0)
    assert has_near_term_debt_maturity(cf) is False


def test_debt_maturity_falls_back_to_maturities_schedule_tag():
    cf = _instant_fact("2026-06-30", 250_000.0, tag="LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths")
    assert has_near_term_debt_maturity(cf) is True


def test_debt_maturity_no_tag_at_all_is_unresolved():
    """Absence of a debt tag is ambiguous (no debt, or just not tagged) --
    must not be silently read as 'no maturity' (the P9 honesty requirement)."""
    cf = {"facts": {"us-gaap": {}}}
    assert has_near_term_debt_maturity(cf) is None
