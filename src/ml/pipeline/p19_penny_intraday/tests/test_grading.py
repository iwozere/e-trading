"""
Tests for P19 Layer 0 grading — the correctness requirements that gate Phase 1.5
sign-off (requirements-v2.md): unknown grades C never A/B, no positive signal
overrides a D disqualifier, insider_conviction renormalises over resolved
signals only.
"""

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p19_penny_intraday.config import P19StructuralConfig
from src.ml.pipeline.p19_penny_intraday.structural.grading import GradingInputs, grade_ticker

_AS_OF = date(2026, 8, 18)


def _company_facts(shares=None, cash=None, ocf=None, proceeds=None, buybacks=None):
    """Minimal companyfacts JSON with just the tags grading.py reads."""
    facts = {"dei": {}, "us-gaap": {}}
    if shares:
        facts["dei"]["EntityCommonStockSharesOutstanding"] = {
            "units": {"shares": [{"end": d.isoformat(), "val": v, "filed": d.isoformat()} for d, v in shares]}
        }
    if cash is not None:
        cash_date, cash_val = cash
        facts["us-gaap"]["CashAndCashEquivalentsAtCarryingValue"] = {
            "units": {"USD": [{"end": cash_date.isoformat(), "val": cash_val}]}
        }
    if ocf:
        facts["us-gaap"]["NetCashProvidedByUsedInOperatingActivities"] = {
            "units": {"USD": [{"start": s.isoformat(), "end": e.isoformat(), "val": v} for s, e, v in ocf]}
        }
    if proceeds:
        facts["us-gaap"]["ProceedsFromIssuanceOfCommonStock"] = {
            "units": {"USD": [{"start": s.isoformat(), "end": e.isoformat(), "val": v} for s, e, v in proceeds]}
        }
    if buybacks:
        facts["us-gaap"]["PaymentsForRepurchaseOfCommonStock"] = {
            "units": {"USD": [{"start": s.isoformat(), "end": e.isoformat(), "val": v} for s, e, v in buybacks]}
        }
    return {"facts": facts}


def _buy_row(name, days_ago, code="P"):
    return {
        "ticker": "AAA",
        "insider_name": name,
        "transaction_code": code,
        "filed_date": (_AS_OF - timedelta(days=days_ago)).isoformat(),
        "is_10b5_1_plan": False,
        "acquired_disposed_code": "A",
        "shares": 1000,
        "price_per_share": 1.0,
        "total_value_usd": 1000.0,
    }


# ── Rule 2: unknown grades C, never A/B (StructuralSignals.md §1, N17) ──────


def test_fpi_unresolvable_grades_c_never_a_or_b():
    """The GRSD case: no Form 4 (FPI exemption), no XBRL, no splits data."""
    inputs = GradingInputs(
        ticker="GRSD",
        cik=None,
        as_of=_AS_OF,
        company_facts=None,
        filings=None,
        splits=None,
        form4_rows=None,
    )
    profile = grade_ticker(inputs)
    assert profile.grade == "C"
    assert profile.coverage == 0.0
    assert profile.insider_conviction == 0.0  # null, not a low positive score


def test_partial_coverage_below_threshold_grades_c():
    inputs = GradingInputs(
        ticker="AAA",
        cik="123",
        as_of=_AS_OF,
        company_facts=None,
        filings=[],  # resolved, but empty -- some signals resolve, most don't
        splits=None,
        form4_rows=None,
    )
    profile = grade_ticker(inputs)
    assert profile.grade == "C"
    assert profile.coverage < 0.4


# ── Rule 1: no positive signal overrides a D disqualifier ───────────────────


def test_recent_reverse_split_grades_d_even_with_clean_everything_else():
    shares = [(date(2024, 6, 30), 8_000_000.0), (_AS_OF, 8_000_000.0)]  # flat -- would otherwise favour A
    inputs = GradingInputs(
        ticker="AAA",
        cik="123",
        as_of=_AS_OF,
        company_facts=_company_facts(shares=shares, cash=(_AS_OF, 5_000_000.0)),
        filings=[],
        splits=[(_AS_OF - timedelta(days=60), 0.1)],  # reverse split 2 months ago
        form4_rows=[_buy_row("A", 5), _buy_row("B", 6), _buy_row("C", 7)],  # would otherwise be P1
    )
    profile = grade_ticker(inputs)
    assert profile.grade == "D"
    assert profile.reverse_splits_24m == 1
    assert any("N1" in d for d in profile.disqualifiers)


def test_grade_d_disqualifier_present_even_with_high_insider_conviction():
    inputs = GradingInputs(
        ticker="AAA",
        cik="123",
        as_of=_AS_OF,
        company_facts=None,
        filings=[
            {"form": "8-K", "items": "3.01", "filingDate": (_AS_OF - timedelta(days=30)).isoformat()},
        ],
        splits=[],
        form4_rows=[_buy_row("A", 5), _buy_row("B", 6), _buy_row("C", 7)],  # P1 fires strongly
    )
    profile = grade_ticker(inputs)
    assert profile.grade == "D"  # N7 (active deficiency notice) wins regardless of P1


# ── insider_conviction renormalisation ───────────────────────────────────────


def test_insider_conviction_renormalises_over_resolved_signals_only():
    """
    An FPI with no XBRL/filings data but real Form 4 data (edge case, but the
    renormalisation logic must not be dragged down by the unresolved P3-P7
    signals -- only P1/P2 (both resolved, both firing) count.
    """
    inputs = GradingInputs(
        ticker="AAA",
        cik="123",
        as_of=_AS_OF,
        company_facts=None,  # P3/P4/P5/P6 unresolved
        filings=None,  # P7 unresolved
        splits=[],
        form4_rows=[_buy_row("A", 5), _buy_row("B", 6), _buy_row("C", 7)],  # P1 + P2 both fire
    )
    profile = grade_ticker(inputs)
    # Only P1 (35) + P2 (20) = 55 of the 120 total weight resolved, both fire.
    assert abs(profile.insider_conviction - 100.0) < 1e-9


def test_insider_conviction_zero_when_nothing_resolved():
    inputs = GradingInputs(
        ticker="AAA", cik=None, as_of=_AS_OF, company_facts=None, filings=None, splits=None, form4_rows=None
    )
    profile = grade_ticker(inputs)
    assert profile.insider_conviction == 0.0


# ── P3 polarity (regression for the aliasing bug caught during review) ──────


def test_p3_fires_on_flat_share_count_not_on_n3n4_disqualifier():
    """
    P3 (flat/declining share count -- positive) must be independent of N3/N4
    (high CAGR -- disqualifier). A flat series should resolve P3=True while
    N3/N4 do not fire, not silently mirror N3/N4's fire state.
    """
    shares = [(date(2024, 6, 30), 8_000_000.0), (_AS_OF, 8_000_000.0)]  # 0% CAGR
    inputs = GradingInputs(
        ticker="AAA",
        cik="123",
        as_of=_AS_OF,
        company_facts=_company_facts(shares=shares),
        filings=[],
        splits=[],
        form4_rows=[],
    )
    profile = grade_ticker(inputs)
    assert not any("N3" in d or "N4" in d for d in profile.disqualifiers)
    assert "P3" not in profile.disqualifier_severities  # positive signal, never a disqualifier


# ── N10/N11 de-cumulation wiring (grading-level smoke test) ─────────────────


def test_runway_computed_from_decumulated_burn():
    cf = _company_facts(
        cash=(_AS_OF, 1_500_000.0),
        ocf=[
            (date(2026, 1, 1), date(2026, 3, 31), -500_000.0),
            (date(2026, 4, 1), date(2026, 6, 30), -500_000.0),
        ],
    )
    inputs = GradingInputs(ticker="AAA", cik="123", as_of=_AS_OF, company_facts=cf, filings=[], splits=[], form4_rows=[])
    profile = grade_ticker(inputs)
    assert profile.runway_quarters == 3.0  # 1.5M / 500k burn -- right at the C threshold boundary (not <)
    assert profile.grade != "D"  # not < 1.5 nor shelf-active


def test_low_runway_with_active_shelf_grades_d():
    cf = _company_facts(
        cash=(_AS_OF, 300_000.0),
        ocf=[(date(2026, 4, 1), date(2026, 6, 30), -500_000.0)],
    )
    filings = [{"form": "S-3", "items": "", "filingDate": (_AS_OF - timedelta(days=100)).isoformat()}]
    inputs = GradingInputs(ticker="AAA", cik="123", as_of=_AS_OF, company_facts=cf, filings=filings, splits=[], form4_rows=[])
    profile = grade_ticker(inputs)
    assert profile.runway_quarters is not None and profile.runway_quarters < 1.5
    assert profile.shelf_active is True
    assert profile.grade == "D"


# ── Custom config is honoured ─────────────────────────────────────────────


def test_custom_thresholds_change_outcome():
    shares = [(date(2024, 6, 30), 8_000_000.0), (_AS_OF, 9_000_000.0)]  # ~6% CAGR
    inputs = GradingInputs(
        ticker="AAA", cik="123", as_of=_AS_OF, company_facts=_company_facts(shares=shares), filings=[], splits=[], form4_rows=[]
    )
    lenient = grade_ticker(inputs, cfg=P19StructuralConfig(share_cagr_c_threshold=0.5, share_cagr_d_threshold=0.9))
    strict = grade_ticker(inputs, cfg=P19StructuralConfig(share_cagr_c_threshold=0.01, share_cagr_d_threshold=0.9))
    assert not any("N4" in d for d in lenient.disqualifiers)
    assert any("N4" in d for d in strict.disqualifiers)


# ── Phase 3: N5 (floating convert), N6 (going concern) ──────────────────────


def test_n5_fires_at_configured_severity_not_hardcoded_d():
    """
    StructuralSignals.md open question 6: precision unmeasured, so the spec's
    own fallback is grade C, not D. Confirm the config knob actually governs
    it (and that the default is C, not D).
    """
    inputs = GradingInputs(
        ticker="AAA", cik="123", as_of=_AS_OF, company_facts=None, filings=[], splits=[], form4_rows=[],
        floating_convert_hit=True,
    )
    default_profile = grade_ticker(inputs)
    assert default_profile.disqualifier_severities["N5"] == "C"
    assert default_profile.grade == "C"

    promoted = grade_ticker(inputs, cfg=P19StructuralConfig(n5_severity="D"))
    assert promoted.disqualifier_severities["N5"] == "D"
    assert promoted.grade == "D"


def test_n5_no_hit_does_not_fire():
    inputs = GradingInputs(
        ticker="AAA", cik="123", as_of=_AS_OF, company_facts=None, filings=[], splits=[], form4_rows=[],
        floating_convert_hit=False,
    )
    profile = grade_ticker(inputs)
    assert "N5" not in profile.disqualifier_severities


def test_n6_going_concern_hit_grades_d():
    inputs = GradingInputs(
        ticker="AAA", cik="123", as_of=_AS_OF, company_facts=None, filings=[], splits=[], form4_rows=[],
        going_concern_hit=True,
    )
    profile = grade_ticker(inputs)
    assert profile.grade == "D"
    assert profile.disqualifier_severities["N6"] == "D"


def test_n5_n6_unresolved_by_default_depress_coverage_not_grade():
    """None (not fetched) must not be silently read as 'no match'."""
    inputs = GradingInputs(ticker="AAA", cik="123", as_of=_AS_OF, company_facts=None, filings=[], splits=[], form4_rows=[])
    profile = grade_ticker(inputs)
    assert "N5" not in profile.disqualifier_severities
    assert "N6" not in profile.disqualifier_severities
    assert profile.floating_convert_flag is None
    assert profile.going_concern_flag is None


# ── Phase 3: N15 (recent IPO + micro float + FPI) ────────────────────────────


def test_n15_fires_only_on_full_conjunction():
    base: Dict[str, Any] = dict(ticker="AAA", cik="123", as_of=_AS_OF, company_facts=None, filings=[], splits=[], form4_rows=[])

    all_three = GradingInputs(
        **base, listing_date=_AS_OF - timedelta(days=200), is_fpi=True, float_shares=2_000_000.0
    )
    assert grade_ticker(all_three).disqualifier_severities.get("N15") == "C"

    # Large float breaks the conjunction even with recent IPO + FPI.
    large_float = GradingInputs(
        **base, listing_date=_AS_OF - timedelta(days=200), is_fpi=True, float_shares=20_000_000.0
    )
    assert "N15" not in grade_ticker(large_float).disqualifier_severities

    # Old listing breaks it even with micro float + FPI.
    old_listing = GradingInputs(
        **base, listing_date=_AS_OF - timedelta(days=900), is_fpi=True, float_shares=2_000_000.0
    )
    assert "N15" not in grade_ticker(old_listing).disqualifier_severities

    # Domestic filer breaks it even with recent IPO + micro float.
    domestic = GradingInputs(
        **base, listing_date=_AS_OF - timedelta(days=200), is_fpi=False, float_shares=2_000_000.0
    )
    assert "N15" not in grade_ticker(domestic).disqualifier_severities


# ── Phase 3: N16 (auditor quality) ───────────────────────────────────────────


def test_n16_whitelisted_auditor_does_not_fire():
    inputs = GradingInputs(
        ticker="AAA", cik="123", as_of=_AS_OF, company_facts=None, filings=[], splits=[], form4_rows=[],
        auditor_name="Marcum LLP",
    )
    profile = grade_ticker(inputs)
    assert "N16" not in profile.disqualifier_severities
    assert profile.auditor_whitelisted is True


def test_n16_unknown_auditor_fires_c():
    inputs = GradingInputs(
        ticker="AAA", cik="123", as_of=_AS_OF, company_facts=None, filings=[], splits=[], form4_rows=[],
        auditor_name="Boladale Lawal & Co",
    )
    profile = grade_ticker(inputs)
    assert profile.disqualifier_severities["N16"] == "C"
    assert profile.auditor_whitelisted is False


# ── Phase 3: P8 (13D/G presence proxy), P9 (debt maturity) ──────────────────


def test_p8_dg_activity_resolves_and_feeds_conviction():
    inputs = GradingInputs(
        ticker="AAA", cik="123", as_of=_AS_OF, company_facts=None, filings=[], splits=[], form4_rows=[],
        dg_activity_2q=True,
    )
    profile = grade_ticker(inputs)
    assert profile.inst_13dg_activity_2q is True
    assert profile.insider_conviction > 0.0


def test_p9_no_near_term_maturity_is_positive():
    inputs = GradingInputs(
        ticker="AAA", cik="123", as_of=_AS_OF, company_facts=None, filings=[], splits=[], form4_rows=[],
        debt_maturity_near_term=False,
    )
    profile = grade_ticker(inputs)
    assert profile.no_debt_maturity_24m is True
    assert profile.insider_conviction > 0.0


def test_p9_near_term_maturity_does_not_fire_positive():
    inputs = GradingInputs(
        ticker="AAA", cik="123", as_of=_AS_OF, company_facts=None, filings=[], splits=[], form4_rows=[],
        debt_maturity_near_term=True,
    )
    profile = grade_ticker(inputs)
    assert profile.no_debt_maturity_24m is False


# ── Phase 3: P11 — SI conditional on grade (the highest-value conditionality) ─


def test_p11_high_si_feeds_conviction_at_clean_grade():
    """At grade A/B (no disqualifiers, decent coverage), high SI + rising DTC
    is squeeze fuel and must feed insider_conviction."""
    inputs = GradingInputs(
        ticker="AAA", cik="123", as_of=_AS_OF, company_facts=None, filings=[], splits=[], form4_rows=[],
        short_interest_pct_float=0.30, days_to_cover=5.0,
    )
    profile = grade_ticker(inputs)
    assert profile.grade in ("A", "B")
    assert profile.insider_conviction > 0.0


def test_p11_high_si_raises_dilution_urgency_not_conviction_at_dirty_grade():
    """At grade C/D, the identical SI numbers must NOT feed insider_conviction
    -- they only bump dilution_urgency (StructuralSignals.md §4 P11)."""
    dirty_base: Dict[str, Any] = dict(
        ticker="AAA", cik="123", as_of=_AS_OF, company_facts=None, filings=[], splits=[], form4_rows=[]
    )
    without_si = grade_ticker(GradingInputs(**dirty_base, going_concern_hit=True))
    with_si = grade_ticker(
        GradingInputs(**dirty_base, going_concern_hit=True, short_interest_pct_float=0.30, days_to_cover=5.0)
    )
    assert with_si.grade == "D" == without_si.grade
    # P11 must never enter the renormalisation at C/D -- adding SI data changes
    # nothing about insider_conviction (whatever P7 alone already contributed).
    assert with_si.insider_conviction == without_si.insider_conviction
    assert with_si.dilution_urgency > without_si.dilution_urgency  # context bump instead
