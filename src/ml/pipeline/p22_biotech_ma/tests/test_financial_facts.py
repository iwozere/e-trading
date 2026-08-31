"""Tests for ingest/financial_facts.py. No live DB — repo is a MagicMock."""

import sys
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.financial_facts import (
    DURATION_DELTA_TAG_MAP,
    FACT_TAG_MAP,
    NormalizedFact,
    extract_fact_series,
    extract_quarterly_delta_series,
    filing_index_url,
    write_financial_facts,
)


def _companyfacts(entries, taxonomy="us-gaap", tag="CashAndCashEquivalentsAtCarryingValue", unit="USD"):
    return {"facts": {taxonomy: {tag: {"units": {unit: entries}}}}}


def test_filing_index_url_matches_the_live_verified_edgar_pattern():
    url = filing_index_url("0000873303", "0001193125-26-335003")
    assert url == "https://www.sec.gov/Archives/edgar/data/873303/000119312526335003/0001193125-26-335003-index.htm"


def test_extract_fact_series_unknown_metric_raises():
    with pytest.raises(ValueError):
        extract_fact_series({}, "0000873303", "not_a_real_metric")


def test_extract_fact_series_missing_tag_returns_empty():
    assert extract_fact_series({"facts": {}}, "0000873303", "cash_and_equivalents") == []


def test_extract_fact_series_basic_extraction():
    payload = _companyfacts(
        [
            {"end": "2024-03-31", "val": 100_000_000, "accn": "0001-24-000001", "filed": "2024-05-01", "form": "10-Q"},
            {"end": "2024-06-30", "val": 90_000_000, "accn": "0001-24-000002", "filed": "2024-08-01", "form": "10-Q"},
        ]
    )

    facts = extract_fact_series(payload, "0000873303", "cash_and_equivalents")

    assert len(facts) == 2
    assert facts[0].period_end == date(2024, 3, 31)
    assert facts[0].value == 100_000_000
    assert facts[0].known_from == datetime(2024, 5, 1, tzinfo=timezone.utc)
    assert facts[0].unit == "USD"
    assert facts[0].source_id == "0001-24-000001"
    # Sorted by known_from ascending.
    assert facts[1].known_from > facts[0].known_from


def test_extract_fact_series_dedups_comparative_column_reruns_keeping_earliest_filed():
    """A later 10-Q re-reporting an unchanged prior-period balance as a comparative column must
    NOT be treated as a new fact known only as of the later filing date."""
    payload = _companyfacts(
        [
            {"end": "2025-12-31", "val": 500_000_000, "accn": "0001-26-A", "filed": "2026-03-01", "form": "10-K"},
            # Same period, same value, re-reported in the next quarter's 10-Q as a comparative column.
            {"end": "2025-12-31", "val": 500_000_000, "accn": "0001-26-B", "filed": "2026-05-01", "form": "10-Q"},
        ]
    )

    facts = extract_fact_series(payload, "0000873303", "cash_and_equivalents")

    assert len(facts) == 1
    assert facts[0].known_from == datetime(2026, 3, 1, tzinfo=timezone.utc)  # the EARLIER filing, not the later one
    assert facts[0].source_id == "0001-26-A"


def test_extract_fact_series_possible_restatement_is_not_written():
    """A later filing reporting a DIFFERENT value for an already-seen period is logged, not written
    (known limitation — see module docstring) — must not silently pick one value over the other."""
    payload = _companyfacts(
        [
            {"end": "2025-12-31", "val": 500_000_000, "accn": "0001-26-A", "filed": "2026-03-01", "form": "10-K"},
            {"end": "2025-12-31", "val": 480_000_000, "accn": "0001-26-B", "filed": "2026-05-01", "form": "10-Q/A"},
        ]
    )

    facts = extract_fact_series(payload, "0000873303", "cash_and_equivalents")

    assert len(facts) == 1
    assert facts[0].value == 500_000_000  # earliest-filed value kept; the restatement is dropped, not silently applied


def test_extract_fact_series_skips_entries_missing_required_fields():
    payload = _companyfacts(
        [
            {"end": "2024-03-31", "val": None, "accn": "0001-24-000001", "filed": "2024-05-01"},
            {"end": None, "val": 100, "accn": "0001-24-000002", "filed": "2024-05-01"},
            {"end": "2024-06-30", "val": 90_000_000, "accn": "0001-24-000003", "filed": "2024-08-01"},
        ]
    )
    facts = extract_fact_series(payload, "0000873303", "cash_and_equivalents")
    assert len(facts) == 1
    assert facts[0].period_end == date(2024, 6, 30)


def test_extract_fact_series_shares_outstanding_uses_dei_units():
    payload = _companyfacts(
        [{"end": "2024-03-31", "val": 100_000_000, "accn": "0001-24-000001", "filed": "2024-05-01"}],
        taxonomy="dei",
        tag="EntityCommonStockSharesOutstanding",
        unit="shares",
    )
    facts = extract_fact_series(payload, "0000873303", "shares_outstanding")
    assert len(facts) == 1
    assert facts[0].unit == "shares"


def test_fact_tag_map_has_the_live_verified_metrics():
    assert FACT_TAG_MAP["cash_and_equivalents"] == [("us-gaap", "CashAndCashEquivalentsAtCarryingValue", "USD")]
    assert FACT_TAG_MAP["shares_outstanding"] == [("dei", "EntityCommonStockSharesOutstanding", "shares")]
    assert FACT_TAG_MAP["short_term_investments"] == [("us-gaap", "ShortTermInvestments", "USD")]
    assert FACT_TAG_MAP["total_debt"] == [
        ("us-gaap", "LongTermDebtNoncurrent", "USD"),
        ("us-gaap", "LongTermDebt", "USD"),
        ("us-gaap", "ConvertibleDebtNoncurrent", "USD"),
    ]


def test_extract_fact_series_merges_all_candidate_tags_not_just_the_first():
    """A filer that migrates from one tag name to another mid-history (live-observed: Alnylam's
    LongTermDebt -> ConvertibleDebtNoncurrent switch) must have BOTH tags' entries merged, not
    just the first candidate that happens to have any data."""
    payload = {
        "facts": {
            "us-gaap": {
                "LongTermDebtNoncurrent": {"units": {"USD": []}},
                "LongTermDebt": {
                    "units": {"USD": [
                        {"end": "2022-06-30", "val": 677_723_000, "accn": "0001-22-A", "filed": "2022-07-28"},
                    ]}
                },
                "ConvertibleDebtNoncurrent": {
                    "units": {"USD": [
                        {"end": "2025-06-30", "val": 1_026_522_000, "accn": "0001-25-A", "filed": "2025-07-31"},
                    ]}
                },
            }
        }
    }

    facts = extract_fact_series(payload, "0001178670", "total_debt")

    assert len(facts) == 2
    period_ends = {f.period_end for f in facts}
    assert date(2022, 6, 30) in period_ends
    assert date(2025, 6, 30) in period_ends


def test_extract_fact_series_short_term_investments_single_tag():
    payload = _companyfacts(
        [{"end": "2024-03-31", "val": 200_000_000, "accn": "0001-24-000001", "filed": "2024-05-01"}],
        taxonomy="us-gaap", tag="ShortTermInvestments", unit="USD",
    )
    facts = extract_fact_series(payload, "0000873303", "short_term_investments")
    assert len(facts) == 1
    assert facts[0].value == 200_000_000


def test_duration_delta_tag_map_has_the_live_verified_metric():
    assert DURATION_DELTA_TAG_MAP["quarterly_opex_burn"] == (
        "us-gaap", "NetCashProvidedByUsedInOperatingActivities", "USD",
    )


def test_extract_quarterly_delta_series_unknown_metric_raises():
    with pytest.raises(ValueError):
        extract_quarterly_delta_series({}, "0001682852", "not_a_real_metric")


def test_extract_quarterly_delta_series_missing_tag_returns_empty():
    assert extract_quarterly_delta_series({"facts": {}}, "0001682852", "quarterly_opex_burn") == []


def test_extract_quarterly_delta_series_derives_quarter_standalone_from_cumulative_ytd():
    """Shape live-verified 2026-08-30 against real Moderna XBRL data: entries are cumulative
    year-to-date, not per-quarter — Q2's standalone value must be Q2_YTD - Q1_YTD, etc."""
    entries = [
        {"start": "2022-01-01", "end": "2022-03-31", "val": 2_763_000_000, "accn": "A-Q1", "filed": "2023-05-04"},
        {"start": "2022-01-01", "end": "2022-06-30", "val": 3_067_000_000, "accn": "A-Q2", "filed": "2023-08-03"},
        {"start": "2022-01-01", "end": "2022-09-30", "val": 3_319_000_000, "accn": "A-Q3", "filed": "2023-11-03"},
        {"start": "2022-01-01", "end": "2022-12-31", "val": 4_981_000_000, "accn": "A-FY", "filed": "2024-02-23"},
    ]
    payload = {"facts": {"us-gaap": {"NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": entries}}}}}

    facts = extract_quarterly_delta_series(payload, "0001682852", "quarterly_opex_burn")

    by_end = {f.period_end: f.value for f in facts}
    assert by_end[date(2022, 3, 31)] == 2_763_000_000  # Q1: no prior baseline, used as-is
    assert by_end[date(2022, 6, 30)] == 3_067_000_000 - 2_763_000_000  # Q2 standalone
    assert by_end[date(2022, 9, 30)] == 3_319_000_000 - 3_067_000_000  # Q3 standalone
    assert by_end[date(2022, 12, 31)] == 4_981_000_000 - 3_319_000_000  # Q4 standalone


def test_extract_quarterly_delta_series_known_from_is_the_later_filing_date():
    entries = [
        {"start": "2022-01-01", "end": "2022-03-31", "val": 100, "accn": "A-Q1", "filed": "2022-05-01"},
        {"start": "2022-01-01", "end": "2022-06-30", "val": 250, "accn": "A-Q2", "filed": "2022-08-01"},
    ]
    payload = {"facts": {"us-gaap": {"NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": entries}}}}}

    facts = extract_quarterly_delta_series(payload, "0001682852", "quarterly_opex_burn")
    by_end = {f.period_end: f for f in facts}

    assert by_end[date(2022, 6, 30)].known_from == datetime(2022, 8, 1, tzinfo=timezone.utc)


def test_extract_quarterly_delta_series_separate_fiscal_years_do_not_mix():
    entries = [
        {"start": "2021-01-01", "end": "2021-12-31", "val": 1000, "accn": "A-FY21", "filed": "2022-02-01"},
        {"start": "2022-01-01", "end": "2022-03-31", "val": 300, "accn": "A-Q1-22", "filed": "2022-05-01"},
    ]
    payload = {"facts": {"us-gaap": {"NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": entries}}}}}

    facts = extract_quarterly_delta_series(payload, "0001682852", "quarterly_opex_burn")
    by_end = {f.period_end: f.value for f in facts}

    # Each fiscal year's first entry is its own baseline — 2022 Q1 must NOT be treated as a
    # delta against 2021's full-year total just because it's chronologically next.
    assert by_end[date(2021, 12, 31)] == 1000
    assert by_end[date(2022, 3, 31)] == 300


def test_extract_quarterly_delta_series_dedupes_comparative_column_reruns():
    entries = [
        {"start": "2022-01-01", "end": "2022-03-31", "val": 100, "accn": "A-Q1", "filed": "2022-05-01"},
        # Re-reported unchanged as a comparative column in a later filing.
        {"start": "2022-01-01", "end": "2022-03-31", "val": 100, "accn": "A-Q1-rerun", "filed": "2022-08-01"},
    ]
    payload = {"facts": {"us-gaap": {"NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": entries}}}}}

    facts = extract_quarterly_delta_series(payload, "0001682852", "quarterly_opex_burn")

    assert len(facts) == 1
    assert facts[0].source_id == "A-Q1"  # earliest-filed kept


def test_write_financial_facts_calls_repo_in_known_from_order():
    repo = MagicMock()
    facts = [
        NormalizedFact(
            metric="cash_and_equivalents", value=90.0, unit="USD", period_end=date(2024, 6, 30),
            known_from=datetime(2024, 8, 1, tzinfo=timezone.utc), source_id="B", source_url="url-b",
        ),
        NormalizedFact(
            metric="cash_and_equivalents", value=100.0, unit="USD", period_end=date(2024, 3, 31),
            known_from=datetime(2024, 5, 1, tzinfo=timezone.utc), source_id="A", source_url="url-a",
        ),
    ]

    count = write_financial_facts(company_id=7, facts=facts, repo=repo)

    assert count == 2
    calls = repo.upsert_financial_fact_bitemporal.call_args_list
    assert calls[0].kwargs["source_id"] == "A"  # earlier known_from written first, despite input order
    assert calls[1].kwargs["source_id"] == "B"
    assert calls[0].kwargs["company_id"] == 7
    assert calls[0].kwargs["valid_from"] == date(2024, 3, 31)
