"""Tests for features/lookahead_audit.py (spec §8.3, mandatory)."""

import random
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.features.lookahead_audit import (
    HIGH_RISK_CATEGORIES,
    AuditSample,
    assert_known_from_is_filing_date_not_period_or_crossing_date,
    assert_lookahead_safe,
    stratified_sample,
)


def _population(counts):
    """counts: {category: n} -> a flat list of AuditSample with distinct company_ids."""
    items = []
    company_id = 0
    for category, n in counts.items():
        for _ in range(n):
            company_id += 1
            items.append(AuditSample(company_id=company_id, as_of=date(2024, 1, 1), source_category=category))
    return items


def test_stratified_sample_guarantees_minimum_per_high_risk_category():
    population = _population({"vendor_fact": 500, "13f_holding": 500, "13d_process_event": 500, "sec_fact": 5000})

    sample = stratified_sample(population, total=200, min_per_high_risk_category=20, rng=random.Random(42))

    for category in HIGH_RISK_CATEGORIES:
        count = sum(1 for s in sample if s.source_category == category)
        assert count >= 20, f"{category} under-represented: {count}"
    assert len(sample) == 200


def test_stratified_sample_uses_all_available_when_fewer_than_floor():
    population = _population({"vendor_fact": 5, "13f_holding": 500, "13d_process_event": 500, "sec_fact": 5000})

    sample = stratified_sample(population, total=200, min_per_high_risk_category=20, rng=random.Random(1))

    assert sum(1 for s in sample if s.source_category == "vendor_fact") == 5


def test_stratified_sample_smaller_population_than_total_returns_everything():
    population = _population({"vendor_fact": 3, "13f_holding": 3})
    sample = stratified_sample(population, total=200, min_per_high_risk_category=20, rng=random.Random(1))
    assert len(sample) == 6


def test_stratified_sample_is_deterministic_with_seeded_rng():
    population = _population({"vendor_fact": 50, "13f_holding": 50, "13d_process_event": 50, "sec_fact": 500})
    sample1 = stratified_sample(population, total=100, rng=random.Random(7))
    sample2 = stratified_sample(population, total=100, rng=random.Random(7))
    assert [s.company_id for s in sample1] == [s.company_id for s in sample2]


def test_assert_lookahead_safe_passes_when_known_from_before_as_of():
    rows = [{"known_from": datetime(2024, 1, 1, tzinfo=timezone.utc), "as_of": date(2024, 1, 2)}]
    assert_lookahead_safe(rows)  # must not raise


def test_assert_lookahead_safe_passes_when_known_from_equals_as_of():
    rows = [{"known_from": datetime(2024, 1, 2, tzinfo=timezone.utc), "as_of": date(2024, 1, 2)}]
    assert_lookahead_safe(rows)  # must not raise — same-day disclosure is legitimate


def test_assert_lookahead_safe_raises_on_violation():
    rows = [{"known_from": datetime(2024, 1, 5, tzinfo=timezone.utc), "as_of": date(2024, 1, 1)}]
    with pytest.raises(AssertionError, match="lookahead violation"):
        assert_lookahead_safe(rows)


def test_assert_lookahead_safe_reports_all_violations_up_to_five():
    rows = [
        {"known_from": datetime(2024, 1, 5, tzinfo=timezone.utc), "as_of": date(2024, 1, 1)} for _ in range(3)
    ]
    with pytest.raises(AssertionError, match=r"3 lookahead violation"):
        assert_lookahead_safe(rows)


def test_assert_known_from_is_filing_date_passes_when_matching():
    rows = [{"known_from": datetime(2024, 3, 1, tzinfo=timezone.utc), "filed_date": date(2024, 3, 1)}]
    assert_known_from_is_filing_date_not_period_or_crossing_date(rows)  # must not raise


def test_assert_known_from_is_filing_date_catches_period_end_used_instead():
    """The exact 13F trap spec §8.3 calls out: known_from silently set to period_end, not filed date."""
    rows = [{"known_from": datetime(2024, 1, 15, tzinfo=timezone.utc), "filed_date": date(2024, 2, 28)}]
    with pytest.raises(AssertionError, match="known_from"):
        assert_known_from_is_filing_date_not_period_or_crossing_date(rows)


def test_assert_known_from_is_filing_date_supports_custom_field_name():
    rows = [{"known_from": datetime(2024, 3, 1, tzinfo=timezone.utc), "crossing_date": date(2024, 2, 20)}]
    with pytest.raises(AssertionError):
        assert_known_from_is_filing_date_not_period_or_crossing_date(rows, filing_date_field="crossing_date")
