"""Tests for features/context.py."""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.features.context import FeatureContext


def test_get_latest_fact_returns_value_from_first_row():
    repo = MagicMock()
    repo.get_financial_facts_as_of.return_value = [{"value": 42.5}, {"value": 10.0}]
    ctx = FeatureContext(as_of=date(2024, 6, 1), repo=repo)

    assert ctx.get_latest_fact(7, "cash_and_equivalents") == 42.5
    repo.get_financial_facts_as_of.assert_called_once_with(7, "cash_and_equivalents", date(2024, 6, 1))


def test_get_latest_fact_returns_none_when_nothing_known():
    """Missing data must propagate as None, never 0.0 (spec §4)."""
    repo = MagicMock()
    repo.get_financial_facts_as_of.return_value = []
    ctx = FeatureContext(as_of=date(2024, 6, 1), repo=repo)

    assert ctx.get_latest_fact(7, "market_cap") is None


def test_get_latest_fact_returns_none_when_value_field_is_none():
    repo = MagicMock()
    repo.get_financial_facts_as_of.return_value = [{"value": None}]
    ctx = FeatureContext(as_of=date(2024, 6, 1), repo=repo)

    assert ctx.get_latest_fact(7, "cash_and_equivalents") is None


def test_get_latest_fact_row_returns_full_row():
    repo = MagicMock()
    repo.get_financial_facts_as_of.return_value = [{"value": 42.5, "period_end": date(2024, 3, 31)}]
    ctx = FeatureContext(as_of=date(2024, 6, 1), repo=repo)

    row = ctx.get_latest_fact_row(7, "cash_and_equivalents")
    assert row is not None
    assert row["period_end"] == date(2024, 3, 31)


def test_get_latest_fact_row_returns_none_when_empty():
    repo = MagicMock()
    repo.get_financial_facts_as_of.return_value = []
    ctx = FeatureContext(as_of=date(2024, 6, 1), repo=repo)

    assert ctx.get_latest_fact_row(7, "market_cap") is None


def test_get_trailing_average_averages_the_most_recent_n_rows():
    repo = MagicMock()
    # Ordered most-recent-first, as get_financial_facts_as_of already returns.
    repo.get_financial_facts_as_of.return_value = [
        {"value": -100.0}, {"value": -80.0}, {"value": -120.0}, {"value": -60.0}, {"value": -9999.0},
    ]
    ctx = FeatureContext(as_of=date(2024, 6, 1), repo=repo)

    avg = ctx.get_trailing_average(7, "quarterly_opex_burn", periods=4)

    assert avg == (-100.0 - 80.0 - 120.0 - 60.0) / 4  # the 5th (oldest) row excluded


def test_get_trailing_average_uses_fewer_rows_when_history_is_shorter_than_window():
    repo = MagicMock()
    repo.get_financial_facts_as_of.return_value = [{"value": -50.0}, {"value": -30.0}]
    ctx = FeatureContext(as_of=date(2024, 6, 1), repo=repo)

    avg = ctx.get_trailing_average(7, "quarterly_opex_burn", periods=4)

    assert avg == -40.0  # averages over the 2 available, not None for lacking a full window


def test_get_trailing_average_returns_none_when_nothing_known():
    repo = MagicMock()
    repo.get_financial_facts_as_of.return_value = []
    ctx = FeatureContext(as_of=date(2024, 6, 1), repo=repo)

    assert ctx.get_trailing_average(7, "quarterly_opex_burn") is None
