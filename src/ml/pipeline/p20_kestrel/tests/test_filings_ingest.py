"""Tests for which EDGAR cache dates the P20 filings ingest reads."""

import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.ingest.filings_ingest import _dates_to_check


def test_weekday_reads_previous_day_only() -> None:
    # Friday 2026-11-06: the 8-K index / Form 4 caches for the 6th only exist on the 7th.
    assert _dates_to_check(date(2026, 11, 6), None) == [date(2026, 11, 5)]


def test_monday_reads_friday_through_sunday() -> None:
    assert _dates_to_check(date(2026, 11, 9), None) == [date(2026, 11, 8), date(2026, 11, 7), date(2026, 11, 6)]


def test_explicit_lookback_never_includes_run_date() -> None:
    run_date = date(2026, 11, 4)
    dates = _dates_to_check(run_date, 2)
    assert dates == [date(2026, 11, 3), date(2026, 11, 2)]
    assert run_date not in dates
