"""
Tests for src.data.utils.refresh_fundamentals_cache staleness/chunk selection and
delisted-symbol tracking.

The staleness tests exercise `_select_symbols_by_staleness` in isolation by
monkeypatching `_get_symbol_cache_age_days` (the module's single per-symbol
cache-age lookup), so no real fundamentals cache/combiner instances are needed.

The delisted-tracking tests exercise the tracking file directly against a `tmp_path`
cache dir, since that mechanism is deliberately just file I/O with no external deps.
"""

from datetime import datetime
from pathlib import Path

import pytest

from src.data.utils import refresh_fundamentals_cache as refresh_module
from src.data.utils.refresh_fundamentals_cache import (
    DELISTED_FAILURE_THRESHOLD,
    _filter_excluded_symbols,
    _select_symbols_by_staleness,
    get_excluded_symbols,
    record_fetch_outcome,
)


def _stub_ages(monkeypatch: pytest.MonkeyPatch, ages: dict) -> None:
    """
    Make `_get_symbol_cache_age_days` return `ages[symbol]` (None if absent), and stub
    out the cache/combiner factories `_select_symbols_by_staleness` otherwise calls --
    the real `FundamentalsCache` constructor creates an on-disk directory as a side
    effect, which a unit test over pure selection logic has no business doing.
    """

    def _fake_age(cache, symbol, data_type="general"):
        del cache, data_type
        return ages.get(symbol)

    monkeypatch.setattr(refresh_module, "_get_symbol_cache_age_days", _fake_age)
    monkeypatch.setattr(refresh_module, "get_fundamentals_cache", lambda *_a, **_kw: None)
    monkeypatch.setattr(refresh_module, "get_fundamentals_combiner", lambda *_a, **_kw: None)


def test_no_knobs_returns_symbols_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no staleness/chunk args, the input list passes through untouched."""
    _stub_ages(monkeypatch, {})
    symbols = ["AAPL", "MSFT", "GOOGL"]

    result = _select_symbols_by_staleness(symbols, cache_dir="unused")

    assert result == symbols


def test_chunk_fraction_alone_keeps_oldest_half(monkeypatch: pytest.MonkeyPatch) -> None:
    """With only --chunk-fraction, the oldest N% (by cache age) is kept, no threshold applied."""
    ages = {"OLDEST": 30.0, "MID": 10.0, "NEWEST": 1.0, "MID2": 5.0}
    _stub_ages(monkeypatch, ages)

    result = _select_symbols_by_staleness(
        list(ages.keys()), cache_dir="unused", chunk_fraction=0.5
    )

    # ceil(4 * 0.5) = 2 oldest symbols, most-stale first.
    assert result == ["OLDEST", "MID"]


def test_missing_cache_sorts_first_and_is_never_dropped_by_chunking(monkeypatch: pytest.MonkeyPatch) -> None:
    """A never-cached symbol (age=None) is treated as infinitely stale."""
    ages = {"HAS_CACHE_OLD": 20.0, "HAS_CACHE_NEW": 1.0}
    _stub_ages(monkeypatch, ages)
    symbols = ["HAS_CACHE_NEW", "NEVER_CACHED", "HAS_CACHE_OLD"]

    result = _select_symbols_by_staleness(symbols, cache_dir="unused", chunk_fraction=0.34)

    # ceil(3 * 0.34) = 2; never-cached must win the top slot over the merely-old one.
    assert result == ["NEVER_CACHED", "HAS_CACHE_OLD"]


def test_stale_threshold_filters_before_chunking(monkeypatch: pytest.MonkeyPatch) -> None:
    """stale_min_days drops symbols that aren't due yet; chunk_fraction then trims the rest."""
    ages = {"VERY_STALE": 30.0, "STALE": 15.0, "FRESH": 2.0}
    _stub_ages(monkeypatch, ages)

    result = _select_symbols_by_staleness(
        list(ages.keys()), cache_dir="unused", stale_min_days=10.0, chunk_fraction=0.5
    )

    # FRESH is excluded outright by the threshold; chunk keeps ceil(2*0.5)=1 of what remains.
    assert result == ["VERY_STALE"]


def test_repeated_runs_round_robin_through_the_backlog(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Simulates two consecutive scheduled runs sharing a chunk_fraction=0.5: the first
    run's chosen half should refresh (age -> ~0), so the second run picks up the half
    left untouched by the first -- together covering the full backlog once.
    """
    ages = {"A": 10.0, "B": 9.0, "C": 8.0, "D": 7.0}
    symbols = list(ages.keys())
    _stub_ages(monkeypatch, ages)

    first_run = _select_symbols_by_staleness(symbols, cache_dir="unused", chunk_fraction=0.5)
    assert first_run == ["A", "B"]

    # First run "refreshed" A and B -- their age drops to ~0; C and D are untouched.
    ages["A"] = 0.0
    ages["B"] = 0.0

    second_run = _select_symbols_by_staleness(symbols, cache_dir="unused", chunk_fraction=0.5)
    assert second_run == ["C", "D"]


def test_invalid_chunk_fraction_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_ages(monkeypatch, {"AAPL": 1.0})

    with pytest.raises(ValueError):
        _select_symbols_by_staleness(["AAPL"], cache_dir="unused", chunk_fraction=1.5)

    with pytest.raises(ValueError):
        _select_symbols_by_staleness(["AAPL"], cache_dir="unused", chunk_fraction=0.0)


# --- Delisted-symbol tracking -------------------------------------------------------


def _freeze_today(monkeypatch: pytest.MonkeyPatch, iso_date: str) -> None:
    """Make `record_fetch_outcome`'s `datetime.now()` report a fixed calendar day."""

    class _FrozenDateTime:
        @classmethod
        def now(cls):
            return datetime.strptime(iso_date, "%Y-%m-%d")

    monkeypatch.setattr(refresh_module, "datetime", _FrozenDateTime)


def test_below_threshold_failures_are_not_excluded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fewer than DELISTED_FAILURE_THRESHOLD distinct-day failures leave a symbol unexcluded."""
    cache_dir = str(tmp_path)

    for day in range(DELISTED_FAILURE_THRESHOLD - 1):
        _freeze_today(monkeypatch, f"2026-01-{day + 1:02d}")
        record_fetch_outcome(cache_dir, "SAVA", had_any_data=False)

    assert get_excluded_symbols(cache_dir) == set()


def test_threshold_failures_on_distinct_days_excludes_symbol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """DELISTED_FAILURE_THRESHOLD failures, each on a distinct day, excludes the symbol."""
    cache_dir = str(tmp_path)

    for day in range(DELISTED_FAILURE_THRESHOLD):
        _freeze_today(monkeypatch, f"2026-01-{day + 1:02d}")
        record_fetch_outcome(cache_dir, "SAVA", had_any_data=False)

    assert get_excluded_symbols(cache_dir) == {"SAVA"}


def test_same_day_reruns_only_count_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Multiple failures recorded on the same calendar day count as a single strike."""
    cache_dir = str(tmp_path)
    _freeze_today(monkeypatch, "2026-01-01")

    for _ in range(DELISTED_FAILURE_THRESHOLD + 5):
        record_fetch_outcome(cache_dir, "SAVA", had_any_data=False)

    assert get_excluded_symbols(cache_dir) == set()


def test_success_clears_prior_failure_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A day where any provider returned data resets the symbol's failure count to zero."""
    cache_dir = str(tmp_path)

    for day in range(DELISTED_FAILURE_THRESHOLD):
        _freeze_today(monkeypatch, f"2026-01-{day + 1:02d}")
        record_fetch_outcome(cache_dir, "SAVA", had_any_data=False)
    assert get_excluded_symbols(cache_dir) == {"SAVA"}

    _freeze_today(monkeypatch, "2026-02-01")
    record_fetch_outcome(cache_dir, "SAVA", had_any_data=True)

    assert get_excluded_symbols(cache_dir) == set()


def test_filter_excluded_symbols_drops_only_tracked_symbols(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_dir = str(tmp_path)
    for day in range(DELISTED_FAILURE_THRESHOLD):
        _freeze_today(monkeypatch, f"2026-01-{day + 1:02d}")
        record_fetch_outcome(cache_dir, "SAVA", had_any_data=False)

    result = _filter_excluded_symbols(["AAPL", "SAVA", "MSFT"], cache_dir, include_excluded=False)

    assert result == ["AAPL", "MSFT"]


def test_filter_excluded_symbols_bypassed_by_include_excluded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = str(tmp_path)
    for day in range(DELISTED_FAILURE_THRESHOLD):
        _freeze_today(monkeypatch, f"2026-01-{day + 1:02d}")
        record_fetch_outcome(cache_dir, "SAVA", had_any_data=False)

    result = _filter_excluded_symbols(["AAPL", "SAVA"], cache_dir, include_excluded=True)

    assert result == ["AAPL", "SAVA"]


def test_no_tracking_file_means_nothing_excluded(tmp_path: Path) -> None:
    """A cache dir that has never recorded a failure has no exclusions."""
    assert get_excluded_symbols(str(tmp_path)) == set()
