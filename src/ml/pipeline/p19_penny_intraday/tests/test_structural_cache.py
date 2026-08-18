"""Tests for the P19 Layer 0 per-ticker structural profile cache."""

import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p19_penny_intraday.models.structural_profile import StructuralProfile
from src.ml.pipeline.p19_penny_intraday.structural.cache import StructuralProfileCache


def test_missing_ticker_is_not_fresh(tmp_path):
    cache = StructuralProfileCache(str(tmp_path), ttl_days=7)
    assert cache.load("AAA") is None
    assert cache.is_fresh("AAA", date(2026, 8, 18)) is False


def test_save_and_load_round_trip(tmp_path):
    cache = StructuralProfileCache(str(tmp_path))
    profile = StructuralProfile(ticker="AAA", cik="123", as_of=date(2026, 8, 18), grade="B", coverage=0.8)
    cache.save(profile)
    loaded = cache.load("aaa")  # case-insensitive
    assert loaded is not None
    assert loaded.ticker == "AAA" and loaded.grade == "B" and loaded.coverage == 0.8


def test_fresh_within_ttl(tmp_path):
    cache = StructuralProfileCache(str(tmp_path), ttl_days=7)
    cache.save(StructuralProfile(ticker="AAA", as_of=date(2026, 8, 12)))
    assert cache.is_fresh("AAA", date(2026, 8, 18)) is True  # 6 days old


def test_stale_beyond_ttl(tmp_path):
    cache = StructuralProfileCache(str(tmp_path), ttl_days=7)
    cache.save(StructuralProfile(ticker="AAA", as_of=date(2026, 8, 1)))
    assert cache.is_fresh("AAA", date(2026, 8, 18)) is False  # 17 days old


def test_stale_due_to_new_filing_within_ttl_window(tmp_path):
    cache = StructuralProfileCache(str(tmp_path), ttl_days=7)
    cache.save(StructuralProfile(ticker="AAA", as_of=date(2026, 8, 15)))
    # Within TTL, but a new filing landed after the cached as_of -> stale.
    assert cache.is_fresh("AAA", date(2026, 8, 18), latest_filing_date=date(2026, 8, 16)) is False
    # No new filing since -> still fresh.
    assert cache.is_fresh("AAA", date(2026, 8, 18), latest_filing_date=date(2026, 8, 10)) is True


def test_clock_skew_treated_as_stale(tmp_path):
    cache = StructuralProfileCache(str(tmp_path), ttl_days=7)
    cache.save(StructuralProfile(ticker="AAA", as_of=date(2026, 9, 1)))  # "future" relative to as_of below
    assert cache.is_fresh("AAA", date(2026, 8, 18)) is False


def test_corrupt_cache_file_treated_as_miss(tmp_path):
    cache = StructuralProfileCache(str(tmp_path))
    (tmp_path / "AAA.json").write_text("not json{{{", encoding="utf-8")
    assert cache.load("AAA") is None
