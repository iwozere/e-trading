"""Tests for the P19 shadow-data report."""

import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p19_penny_intraday import shadow_report as sr
from src.ml.pipeline.p19_penny_intraday.models.intraday_signal import IntradaySignal
from src.ml.pipeline.p19_penny_intraday.shadow_store import ShadowStore


def _sig(ticker, **kw):
    return IntradaySignal(ticker=ticker, ts=datetime(2026, 6, 29, 14, 0, tzinfo=UTC), **kw)


def test_report_missing_db(tmp_path):
    s = sr.report(str(tmp_path / "nope.sqlite"))
    assert s["error"] == "no shadow store yet"


def test_report_basic_stats(tmp_path):
    db = str(tmp_path / "s.sqlite")
    st = ShadowStore(db)
    st.append_many(
        "2026-06-29",
        [
            _sig(
                "AAA", source="p17", pct_from_open=0.10, rvol_so_far=2.0, day_volume=300_000, avg_volume_30d=1_000_000
            ),
            _sig("BBB", source="gapper", pct_from_open=-0.05, rvol_so_far=0.0, day_volume=50_000, avg_volume_30d=0),
        ],
    )
    s = sr.report(db, "2026-06-29")
    assert s["rows"] == 2 and s["distinct_tickers"] == 2
    assert s["by_source"] == {"p17": 1, "gapper": 1}
    assert s["rvol_so_far"]["n"] == 1 and s["rvol_so_far"]["median"] == 2.0
    assert s["gappers_zero_rvol"] == 1
    assert "baseline enrichment gap" in " ".join(s["flags"])


def test_report_flags_volume_unit_mismatch(tmp_path):
    db = str(tmp_path / "s.sqlite")
    st = ShadowStore(db)
    # day_volume 100x the 30d average → lot-size mismatch should be flagged
    st.append(
        "2026-06-29", _sig("AAA", source="p17", rvol_so_far=1.0, day_volume=100_000_000, avg_volume_30d=1_000_000)
    )
    s = sr.report(db, "2026-06-29")
    assert any("100x HIGH" in f for f in s["flags"])


def test_format_report_runs(tmp_path):
    db = str(tmp_path / "s.sqlite")
    ShadowStore(db).append(
        "2026-06-29", _sig("AAA", source="p17", rvol_so_far=1.5, day_volume=500_000, avg_volume_30d=1_000_000)
    )
    text = sr.format_report(sr.report(db, "2026-06-29"))
    assert "P19 shadow report" in text and "AAA" not in text  # summary, not row dump


# ── v2: per-grade / coverage reporting ──────────────────────────────────────


def test_by_grade_counts_and_unprofiled(tmp_path):
    db = str(tmp_path / "s.sqlite")
    st = ShadowStore(db)
    st.append_many(
        "2026-06-29",
        [
            _sig("AAA", structural_grade="D", structural_coverage=0.9),
            _sig("BBB", structural_grade="B", structural_coverage=0.8),
            _sig("CCC"),  # never profiled -- structural_grade stays ""
        ],
    )
    s = sr.report(db, "2026-06-29")
    assert s["by_grade"] == {"D": 1, "B": 1, "unprofiled": 1}
    assert s["unprofiled_count"] == 1


def test_low_coverage_flag_fires_above_threshold(tmp_path):
    db = str(tmp_path / "s.sqlite")
    st = ShadowStore(db)
    st.append_many(
        "2026-06-29",
        [
            _sig("AAA", structural_grade="C", structural_coverage=0.1),
            _sig("BBB", structural_grade="C", structural_coverage=0.2),
        ],
    )
    s = sr.report(db, "2026-06-29")
    assert s["low_coverage_count"] == 2
    assert any("coverage" in f for f in s["flags"])


def test_small_grade_sample_flagged_non_conclusive(tmp_path):
    db = str(tmp_path / "s.sqlite")
    st = ShadowStore(db)
    st.append("2026-06-29", _sig("AAA", structural_grade="A", structural_coverage=0.9))
    s = sr.report(db, "2026-06-29")
    assert any("n=1 < 30" in f for f in s["flags"])


def test_grades_denormalised_across_polls_count_once_per_ticker(tmp_path):
    """Same ticker, multiple polls same day -> counted once in by_grade, not once per poll."""
    db = str(tmp_path / "s.sqlite")
    st = ShadowStore(db)
    st.append_many(
        "2026-06-29",
        [
            _sig("AAA", structural_grade="B", structural_coverage=0.8),
            _sig("AAA", structural_grade="B", structural_coverage=0.8),
            _sig("AAA", structural_grade="B", structural_coverage=0.8),
        ],
    )
    s = sr.report(db, "2026-06-29")
    assert s["by_grade"] == {"B": 1}


# ── Phase 3: FPI share reporting (StructuralSignals.md §2) ──────────────────


def test_fpi_share_stats_and_flag(tmp_path):
    db = str(tmp_path / "s.sqlite")
    st = ShadowStore(db)
    st.append_many(
        "2026-06-29",
        [
            _sig("AAA", structural_grade="C", is_fpi=True, structural_coverage=0.4),
            _sig("BBB", structural_grade="C", is_fpi=True, structural_coverage=0.4),
            _sig("CCC", structural_grade="B", is_fpi=False, structural_coverage=0.9),
            _sig("DDD", structural_grade="A", is_fpi=False, structural_coverage=0.9),
            _sig("EEE", structural_grade="B", is_fpi=False, structural_coverage=0.9),
        ],
    )
    s = sr.report(db, "2026-06-29")
    assert s["fpi_count"] == 2
    assert s["fpi_grade_c_count"] == 2
    assert any("are FPIs" in f for f in s["flags"])  # 2/5 = 40% > 20% threshold


def test_no_fpi_names_no_flag(tmp_path):
    db = str(tmp_path / "s.sqlite")
    st = ShadowStore(db)
    st.append("2026-06-29", _sig("AAA", structural_grade="A", is_fpi=False, structural_coverage=0.9))
    s = sr.report(db, "2026-06-29")
    assert s.get("fpi_count", 0) == 0
    assert not any("are FPIs" in f for f in s["flags"])


# ── --hits: T1+ momentum hits table ─────────────────────────────────────────


def _poll(ticker, hh, mm, **kw):
    return IntradaySignal(ticker=ticker, ts=datetime(2026, 6, 29, hh, mm, tzinfo=UTC), **kw)


def test_hits_missing_db(tmp_path):
    assert sr.hits(str(tmp_path / "nope.sqlite"))["error"] == "no shadow store yet"


def test_hits_aggregates_per_ticker_and_skips_t0(tmp_path):
    db = str(tmp_path / "s.sqlite")
    ShadowStore(db).append_many(
        "2026-06-29",
        [
            _poll("AAA", 14, 0, momentum_tier="T0", momentum_score=10.0, price=1.0, pct_from_open=0.05),
            _poll("AAA", 14, 15, momentum_tier="T1", momentum_score=30.0, price=1.2, pct_from_open=0.25, rvol_so_far=6.0),
            _poll("AAA", 14, 30, momentum_tier="T3", momentum_score=90.0, price=1.5, pct_from_open=0.50, rvol_so_far=9.0),
            _poll("BBB", 15, 0, momentum_tier="T2", momentum_score=50.0, price=2.0, pct_from_open=-0.35),
            _poll("CCC", 15, 0, momentum_tier="T0", momentum_score=5.0),
        ],
    )
    h = sr.hits(db, "2026-06-29")
    assert [x["ticker"] for x in h["hits"]] == ["AAA", "BBB"]  # T3 before T2; CCC never hit
    aaa = h["hits"][0]
    assert aaa["first_hit_et"] == "10:15"  # 14:15 UTC in June = EDT
    assert aaa["first_hit_price"] == 1.2
    assert aaa["peak_tier"] == "T3" and aaa["peak_score"] == 90.0
    assert aaa["max_move"] == 0.50 and aaa["peak_rvol"] == 9.0
    assert h["hits"][1]["max_move"] == -0.35  # drops are hits too; sign kept


def test_hits_defaults_to_latest_date(tmp_path):
    db = str(tmp_path / "s.sqlite")
    st = ShadowStore(db)
    st.append("2026-06-26", _sig("OLD", momentum_tier="T1", momentum_score=30.0))
    st.append("2026-06-29", _sig("NEW", momentum_tier="T1", momentum_score=30.0))
    h = sr.hits(db)
    assert h["date"] == "2026-06-29" and [x["ticker"] for x in h["hits"]] == ["NEW"]


def test_format_hits_table_and_empty(tmp_path):
    db = str(tmp_path / "s.sqlite")
    ShadowStore(db).append_many(
        "2026-06-29",
        [
            _sig("AAA", momentum_tier="T2", momentum_score=55.0, price=1.0, pct_from_open=-0.30),
            _sig("ZZZ", momentum_tier="T0"),
        ],
    )
    text = sr.format_hits(sr.hits(db, "2026-06-29"))
    assert "1 name(s)" in text and "AAA" in text and "-30%" in text and "ZZZ" not in text
    assert "no names crossed T1" in sr.format_hits({"db": db, "date": "2026-06-30", "hits": []})
