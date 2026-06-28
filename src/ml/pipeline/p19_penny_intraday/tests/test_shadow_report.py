"""Tests for the P19 shadow-data report."""

from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p19_penny_intraday import shadow_report as sr
from src.ml.pipeline.p19_penny_intraday.shadow_store import ShadowStore
from src.ml.pipeline.p19_penny_intraday.models.intraday_signal import IntradaySignal


def _sig(ticker, **kw):
    return IntradaySignal(ticker=ticker, ts=datetime(2026, 6, 29, 14, 0, tzinfo=timezone.utc), **kw)


def test_report_missing_db(tmp_path):
    s = sr.report(str(tmp_path / "nope.sqlite"))
    assert s["error"] == "no shadow store yet"


def test_report_basic_stats(tmp_path):
    db = str(tmp_path / "s.sqlite")
    st = ShadowStore(db)
    st.append_many("2026-06-29", [
        _sig("AAA", source="p17", pct_from_open=0.10, rvol_so_far=2.0,
             day_volume=300_000, avg_volume_30d=1_000_000),
        _sig("BBB", source="gapper", pct_from_open=-0.05, rvol_so_far=0.0,
             day_volume=50_000, avg_volume_30d=0),
    ])
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
    st.append("2026-06-29", _sig("AAA", source="p17", rvol_so_far=1.0,
                                 day_volume=100_000_000, avg_volume_30d=1_000_000))
    s = sr.report(db, "2026-06-29")
    assert any("100x HIGH" in f for f in s["flags"])


def test_format_report_runs(tmp_path):
    db = str(tmp_path / "s.sqlite")
    ShadowStore(db).append("2026-06-29", _sig("AAA", source="p17", rvol_so_far=1.5,
                                              day_volume=500_000, avg_volume_30d=1_000_000))
    text = sr.format_report(sr.report(db, "2026-06-29"))
    assert "P19 shadow report" in text and "AAA" not in text  # summary, not row dump
