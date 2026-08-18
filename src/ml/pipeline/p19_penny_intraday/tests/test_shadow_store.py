"""Tests for the P19 SQLite shadow store."""

import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p19_penny_intraday.models.intraday_signal import IntradaySignal
from src.ml.pipeline.p19_penny_intraday.shadow_store import _V1_COLUMNS, ShadowStore


def _sig(ticker="AAA", **kw):
    return IntradaySignal(ticker=ticker, ts=datetime(2026, 6, 24, 14, 30, tzinfo=UTC), **kw)


def test_append_count_and_tickers(tmp_path):
    st = ShadowStore(str(tmp_path / "s.sqlite"))
    st.append_many("2026-06-24", [_sig("AAA", rvol_so_far=2.0), _sig("BBB")])
    st.append("2026-06-25", _sig("CCC"))
    assert st.count("2026-06-24") == 2
    assert st.count() == 3
    assert set(st.tickers_for_date("2026-06-24")) == {"AAA", "BBB"}


def test_round_trip_fields(tmp_path):
    st = ShadowStore(str(tmp_path / "s.sqlite"))
    st.append(
        "2026-06-24",
        _sig("AAA", price=3.0, pct_from_open=0.2, rvol_so_far=5.1, day_volume=300000, source="p17", tier="B"),
    )
    cur = st._conn.execute("SELECT ticker, price, rvol_so_far, source, tier FROM shadow_log")
    row = cur.fetchone()
    assert row == ("AAA", 3.0, 5.1, "p17", "B")


def test_update_eod_excludes_from_pending(tmp_path):
    st = ShadowStore(str(tmp_path / "s.sqlite"))
    st.append("2026-06-24", _sig("AAA"))
    n = st.update_eod("2026-06-24", "AAA", {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5})
    assert n == 1
    assert st.tickers_for_date("2026-06-24") == []  # eod_close set → no longer pending


def test_reopen_persists(tmp_path):
    p = str(tmp_path / "s.sqlite")
    ShadowStore(p).append("2026-06-24", _sig("AAA"))
    assert ShadowStore(p).count() == 1  # schema + data persist across opens


def test_v1_table_migrates_additively_v2_columns_null(tmp_path):
    """
    Simulates the real situation on the Pi (design-v2.md decision #3): a
    shadow.sqlite created before schema v2, with real v1-shaped rows already in
    it. Opening it with the current ShadowStore must add the v2 columns without
    touching existing data, and old rows must read back NULL in every new column.
    """
    p = str(tmp_path / "v1_shaped.sqlite")
    conn = sqlite3.connect(p)
    v1_cols = ", ".join(f"{c} TEXT" for c in _V1_COLUMNS if c != "date")
    conn.execute(f"CREATE TABLE shadow_log (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, {v1_cols})")
    conn.execute("INSERT INTO shadow_log (date, ticker, source, price) VALUES (?, ?, ?, ?)", ("2026-06-24", "AAA", "p17", "3.0"))
    conn.commit()
    conn.close()

    st = ShadowStore(p)
    assert st.count("2026-06-24") == 1
    row = st._conn.execute(
        "SELECT ticker, structural_grade, momentum_tier, dilution_urgency, close_retention FROM shadow_log"
    ).fetchone()
    assert row[0] == "AAA"
    assert row[1] is None and row[2] is None and row[3] is None and row[4] is None

    # New rows after migration write v2 fields normally.
    st.append("2026-08-18", _sig("BBB", momentum_tier="T2", structural_grade="B"))
    new_row = st._conn.execute(
        "SELECT structural_grade, momentum_tier FROM shadow_log WHERE ticker='BBB'"
    ).fetchone()
    assert new_row == ("B", "T2")


def test_migration_is_idempotent_on_reopen(tmp_path):
    p = str(tmp_path / "s.sqlite")
    ShadowStore(p).append("2026-06-24", _sig("AAA"))
    ShadowStore(p)  # reopen -- migration pass runs again, must not error or duplicate columns
    st = ShadowStore(p)
    cols = [r[1] for r in st._conn.execute("PRAGMA table_info(shadow_log)").fetchall()]
    assert cols.count("structural_grade") == 1


# ── Outcome labels (v2 spec §12.2) ──────────────────────────────────────────


def test_polls_for_date_ticker_ordered_by_ts(tmp_path):
    st = ShadowStore(str(tmp_path / "s.sqlite"))
    later = IntradaySignal(ticker="AAA", ts=datetime(2026, 8, 18, 15, 0, tzinfo=UTC), price=2.0)
    earlier = IntradaySignal(ticker="AAA", ts=datetime(2026, 8, 18, 14, 0, tzinfo=UTC), price=1.0)
    st.append("2026-08-18", later)
    st.append("2026-08-18", earlier)
    polls = st.polls_for_date_ticker("2026-08-18", "AAA")
    assert [p["price"] for p in polls] == [1.0, 2.0]


def test_same_day_labels_write_and_needing_query(tmp_path):
    st = ShadowStore(str(tmp_path / "s.sqlite"))
    st.append("2026-08-18", _sig("AAA"))
    st.update_eod("2026-08-18", "AAA", {"open": 1.0, "high": 2.0, "low": 0.9, "close": 1.8})
    assert st.tickers_for_date_needing_labels("2026-08-18") == ["AAA"]

    n = st.update_same_day_labels(
        "2026-08-18", "AAA", {"high_time": "14:32", "close_retention": 0.8, "mae_from_alert": -0.05, "mfe_from_alert": 1.0}
    )
    assert n == 1
    assert st.tickers_for_date_needing_labels("2026-08-18") == []
    row = st._conn.execute("SELECT high_time, close_retention FROM shadow_log WHERE ticker='AAA'").fetchone()
    assert row == ("14:32", 0.8)


def test_forward_labels_write_and_needing_query(tmp_path):
    st = ShadowStore(str(tmp_path / "s.sqlite"))
    st.append("2026-08-01", _sig("AAA"))
    st.update_eod("2026-08-01", "AAA", {"open": 1.0, "high": 1.5, "low": 0.9, "close": 1.2})
    assert st.tickers_needing_label_backfill("2026-08-01") == ["AAA"]
    assert "2026-08-01" in st.dates_needing_label_backfill()

    n = st.update_forward_labels(
        "2026-08-01",
        "AAA",
        {"ret_t1": 0.1, "ret_t3": 0.2, "ret_t5": -0.1, "ret_t10": 0.3, "dilution_event_within_5d": True, "reverse_split_within_180d": False},
    )
    assert n == 1
    assert st.tickers_needing_label_backfill("2026-08-01") == []
    assert "2026-08-01" not in st.dates_needing_label_backfill()
    row = st._conn.execute(
        "SELECT ret_t10, dilution_event_within_5d, reverse_split_within_180d FROM shadow_log WHERE ticker='AAA'"
    ).fetchone()
    assert row == (0.3, 1, 0)
