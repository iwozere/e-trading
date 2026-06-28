"""Tests for the P19 SQLite shadow store."""

from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p19_penny_intraday.shadow_store import ShadowStore
from src.ml.pipeline.p19_penny_intraday.models.intraday_signal import IntradaySignal


def _sig(ticker="AAA", **kw):
    return IntradaySignal(ticker=ticker, ts=datetime(2026, 6, 24, 14, 30, tzinfo=timezone.utc), **kw)


def test_append_count_and_tickers(tmp_path):
    st = ShadowStore(str(tmp_path / "s.sqlite"))
    st.append_many("2026-06-24", [_sig("AAA", rvol_so_far=2.0), _sig("BBB")])
    st.append("2026-06-25", _sig("CCC"))
    assert st.count("2026-06-24") == 2
    assert st.count() == 3
    assert set(st.tickers_for_date("2026-06-24")) == {"AAA", "BBB"}


def test_round_trip_fields(tmp_path):
    st = ShadowStore(str(tmp_path / "s.sqlite"))
    st.append("2026-06-24", _sig("AAA", price=3.0, pct_from_open=0.2, rvol_so_far=5.1,
                                 day_volume=300000, source="p17", tier="B"))
    cur = st._conn.execute("SELECT ticker, price, rvol_so_far, source, tier FROM shadow_log")
    row = cur.fetchone()
    assert row == ("AAA", 3.0, 5.1, "p17", "B")


def test_update_eod_excludes_from_pending(tmp_path):
    st = ShadowStore(str(tmp_path / "s.sqlite"))
    st.append("2026-06-24", _sig("AAA"))
    n = st.update_eod("2026-06-24", "AAA", {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5})
    assert n == 1
    assert st.tickers_for_date("2026-06-24") == []   # eod_close set → no longer pending


def test_reopen_persists(tmp_path):
    p = str(tmp_path / "s.sqlite")
    ShadowStore(p).append("2026-06-24", _sig("AAA"))
    assert ShadowStore(p).count() == 1               # schema + data persist across opens
