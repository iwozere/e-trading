"""Tests for the P24 live scan pieces: universe filter, bar prep, scan, paper book, message, session date."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd

from src.ml.pipeline.p24_falling_knife.config import DAILY, DAILY_PARAMS, WEEKLY, CostModel, UniverseParams, live_params
from src.ml.pipeline.p24_falling_knife.jobs.run_p24_scan import default_as_of, is_stale, session_of
from src.ml.pipeline.p24_falling_knife.live import filter_universe, is_live_common_stock, merge_fresh, prepare_live_daily
from src.ml.pipeline.p24_falling_knife.notify import compose
from src.ml.pipeline.p24_falling_knife.paper import (
    CLOSED,
    OPEN,
    PENDING_ENTRY,
    PaperPosition,
    add_signals,
    events_on,
    load_state,
    save_state,
    update_position,
)
from src.ml.pipeline.p24_falling_knife.scan import build_series, scan_symbol, signal_row
from src.ml.pipeline.p24_falling_knife.tests.helpers import KNIFE_LEGS, bars_from_closes, path_from_legs

PARAMS = DAILY_PARAMS.with_overrides(use_shakeout_rule=False)
UNIVERSE = UniverseParams()


def _daily() -> pd.DataFrame:
    return prepare_live_daily(bars_from_closes(path_from_legs(KNIFE_LEGS), start="2020-08-03", volume=2_000_000.0))


def _signal_day(daily: pd.DataFrame) -> date:
    scan = build_series("KNF", daily, DAILY, PARAMS, UNIVERSE, daily.index[-1].date())
    assert scan is not None and scan.signals
    return scan.ctx.index[scan.signals[0].t].date()


def test_live_universe_filter() -> None:
    names = {"AAPL": "Apple Inc. - Common Stock", "ABCDW": "Abc Corp - Warrants", "XYZU": "Xyz Acquisition Corp - Units",
             "PFD": "Big Bank 6% Cumulative Preferred", "ZXZZT": "NASDAQ TEST STOCK", "INTT": "inTEST Corporation Common Stock"}
    assert filter_universe(list(names) + ["BRK.B", "NEW"], names) == ["AAPL", "INTT", "NEW"]
    assert is_live_common_stock("MSFT", None) and not is_live_common_stock("ABCDR", None)


def test_prepare_live_daily_adds_raw_columns_and_dedups() -> None:
    idx = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-05", "2026-01-06"])
    frame = pd.DataFrame({"open": [1, 2, 2.5, 3], "high": [1, 2, 3, 3], "low": [1, 2, 2, 3], "close": [1, 2, 2.6, 0],
                          "volume": [10, 20, 30, 40]}, index=idx)
    bars = prepare_live_daily(frame)
    assert [d.strftime("%Y-%m-%d") for d in bars.index] == ["2026-01-02", "2026-01-05"]
    assert bars["close_raw"].iloc[-1] == 2.6 and not bars["jump_no_split"].any()


def test_scan_finds_signal_only_on_its_session() -> None:
    daily = _daily()
    day = _signal_day(daily)
    upto = daily[daily.index <= pd.Timestamp(day)]
    scan = scan_symbol("KNF", upto, DAILY, PARAMS, UNIVERSE, day, forming_max_missing=2)
    assert scan is not None and scan.signal is not None
    row = signal_row(scan, scan.signal)
    assert row["ticker"] == "KNF" and row["signal_date"] == day and row["drop_pct"] >= 0.40
    later = daily.index[daily.index > pd.Timestamp(day)][0].date()
    after = scan_symbol("KNF", daily[daily.index <= pd.Timestamp(later)], DAILY, PARAMS, UNIVERSE, later, 2)
    assert after is not None and after.signal is None
    assert scan_symbol("KNF", upto, DAILY, PARAMS, UNIVERSE, later, 2) is None  # no bar for that session


def test_paper_lifecycle_is_replayed_and_idempotent(tmp_path: Path) -> None:
    daily = _daily()
    day = _signal_day(daily)
    sessions = [d.date() for d in daily.index if d >= pd.Timestamp(day)]
    scan = scan_symbol("KNF", daily[daily.index <= pd.Timestamp(day)], DAILY, PARAMS, UNIVERSE, day, 2)
    assert scan is not None and scan.signal is not None
    positions: List[PaperPosition] = []
    rows = [signal_row(scan, scan.signal)]
    assert len(add_signals(positions, rows, 2_000.0)) == 1
    assert add_signals(positions, rows, 2_000.0) == []
    pos = positions[0]
    update_position(pos, scan, PARAMS, CostModel())
    assert pos.status == PENDING_ENTRY

    nxt = sessions[1]
    scan2 = build_series("KNF", daily[daily.index <= pd.Timestamp(nxt)], DAILY, PARAMS, UNIVERSE, nxt)
    assert scan2 is not None
    update_position(pos, scan2, PARAMS, CostModel())
    assert pos.status == OPEN and pos.entry_date == nxt.isoformat()
    assert [p.key for p in events_on(positions, DAILY, nxt)["entered"]] == [pos.key]

    full = build_series("KNF", daily, DAILY, PARAMS, UNIVERSE, sessions[-1])
    assert full is not None
    update_position(pos, full, PARAMS, CostModel())
    first = (pos.status, pos.exit_date, pos.net_return)
    update_position(pos, full, PARAMS, CostModel())
    assert (pos.status, pos.exit_date, pos.net_return) == first
    if pos.status == CLOSED:
        assert events_on(positions, DAILY, date.fromisoformat(pos.exit_date or ""))["exited"] == [pos]

    state = tmp_path / "state.json"
    save_state(state, positions)
    assert [p.key for p in load_state(state)] == [pos.key]


def test_compose_only_when_something_happened() -> None:
    assert compose(DAILY, date(2026, 11, 6), [], [], [], 3) is None
    row = {"ticker": "KNF", "timeframe": DAILY, "close": 9.6, "drop_pct": 0.55, "peak_date": date(2026, 6, 1), "cci_3": -180.0,
           "cci_2": -120.0, "score": 0.7, "suspect_gap": False}
    message = compose(DAILY, date(2026, 11, 6), [row], [], [], 3)
    assert message is not None
    assert "KNF daily" in message["text"] and "-55.0%" in message["text"] and "<html>" in message["html"]


def test_session_of_weekly_uses_last_complete_week() -> None:
    idx = pd.bdate_range("2026-10-26", "2026-11-04")  # Mon 10-26 .. Wed 11-04
    spy = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
    assert session_of(spy, DAILY, date(2026, 11, 4)) == date(2026, 11, 4)
    assert session_of(spy, WEEKLY, date(2026, 11, 4)) == date(2026, 10, 30)


def test_live_params_apply_frozen_calibration() -> None:
    daily = live_params(DAILY)
    assert (daily.w_narrow, daily.cci_div_min, daily.zz_pct, daily.use_shakeout_rule) == (0.40, 40.0, 0.20, False)
    assert live_params(WEEKLY).use_shakeout_rule is False and live_params(WEEKLY).w_narrow == 0.60


def test_merge_fresh_overlays_recent_bars() -> None:
    cached = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=pd.to_datetime(["2026-10-28", "2026-10-29", "2026-10-30"]))
    fresh = pd.DataFrame({"close": [3.5, 4.0]}, index=pd.to_datetime(["2026-10-30", "2026-11-06"]))
    merged = merge_fresh({"A": cached, "B": pd.DataFrame()}, {"A": fresh, "C": fresh})
    assert sorted(merged) == ["A", "C"]
    assert merged["A"]["close"].tolist() == [1.0, 2.0, 3.5, 4.0]


def test_is_stale_allows_holidays_only() -> None:
    assert not is_stale(date(2026, 11, 6), date(2026, 11, 6))
    assert not is_stale(date(2026, 4, 2), date(2026, 4, 6))  # Thursday before Good Friday, run Monday
    assert is_stale(date(2026, 10, 30), date(2026, 11, 6))


def test_default_as_of_waits_for_the_close() -> None:
    assert default_as_of(datetime(2026, 11, 6, 18, 50, tzinfo=timezone.utc)) == date(2026, 11, 5)
    assert default_as_of(datetime(2026, 11, 6, 22, 45, tzinfo=timezone.utc)) == date(2026, 11, 6)
