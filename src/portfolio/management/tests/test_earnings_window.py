"""Unit tests for `earnings_window` trigger math."""

from datetime import date, datetime, timezone

from src.portfolio.management.earnings_window import (
    TRIGGER_T_MINUS_1_DAY,
    TRIGGER_T_MINUS_1_HOUR,
    EarningsEvent,
    matched_trigger,
    resolve_anchor_utc,
)


def test_bmo_anchors_at_market_open_et():
    """BMO earnings anchor at 09:30 ET (13:30 UTC in August, EDT)."""
    event = EarningsEvent(ticker="AAA", earnings_date=date(2026, 8, 20), session="bmo")
    anchor = resolve_anchor_utc(event)
    assert anchor == datetime(2026, 8, 20, 13, 30, tzinfo=timezone.utc)


def test_amc_anchors_at_market_close_et():
    """AMC earnings anchor at 16:00 ET (20:00 UTC in August, EDT)."""
    event = EarningsEvent(ticker="AAA", earnings_date=date(2026, 8, 20), session="amc")
    anchor = resolve_anchor_utc(event)
    assert anchor == datetime(2026, 8, 20, 20, 0, tzinfo=timezone.utc)


def test_unknown_session_defaults_to_market_open_like_bmo():
    """Unknown session anchors at market open — the earlier, safer default."""
    unknown = EarningsEvent(ticker="AAA", earnings_date=date(2026, 8, 20))
    bmo = EarningsEvent(ticker="AAA", earnings_date=date(2026, 8, 20), session="bmo")
    assert resolve_anchor_utc(unknown) == resolve_anchor_utc(bmo)


def test_winter_dst_offset():
    """EST (winter) is UTC-5, not UTC-4 — 09:30 ET becomes 14:30 UTC."""
    event = EarningsEvent(ticker="AAA", earnings_date=date(2026, 1, 20), session="bmo")
    anchor = resolve_anchor_utc(event)
    assert anchor == datetime(2026, 1, 20, 14, 30, tzinfo=timezone.utc)


def test_matched_trigger_t_minus_1_day():
    anchor = datetime(2026, 8, 20, 13, 30, tzinfo=timezone.utc)
    now = anchor.replace(day=19)  # exactly 1 day before
    assert matched_trigger(now, anchor, window_minutes=15) == TRIGGER_T_MINUS_1_DAY


def test_matched_trigger_t_minus_1_hour():
    anchor = datetime(2026, 8, 20, 13, 30, tzinfo=timezone.utc)
    now = datetime(2026, 8, 20, 12, 30, tzinfo=timezone.utc)  # exactly 1 hour before
    assert matched_trigger(now, anchor, window_minutes=15) == TRIGGER_T_MINUS_1_HOUR


def test_matched_trigger_within_window_but_not_exact():
    anchor = datetime(2026, 8, 20, 13, 30, tzinfo=timezone.utc)
    now = datetime(2026, 8, 20, 12, 35, tzinfo=timezone.utc)  # 55 min before -> within 15-min window of T-1h
    assert matched_trigger(now, anchor, window_minutes=15) == TRIGGER_T_MINUS_1_HOUR


def test_matched_trigger_none_outside_both_windows():
    anchor = datetime(2026, 8, 20, 13, 30, tzinfo=timezone.utc)
    now = datetime(2026, 8, 20, 9, 0, tzinfo=timezone.utc)  # ~4.5h before, in neither window
    assert matched_trigger(now, anchor, window_minutes=15) is None


def test_matched_trigger_none_far_in_past_or_future():
    anchor = datetime(2026, 8, 20, 13, 30, tzinfo=timezone.utc)
    assert matched_trigger(datetime(2026, 8, 1, tzinfo=timezone.utc), anchor, 15) is None
    assert matched_trigger(datetime(2026, 9, 1, tzinfo=timezone.utc), anchor, 15) is None
