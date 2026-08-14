"""Unit tests for notifier message formatting."""

from datetime import UTC, date, datetime

from src.portfolio.management.coverage import CoverageRow, CoverageStatus
from src.portfolio.management.earnings_window import TRIGGER_T_MINUS_1_DAY, TRIGGER_T_MINUS_1_HOUR, EarningsEvent
from src.portfolio.management.notifier import TriggeredReminder, format_html, format_plain_text

AS_OF = datetime(2026, 8, 19, 13, 30, tzinfo=UTC)


def _reminder(ticker: str, trigger: str, status: CoverageStatus, protected: float = 0, qty: float = 100) -> TriggeredReminder:
    return TriggeredReminder(
        event=EarningsEvent(ticker=ticker, earnings_date=date(2026, 8, 20), session="bmo"),
        trigger=trigger,
        coverage=CoverageRow(ticker=ticker, position_qty=qty, protected_qty=protected, status=status),
    )


def test_plain_text_zero_reminders_returns_header_only():
    text = format_plain_text([], as_of=AS_OF)
    assert "0 ticker(s) with earnings coming up" in text
    assert "\n\n" not in text  # no trailing blank body


def test_plain_text_contains_ticker_trigger_and_status():
    reminders = [
        _reminder("AAA", TRIGGER_T_MINUS_1_DAY, CoverageStatus.UNCOVERED, protected=0),
        _reminder("BBB", TRIGGER_T_MINUS_1_HOUR, CoverageStatus.COVERED, protected=100),
    ]

    text = format_plain_text(reminders, as_of=AS_OF)

    assert "2 ticker(s) with earnings coming up" in text
    assert "AAA" in text and "earnings in ~1 day" in text and "UNCOVERED" in text
    assert "BBB" in text and "earnings in ~1 hour" in text
    assert "0/100 shares protected" in text
    assert "100/100 shares protected" in text


def test_plain_text_omits_session_when_unknown():
    reminder = TriggeredReminder(
        event=EarningsEvent(ticker="AAA", earnings_date=date(2026, 8, 20)),  # session defaults to "unknown"
        trigger=TRIGGER_T_MINUS_1_DAY,
        coverage=CoverageRow(ticker="AAA", position_qty=10, protected_qty=0, status=CoverageStatus.UNCOVERED),
    )

    text = format_plain_text([reminder], as_of=AS_OF)

    assert "(BMO)" not in text and "(AMC)" not in text


def test_html_contains_table_row_per_reminder():
    reminders = [_reminder("AAA", TRIGGER_T_MINUS_1_DAY, CoverageStatus.PARTIALLY_COVERED, protected=40, qty=100)]

    html = format_html(reminders, as_of=AS_OF)

    assert "<table" in html
    assert "AAA" in html
    assert "PARTIALLY covered" in html
    assert "40/100" in html


def test_html_zero_reminders_has_no_table():
    html = format_html([], as_of=AS_OF)
    assert "<table" not in html
