"""Unit tests for notifier message formatting."""

from datetime import datetime, timezone

from src.portfolio.pnl_alert.notifier import format_html, format_plain_text
from src.portfolio.pnl_alert.pnl_evaluator import AlertRow


def _row(symbol: str, avg: float, now: float, qty: float = 1.0, source: str = "ibkr") -> AlertRow:
    return AlertRow(
        symbol=symbol,
        avg_price=avg,
        current_price=now,
        quantity=qty,
        pnl_abs=(now - avg) * qty,
        pnl_pct=(now - avg) / avg,
        source=source,
    )


AS_OF = datetime(2026, 4, 20, tzinfo=timezone.utc)


def test_plain_text_contains_header_and_rows():
    """The header, row lines, and source summary all appear."""
    rows = [
        _row("NVDA", 120.0, 156.4, qty=10, source="ibkr"),
        _row("AAPL", 150.0, 180.15, qty=10, source="ibkr"),
        _row("MSFT", 310.0, 352.70, qty=1, source="watchlist"),
    ]

    text = format_plain_text(rows, threshold_pct=0.10, as_of=AS_OF)

    assert "Portfolio PnL Alert - 2026-04-20" in text
    assert "3 position(s) above +10.00% threshold" in text
    assert "1. NVDA" in text
    assert "2. AAPL" in text
    assert "3. MSFT" in text
    assert "Sources: ibkr=2, watchlist=1" in text


def test_plain_text_zero_rows_returns_header_only():
    """Zero rows produce just the header text (no bullet list, no footer)."""
    text = format_plain_text([], threshold_pct=0.10, as_of=AS_OF)

    assert "Portfolio PnL Alert - 2026-04-20" in text
    assert "0 position(s) above" in text
    assert "Sources:" not in text


def test_html_has_table_markup():
    """HTML formatting wraps rows in a `<table>` element."""
    rows = [_row("NVDA", 120.0, 156.4, qty=10)]
    html = format_html(rows, threshold_pct=0.10, as_of=AS_OF)

    assert "<table" in html
    assert "<th>Ticker</th>" in html
    assert "NVDA" in html


def test_money_and_pct_signs_are_formatted():
    """Signed formatting uses explicit + or - sign."""
    rows = [_row("NVDA", 100.0, 130.0)]
    text = format_plain_text(rows, threshold_pct=0.10, as_of=AS_OF)

    assert "+$30.00" in text
    assert "+30.00%" in text
