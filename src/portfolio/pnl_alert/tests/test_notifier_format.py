"""Unit tests for notifier message formatting."""

from datetime import UTC, datetime

from src.portfolio.pnl_alert.insider_activity import InsiderTransaction
from src.portfolio.pnl_alert.notifier import format_html, format_plain_text
from src.portfolio.pnl_alert.pnl_evaluator import AlertRow

AS_OF = datetime(2026, 4, 20, tzinfo=UTC)


def _row(symbol: str, avg: float, now: float, qty: float = 1.0, source: str = "ibkr", flagged: bool | None = None) -> AlertRow:
    pnl_pct = (now - avg) / avg
    return AlertRow(
        symbol=symbol,
        avg_price=avg,
        current_price=now,
        quantity=qty,
        pnl_abs=(now - avg) * qty,
        pnl_pct=pnl_pct,
        source=source,
        flagged=flagged if flagged is not None else pnl_pct >= 0.10,
    )


def _txn(
    ticker: str,
    date: str = "2026-04-15",
    code: str = "P",
    insider: str = "Jensen Huang",
    role: str = "Officer (CEO)",
    shares: int = 1000,
    price: float = 25.0,
    plan: bool = False,
) -> InsiderTransaction:
    return InsiderTransaction(
        ticker=ticker,
        insider_name=insider,
        role=role,
        transaction_code=code,
        acquired_disposed_code="A" if code in ("P", "A") else "D",
        shares=shares,
        price_per_share=price,
        total_value_usd=shares * price,
        transaction_date=date,
        filed_date=date,
        is_10b5_1_plan=plan,
    )


def test_plain_text_contains_header_and_all_rows():
    """The header, every row (not just flagged ones), and source summary all appear."""
    rows = [
        _row("NVDA", 120.0, 156.4, qty=10, source="ibkr"),
        _row("AAPL", 150.0, 180.15, qty=10, source="ibkr"),
        _row("MSFT", 310.0, 312.0, qty=1, source="watchlist"),
    ]

    text = format_plain_text(rows, threshold_pct=0.10, as_of=AS_OF)

    assert "Portfolio PnL Digest - 2026-04-20" in text
    assert "3 position(s), 2 flagged >= +10.00% threshold" in text
    assert "1. NVDA" in text
    assert "2. AAPL" in text
    assert "3. MSFT" in text
    assert "Sources: ibkr=2, watchlist=1" in text


def test_plain_text_flags_only_qualifying_rows():
    """Rows below threshold are still printed but without the flag marker."""
    rows = [
        _row("NVDA", 120.0, 156.4, qty=10),  # +30%, flagged
        _row("MSFT", 310.0, 312.0, qty=1),  # +0.6%, not flagged
    ]

    text = format_plain_text(rows, threshold_pct=0.10, as_of=AS_OF)
    lines = text.splitlines()
    nvda_line = next(line for line in lines if "NVDA" in line)
    msft_line = next(line for line in lines if "MSFT" in line)

    assert "\U0001f53a" in nvda_line
    assert "\U0001f53a" not in msft_line


def test_plain_text_zero_rows_returns_header_only():
    """Zero rows (no priced holdings) produce just the header text (no bullet list, no footer)."""
    text = format_plain_text([], threshold_pct=0.10, as_of=AS_OF)

    assert "Portfolio PnL Digest - 2026-04-20" in text
    assert "0 position(s), 0 flagged" in text
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


def test_insider_activity_section_appears_under_its_ticker_plain_text():
    """A ticker with insider transactions gets an indented sub-section; others don't."""
    rows = [_row("NVDA", 120.0, 156.4, qty=10), _row("AAPL", 150.0, 152.0, qty=10)]
    insider_by_ticker = {"NVDA": [_txn("NVDA", code="P", insider="Jensen Huang", role="Officer (CEO)")]}

    text = format_plain_text(rows, threshold_pct=0.10, as_of=AS_OF, insider_by_ticker=insider_by_ticker)

    assert "Insider activity (30d):" in text
    assert "Jensen Huang (Officer (CEO))" in text
    assert "Buy" in text
    lines = text.splitlines()
    aapl_idx = next(i for i, line in enumerate(lines) if "2. AAPL" in line)
    # AAPL has no insider activity — the next line must not be an insider section.
    assert aapl_idx == len(lines) - 1 or "Insider activity" not in lines[aapl_idx + 1]


def test_10b5_1_plan_transactions_are_labeled_separately():
    """10b5-1 plan trades are shown but tagged, not filtered out or mixed in silently."""
    rows = [_row("NVDA", 120.0, 156.4, qty=10)]
    insider_by_ticker = {"NVDA": [_txn("NVDA", code="S", plan=True)]}

    text = format_plain_text(rows, threshold_pct=0.10, as_of=AS_OF, insider_by_ticker=insider_by_ticker)

    assert "[10b5-1 plan]" in text


def test_ticker_with_no_insider_activity_has_no_section():
    """A ticker absent from insider_by_ticker gets no sub-section at all."""
    rows = [_row("NVDA", 120.0, 156.4, qty=10)]

    text = format_plain_text(rows, threshold_pct=0.10, as_of=AS_OF, insider_by_ticker={})

    assert "Insider activity" not in text


def test_html_insider_section_and_plan_badge():
    """HTML output nests an insider-activity table and marks 10b5-1 plan trades."""
    rows = [_row("NVDA", 120.0, 156.4, qty=10)]
    insider_by_ticker = {"NVDA": [_txn("NVDA", code="S", plan=True)]}

    html = format_html(rows, threshold_pct=0.10, as_of=AS_OF, insider_by_ticker=insider_by_ticker)

    assert "10b5-1 plan" in html
    assert "Sell" in html
