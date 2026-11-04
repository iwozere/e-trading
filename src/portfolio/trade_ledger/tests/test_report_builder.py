"""Unit tests for the report view, HTML / plain-text rendering and the XLSX workbook (synthetic events)."""

from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import List, Optional

from openpyxl import load_workbook

from src.portfolio.trade_ledger.ledger_engine import build_ledger
from src.portfolio.trade_ledger.models import (
    CashTransaction,
    Execution,
    LedgerEvents,
    LedgerResult,
    ReconciliationLine,
    ReconciliationReport,
)
from src.portfolio.trade_ledger.reconciler import STATUS_NO_SNAPSHOT, STATUS_OK
from src.portfolio.trade_ledger.report_builder import ReportView, build_report_view, build_ticker_pnl, format_html, format_plain_text
from src.portfolio.trade_ledger.tests.event_factories import cash, trade
from src.portfolio.trade_ledger.xlsx_writer import write_xlsx

D = Decimal
_END = date(2025, 12, 31)


def _ledger(executions: List[Execution], cash_rows: Optional[List[CashTransaction]] = None) -> LedgerResult:
    return build_ledger(LedgerEvents(executions=executions, cash_transactions=cash_rows or []), date(2025, 12, 1), _END)


def _recon(result: LedgerResult, mark: Optional[str]) -> ReconciliationReport:
    lines = [
        ReconciliationLine(
            position_key=key, symbol=pos.symbol, ledger_shares=pos.shares, ibkr_shares=pos.shares, wac_incl=pos.wac_incl,
            ibkr_cost_basis_price=None, ibkr_mark_price=D(mark) if mark is not None else None,
        )
        for key, pos in result.positions.items()
        if pos.shares > 0
    ]
    return ReconciliationReport(as_of=_END, snapshot_date=date(2025, 12, 31), status=STATUS_OK, lines=lines)


def _sample() -> LedgerResult:
    return _ledger(
        [
            trade("2023-01-05 10:00", "10", "10", commission="-1", conid="200", symbol="OLD"),
            trade("2023-06-05 10:00", "-10", "8", commission="-1", conid="200", symbol="OLD"),
            trade("2025-02-05 10:00", "10", "20", commission="-1"),
        ],
        [cash("2025-06-10 20:20", "3")],
    )


def test_ticker_pnl_combines_realized_dividends_and_unrealized() -> None:
    result = _sample()

    rows = build_ticker_pnl(result, _recon(result, "25"))

    assert [r.symbol for r in rows] == ["ABC", "OLD"]  # best total first
    abc, old = rows
    assert abc.close == D(25)
    assert abc.market_value == D(250)
    assert abc.unrealized == D(250) - D(201)
    assert abc.dividends == D(3)
    assert abc.total == D(49) + D(3)
    assert old.shares == D(0)
    assert old.close is None and old.unrealized is None
    assert old.realized == D(80) - D(1) - D(101)
    assert old.total == old.realized
    assert not old.missing_price


def test_held_ticker_without_mark_price_is_flagged_and_excludes_unrealized() -> None:
    result = _sample()

    abc = next(r for r in build_ticker_pnl(result, _recon(result, None)) if r.symbol == "ABC")

    assert abc.missing_price
    assert abc.market_value is None
    assert abc.total == D(3)


def test_report_view_totals_per_currency_and_history_start() -> None:
    result = _sample()

    view = build_report_view(result, _recon(result, "25"), datetime(2026, 1, 2, tzinfo=timezone.utc))

    assert view.history_start == date(2023, 1, 5)
    (usd,) = view.ticker_pnl_totals
    assert usd.currency == "USD"
    assert usd.market_value == D(250)
    assert usd.total == D(52) + D(-22)
    assert usd.commissions == D(3)


def _view() -> ReportView:
    result = _sample()
    return build_report_view(result, _recon(result, "25"), datetime(2026, 1, 2, tzinfo=timezone.utc))


def test_html_ends_with_ticker_pnl_table() -> None:
    html = format_html(_view())

    assert "P&amp;L per ticker since 2023-01-05" in html
    assert "IBKR closing price of 2025-12-31" in html
    assert html.index("P&amp;L per ticker") > html.index("Positions")
    assert "Total USD" in html


def test_html_without_snapshot_says_no_closing_price() -> None:
    result = _sample()
    rec = ReconciliationReport(as_of=_END, snapshot_date=None, status=STATUS_NO_SNAPSHOT)

    html = format_html(build_report_view(result, rec, datetime(2026, 1, 2, tzinfo=timezone.utc)))

    assert "no IBKR closing price this month" in html


def test_plain_text_lists_every_ticker_and_total() -> None:
    text = format_plain_text(_view())

    section = text[text.index("P&L per ticker"):]
    assert "ABC" in section and "OLD" in section and "TOTAL" in section


def test_xlsx_has_ticker_pnl_sheet_with_totals_row(tmp_path: Path) -> None:
    result = _sample()
    view = build_report_view(result, _recon(result, "25"), datetime(2026, 1, 2, tzinfo=timezone.utc))

    path = write_xlsx(view, result, tmp_path / "out" / "ledger.xlsx")

    ws = load_workbook(path)["P&L per Ticker"]
    values = [[c.value for c in row] for row in ws.iter_rows(min_row=2)]
    assert [v[0] for v in values] == ["ABC", "OLD", "Total USD"]
    assert values[0][9] == 52  # Total P&L as a number, not text
    assert values[2][9] == 30


def _ytd_sample() -> LedgerResult:
    return _ledger(
        [
            trade("2024-11-05 10:00", "10", "10", commission="-1", conid="300", symbol="PRIOR"),
            trade("2025-03-05 10:00", "10", "10", commission="-1", conid="200", symbol="GONE"),
            trade("2025-04-05 10:00", "-10", "12", commission="-1", conid="200", symbol="GONE"),
            trade("2025-12-05 10:00", "10", "20", commission="-1"),
        ],
        [cash("2025-06-10 20:20", "2", conid="300", symbol="PRIOR")],
    )


def test_ytd_includes_positions_closed_earlier_in_the_year() -> None:
    result = _ytd_sample()

    view = build_report_view(result, _recon(result, "25"), datetime(2026, 1, 2, tzinfo=timezone.utc))

    assert [y.symbol for y in view.ytd] == ["ABC", "GONE", "PRIOR"]
    abc, gone, prior = view.ytd
    assert gone.closed and gone.bought == D(10) and gone.sold == D(10)
    assert gone.realized == D(120) - D(1) - D(101)
    assert gone.executions == 2
    assert not prior.closed and prior.executions == 0 and prior.dividends == D(2)  # dividend only, bought last year
    assert abc.realized == D(0) and abc.commissions == D(1)
    assert "GONE" not in [p.symbol for p in view.positions]  # month table only shows month activity / holdings
    (usd,) = view.ytd_totals
    assert (usd.executions, usd.commissions, usd.total) == (3, D(3), D(18) + D(2))


def test_ytd_section_in_html_text_and_xlsx(tmp_path: Path) -> None:
    result = _ytd_sample()
    view = build_report_view(result, _recon(result, "25"), datetime(2026, 1, 2, tzinfo=timezone.utc))

    html = format_html(view)
    assert html.index("Year to date 2025") < html.index("P&amp;L per ticker")
    assert "Year to date 2025" in format_plain_text(view)
    ws = load_workbook(write_xlsx(view, result, tmp_path / "l.xlsx"))["YTD 2025"]
    assert [row[0].value for row in ws.iter_rows(min_row=2)] == ["ABC", "GONE", "PRIOR", "Total USD"]


def test_pnl_totals_are_blank_when_currency_holds_nothing() -> None:
    result = _ledger(
        [
            trade("2023-01-05 10:00", "10", "10", commission="-1", conid="400", symbol="HELN"),
            trade("2023-06-05 10:00", "-10", "12", commission="-1", conid="400", symbol="HELN"),
        ]
    )

    view = build_report_view(result, _recon(result, None), datetime(2026, 1, 2, tzinfo=timezone.utc))

    (t,) = view.ticker_pnl_totals
    assert t.market_value is None and t.unrealized is None
    assert t.total == D(18)


def test_full_account_ids_are_shown() -> None:
    html = format_html(_view())

    assert "U0000001" in html
