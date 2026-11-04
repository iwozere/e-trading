"""
XLSX attachment of the monthly trade-ledger report.

Sheets: Summary (the month, as in the email), Executions (full history, each
row annotated with the position state after it), Positions (held at month
end), Cycles (closed position cycles), Corporate Actions, Reconciliation,
YTD (per ticker this year, incl. closed), P&L per Ticker (lifetime, unrealized at IBKR's month-end close) and Warnings. Values are written as numbers (Decimal), never as formatted text,
so the workbook stays sortable / filterable.
"""

from pathlib import Path
from typing import Any, List, Optional, Sequence

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from src.notification.logger import setup_logger
from src.portfolio.trade_ledger.models import LedgerResult
from src.portfolio.trade_ledger.report_builder import ReportView

_logger = setup_logger(__name__)

_HEADER_FONT = Font(bold=True)
_HEADER_FILL = PatternFill("solid", fgColor="F6F8FA")
_QTY = "#,##0.########"
_MONEY = "#,##0.00"
_PRICE = "#,##0.0000"
_DATE = "yyyy-mm-dd"
_DATETIME = "yyyy-mm-dd hh:mm:ss"


def _sheet(wb: Workbook, title: str, headers: Sequence[str], formats: Sequence[Optional[str]], rows: List[Sequence[Any]]) -> Worksheet:
    """
    Append a sheet with a bold frozen header, autofilter, number formats and sensible widths.

    Args:
        wb: Workbook to add the sheet to.
        title: Sheet title.
        headers: Column headers.
        formats: Number format per column (None = general).
        rows: Row values (Decimal / date / datetime / str / None).

    Returns:
        The new worksheet.
    """
    ws = wb.create_sheet(title)
    ws.append(list(headers))
    for cell in ws[1]:
        cell.font = _HEADER_FONT
        cell.fill = _HEADER_FILL
    for values in rows:
        ws.append(list(values))
    for col, fmt in enumerate(formats, start=1):
        letter = get_column_letter(col)
        if fmt:
            for (cell,) in ws.iter_rows(min_row=2, min_col=col, max_col=col):
                cell.number_format = fmt
        longest = max([len(str(headers[col - 1]))] + [len(str(r[col - 1])) for r in rows[:500] if r[col - 1] is not None])
        ws.column_dimensions[letter].width = min(max(10, longest + 2), 60)
    ws.freeze_panes = "A2"
    if rows:
        ws.auto_filter.ref = ws.dimensions
    return ws


def _summary_rows(view: ReportView) -> List[Sequence[Any]]:
    rows: List[Sequence[Any]] = []
    for p in view.positions:
        rows.append([
            p.symbol, p.isin, p.currency, p.shares, p.wac_excl, p.wac_incl, view.ibkr_basis.get(p.position_key),
            p.month.shares_bought, p.month.shares_sold, p.month.commissions, p.month.realized_excl, p.month.realized_incl,
            p.ytd.commissions, p.ytd.realized_excl, p.ytd.realized_incl, p.cycle_no, p.cycle_open_date,
        ])
    return rows


def write_xlsx(view: ReportView, result: LedgerResult, path: Path) -> Path:
    """
    Write the report workbook.

    Args:
        view: Month view (Summary, Reconciliation, month warnings).
        result: Full ledger (history sheets).
        path: Output file path; parent directories are created.

    Returns:
        `path`.
    """
    wb = Workbook()
    default = wb.active
    if default is not None:
        wb.remove(default)

    _sheet(
        wb, "Summary",
        ["Symbol", "ISIN", "Ccy", "Shares", "Avg excl. comm.", "Avg incl. comm.", "IBKR basis", "Bought (month)", "Sold (month)",
         "Commission (month)", "Realized excl. (month)", "Realized incl. (month)", "Commission (YTD)", "Realized excl. (YTD)",
         "Realized incl. (YTD)", "Cycle", "Cycle opened"],
        [None, None, None, _QTY, _PRICE, _PRICE, _PRICE, _QTY, _QTY, _MONEY, _MONEY, _MONEY, _MONEY, _MONEY, _MONEY, None, _DATE],
        _summary_rows(view),
    )

    exec_rows: List[Sequence[Any]] = []
    for row in result.execution_rows:
        ex = row.execution
        exec_rows.append([
            ex.trade_datetime, ex.account_id, ex.symbol, ex.isin, ex.buy_sell, ex.quantity, ex.trade_price, abs(ex.proceeds),
            abs(ex.ib_commission), abs(ex.taxes), ex.net_cash, ex.currency, row.shares_after, row.wac_excl_after, row.wac_incl_after,
            row.realized_excl, row.realized_incl, row.cycle_no, ex.fifo_pnl_realized, ex.notes, ex.trade_id, ex.source_file,
        ])
    _sheet(
        wb, "Executions",
        ["Date/time", "Account", "Symbol", "ISIN", "Side", "Quantity", "Price", "Gross", "Commission", "Taxes", "Net cash", "Ccy",
         "Shares after", "Avg excl. after", "Avg incl. after", "Realized excl.", "Realized incl.", "Cycle", "IBKR FIFO P&L", "Notes",
         "Trade ID", "Source file"],
        [_DATETIME, None, None, None, None, _QTY, _PRICE, _MONEY, _MONEY, _MONEY, _MONEY, None, _QTY, _PRICE, _PRICE, _MONEY, _MONEY,
         None, _MONEY, None, None, None],
        exec_rows,
    )

    held = sorted((p for p in result.positions.values() if p.shares != 0), key=lambda p: p.symbol)
    _sheet(
        wb, "Positions",
        ["Symbol", "ISIN", "Ccy", "Shares", "Cost excl.", "Cost incl.", "Avg excl.", "Avg incl.", "IBKR basis", "Cycle", "Cycle opened"],
        [None, None, None, _QTY, _MONEY, _MONEY, _PRICE, _PRICE, _PRICE, None, _DATE],
        [[p.symbol, p.isin, p.currency, p.shares, p.cost_excl, p.cost_incl, p.wac_excl, p.wac_incl, view.ibkr_basis.get(p.position_key),
          p.cycle_no, p.cycle_open_date] for p in held],
    )

    _sheet(
        wb, "Cycles",
        ["Symbol", "Ccy", "Cycle", "Opened", "Closed", "Shares bought", "Commissions", "Realized excl.", "Realized incl."],
        [None, None, None, _DATE, _DATE, _QTY, _MONEY, _MONEY, _MONEY],
        [[c.symbol, c.currency, c.cycle_no, c.open_date, c.close_date, c.shares_bought, c.commissions, c.realized_excl, c.realized_incl]
         for c in result.closed_cycles],
    )

    _sheet(
        wb, "Corporate Actions",
        ["Date/time", "Symbol", "Type", "Shares before", "Shares after", "Avg incl. before", "Avg incl. after", "Proceeds", "Realized incl.",
         "Description"],
        [_DATETIME, None, None, _QTY, _QTY, _PRICE, _PRICE, _MONEY, _MONEY, None],
        [[c.action_datetime, c.symbol, c.action_type, c.shares_before, c.shares_after, c.wac_incl_before, c.wac_incl_after, c.proceeds,
          c.realized_incl, c.description] for c in result.corporate_action_rows],
    )

    rec = view.reconciliation
    _sheet(
        wb, "Reconciliation",
        ["Snapshot", "Status", "Symbol", "Ledger shares", "IBKR shares", "Match", "Avg incl.", "IBKR basis"],
        [_DATE, None, None, _QTY, _QTY, None, _PRICE, _PRICE],
        [[rec.snapshot_date, rec.status, line.symbol, line.ledger_shares, line.ibkr_shares, "yes" if line.shares_match else "NO",
          line.wac_incl, line.ibkr_cost_basis_price] for line in rec.lines] or [[rec.snapshot_date, rec.status, None, None, None, None, None, None]],
    )

    ytd_rows: List[Sequence[Any]] = [
        [r.symbol, r.currency, r.bought, r.sold, r.shares, "closed" if r.closed else "held", r.executions, r.commissions, r.realized, r.dividends,
         r.total]
        for r in view.ytd
    ]
    ytd_rows += [
        [f"Total {t.currency}", t.currency, None, None, None, None, t.executions, t.commissions, t.realized, t.dividends, t.total]
        for t in view.ytd_totals
    ]
    _sheet(
        wb, f"YTD {view.period_end.year}",
        ["Symbol", "Ccy", "Bought", "Sold", "Held", "Status", "Executions", "Commissions", "Realized", "Dividends (net)", "Total"],
        [None, None, _QTY, _QTY, _QTY, None, None, _MONEY, _MONEY, _MONEY, _MONEY],
        ytd_rows,
    )

    pnl_rows: List[Sequence[Any]] = [
        [r.symbol, r.currency, r.shares, r.wac_incl, r.close, r.market_value, r.unrealized, r.realized, r.dividends, r.total, r.commissions]
        for r in view.ticker_pnl
    ]
    pnl_rows += [
        [f"Total {t.currency}", t.currency, None, None, None, t.market_value, t.unrealized, t.realized, t.dividends, t.total, t.commissions]
        for t in view.ticker_pnl_totals
    ]
    _sheet(
        wb, "P&L per Ticker",
        ["Symbol", "Ccy", "Shares", "Avg incl. comm.", "Close", "Market value", "Unrealized", "Realized", "Dividends (net)", "Total P&L",
         "Commissions"],
        [None, None, _QTY, _PRICE, _PRICE, _MONEY, _MONEY, _MONEY, _MONEY, _MONEY, _MONEY],
        pnl_rows,
    )

    _sheet(
        wb, "Warnings",
        ["Date/time", "Symbol", "Kind", "Message"],
        [_DATETIME, None, None, None],
        [[w.when, w.symbol, w.kind, w.message] for w in result.warnings],
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    _logger.info("Wrote trade-ledger workbook %s (%d executions)", path, len(exec_rows))
    return path

