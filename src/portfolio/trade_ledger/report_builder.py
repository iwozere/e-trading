"""
Monthly trade-ledger report: view model + HTML and plain-text email bodies.

`build_report_view` derives everything the email shows from a `LedgerResult`
and a `ReconciliationReport` once; `format_html` / `format_plain_text` only
render it. HTML is email-safe: inline styles, tables, no scripts or external
resources (Gmail / Outlook strip those).
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from html import escape
from typing import Dict, List, Optional

from src.portfolio.trade_ledger.models import (
    ClosedCycle,
    CorporateActionRow,
    DepositTotals,
    ExecutionRow,
    LedgerResult,
    LedgerWarning,
    PositionSummary,
    ReconciliationReport,
)
from src.portfolio.trade_ledger.reconciler import STATUS_MISMATCH, STATUS_NO_SNAPSHOT, STATUS_OK

_ZERO = Decimal(0)

# Colours (inline; email clients ignore <style> blocks unreliably).
_INK = "#1f2328"
_MUTED = "#656d76"
_LINE = "#d0d7de"
_HEAD_BG = "#f6f8fa"
_GAIN = "#1a7f37"
_LOSS = "#cf222e"
_WARN_BG = "#fff8c5"
_OK_BG = "#dafbe1"
_BAD_BG = "#ffebe9"


@dataclass
class CurrencyTotals:
    """Month / YTD totals for one currency."""

    currency: str
    executions_month: int = 0
    commissions_month: Decimal = _ZERO
    commissions_ytd: Decimal = _ZERO
    realized_month: Decimal = _ZERO
    realized_ytd: Decimal = _ZERO


@dataclass(frozen=True)
class TickerPnl:
    """Lifetime P&L of one ticker: realized (all cycles) + net dividends + unrealized at the month-end close."""

    position_key: str
    symbol: str
    currency: str
    shares: Decimal
    wac_incl: Optional[Decimal]
    close: Optional[Decimal]
    market_value: Optional[Decimal]
    unrealized: Optional[Decimal]
    realized: Decimal
    dividends: Decimal
    commissions: Decimal
    total: Decimal

    @property
    def missing_price(self) -> bool:
        """True when shares are held but no closing price was available (total excludes unrealized)."""
        return self.shares > 0 and self.unrealized is None


@dataclass
class PnlTotals:
    """Sum of the P&L-per-ticker table for one currency."""

    currency: str
    market_value: Optional[Decimal] = None
    unrealized: Optional[Decimal] = None
    realized: Decimal = _ZERO
    dividends: Decimal = _ZERO
    commissions: Decimal = _ZERO
    total: Decimal = _ZERO


@dataclass(frozen=True)
class YtdTicker:
    """Year-to-date activity of one ticker (traded, realized or paid dividends this year)."""

    position_key: str
    symbol: str
    currency: str
    shares: Decimal
    bought: Decimal
    sold: Decimal
    executions: int
    commissions: Decimal
    realized: Decimal
    dividends: Decimal

    @property
    def closed(self) -> bool:
        """True when nothing is held at the report month end."""
        return self.shares == 0

    @property
    def total(self) -> Decimal:
        """Realized P&L + net dividends of the year."""
        return self.realized + self.dividends


@dataclass
class YtdTotals:
    """Sum of the year-to-date table for one currency."""

    currency: str
    executions: int = 0
    commissions: Decimal = _ZERO
    realized: Decimal = _ZERO
    dividends: Decimal = _ZERO

    @property
    def total(self) -> Decimal:
        """Realized P&L + net dividends of the year."""
        return self.realized + self.dividends


@dataclass
class ReportView:
    """Everything the monthly report shows, derived once."""

    period_start: date
    period_end: date
    generated_at: datetime
    accounts: List[str]
    executions: List[ExecutionRow]
    positions: List[PositionSummary]
    ibkr_basis: Dict[str, Decimal]
    closed_cycles: List[ClosedCycle]
    corporate_actions: List[CorporateActionRow]
    warnings: List[LedgerWarning]
    reconciliation: ReconciliationReport
    totals: List[CurrencyTotals] = field(default_factory=list)
    ticker_pnl: List[TickerPnl] = field(default_factory=list)
    ticker_pnl_totals: List[PnlTotals] = field(default_factory=list)
    history_start: Optional[date] = None
    ytd: List[YtdTicker] = field(default_factory=list)
    ytd_totals: List[YtdTotals] = field(default_factory=list)
    deposits: List[DepositTotals] = field(default_factory=list)

    @property
    def title(self) -> str:
        """Report title, e.g. 'Trade Ledger — October 2026'."""
        return f"Trade Ledger — {self.period_start.strftime('%B %Y')}"


def build_ticker_pnl(result: LedgerResult, reconciliation: ReconciliationReport) -> List[TickerPnl]:
    """
    Lifetime P&L per ticker, best first.

    Unrealized = shares x IBKR month-end mark price - remaining cost incl.
    commission; None when shares are held but IBKR gave no mark price.

    Args:
        result: Ledger up to the report month end.
        reconciliation: Supplies IBKR's mark prices of the snapshot date.

    Returns:
        One row per ticker ever traded or paid a dividend, sorted by total P&L descending.
    """
    marks = reconciliation.mark_price_by_key()
    rows: List[TickerPnl] = []
    for key, life in result.lifetime_totals.items():
        pos = result.positions[key]
        held = pos.shares > 0 and not pos.is_short
        close = marks.get(key) if held else None
        unrealized = pos.shares * close - pos.cost_incl if held and close is not None else None
        rows.append(
            TickerPnl(
                position_key=key,
                symbol=pos.symbol,
                currency=pos.currency,
                shares=pos.shares,
                wac_incl=pos.wac_incl,
                close=close,
                market_value=pos.shares * close if close is not None else None,
                unrealized=unrealized,
                realized=life.realized_incl,
                dividends=life.dividends,
                commissions=life.commissions,
                total=life.realized_incl + life.dividends + (unrealized or _ZERO),
            )
        )
    rows.sort(key=lambda r: (-r.total, r.symbol))
    return rows


def _pnl_totals(rows: List[TickerPnl]) -> List[PnlTotals]:
    """Sum the P&L-per-ticker table per currency."""
    totals: Dict[str, PnlTotals] = {}
    for r in rows:
        t = totals.setdefault(r.currency, PnlTotals(currency=r.currency))
        if r.market_value is not None:
            t.market_value = (t.market_value or _ZERO) + r.market_value
        if r.unrealized is not None:
            t.unrealized = (t.unrealized or _ZERO) + r.unrealized
        t.realized += r.realized
        t.dividends += r.dividends
        t.commissions += r.commissions
        t.total += r.total
    return sorted(totals.values(), key=lambda t: t.currency)


def build_ytd(result: LedgerResult) -> List[YtdTicker]:
    """
    Year-to-date activity per ticker, including positions closed earlier in the year.

    Args:
        result: Ledger up to the report month end.

    Returns:
        One row per ticker with executions, realized P&L or dividends this year,
        sorted by symbol.
    """
    rows: List[YtdTicker] = []
    for key, ytd in result.ytd_totals.items():
        if not (ytd.executions or ytd.realized_incl or ytd.dividends):
            continue
        pos = result.positions[key]
        rows.append(
            YtdTicker(
                position_key=key,
                symbol=pos.symbol,
                currency=pos.currency,
                shares=pos.shares,
                bought=ytd.shares_bought,
                sold=ytd.shares_sold,
                executions=ytd.executions,
                commissions=ytd.commissions,
                realized=ytd.realized_incl,
                dividends=ytd.dividends,
            )
        )
    rows.sort(key=lambda r: (r.symbol, r.position_key))
    return rows


def _ytd_totals(rows: List[YtdTicker]) -> List[YtdTotals]:
    """Sum the year-to-date table per currency."""
    totals: Dict[str, YtdTotals] = {}
    for r in rows:
        t = totals.setdefault(r.currency, YtdTotals(currency=r.currency))
        t.executions += r.executions
        t.commissions += r.commissions
        t.realized += r.realized
        t.dividends += r.dividends
    return sorted(totals.values(), key=lambda t: t.currency)


def build_report_view(result: LedgerResult, reconciliation: ReconciliationReport, generated_at: datetime) -> ReportView:
    """
    Derive the report content for `result.period_start`..`result.period_end`.

    Args:
        result: Ledger built for the report month.
        reconciliation: Reconciliation of the same month.
        generated_at: Timestamp shown in the footer.

    Returns:
        ReportView with month executions (oldest first), positions held at month
        end first then positions closed in the month, totals per currency and the
        warnings that fall inside the month.
    """
    start, end = result.period_start, result.period_end

    def in_month(day: date) -> bool:
        return start <= day <= end

    executions = [row for row in result.execution_rows if in_month(row.execution.trade_date)]
    positions = sorted(result.summaries, key=lambda s: (s.shares == 0, s.symbol))

    totals: Dict[str, CurrencyTotals] = {}
    for key, pos in result.positions.items():
        cur = totals.setdefault(pos.currency, CurrencyTotals(currency=pos.currency))
        month = result.month_totals.get(key)
        ytd = result.ytd_totals.get(key)
        if month is not None:
            cur.executions_month += month.executions
            cur.commissions_month += month.commissions
            cur.realized_month += month.realized_incl
        if ytd is not None:
            cur.commissions_ytd += ytd.commissions
            cur.realized_ytd += ytd.realized_incl

    accounts = sorted({row.execution.account_id for row in result.execution_rows})
    ticker_pnl = build_ticker_pnl(result, reconciliation)
    ytd_rows = build_ytd(result)
    return ReportView(
        period_start=start,
        period_end=end,
        generated_at=generated_at,
        accounts=accounts,
        executions=executions,
        positions=positions,
        ibkr_basis=reconciliation.ibkr_cost_basis_by_key(),
        closed_cycles=[c for c in result.closed_cycles if in_month(c.close_date)],
        corporate_actions=[c for c in result.corporate_action_rows if in_month(c.action_datetime.date())],
        warnings=[w for w in result.warnings if in_month(w.when.date())],
        reconciliation=reconciliation,
        totals=sorted((t for t in totals.values() if t.executions_month or t.realized_ytd or t.commissions_ytd), key=lambda t: t.currency),
        ticker_pnl=ticker_pnl,
        ticker_pnl_totals=_pnl_totals(ticker_pnl),
        history_start=min((row.execution.trade_date for row in result.execution_rows), default=None),
        ytd=ytd_rows,
        ytd_totals=_ytd_totals(ytd_rows),
        deposits=sorted(result.deposits.values(), key=lambda d: d.currency),
    )


# --------------------------------------------------------------------------
# Number formatting
# --------------------------------------------------------------------------


def fmt_qty(value: Decimal) -> str:
    """Share quantity without trailing zeros, thousands separated (0.0002, 1,540)."""
    return f"{value.normalize():,f}"


def fmt_money(value: Decimal) -> str:
    """Money with 2 decimals."""
    return f"{value:,.2f}"


def fmt_signed(value: Decimal) -> str:
    """Signed money with 2 decimals (+12.34 / -5.00)."""
    return f"{value:+,.2f}"


def fmt_price(value: Optional[Decimal]) -> str:
    """Per-share price with 4 decimals (averages differ from each other by cents); '—' when unknown."""
    if value is None:
        return "—"
    return f"{value:,.4f}"


def fmt_trade_price(value: Decimal) -> str:
    """Execution price as IBKR reported it, at least 2 decimals."""
    text = f"{value.normalize():f}"
    if "." not in text:
        return f"{value:,.2f}"
    return f"{value:,.{max(2, len(text.split('.')[1]))}f}"


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

# Cell styling lives on the <table> (inherited) and in attributes, not on every
# <td>: Gmail clips HTML bodies over ~102 KB, and per-cell inline styles alone
# pushed the report past that.
_TABLE = f"border-collapse:collapse;width:100%;font-size:13px;color:{_INK};white-space:nowrap;font-variant-numeric:tabular-nums;"
_TH = f"border-bottom:2px solid {_LINE};background:{_HEAD_BG};font-weight:600;"
_ZEBRA = _HEAD_BG
_H2 = f"font-size:16px;margin:24px 0 8px 0;color:{_INK};"


def _th(label: str, numeric: bool = False) -> str:
    return f"<th align='{'right' if numeric else 'left'}' style='{_TH}'>{escape(label)}</th>"


def _td(text: str, numeric: bool = False, extra: str = "") -> str:
    align = " align='right'" if numeric else ""
    style = f" style='{extra}'" if extra else ""
    return f"<td{align}{style}>{text}</td>"


def _pnl_td(value: Optional[Decimal], extra: str = "") -> str:
    if value is None:
        return _td("", numeric=True, extra=extra)
    colour = _GAIN if value > 0 else _LOSS if value < 0 else _MUTED
    return _td(escape(fmt_signed(value)), numeric=True, extra=f"color:{colour};{extra}")


def _table(headers: List[str], numeric: List[bool], rows: List[str]) -> str:
    head = "".join(_th(h, n) for h, n in zip(headers, numeric))
    body = "".join(row.replace("<tr>", f"<tr bgcolor='{_ZEBRA}'>", 1) if i % 2 else row for i, row in enumerate(rows))
    return f"<table cellpadding='5' cellspacing='0' style='{_TABLE}'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _tile(label: str, value: str, colour: str = _INK) -> str:
    return (
        f"<td style='padding:10px 14px;border:1px solid {_LINE};vertical-align:top;'>"
        f"<div style='font-size:12px;color:{_MUTED};'>{escape(label)}</div>"
        f"<div style='font-size:18px;font-weight:600;color:{colour};font-variant-numeric:tabular-nums;'>{escape(value)}</div></td>"
    )


def _tiles_html(view: ReportView) -> str:
    buys = sum(1 for r in view.executions if r.execution.quantity > 0)
    sells = len(view.executions) - buys
    tiles = [_tile("Executions", f"{len(view.executions)}  ({buys} buy / {sells} sell)")]
    for t in view.totals:
        tiles.append(_tile(f"Commissions {t.currency} (month / YTD)", f"{fmt_money(t.commissions_month)} / {fmt_money(t.commissions_ytd)}"))
        colour = _GAIN if t.realized_month > 0 else _LOSS if t.realized_month < 0 else _INK
        tiles.append(_tile(f"Realized P&L {t.currency} month", fmt_signed(t.realized_month), colour))
        colour = _GAIN if t.realized_ytd > 0 else _LOSS if t.realized_ytd < 0 else _INK
        tiles.append(_tile(f"Realized P&L {t.currency} YTD", fmt_signed(t.realized_ytd), colour))
    for d in view.deposits:
        tiles.append(_tile(f"Net deposits {d.currency} (month / YTD / total)", f"{fmt_money(d.month)} / {fmt_money(d.ytd)} / {fmt_money(d.lifetime)}"))
    return f"<table cellpadding='0' cellspacing='0' style='border-collapse:collapse;margin:8px 0;'><tr>{''.join(tiles)}</tr></table>"


def _reconciliation_html(view: ReportView) -> str:
    rec = view.reconciliation
    if rec.status == STATUS_OK:
        text = f"✔ Share counts match IBKR's positions of {rec.snapshot_date} for all {len(rec.lines)} positions."
        bg = _OK_BG
    elif rec.status == STATUS_NO_SNAPSHOT:
        text = "No IBKR position snapshot for this month — share counts not cross-checked."
        bg = _WARN_BG
    else:
        details = ", ".join(f"{m.symbol}: ours {fmt_qty(m.ledger_shares)} vs IBKR {fmt_qty(m.ibkr_shares)}" for m in rec.mismatches)
        text = f"✖ Share counts differ from IBKR ({rec.snapshot_date}): {details}"
        bg = _BAD_BG
    return f"<p style='padding:8px 12px;background:{bg};border-radius:6px;font-size:13px;margin:8px 0;'>{escape(text)}</p>"


def _executions_html(view: ReportView) -> str:
    if not view.executions:
        return f"<p style='color:{_MUTED};font-size:13px;'>No executions this month.</p>"
    rows = []
    for row in view.executions:
        ex = row.execution
        side_colour = _GAIN if ex.quantity > 0 else _LOSS
        rows.append(
            "<tr>"
            + _td(escape(ex.trade_datetime.strftime("%Y-%m-%d %H:%M")))
            + _td(escape(ex.account_id), extra=f"color:{_MUTED};")
            + _td(f"<b>{escape(ex.symbol)}</b>")
            + _td(escape(ex.buy_sell), extra=f"color:{side_colour};font-weight:600;")
            + _td(escape(fmt_qty(abs(ex.quantity))), numeric=True)
            + _td(escape(fmt_trade_price(ex.trade_price)), numeric=True)
            + _td(escape(fmt_money(abs(ex.proceeds))), numeric=True)
            + _td(escape(fmt_money(abs(ex.ib_commission) + abs(ex.taxes))), numeric=True)
            + _td(escape(fmt_money(ex.net_cash)), numeric=True)
            + _td(escape(ex.currency), extra=f"color:{_MUTED};")
            + _pnl_td(row.realized_incl)
            + "</tr>"
        )
    headers = ["Date", "Account", "Symbol", "Side", "Qty", "Price", "Gross", "Commission", "Net cash", "Ccy", "Realized P&L"]
    numeric = [False, False, False, False, True, True, True, True, True, False, True]
    return _table(headers, numeric, rows)


def _positions_html(view: ReportView) -> str:
    if not view.positions:
        return f"<p style='color:{_MUTED};font-size:13px;'>No positions.</p>"
    rows = []
    for p in view.positions:
        ibkr = view.ibkr_basis.get(p.position_key)
        closed = p.shares == 0
        style = f"color:{_MUTED};" if closed else ""
        rows.append(
            "<tr>"
            + _td(f"<b>{escape(p.symbol)}</b>" + (" <span style='font-size:11px;'>(closed)</span>" if closed else ""), extra=style)
            + _td(escape(fmt_qty(p.shares)), numeric=True, extra=style)
            + _td(escape(fmt_price(p.wac_excl)), numeric=True, extra=style)
            + _td(f"<b>{escape(fmt_price(p.wac_incl))}</b>", numeric=True, extra=style)
            + _td(escape(fmt_price(ibkr)), numeric=True, extra=f"color:{_MUTED};")
            + _td(escape(fmt_qty(p.month.shares_bought)) if p.month.shares_bought else "", numeric=True)
            + _td(escape(fmt_qty(p.month.shares_sold)) if p.month.shares_sold else "", numeric=True)
            + _td(escape(fmt_money(p.month.commissions)) if p.month.executions else "", numeric=True)
            + (_pnl_td(p.month.realized_incl) if p.month.realized_incl else _td("", numeric=True))
            + _td(escape(p.currency), extra=f"color:{_MUTED};")
            + "</tr>"
        )
    headers = ["Symbol", "Shares", "Avg excl. comm.", "Avg incl. comm.", "IBKR basis", "Bought", "Sold", "Commission", "Realized month", "Ccy"]
    numeric = [False, True, True, True, True, True, True, True, True, False]
    return _table(headers, numeric, rows)


def _ytd_html(view: ReportView) -> str:
    rows = []
    for r in view.ytd:
        muted = f"color:{_MUTED};" if r.closed else ""
        rows.append(
            "<tr>"
            + _td(f"<b>{escape(r.symbol)}</b>" + (" <span style='font-size:11px;'>(closed)</span>" if r.closed else ""), extra=muted)
            + _td(escape(fmt_qty(r.bought)) if r.bought else "", numeric=True)
            + _td(escape(fmt_qty(r.sold)) if r.sold else "", numeric=True)
            + _td(escape(fmt_qty(r.shares)) if not r.closed else "", numeric=True)
            + _td(str(r.executions) if r.executions else "", numeric=True)
            + _td(escape(fmt_money(r.commissions)) if r.commissions else "", numeric=True)
            + (_pnl_td(r.realized) if r.realized else _td("", numeric=True))
            + (_pnl_td(r.dividends) if r.dividends else _td("", numeric=True))
            + _pnl_td(r.total, extra="font-weight:600;")
            + _td(escape(r.currency), extra=f"color:{_MUTED};")
            + "</tr>"
        )
    bold = f"font-weight:700;border-top:2px solid {_LINE};"
    for t in view.ytd_totals:
        rows.append(
            "<tr>"
            + _td(f"Total {escape(t.currency)}", extra=bold)
            + _td("", extra=bold)
            + _td("", extra=bold)
            + _td("", extra=bold)
            + _td(str(t.executions), numeric=True, extra=bold)
            + _td(escape(fmt_money(t.commissions)), numeric=True, extra=bold)
            + _pnl_td(t.realized, extra=bold)
            + _pnl_td(t.dividends, extra=bold)
            + _pnl_td(t.total, extra=bold)
            + _td(escape(t.currency), extra=bold)
            + "</tr>"
        )
    headers = ["Symbol", "Bought", "Sold", "Held", "Executions", "Commissions", "Realized", "Dividends (net)", "Total", "Ccy"]
    numeric = [False, True, True, True, True, True, True, True, True, False]
    return _table(headers, numeric, rows)


def _cycles_html(view: ReportView) -> str:
    rows = [
        "<tr>"
        + _td(f"<b>{escape(c.symbol)}</b>")
        + _td(escape(c.open_date.isoformat() if c.open_date else "—"))
        + _td(escape(c.close_date.isoformat()))
        + _td(escape(fmt_qty(c.shares_bought)), numeric=True)
        + _td(escape(fmt_money(c.commissions)), numeric=True)
        + _pnl_td(c.realized_incl)
        + _td(escape(c.currency), extra=f"color:{_MUTED};")
        + "</tr>"
        for c in view.closed_cycles
    ]
    return _table(["Symbol", "Opened", "Closed", "Shares bought", "Commissions", "Realized P&L", "Ccy"], [False, False, False, True, True, True, False], rows)


def _corporate_actions_html(view: ReportView) -> str:
    rows = [
        "<tr>"
        + _td(escape(c.action_datetime.date().isoformat()))
        + _td(f"<b>{escape(c.symbol)}</b>")
        + _td(escape(c.action_type))
        + _td(escape(f"{fmt_qty(c.shares_before)} → {fmt_qty(c.shares_after)}"), numeric=True)
        + _td(escape(f"{fmt_price(c.wac_incl_before)} → {fmt_price(c.wac_incl_after)}"), numeric=True)
        + _td(escape(c.description or ""), extra=f"color:{_MUTED};white-space:normal;")
        + "</tr>"
        for c in view.corporate_actions
    ]
    return _table(["Date", "Symbol", "Type", "Shares", "Avg incl. comm.", "Description"], [False, False, False, True, True, False], rows)


def _warnings_html(view: ReportView) -> str:
    items = "".join(
        f"<li>{escape(w.when.date().isoformat())} <b>{escape(w.symbol)}</b> [{escape(w.kind)}]: {escape(w.message)}</li>" for w in view.warnings
    )
    return f"<ul style='font-size:13px;background:{_WARN_BG};padding:8px 8px 8px 28px;border-radius:6px;'>{items}</ul>"


def _ticker_pnl_html(view: ReportView) -> str:
    rows = []
    for r in view.ticker_pnl:
        held = r.shares > 0
        muted = "" if held else f"color:{_MUTED};"
        close = escape(fmt_trade_price(r.close)) if r.close is not None else ("n/a" if held else "")
        rows.append(
            "<tr>"
            + _td(f"<b>{escape(r.symbol)}</b>", extra=muted)
            + _td(escape(fmt_qty(r.shares)) if held else "", numeric=True)
            + _td(escape(fmt_price(r.wac_incl)) if held else "", numeric=True)
            + _td(close, numeric=True)
            + _td(escape(fmt_money(r.market_value)) if r.market_value is not None else "", numeric=True)
            + _pnl_td(r.unrealized)
            + (_pnl_td(r.realized) if r.realized else _td("", numeric=True))
            + (_pnl_td(r.dividends) if r.dividends else _td("", numeric=True))
            + _pnl_td(r.total, extra="font-weight:600;")
            + _td(escape(fmt_money(r.commissions)), numeric=True, extra=f"color:{_MUTED};")
            + _td(escape(r.currency), extra=f"color:{_MUTED};")
            + "</tr>"
        )
    bold = f"font-weight:700;border-top:2px solid {_LINE};"
    for t in view.ticker_pnl_totals:
        rows.append(
            "<tr>"
            + _td(f"Total {escape(t.currency)}", extra=bold)
            + _td("", extra=bold)
            + _td("", extra=bold)
            + _td("", extra=bold)
            + _td(escape(fmt_money(t.market_value)) if t.market_value is not None else "", numeric=True, extra=bold)
            + _pnl_td(t.unrealized, extra=bold)
            + _pnl_td(t.realized, extra=bold)
            + _pnl_td(t.dividends, extra=bold)
            + _pnl_td(t.total, extra=bold)
            + _td(escape(fmt_money(t.commissions)), numeric=True, extra=bold)
            + _td(escape(t.currency), extra=bold)
            + "</tr>"
        )
    headers = ["Ticker", "Shares", "Avg incl. comm.", "Close", "Market value", "Unrealized", "Realized", "Dividends (net)", "Total P&L",
               "Commissions", "Ccy"]
    numeric = [False, True, True, True, True, True, True, True, True, True, False]
    return _table(headers, numeric, rows)


def format_html(view: ReportView) -> str:
    """
    Render the report as a standalone HTML document (the copy saved next to the XLSX).

    Args:
        view: Report content from `build_report_view`.

    Returns:
        Complete HTML document wrapping `format_html_body`.
    """
    body = format_html_body(view)
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{escape(view.title)}</title></head><body>{body}</body></html>"


def format_html_body(view: ReportView) -> str:
    """
    Render the email HTML body (a single <div>; the email channel adds <html>/<body>).

    Args:
        view: Report content from `build_report_view`.

    Returns:
        HTML fragment with inline styles only.
    """
    parts = [
        f"<div style='font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:{_INK};max-width:1100px;'>",
        f"<h1 style='font-size:20px;margin:0 0 4px 0;'>{escape(view.title)}</h1>",
        f"<div style='color:{_MUTED};font-size:13px;'>{escape(view.period_start.isoformat())} – {escape(view.period_end.isoformat())}"
        f" · accounts {escape(', '.join(view.accounts))} combined</div>",
        _tiles_html(view),
        _reconciliation_html(view),
        f"<h2 style='{_H2}'>Executions this month</h2>",
        _executions_html(view),
        f"<h2 style='{_H2}'>Positions</h2>",
        f"<p style='color:{_MUTED};font-size:12px;margin:0 0 6px 0;'>Averages are weighted over all buys of the current position cycle and do not"
        " change on a sell. IBKR basis is IBKR's lot-based (FIFO) figure, shown for comparison.</p>",
        _positions_html(view),
    ]
    if view.closed_cycles:
        parts += [f"<h2 style='{_H2}'>Positions closed this month</h2>", _cycles_html(view)]
    if view.corporate_actions:
        parts += [f"<h2 style='{_H2}'>Corporate actions</h2>", _corporate_actions_html(view)]
    if view.ytd:
        parts += [
            f"<h2 style='{_H2}'>Year to date {view.period_end.year}</h2>",
            f"<p style='color:{_MUTED};font-size:12px;margin:0 0 6px 0;'>Every ticker traded or paying dividends since 1 January, including"
            " positions already closed. Realized is after commissions.</p>",
            _ytd_html(view),
        ]
    if view.warnings:
        parts += [f"<h2 style='{_H2}'>Warnings</h2>", _warnings_html(view)]
    if view.ticker_pnl:
        since = f" since {view.history_start.isoformat()}" if view.history_start else ""
        snapshot = view.reconciliation.snapshot_date
        close = f"IBKR closing price of {snapshot.isoformat()}" if snapshot else "no IBKR closing price this month"
        parts += [
            f"<h2 style='{_H2}'>P&amp;L per ticker{escape(since)}</h2>",
            f"<p style='color:{_MUTED};font-size:12px;margin:0 0 6px 0;'>Total = realized (all cycles, after commissions) + net dividends"
            f" + unrealized. Unrealized = shares \u00d7 (close \u2212 average incl. commission), {escape(close)}; selling costs not deducted.</p>",
            _ticker_pnl_html(view),
        ]
    parts.append(
        f"<p style='color:{_MUTED};font-size:12px;margin-top:24px;'>Full history in the attached XLSX. "
        f"Generated {escape(view.generated_at.strftime('%Y-%m-%d %H:%M UTC'))}.</p></div>"
    )
    return "".join(parts)


# --------------------------------------------------------------------------
# Plain text (fallback body and Telegram-safe summary)
# --------------------------------------------------------------------------


def format_plain_text(view: ReportView) -> str:
    """
    Render a short plain-text version of the report.

    Args:
        view: Report content from `build_report_view`.

    Returns:
        Plain-text body (summary, executions, positions held).
    """
    lines = [view.title, f"{view.period_start} – {view.period_end}", ""]
    for t in view.totals:
        lines.append(
            f"{t.currency}: {t.executions_month} executions, commissions {fmt_money(t.commissions_month)} (YTD {fmt_money(t.commissions_ytd)}), "
            f"realized {fmt_signed(t.realized_month)} (YTD {fmt_signed(t.realized_ytd)})"
        )
    for d in view.deposits:
        since = f" since {d.first_date}" if d.first_date else ""
        lines.append(f"Net deposits {d.currency}: month {fmt_money(d.month)}, YTD {fmt_money(d.ytd)}, total {fmt_money(d.lifetime)}{since}")
    rec = view.reconciliation
    if rec.status == STATUS_OK:
        lines.append(f"Reconciliation: OK vs IBKR {rec.snapshot_date}")
    elif rec.status == STATUS_MISMATCH:
        lines.append(f"Reconciliation: {len(rec.mismatches)} MISMATCH(ES) vs IBKR {rec.snapshot_date}")
    else:
        lines.append("Reconciliation: no IBKR snapshot this month")
    lines += ["", "Executions:"]
    for row in view.executions:
        ex = row.execution
        realized = f"  P&L {fmt_signed(row.realized_incl)}" if row.realized_incl is not None else ""
        lines.append(
            f"  {ex.trade_date} {ex.buy_sell:<4} {ex.symbol:<6} {fmt_qty(abs(ex.quantity)):>8} @ {fmt_trade_price(ex.trade_price)}"
            f"  comm {fmt_money(abs(ex.ib_commission) + abs(ex.taxes))}{realized}"
        )
    if not view.executions:
        lines.append("  none")
    lines += ["", "Positions held (avg excl / incl commission):"]
    for p in view.positions:
        if p.shares:
            lines.append(f"  {p.symbol:<6} {fmt_qty(p.shares):>8}  {fmt_price(p.wac_excl)} / {fmt_price(p.wac_incl)} {p.currency}")
    if view.ytd:
        lines += ["", f"Year to date {view.period_end.year} (realized + dividends = total):"]
        for y in view.ytd:
            closed = " (closed)" if y.closed else ""
            lines.append(f"  {y.symbol:<8} {fmt_signed(y.realized):>12} {fmt_signed(y.dividends):>10} = {fmt_signed(y.total):>12} {y.currency}{closed}")
        for yt in view.ytd_totals:
            lines.append(f"  {'TOTAL':<8} {fmt_signed(yt.realized):>12} {fmt_signed(yt.dividends):>10} = {fmt_signed(yt.total):>12} {yt.currency}")
    if view.ticker_pnl:
        lines += ["", "P&L per ticker (realized + dividends + unrealized = total):"]
        for r in view.ticker_pnl:
            unreal = fmt_signed(r.unrealized) if r.unrealized is not None else ("n/a" if r.shares > 0 else "-")
            lines.append(f"  {r.symbol:<8} {fmt_signed(r.realized):>12} {fmt_signed(r.dividends):>10} {unreal:>12} = {fmt_signed(r.total):>12} {r.currency}")
        for pt in view.ticker_pnl_totals:
            unreal_total = fmt_signed(pt.unrealized) if pt.unrealized is not None else "-"
            lines.append(f"  {'TOTAL':<8} {fmt_signed(pt.realized):>12} {fmt_signed(pt.dividends):>10} {unreal_total:>12} = {fmt_signed(pt.total):>12} {pt.currency}")
    if view.warnings or rec.status == STATUS_MISMATCH:
        lines += ["", f"Warnings: {len(view.warnings)} (see HTML / XLSX)"]
    return "\n".join(lines)

