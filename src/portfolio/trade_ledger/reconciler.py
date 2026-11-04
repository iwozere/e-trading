"""
Reconciliation of the ledger against IBKR's own open positions.

Pure — no I/O. IBKR's Open Positions section (SUMMARY level) is a snapshot at
the end of each downloaded period, dated the last *business* day (e.g.
2022-12-30, 2026-10-30), not always the calendar month end. The ledger is
therefore rebuilt as of the snapshot date itself before comparing, so trades
and corporate actions can never fall between the two.

Share counts must match exactly (all accounts combined). IBKR's cost basis is
carried along for display only: it is lot-based (FIFO, per account) and is
expected to differ from our weighted average after partial sells. IBKR's mark
(closing) price on the snapshot date values open positions in the report.
"""

from collections import defaultdict
from datetime import date
from decimal import Decimal
from typing import Dict, List, Optional

from src.notification.logger import setup_logger
from src.portfolio.trade_ledger.ledger_engine import build_ledger
from src.portfolio.trade_ledger.models import (
    IbkrOpenPosition,
    LedgerEvents,
    LedgerResult,
    ReconciliationLine,
    ReconciliationReport,
)

_logger = setup_logger(__name__)

STATUS_OK = "ok"
STATUS_MISMATCH = "mismatch"
STATUS_NO_SNAPSHOT = "no_snapshot"


def _snapshot_dates(positions: List[IbkrOpenPosition]) -> List[date]:
    """Distinct snapshot dates, oldest first."""
    return sorted({p.report_date for p in positions})


def find_snapshot_date(positions: List[IbkrOpenPosition], period_start: date, period_end: date) -> Optional[date]:
    """
    Latest IBKR snapshot date within [period_start, period_end].

    Args:
        positions: All stored IBKR open positions.
        period_start: First day of the period.
        period_end: Last day of the period.

    Returns:
        The snapshot date, or None if IBKR has no snapshot in the period.
    """
    in_period = [d for d in _snapshot_dates(positions) if period_start <= d <= period_end]
    return in_period[-1] if in_period else None


def compare_snapshot(ledger: LedgerResult, positions: List[IbkrOpenPosition], snapshot_date: date) -> ReconciliationReport:
    """
    Compare a ledger built up to `snapshot_date` with IBKR's snapshot of that date.

    Args:
        ledger: Ledger whose `period_end` equals `snapshot_date`.
        positions: All stored IBKR open positions (other dates are ignored).
        snapshot_date: IBKR snapshot date to compare against.

    Returns:
        ReconciliationReport with one line per position held by either side.

    Raises:
        ValueError: If the ledger was not built up to `snapshot_date`.
    """
    if ledger.period_end != snapshot_date:
        raise ValueError(f"ledger period_end {ledger.period_end} != snapshot date {snapshot_date}")

    ibkr_shares: Dict[str, Decimal] = defaultdict(Decimal)
    ibkr_money: Dict[str, Decimal] = defaultdict(Decimal)
    ibkr_symbol: Dict[str, str] = {}
    ibkr_mark: Dict[str, Decimal] = {}
    money_complete: Dict[str, bool] = defaultdict(lambda: True)
    for p in positions:
        if p.report_date != snapshot_date:
            continue
        key = ledger.conid_to_key.get(p.conid, p.conid)
        ibkr_shares[key] += p.position
        ibkr_symbol[key] = p.symbol
        if p.mark_price is not None:
            ibkr_mark[key] = p.mark_price
        if p.cost_basis_money is None:
            money_complete[key] = False
        else:
            ibkr_money[key] += p.cost_basis_money

    held = {key: pos for key, pos in ledger.positions.items() if pos.shares != 0}
    lines: List[ReconciliationLine] = []
    for key in sorted(set(held) | set(ibkr_shares)):
        pos = held.get(key)
        shares = ibkr_shares.get(key, Decimal(0))
        basis = ibkr_money[key] / shares if shares and money_complete[key] else None
        lines.append(
            ReconciliationLine(
                position_key=key,
                symbol=pos.symbol if pos else ibkr_symbol.get(key, key),
                ledger_shares=pos.shares if pos else Decimal(0),
                ibkr_shares=shares,
                wac_incl=pos.wac_incl if pos else None,
                ibkr_cost_basis_price=basis,
                ibkr_mark_price=ibkr_mark.get(key),
            )
        )
    lines.sort(key=lambda line: line.symbol)

    report = ReconciliationReport(as_of=snapshot_date, snapshot_date=snapshot_date, status=STATUS_OK, lines=lines)
    if report.mismatches:
        report.status = STATUS_MISMATCH
        for line in report.mismatches:
            _logger.warning(
                "Reconciliation mismatch %s on %s: ledger %s vs IBKR %s shares",
                line.symbol,
                snapshot_date,
                line.ledger_shares,
                line.ibkr_shares,
            )
    return report


def reconcile_period(events: LedgerEvents, period_start: date, period_end: date) -> ReconciliationReport:
    """
    Reconcile the ledger against IBKR's latest snapshot within a period.

    Args:
        events: Every stored event.
        period_start: First day of the report period.
        period_end: Last day of the report period.

    Returns:
        ReconciliationReport; status "no_snapshot" when IBKR has none in the period.
    """
    snapshot = find_snapshot_date(events.open_positions, period_start, period_end)
    if snapshot is None:
        _logger.info("No IBKR open-positions snapshot between %s and %s; reconciliation skipped", period_start, period_end)
        return ReconciliationReport(as_of=period_end, snapshot_date=None, status=STATUS_NO_SNAPSHOT)
    ledger = build_ledger(events, snapshot.replace(day=1), snapshot)
    report = compare_snapshot(ledger, events.open_positions, snapshot)
    report.as_of = period_end
    return report


def reconcile_all(events: LedgerEvents) -> List[ReconciliationReport]:
    """
    Reconcile the ledger against every IBKR snapshot in the data (full-history regression check).

    Args:
        events: Every stored event.

    Returns:
        One report per snapshot date, oldest first.
    """
    reports: List[ReconciliationReport] = []
    for snapshot in _snapshot_dates(events.open_positions):
        ledger = build_ledger(events, snapshot.replace(day=1), snapshot)
        reports.append(compare_snapshot(ledger, events.open_positions, snapshot))
    return reports

