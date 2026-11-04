"""Unit tests for reconciliation against IBKR open-position snapshots (synthetic data)."""

from datetime import date
from decimal import Decimal
from typing import Optional

import pytest

from src.portfolio.trade_ledger.ledger_engine import build_ledger
from src.portfolio.trade_ledger.models import IbkrOpenPosition, LedgerEvents
from src.portfolio.trade_ledger.reconciler import (
    STATUS_MISMATCH,
    STATUS_NO_SNAPSHOT,
    STATUS_OK,
    compare_snapshot,
    find_snapshot_date,
    reconcile_all,
    reconcile_period,
)
from src.portfolio.trade_ledger.tests.event_factories import ACC, ACC2, leg, trade

D = Decimal


def _snap(
    day: date, conid: str, position: str, money: Optional[str], symbol: str = "ABC", account: str = ACC, mark: Optional[str] = None
) -> IbkrOpenPosition:
    return IbkrOpenPosition(
        account_id=account, report_date=day, conid=conid, isin=None, symbol=symbol, position=D(position),
        cost_basis_price=None, cost_basis_money=D(money) if money is not None else None, mark_price=D(mark) if mark is not None else None,
        currency="USD", source_file="t.xml",
    )


def _events(*snapshots: IbkrOpenPosition) -> LedgerEvents:
    return LedgerEvents(
        executions=[
            trade("2025-01-02 10:00", "10", "100", commission="-1"),
            trade("2025-01-03 10:00", "5", "100", commission="0", account=ACC2),
            trade("2025-12-31 15:00", "1", "100", commission="0"),  # after the 30 Dec snapshot
        ],
        open_positions=list(snapshots),
    )


def test_matching_snapshot_combines_accounts_and_reports_ibkr_basis() -> None:
    events = _events(_snap(date(2025, 12, 30), "100", "10", "1001"), _snap(date(2025, 12, 30), "100", "5", "500", account=ACC2))

    report = reconcile_period(events, date(2025, 12, 1), date(2025, 12, 31))

    assert report.status == STATUS_OK
    assert report.snapshot_date == date(2025, 12, 30)  # last business day, not the 31st
    assert report.as_of == date(2025, 12, 31)
    line = report.lines[0]
    assert (line.ledger_shares, line.ibkr_shares) == (D(15), D(15))  # 31 Dec trade not counted
    assert line.ibkr_cost_basis_price == D(1501) / D(15)
    assert report.ibkr_cost_basis_by_key() == {"100": D(1501) / D(15)}


def test_share_difference_is_a_mismatch() -> None:
    events = _events(_snap(date(2025, 12, 30), "100", "14", "1400"))

    report = reconcile_period(events, date(2025, 12, 1), date(2025, 12, 31))

    assert report.status == STATUS_MISMATCH
    assert [(m.ledger_shares, m.ibkr_shares) for m in report.mismatches] == [(D(15), D(14))]


def test_position_only_at_ibkr_is_a_mismatch() -> None:
    events = _events(_snap(date(2025, 12, 30), "100", "15", "1501"), _snap(date(2025, 12, 30), "999", "3", "30", symbol="GHOST"))

    report = reconcile_period(events, date(2025, 12, 1), date(2025, 12, 31))

    assert [(m.symbol, m.ledger_shares, m.ibkr_shares) for m in report.mismatches] == [("GHOST", D(0), D(3))]


def test_no_snapshot_in_period() -> None:
    report = reconcile_period(_events(_snap(date(2025, 11, 28), "100", "15", "1501")), date(2025, 12, 1), date(2025, 12, 31))

    assert report.status == STATUS_NO_SNAPSHOT
    assert report.snapshot_date is None
    assert report.lines == []


def test_missing_ibkr_money_gives_no_basis() -> None:
    events = _events(_snap(date(2025, 12, 30), "100", "10", None), _snap(date(2025, 12, 30), "100", "5", "500", account=ACC2))

    line = reconcile_period(events, date(2025, 12, 1), date(2025, 12, 31)).lines[0]

    assert line.shares_match
    assert line.ibkr_cost_basis_price is None


def test_snapshot_on_new_conid_after_split_maps_to_same_position() -> None:
    events = LedgerEvents(
        executions=[trade("2025-01-02 10:00", "1000", "1", commission="0")],
        corporate_actions=[leg("2025-07-02 20:25", "100", "-1000"), leg("2025-07-02 20:25", "200", "10")],
        open_positions=[_snap(date(2025, 12, 31), "200", "10", "1000")],
    )

    report = reconcile_period(events, date(2025, 12, 1), date(2025, 12, 31))

    assert report.status == STATUS_OK
    assert report.lines[0].wac_incl == report.lines[0].ibkr_cost_basis_price == D(100)


def test_reconcile_all_checks_every_snapshot() -> None:
    events = _events(_snap(date(2025, 1, 2), "100", "10", "1001"), _snap(date(2025, 12, 30), "100", "15", "1501"))

    reports = reconcile_all(events)

    assert [(r.snapshot_date, r.status) for r in reports] == [(date(2025, 1, 2), STATUS_OK), (date(2025, 12, 30), STATUS_OK)]


def test_find_snapshot_date_picks_latest_in_period() -> None:
    snaps = [_snap(date(2025, 10, 31), "1", "1", "1"), _snap(date(2025, 11, 14), "1", "1", "1"), _snap(date(2025, 11, 28), "1", "1", "1")]

    assert find_snapshot_date(snaps, date(2025, 11, 1), date(2025, 11, 30)) == date(2025, 11, 28)
    assert find_snapshot_date(snaps, date(2025, 12, 1), date(2025, 12, 31)) is None


def test_compare_snapshot_rejects_ledger_for_other_date() -> None:
    ledger = build_ledger(LedgerEvents(), date(2025, 12, 1), date(2025, 12, 31))

    with pytest.raises(ValueError):
        compare_snapshot(ledger, [], date(2025, 12, 30))


def test_mark_price_is_reported_per_position() -> None:
    events = _events(_snap(date(2025, 12, 30), "100", "10", "1001", mark="120.5"), _snap(date(2025, 12, 30), "100", "5", "500", account=ACC2, mark="120.5"))

    report = reconcile_period(events, date(2025, 12, 1), date(2025, 12, 31))

    assert report.mark_price_by_key() == {"100": D("120.5")}
    assert report.lines[0].ibkr_mark_price == D("120.5")
