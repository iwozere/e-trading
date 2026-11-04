"""Unit tests for the ledger engine (synthetic events, pure — no I/O)."""

from datetime import date
from decimal import Decimal
from typing import List, Optional

import pytest

from src.portfolio.trade_ledger.ledger_engine import build_ledger
from src.portfolio.trade_ledger.models import (
    CASH_PAYMENT_IN_LIEU,
    CASH_WITHHOLDING_TAX,
    CashTransaction,
    CorporateActionLeg,
    Execution,
    LedgerEvents,
    LedgerResult,
    PositionState,
    TransferEvent,
)
from src.portfolio.trade_ledger.tests.event_factories import ACC, ACC2, cash, deposit, leg, refund, trade, transfer

D = Decimal


def _run(
    executions: Optional[List[Execution]] = None,
    legs: Optional[List[CorporateActionLeg]] = None,
    transfers: Optional[List[TransferEvent]] = None,
    adjustments: Optional[List[CashTransaction]] = None,
    start: date = date(2025, 1, 1),
    end: date = date(2025, 12, 31),
) -> LedgerResult:
    events = LedgerEvents(
        executions=executions or [], corporate_actions=legs or [], transfers=transfers or [], cash_transactions=adjustments or []
    )
    return build_ledger(events, start, end)


def _only(result: LedgerResult) -> PositionState:
    assert len(result.positions) == 1
    return next(iter(result.positions.values()))


def test_buys_average_excl_and_incl_commission() -> None:
    pos = _only(_run([trade("2025-01-02 10:00", "10", "100"), trade("2025-01-03 10:00", "30", "120")]))

    assert pos.shares == D(40)
    assert pos.wac_excl == D(4600) / D(40)  # 115
    assert pos.wac_incl == D(4602) / D(40)  # 115.05


def test_sell_keeps_average_and_realizes_against_it() -> None:
    result = _run([
        trade("2025-01-02 10:00", "10", "100"),
        trade("2025-01-03 10:00", "30", "120"),
        trade("2025-02-01 10:00", "-20", "130"),
    ])
    pos = _only(result)
    sell_row = result.execution_rows[-1]

    assert pos.shares == D(20)
    assert pos.wac_incl == D("115.05")
    assert pos.wac_excl == D(115)
    assert sell_row.realized_excl == D(2600) - D(20) * D(115)
    assert sell_row.realized_incl == D(2600) - D(1) - D(20) * D("115.05")


def test_full_close_records_cycle_and_rebuy_starts_fresh() -> None:
    result = _run([
        trade("2025-01-02 10:00", "10", "100"),
        trade("2025-02-01 10:00", "-10", "110"),
        trade("2025-03-01 10:00", "5", "50"),
    ])
    pos = _only(result)

    assert len(result.closed_cycles) == 1
    cycle = result.closed_cycles[0]
    assert cycle.cycle_no == 1
    assert cycle.realized_incl == D(1100) - D(1) - D(1001)
    assert cycle.commissions == D(2)
    assert cycle.close_date == date(2025, 2, 1)
    assert pos.cycle_no == 2
    assert pos.wac_incl == D(251) / D(5)  # not polluted by the first cycle
    assert pos.cycle_open_date == date(2025, 3, 1)


def test_reverse_split_moves_position_to_new_conid_and_rescales_average() -> None:
    result = _run(
        [trade("2025-01-02 10:00", "1000", "1", commission="0"), trade("2025-09-01 10:00", "-5", "300", commission="0", conid="200")],
        legs=[leg("2025-07-02 20:25", "100", "-1000", symbol="ABC.OLD"), leg("2025-07-02 20:25", "200", "10")],
    )
    pos = _only(result)
    ca = result.corporate_action_rows[0]

    assert (ca.shares_before, ca.shares_after) == (D(1000), D(10))
    assert ca.wac_incl_after == D(100)
    assert pos.shares == D(5)  # later sell on the NEW conid hit the same position
    assert pos.symbol == "ABC"
    assert result.execution_rows[-1].realized_incl == D(1500) - D(500)
    assert result.conid_to_key == {"100": "100", "200": "100"}
    assert result.warnings == []


def test_chained_reverse_splits_keep_total_cost() -> None:
    result = _run(
        [trade("2025-01-02 10:00", "10", "1414", commission="-1")],
        legs=[
            leg("2025-08-01 20:25", "100", "-10", action_id="A1", symbol="X.OLD"),
            leg("2025-08-01 20:25", "200", "0.04", action_id="A1", symbol="X"),
            leg("2025-09-19 20:25", "200", "-0.04", action_id="A2", symbol="X.OLD"),
            leg("2025-09-19 20:25", "300", "0.0002", action_id="A2", symbol="X"),
        ],
    )
    pos = _only(result)

    assert pos.shares == D("0.0002")
    assert pos.cost_incl == D(14141)
    assert pos.wac_incl == D(14141) / D("0.0002")


def test_trade_at_same_timestamp_applies_before_corporate_action() -> None:
    result = _run(
        [trade("2025-07-02 20:25", "10", "1", commission="0")],
        legs=[leg("2025-07-02 20:25", "100", "-10"), leg("2025-07-02 20:25", "200", "1")],
    )

    assert _only(result).shares == D(1)


def test_internal_transfer_between_own_accounts_is_noop() -> None:
    result = _run(
        [trade("2025-01-02 10:00", "24", "100", commission="0"), trade("2025-01-05 10:00", "1", "100", commission="0", account=ACC2)],
        transfers=[transfer("2025-06-01 06:00", "-24", ACC, ACC2), transfer("2025-06-01 06:00", "24", ACC2, ACC)],
    )
    pos = _only(result)

    assert pos.shares == D(25)
    assert pos.wac_incl == D(100)
    assert result.warnings == []


def test_external_transfer_in_adds_shares_at_position_amount_and_warns() -> None:
    result = _run(transfers=[transfer("2025-03-01 10:00", "10", ACC, "OTHERBROKER", kind="ACATS", amount="1500")])
    pos = _only(result)

    assert pos.shares == D(10)
    assert pos.wac_incl == D(150)
    assert [w.kind for w in result.warnings] == ["external_transfer"]


def test_selling_more_than_held_warns_short() -> None:
    result = _run([trade("2025-01-02 10:00", "5", "10", commission="0"), trade("2025-01-03 10:00", "-8", "12", commission="0")])
    pos = _only(result)

    assert pos.shares == D(-3)
    assert pos.is_short
    assert pos.wac_incl is None
    assert result.execution_rows[-1].realized_incl == D(60) - D(50)  # only the 5 held shares realize
    assert [w.kind for w in result.warnings] == ["short"]


def test_cash_merger_without_new_shares_is_a_sale() -> None:
    result = _run(
        [trade("2025-01-02 10:00", "10", "20", commission="0")],
        legs=[leg("2025-05-01 20:25", "100", "-10", action_type="TO", proceeds="250")],
    )
    pos = _only(result)

    assert pos.shares == D(0)
    assert result.closed_cycles[0].realized_incl == D(50)
    assert result.corporate_action_rows[0].realized_incl == D(50)
    assert [w.kind for w in result.warnings] == ["corporate_action_type"]


def test_cash_in_lieu_on_split_is_realized_and_warned() -> None:
    result = _run(
        [trade("2025-01-02 10:00", "105", "1", commission="0")],
        legs=[leg("2025-05-01 20:25", "100", "-105"), leg("2025-05-01 20:25", "200", "10", proceeds="2.5")],
    )

    assert _only(result).cost_incl == D(105)
    assert result.summaries[0].ytd.realized_incl == D("2.5")
    assert [w.kind for w in result.warnings] == ["cash_in_lieu"]


def test_shares_received_without_removal_get_zero_cost_and_warn() -> None:
    result = _run(legs=[leg("2025-05-01 20:25", "100", "5", action_type="SO", symbol="SPIN")])
    pos = _only(result)

    assert pos.shares == D(5)
    assert pos.wac_incl == D(0)
    assert {w.kind for w in result.warnings} == {"corporate_action_type", "zero_cost_shares"}


def test_commission_refund_on_held_buy_lowers_average_incl() -> None:
    buy = trade("2025-01-05 10:00", "10", "20", commission="-1")
    result = _run([buy], adjustments=[refund("2025-02-06 00:00", buy, "1")])
    pos = _only(result)

    assert pos.wac_incl == D(20)
    assert pos.wac_excl == D(20)
    assert result.summaries[0].ytd.commissions == D(0)
    assert result.summaries[0].ytd.realized_incl == D(0)


def test_commission_refund_after_position_closed_is_realized() -> None:
    buy = trade("2025-01-05 10:00", "10", "20", commission="-1")
    sell = trade("2025-01-12 10:00", "-10", "30", commission="-1")
    result = _run([buy, sell], adjustments=[refund("2025-01-12 23:00", sell, "0.5")])

    assert result.closed_cycles[0].realized_incl == D(300) - D(1) - D(201)  # cycle record is not reopened
    assert result.summaries[0].ytd.realized_incl == D(98) + D("0.5")


def test_events_after_period_end_are_ignored() -> None:
    result = _run([trade("2025-01-02 10:00", "10", "1"), trade("2026-01-02 10:00", "10", "1")], end=date(2025, 12, 31))

    assert _only(result).shares == D(10)
    assert len(result.execution_rows) == 1


def test_month_and_ytd_totals() -> None:
    result = _run(
        [
            trade("2024-12-20 10:00", "10", "10"),
            trade("2025-02-10 10:00", "10", "12"),
            trade("2025-03-05 10:00", "-5", "15"),
            trade("2025-03-06 10:00", "2", "14"),
        ],
        start=date(2025, 3, 1),
        end=date(2025, 3, 31),
    )
    line = result.summaries[0]

    assert line.month.executions == 2
    assert line.month.shares_bought == D(2)
    assert line.month.shares_sold == D(5)
    assert line.month.commissions == D(2)
    assert line.ytd.executions == 3  # the 2024 buy is outside YTD
    assert line.ytd.shares_bought == D(12)
    assert line.month.realized_incl == D(75) - D(1) - D(5) * D(222) / D(20)


def test_summaries_list_held_or_active_positions_only() -> None:
    result = _run(
        [
            trade("2025-01-02 10:00", "1", "1", conid="1", symbol="OLDCLOSED"),
            trade("2025-01-03 10:00", "-1", "1", conid="1", symbol="OLDCLOSED"),
            trade("2025-03-02 10:00", "1", "1", conid="2", symbol="CLOSEDINMONTH"),
            trade("2025-03-03 10:00", "-1", "2", conid="2", symbol="CLOSEDINMONTH"),
            trade("2025-01-04 10:00", "1", "1", conid="3", symbol="HELD"),
        ],
        start=date(2025, 3, 1),
        end=date(2025, 3, 31),
    )

    assert [line.symbol for line in result.summaries] == ["CLOSEDINMONTH", "HELD"]


def test_rejects_inverted_period() -> None:
    with pytest.raises(ValueError):
        build_ledger(LedgerEvents(), date(2025, 2, 1), date(2025, 1, 31))


def test_dividends_and_withholding_are_net_and_do_not_change_average() -> None:
    buy = trade("2025-01-05 10:00", "10", "20", commission="-1")
    result = _run(
        [buy],
        adjustments=[
            cash("2025-03-10 20:20", "5"),
            cash("2025-03-10 20:20", "-0.75", cash_type=CASH_WITHHOLDING_TAX),
            cash("2024-12-10 20:20", "1", cash_type=CASH_PAYMENT_IN_LIEU),
        ],
        start=date(2025, 3, 1),
        end=date(2025, 3, 31),
    )

    pos = _only(result)
    assert pos.wac_incl == D("20.1")
    assert result.month_totals["100"].dividends == D("4.25")
    assert result.ytd_totals["100"].dividends == D("4.25")
    assert result.lifetime_totals["100"].dividends == D("5.25")
    assert result.lifetime_totals["100"].realized_incl == D(0)


def test_lifetime_totals_span_all_years_and_cycles() -> None:
    result = _run(
        [
            trade("2023-01-05 10:00", "10", "10", commission="-1"),
            trade("2023-06-05 10:00", "-10", "15", commission="-1"),
            trade("2025-02-05 10:00", "10", "20", commission="-1"),
            trade("2025-02-06 10:00", "-5", "22", commission="-1"),
        ]
    )

    life = result.lifetime_totals["100"]
    assert life.realized_incl == D(48) + (D(110) - D(1) - D("100.5"))  # cycle 1: 150-1-101; cycle 2: half of 201
    assert life.commissions == D(4)
    assert life.executions == 4
    assert result.ytd_totals["100"].executions == 2


def test_deposits_are_summed_per_currency_for_month_ytd_and_lifetime() -> None:
    result = _run(
        [trade("2025-03-05 10:00", "10", "20")],
        adjustments=[
            deposit("2024-11-02 19:00", "5000"),
            deposit("2025-01-07 19:00", "3000"),
            deposit("2025-03-04 19:00", "2200"),
            deposit("2025-03-20 19:00", "-500"),  # withdrawal
            deposit("2025-03-21 19:00", "1000", account=ACC2),
            deposit("2025-03-22 15:00", "0.01", currency="USD", description="ADJUSTMENT: COMMISSION"),
            deposit("2025-04-01 19:00", "999"),  # after the period
        ],
        start=date(2025, 3, 1),
        end=date(2025, 3, 31),
    )

    (chf,) = result.deposits.values()
    assert chf.currency == "CHF"
    assert chf.month == D(2700) and chf.count_month == 3
    assert chf.ytd == D(5700)
    assert chf.lifetime == D(10700)
    assert chf.first_date == date(2024, 11, 2)
    assert len(result.deposit_rows) == 5
    assert list(result.positions) == ["100"]  # deposits never create positions
    assert result.warnings == []


def test_internal_cash_transfer_between_own_accounts_nets_to_zero() -> None:
    result = _run(
        [trade("2025-03-05 10:00", "10", "20")],
        adjustments=[deposit("2025-03-10 12:00", "-1000"), deposit("2025-03-10 12:00", "1000", account=ACC2)],
        start=date(2025, 3, 1),
        end=date(2025, 3, 31),
    )

    assert result.deposits["CHF"].month == D(0)
