"""Unit tests for the Flex Query XML parser (synthetic data, no real account ids)."""

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from src.portfolio.trade_ledger.models import CASH_COMMISSION_ADJUSTMENT, CASH_DIVIDEND
from src.portfolio.trade_ledger.tests.flex_fixture import FLEX_XML as _XML
from src.portfolio.trade_ledger.xml_parser import (
    FlexParseError,
    mask_account,
    parse_flex_bytes,
    parse_flex_datetime,
    parse_flex_file,
)


def test_parses_one_statement_per_account() -> None:
    result = parse_flex_bytes(_XML, "fixture.xml")

    assert [s.account_id for s in result.statements] == ["U0000001", "U0000002"]
    assert result.statements[0].from_date == date(2025, 1, 1)
    assert result.statements[0].to_date == date(2025, 12, 31)


def test_execution_fields_are_decimal_and_parsed() -> None:
    ex = parse_flex_bytes(_XML, "fixture.xml").statements[0].executions[0]

    assert ex.trade_id == "9001"
    assert ex.symbol == "ABC"
    assert ex.conid == "111"
    assert ex.buy_sell == "BUY"
    assert ex.quantity == Decimal("15")
    assert ex.trade_price == Decimal("22")
    assert ex.ib_commission == Decimal("-1.000525")
    assert ex.net_cash == Decimal("-331.000525")
    assert ex.trade_date == date(2025, 5, 21)
    assert ex.trade_datetime == datetime(2025, 5, 21, 15, 7, 56)
    assert ex.notes == "P"
    assert ex.source_file == "fixture.xml"


def test_empty_optional_numbers() -> None:
    sell = parse_flex_bytes(_XML, "fixture.xml").statements[0].executions[1]

    assert sell.taxes == Decimal(0)
    assert sell.cost is None
    assert sell.notes is None
    assert sell.ib_order_id is None


def test_skips_cash_and_malformed_trades() -> None:
    result = parse_flex_bytes(_XML, "fixture.xml")

    assert [e.trade_id for e in result.statements[0].executions] == ["9001", "9002"]
    assert result.skipped["trade_non_stk"] == 1
    assert result.skipped["trade_malformed"] == 1


def test_corporate_action_legs_share_action_id() -> None:
    legs = parse_flex_bytes(_XML, "fixture.xml").statements[0].corporate_actions

    assert [(leg.conid, leg.quantity) for leg in legs] == [("111", Decimal("-10")), ("112", Decimal("1"))]
    assert {leg.action_id for leg in legs} == {"777"}
    assert legs[0].action_type == "RS"
    assert legs[0].report_date == date(2025, 7, 3)
    assert legs[0].cost_basis is None


def test_transfer_counterparty_and_date_fallback() -> None:
    result = parse_flex_bytes(_XML, "fixture.xml")
    out_leg = result.statements[0].transfers[0]
    in_leg = result.statements[1].transfers[0]

    assert out_leg.direction == "OUT"
    assert out_leg.counterparty_account == "U0000002"
    assert out_leg.quantity == Decimal("-24")
    assert in_leg.transfer_datetime == datetime(2025, 12, 29)  # empty dateTime -> falls back to date


def test_open_positions_summary_only_with_report_date_fallback() -> None:
    result = parse_flex_bytes(_XML, "fixture.xml")
    first = result.statements[0].open_positions
    second = result.statements[1].open_positions

    assert len(first) == 1
    assert first[0].cost_basis_price == Decimal("221.0")
    assert result.skipped["open_position_not_summary"] == 1
    assert second[0].report_date == date(2025, 12, 31)  # missing reportDate -> statement toDate


def test_rejects_non_flex_document() -> None:
    with pytest.raises(FlexParseError, match="FlexQueryResponse"):
        parse_flex_bytes(b"<FlexStatementResponse><Status>Fail</Status></FlexStatementResponse>", "x.xml")


def test_rejects_invalid_xml() -> None:
    with pytest.raises(FlexParseError, match="not valid XML"):
        parse_flex_bytes(b"not xml", "x.xml")


def test_parse_flex_file_uses_file_name(tmp_path: Path) -> None:
    path = tmp_path / "My_Trades_2025.xml"
    path.write_bytes(_XML)

    result = parse_flex_file(path)

    assert result.source_file == "My_Trades_2025.xml"
    assert result.statements[0].executions[0].source_file == "My_Trades_2025.xml"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("20250521;150756", datetime(2025, 5, 21, 15, 7, 56)), ("20250521", datetime(2025, 5, 21))],
)
def test_parse_flex_datetime(raw: str, expected: datetime) -> None:
    assert parse_flex_datetime(raw) == expected


def test_parse_flex_datetime_rejects_other_formats() -> None:
    with pytest.raises(FlexParseError):
        parse_flex_datetime("2025-05-21 15:07:56")


@pytest.mark.parametrize(("raw", "expected"), [("U1234567", "U1***67"), ("U1", "***")])
def test_mask_account(raw: str, expected: str) -> None:
    assert mask_account(raw) == expected


def test_only_ledger_cash_types_are_read_from_cash_transactions() -> None:
    rows = parse_flex_bytes(_XML, "fixture.xml").statements[0].cash_transactions

    assert [r.cash_type for r in rows] == [CASH_COMMISSION_ADJUSTMENT, CASH_DIVIDEND]  # broker interest is ignored
    adj, div = rows
    assert adj.trade_id == "9001"
    assert adj.amount == Decimal("1")
    assert adj.cash_datetime == datetime(2025, 6, 6)
    assert div.trade_id is None
    assert div.amount == Decimal("2.5")
    assert div.conid == "111"


def test_open_position_reads_mark_price() -> None:
    (pos,) = parse_flex_bytes(_XML, "fixture.xml").statements[0].open_positions

    assert pos.mark_price == Decimal("250.5")
