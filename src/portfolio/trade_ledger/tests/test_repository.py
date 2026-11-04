"""Unit tests for the trade-ledger storage adapter (in-memory fake, no DB)."""

from src.portfolio.trade_ledger.repository import load_events, store_parse_result
from src.portfolio.trade_ledger.tests.fake_storage import FakeStorage
from src.portfolio.trade_ledger.tests.flex_fixture import FLEX_XML as _XML
from src.portfolio.trade_ledger.xml_parser import parse_flex_bytes


def test_store_then_load_round_trips_dataclasses() -> None:
    parsed = parse_flex_bytes(_XML, "fixture.xml")
    storage = FakeStorage()

    store_parse_result(parsed, storage)
    events = load_events(storage)

    assert events.executions == parsed.statements[0].executions
    assert events.corporate_actions == parsed.statements[0].corporate_actions
    assert events.transfers == parsed.statements[0].transfers + parsed.statements[1].transfers
    assert events.cash_transactions == parsed.statements[0].cash_transactions
    assert events.open_positions == parsed.statements[0].open_positions + parsed.statements[1].open_positions


def test_storing_same_file_twice_inserts_nothing_new() -> None:
    parsed = parse_flex_bytes(_XML, "fixture.xml")
    storage = FakeStorage()

    first = store_parse_result(parsed, storage)
    second = store_parse_result(parsed, storage)

    assert first == {"executions": 2, "corporate_actions": 2, "transfers": 2, "cash_transactions": 2, "open_positions": 2}
    assert second == {"executions": 0, "corporate_actions": 0, "transfers": 0, "cash_transactions": 0, "open_positions": 0}
