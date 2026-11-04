"""Tests for TradeLedgerRepo (needs the Postgres test DB, see conftest.py)."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict

from sqlalchemy.orm import Session

from src.data.db.repos.repo_trade_ledger import TradeLedgerRepo


def _execution(trade_id: str, quantity: str, hour: int) -> Dict[str, Any]:
    return {
        "account_id": "U0000001", "trade_id": trade_id, "transaction_id": None, "ib_order_id": None, "ib_exec_id": None,
        "conid": "111", "isin": None, "symbol": "ABC", "description": None, "listing_exchange": None, "sub_category": None,
        "currency": "USD", "fx_rate_to_base": Decimal("0.79"), "trade_date": date(2025, 5, 21),
        "trade_datetime": datetime(2025, 5, 21, hour), "buy_sell": "BUY", "quantity": Decimal(quantity),
        "trade_price": Decimal("22"), "proceeds": Decimal("-330"), "ib_commission": Decimal("-1.000525"),
        "ib_commission_currency": "USD", "taxes": Decimal(0), "net_cash": Decimal("-331.000525"), "cost": None,
        "fifo_pnl_realized": None, "open_close": "O", "notes": None, "source_file": "a.xml",
    }


def test_upsert_executions_is_idempotent_and_refreshes(db_session: Session) -> None:
    repo = TradeLedgerRepo(db_session)
    assert repo.upsert_executions([_execution("2", "5", 11), _execution("1", "15", 10)]) == 2

    refreshed = _execution("1", "15", 10) | {"notes": "P", "source_file": "b.xml"}
    assert repo.upsert_executions([refreshed]) == 0

    rows = repo.load_executions()
    assert [r["trade_id"] for r in rows] == ["1", "2"]  # oldest first
    assert rows[0]["notes"] == "P"
    assert rows[0]["ib_commission"] == Decimal("-1.000525")  # exact, no float rounding
    assert "id" not in rows[0] and "imported_at" not in rows[0]


def test_upsert_open_positions_keyed_by_report_date(db_session: Session) -> None:
    repo = TradeLedgerRepo(db_session)
    row = {"account_id": "U0000001", "report_date": date(2025, 12, 31), "conid": "112", "isin": None, "symbol": "ABC",
           "position": Decimal("0.0002"), "cost_basis_price": None, "cost_basis_money": None, "mark_price": Decimal("1.25"), "currency": "USD",
           "source_file": "a.xml"}

    assert repo.upsert_open_positions([row, row | {"report_date": date(2026, 10, 30)}]) == 2
    assert repo.upsert_open_positions([row]) == 0
    assert [r["position"] for r in repo.load_open_positions()] == [Decimal("0.0002"), Decimal("0.0002")]
