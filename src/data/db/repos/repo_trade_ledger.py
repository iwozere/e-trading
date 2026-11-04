"""
Repository layer for the trade ledger tables.

Accepts a SQLAlchemy Session in __init__ — never opens its own session; the
surrounding service commits on success. Rows are plain dicts whose keys equal
the column names (and the field names of ``src.portfolio.trade_ledger.models``).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Type

from sqlalchemy import Result, literal_column, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session
from sqlalchemy.sql.dml import ReturningInsert

from src.data.db.core.base import Base
from src.data.db.models.model_trade_ledger import (
    TradeLedgerCashTransaction,
    TradeLedgerCorporateAction,
    TradeLedgerExecution,
    TradeLedgerOpenPosition,
    TradeLedgerTransfer,
)

# Postgres allows 65,535 bind parameters per statement; 1,000 rows x ~30 columns stays well below.
_INSERT_CHUNK = 1_000
_NEVER_UPDATED = frozenset({"id", "imported_at"})


def _chunks(rows: Sequence[Dict[str, Any]], size: int) -> Iterable[Sequence[Dict[str, Any]]]:
    """Yield consecutive slices of rows."""
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


class TradeLedgerRepo:
    """Repository operations for the trade ledger, bound to a single Session."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def _upsert(self, model: Type[Base], constraint: str, rows: Sequence[Dict[str, Any]]) -> int:
        """
        Insert rows; rows whose unique key exists are refreshed with the new values.

        Args:
            model: ORM model class.
            constraint: Name of the unique constraint identifying a row.
            rows: Dicts keyed by column name.

        Returns:
            Number of rows that were newly inserted (refreshed rows are not counted).
        """
        inserted = 0
        for chunk in _chunks(rows, _INSERT_CHUNK):
            insert = pg_insert(model).values(list(chunk))
            columns = [c for c in chunk[0] if c not in _NEVER_UPDATED]
            stmt: ReturningInsert[Any] = insert.on_conflict_do_update(
                constraint=constraint,
                set_={c: insert.excluded[c] for c in columns},
            ).returning(literal_column("(xmax = 0)").label("inserted"))  # xmax = 0 only on freshly inserted rows
            result: Result[Any] = self.session.execute(stmt)
            inserted += sum(1 for row in result if row.inserted)
        self.session.flush()
        return inserted

    def upsert_executions(self, rows: Sequence[Dict[str, Any]]) -> int:
        """Upsert executions on (account_id, trade_id); returns newly inserted count."""
        return self._upsert(TradeLedgerExecution, "uq_trade_ledger_executions_key", rows)

    def upsert_corporate_actions(self, rows: Sequence[Dict[str, Any]]) -> int:
        """Upsert corporate-action legs on (account_id, transaction_id); returns newly inserted count."""
        return self._upsert(TradeLedgerCorporateAction, "uq_trade_ledger_corporate_actions_key", rows)

    def upsert_transfers(self, rows: Sequence[Dict[str, Any]]) -> int:
        """Upsert transfers on (account_id, transaction_id); returns newly inserted count."""
        return self._upsert(TradeLedgerTransfer, "uq_trade_ledger_transfers_key", rows)

    def upsert_cash_transactions(self, rows: Sequence[Dict[str, Any]]) -> int:
        """Upsert per-security cash transactions on (account_id, transaction_id); returns newly inserted count."""
        return self._upsert(TradeLedgerCashTransaction, "uq_trade_ledger_cash_transactions_key", rows)

    def upsert_open_positions(self, rows: Sequence[Dict[str, Any]]) -> int:
        """Upsert open positions on (account_id, report_date, conid); returns newly inserted count."""
        return self._upsert(TradeLedgerOpenPosition, "uq_trade_ledger_open_positions_key", rows)

    def _load(self, model: Type[Base], order_by: Sequence[str]) -> List[Dict[str, Any]]:
        """Return every row of `model` as a dict (without id / imported_at), ordered by `order_by`."""
        table = model.__table__
        columns = [c for c in table.columns if c.name not in _NEVER_UPDATED]
        stmt = select(*columns).order_by(*(table.c[name] for name in order_by))
        return [dict(row._mapping) for row in self.session.execute(stmt)]

    def load_executions(self) -> List[Dict[str, Any]]:
        """Return all executions, oldest first."""
        return self._load(TradeLedgerExecution, ("trade_datetime", "trade_id"))

    def load_corporate_actions(self) -> List[Dict[str, Any]]:
        """Return all corporate-action legs, oldest first."""
        return self._load(TradeLedgerCorporateAction, ("action_datetime", "transaction_id"))

    def load_transfers(self) -> List[Dict[str, Any]]:
        """Return all transfers, oldest first."""
        return self._load(TradeLedgerTransfer, ("transfer_datetime", "transaction_id"))

    def load_cash_transactions(self) -> List[Dict[str, Any]]:
        """Return all per-security cash transactions, oldest first."""
        return self._load(TradeLedgerCashTransaction, ("cash_datetime", "transaction_id"))

    def load_open_positions(self) -> List[Dict[str, Any]]:
        """Return all period-end open positions, oldest report date first."""
        return self._load(TradeLedgerOpenPosition, ("report_date", "account_id", "conid"))
