"""
Repository layer for ``signal_ledger`` and ``signal_book_nav``.

Accepts a SQLAlchemy Session in __init__ — never opens its own session; the
surrounding service commits on success.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from sqlalchemy import Result, delete, literal_column, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.data.db.models.model_signal_ledger import SignalBookNav, SignalLedger

# Postgres allows 65,535 bind parameters per statement; 2,000 rows x 9 columns stays well below.
_INSERT_CHUNK = 2_000

_RETURN_COLUMNS = tuple(f"{prefix}_{h}d" for prefix in ("ret", "vt", "spy", "iwm") for h in (1, 5, 10, 20))
_SIGNAL_COLUMNS = ("id", "pipeline", "source", "signal_date", "ticker", "side", "rank", "score", "horizon_days", "meta")
_OUTCOME_COLUMNS = ("status", "entry_date", "entry_price", "data_flag", *_RETURN_COLUMNS)


def _chunks(rows: Sequence[Dict[str, Any]], size: int) -> Iterable[Sequence[Dict[str, Any]]]:
    """Yield consecutive slices of rows."""
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


class SignalLedgerRepo:
    """Repository operations for the signal ledger, bound to a single Session."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_signals(self, rows: Sequence[Dict[str, Any]]) -> int:
        """
        Insert new signals; existing keys only refresh rank/score/horizon/meta, never outcomes.

        Args:
            rows: Dicts with ``pipeline, source, signal_date, ticker, side, rank, score, horizon_days, meta``.

        Returns:
            Number of rows that were newly inserted.
        """
        inserted = 0
        for chunk in _chunks(rows, _INSERT_CHUNK):
            insert = pg_insert(SignalLedger).values(list(chunk))
            result: Result[Any] = self.session.execute(insert.on_conflict_do_update(
                constraint="uq_signal_ledger_key",
                set_={
                    "rank": insert.excluded.rank,
                    "score": insert.excluded.score,
                    "horizon_days": insert.excluded.horizon_days,
                    "meta": insert.excluded.meta,
                },
            ).returning(literal_column("(xmax = 0)").label("inserted")))  # xmax = 0 only on freshly inserted rows
            inserted += sum(1 for row in result if row.inserted)
        self.session.flush()
        return inserted

    def get_pending_signals(self) -> List[Dict[str, Any]]:
        """Return ``id, signal_date, ticker`` of every signal whose outcome is not final."""
        stmt = select(SignalLedger.id, SignalLedger.signal_date, SignalLedger.ticker).where(SignalLedger.status == "pending")
        return [dict(row._mapping) for row in self.session.execute(stmt)]

    def update_outcomes(self, outcomes: Sequence[Dict[str, Any]]) -> int:
        """
        Write evaluated outcomes by primary key.

        Args:
            outcomes: Dicts with ``id`` plus outcome columns (status, entry_date, entry_price, returns).

        Returns:
            Number of rows updated.
        """
        if not outcomes:
            return 0
        now = datetime.now(timezone.utc)
        rows = [{"id": o["id"], "evaluated_at": now, **{k: o.get(k) for k in _OUTCOME_COLUMNS}} for o in outcomes]
        for chunk in _chunks(rows, _INSERT_CHUNK):
            self.session.execute(update(SignalLedger), list(chunk))
        self.session.flush()
        return len(rows)

    def list_signals(
        self,
        source: Optional[str] = None,
        since: Optional[date] = None,
        limit: Optional[int] = None,
        newest_first: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Return signals with their outcomes as plain dicts.

        Args:
            source: Only this source.
            since: Only signals dated on/after this.
            limit: Maximum rows.
            newest_first: Order by signal_date descending instead of ascending.

        Returns:
            Rows with signal and outcome columns.
        """
        columns = [getattr(SignalLedger, c) for c in (*_SIGNAL_COLUMNS, *_OUTCOME_COLUMNS)]
        stmt = select(*columns)
        if source:
            stmt = stmt.where(SignalLedger.source == source)
        if since:
            stmt = stmt.where(SignalLedger.signal_date >= since)
        order = SignalLedger.signal_date.desc() if newest_first else SignalLedger.signal_date
        stmt = stmt.order_by(order, SignalLedger.source, SignalLedger.rank, SignalLedger.ticker)
        if limit:
            stmt = stmt.limit(limit)
        return [dict(row._mapping) for row in self.session.execute(stmt)]

    def replace_book_nav(self, source: str, rows: Sequence[Dict[str, Any]]) -> int:
        """
        Replace the whole NAV history of one stream's paper book.

        Args:
            source: Stream source.
            rows: Dicts with ``date, nav, benchmark_nav, invested, positions, opened, closed``.

        Returns:
            Number of rows written.
        """
        self.session.execute(delete(SignalBookNav).where(SignalBookNav.source == source))
        values = [{"source": source, "nav_date": r["date"], **{k: r[k] for k in ("nav", "benchmark_nav", "invested", "positions", "opened", "closed")}} for r in rows]
        for chunk in _chunks(values, _INSERT_CHUNK):
            self.session.execute(pg_insert(SignalBookNav).values(list(chunk)))
        self.session.flush()
        return len(values)

    def get_book_nav(self, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return paper-book NAV rows (``source, date, nav, ...``), oldest first."""
        stmt = select(SignalBookNav)
        if source:
            stmt = stmt.where(SignalBookNav.source == source)
        stmt = stmt.order_by(SignalBookNav.source, SignalBookNav.nav_date)
        return [
            {"source": r.source, "date": r.nav_date, "nav": r.nav, "benchmark_nav": r.benchmark_nav, "invested": r.invested,
             "positions": r.positions, "opened": r.opened, "closed": r.closed}
            for r in self.session.scalars(stmt)
        ]
