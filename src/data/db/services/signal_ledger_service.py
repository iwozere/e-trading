"""
SignalLedgerService — Unit-of-Work service for ``signal_ledger`` / ``signal_book_nav``.

Each method opens its own Unit-of-Work via @with_uow and delegates to SignalLedgerRepo.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Sequence

from src.data.db.services.base_service import BaseDBService, with_uow


class SignalLedgerService(BaseDBService):
    """Unit-of-Work service for the cross-pipeline signal ledger."""

    @with_uow
    def upsert_signals(self, rows: Sequence[Dict[str, Any]]) -> int:
        """Upsert harvested signals; returns the number of newly inserted rows."""
        return self.repos.signal_ledger.upsert_signals(rows)

    @with_uow
    def get_pending_signals(self) -> List[Dict[str, Any]]:
        """Return ``id, signal_date, ticker`` of signals whose outcome is not final."""
        return self.repos.signal_ledger.get_pending_signals()

    @with_uow
    def update_outcomes(self, outcomes: Sequence[Dict[str, Any]]) -> int:
        """Write evaluated outcomes by id; returns the row count."""
        return self.repos.signal_ledger.update_outcomes(outcomes)

    @with_uow
    def list_signals(
        self,
        source: Optional[str] = None,
        since: Optional[date] = None,
        limit: Optional[int] = None,
        newest_first: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return signals with outcomes (see SignalLedgerRepo.list_signals)."""
        return self.repos.signal_ledger.list_signals(source=source, since=since, limit=limit, newest_first=newest_first)

    @with_uow
    def replace_book_nav(self, source: str, rows: Sequence[Dict[str, Any]]) -> int:
        """Replace one stream's paper-book NAV history; returns the row count."""
        return self.repos.signal_ledger.replace_book_nav(source, rows)

    @with_uow
    def get_book_nav(self, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return paper-book NAV rows, oldest first."""
        return self.repos.signal_ledger.get_book_nav(source)
