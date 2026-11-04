"""
TradeLedgerService — Unit-of-Work service for the trade ledger tables.

Each method opens its own Unit-of-Work via @with_uow and delegates to TradeLedgerRepo.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from src.data.db.services.base_service import BaseDBService, with_uow


class TradeLedgerService(BaseDBService):
    """Unit-of-Work service for the monthly trade ledger report."""

    @with_uow
    def upsert_events(
        self,
        executions: Sequence[Dict[str, Any]],
        corporate_actions: Sequence[Dict[str, Any]],
        transfers: Sequence[Dict[str, Any]],
        cash_transactions: Sequence[Dict[str, Any]],
        open_positions: Sequence[Dict[str, Any]],
    ) -> Dict[str, int]:
        """
        Upsert one import's events in a single transaction.

        Args:
            executions: Execution rows.
            corporate_actions: Corporate-action leg rows.
            transfers: Transfer rows.
            cash_transactions: Commission adjustment / dividend / withholding-tax rows.
            open_positions: Period-end open position rows.

        Returns:
            Newly inserted row count per table kind.
        """
        repo = self.repos.trade_ledger
        return {
            "executions": repo.upsert_executions(executions) if executions else 0,
            "corporate_actions": repo.upsert_corporate_actions(corporate_actions) if corporate_actions else 0,
            "transfers": repo.upsert_transfers(transfers) if transfers else 0,
            "cash_transactions": repo.upsert_cash_transactions(cash_transactions) if cash_transactions else 0,
            "open_positions": repo.upsert_open_positions(open_positions) if open_positions else 0,
        }

    @with_uow
    def load_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return every stored event, oldest first, keyed by table kind."""
        repo = self.repos.trade_ledger
        return {
            "executions": repo.load_executions(),
            "corporate_actions": repo.load_corporate_actions(),
            "transfers": repo.load_transfers(),
            "cash_transactions": repo.load_cash_transactions(),
            "open_positions": repo.load_open_positions(),
        }
