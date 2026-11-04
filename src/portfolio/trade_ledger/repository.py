"""
Storage adapter for the trade ledger.

Converts parsed Flex dataclasses to DB rows and back. All persistence goes
through `TradeLedgerService` (src.data.db); this module holds no SQL. Column
names equal dataclass field names, so conversion is a plain `asdict` /
`Model(**row)`.
"""

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Protocol, Sequence

from src.notification.logger import setup_logger
from src.portfolio.trade_ledger.models import (
    CashTransaction,
    CorporateActionLeg,
    Execution,
    IbkrOpenPosition,
    LedgerEvents,
    ParseResult,
    TransferEvent,
)

_logger = setup_logger(__name__)


class TradeLedgerStorage(Protocol):
    """The subset of `TradeLedgerService` this module uses (lets tests pass a fake)."""

    def upsert_events(
        self,
        executions: Sequence[Dict[str, Any]],
        corporate_actions: Sequence[Dict[str, Any]],
        transfers: Sequence[Dict[str, Any]],
        cash_transactions: Sequence[Dict[str, Any]],
        open_positions: Sequence[Dict[str, Any]],
    ) -> Dict[str, int]:
        """Upsert one import's events; returns newly inserted counts."""
        ...

    def load_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return every stored event keyed by table kind."""
        ...


def _default_storage() -> TradeLedgerStorage:
    """Build the real DB-backed service (imported lazily so unit tests need no DB)."""
    from src.data.db.services.trade_ledger_service import TradeLedgerService

    return TradeLedgerService()


def store_parse_result(result: ParseResult, storage: Optional[TradeLedgerStorage] = None) -> Dict[str, int]:
    """
    Persist everything parsed from one Flex file in a single transaction.

    Idempotent: rows are keyed on IBKR ids, so storing the same file twice
    inserts nothing new the second time.

    Args:
        result: Parsed Flex file.
        storage: Storage backend; defaults to `TradeLedgerService`.

    Returns:
        Newly inserted row count per table kind.
    """
    storage = storage or _default_storage()
    inserted = storage.upsert_events(
        executions=[asdict(e) for s in result.statements for e in s.executions],
        corporate_actions=[asdict(c) for s in result.statements for c in s.corporate_actions],
        transfers=[asdict(t) for s in result.statements for t in s.transfers],
        cash_transactions=[asdict(a) for s in result.statements for a in s.cash_transactions],
        open_positions=[asdict(p) for s in result.statements for p in s.open_positions],
    )
    _logger.info("Stored %s: newly inserted %s", result.source_file, inserted)
    return inserted


def load_events(storage: Optional[TradeLedgerStorage] = None) -> LedgerEvents:
    """
    Load every stored event as dataclasses.

    Args:
        storage: Storage backend; defaults to `TradeLedgerService`.

    Returns:
        LedgerEvents, each list oldest first.
    """
    storage = storage or _default_storage()
    rows = storage.load_all()
    return LedgerEvents(
        executions=[Execution(**r) for r in rows["executions"]],
        corporate_actions=[CorporateActionLeg(**r) for r in rows["corporate_actions"]],
        transfers=[TransferEvent(**r) for r in rows["transfers"]],
        cash_transactions=[CashTransaction(**r) for r in rows["cash_transactions"]],
        open_positions=[IbkrOpenPosition(**r) for r in rows["open_positions"]],
    )
