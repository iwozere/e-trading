"""In-memory stand-in for TradeLedgerService shared by the trade-ledger tests (no DB)."""

from typing import Any, Dict, List, Sequence, Tuple

_KEYS: Dict[str, Tuple[str, ...]] = {
    "executions": ("account_id", "trade_id"),
    "corporate_actions": ("account_id", "transaction_id"),
    "transfers": ("account_id", "transaction_id"),
    "cash_transactions": ("account_id", "transaction_id"),
    "open_positions": ("account_id", "report_date", "conid"),
}


class FakeStorage:
    """Mimics TradeLedgerService's upsert-on-unique-key semantics in memory."""

    def __init__(self) -> None:
        self.tables: Dict[str, Dict[Tuple[Any, ...], Dict[str, Any]]] = {kind: {} for kind in _KEYS}

    def upsert_events(
        self,
        executions: Sequence[Dict[str, Any]],
        corporate_actions: Sequence[Dict[str, Any]],
        transfers: Sequence[Dict[str, Any]],
        cash_transactions: Sequence[Dict[str, Any]],
        open_positions: Sequence[Dict[str, Any]],
    ) -> Dict[str, int]:
        """Upsert rows on their unique keys; return the newly inserted count per kind."""
        batches = {
            "executions": executions,
            "corporate_actions": corporate_actions,
            "transfers": transfers,
            "cash_transactions": cash_transactions,
            "open_positions": open_positions,
        }
        inserted: Dict[str, int] = {}
        for kind, rows in batches.items():
            table = self.tables[kind]
            new = 0
            for row in rows:
                key = tuple(row[k] for k in _KEYS[kind])
                new += key not in table
                table[key] = dict(row)
            inserted[kind] = new
        return inserted

    def load_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return every stored row keyed by table kind."""
        return {kind: list(table.values()) for kind, table in self.tables.items()}
