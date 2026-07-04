"""Single return type for trade persistence (`create_trade` / `add_trade`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class CreatedTrade:
    """Result of inserting a trade row. Callers use `.id`; full DB shape is in `.row`."""

    id: str
    row: Dict[str, Any]

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> CreatedTrade:
        if not row or row.get("id") is None:
            raise ValueError("Trade row must be a non-empty dict with an 'id' key")
        return cls(id=str(row["id"]), row=dict(row))

    @classmethod
    def synthetic(cls, trade_id: str, row: Dict[str, Any] | None = None) -> CreatedTrade:
        """Test / offline mocks without a real database row."""
        return cls(id=str(trade_id), row=dict(row or {}))
