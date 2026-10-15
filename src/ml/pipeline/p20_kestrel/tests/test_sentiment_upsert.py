"""DB-free tests for KestrelRepo statement construction (compiled against the PostgreSQL dialect)."""

from datetime import date
from typing import Any, List, cast

from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from src.data.db.repos.repo_kestrel import KestrelRepo


class _CompilingSession:
    """Compiles each executed statement for PostgreSQL, as a real session would before sending it."""

    def __init__(self) -> None:
        self.compiled: List[Any] = []

    def execute(self, stmt: Any) -> None:
        self.compiled.append(stmt.compile(dialect=postgresql.dialect()))


def test_upsert_sentiment_accepts_rows_with_different_columns() -> None:
    """Social poll mixes StockTwits rows (with bullish_ratio) and ApeWisdom rows (without) in one batch."""
    session = _CompilingSession()
    rows = [
        {"ticker": "AAPL", "date": date(2026, 10, 15), "source": "stocktwits", "mentions": 610.0,
         "bullish_ratio": 0.7},
        {"ticker": "AAPL", "date": date(2026, 10, 15), "source": "apewisdom", "mentions": 12},
    ]

    assert KestrelRepo(cast(Session, session)).upsert_sentiment(rows) == 2

    params = session.compiled[0].params
    assert params["bullish_ratio_m0"] == 0.7 and params["bullish_ratio_m1"] is None
    assert "bullish_ratio" not in rows[1]  # caller's dicts are not mutated
