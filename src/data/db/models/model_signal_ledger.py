"""
SQLAlchemy ORM models for the signal ledger (src/ml/pipeline/signal_ledger).

``signal_ledger`` holds every buy/sell recommendation any pipeline emitted, with
its forward returns filled in once they mature. ``signal_book_nav`` holds the
daily NAV of the simulated paper book of each signal stream (fully rebuilt by
the nightly job). No existing tables are modified.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Date,
    DateTime,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.data.db.core.base import Base
from src.data.db.core.json_types import JsonType


def _ret() -> Mapped[float | None]:
    """A nullable float return column."""
    return mapped_column(Numeric(asdecimal=False))


class SignalLedger(Base):
    """One pipeline signal and its forward returns (raw ticker return and benchmarks over the same window)."""

    __tablename__ = "signal_ledger"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    pipeline: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    signal_date: Mapped[date] = mapped_column(Date, nullable=False)
    ticker: Mapped[str] = mapped_column(Text, nullable=False)
    side: Mapped[str] = mapped_column(Text, nullable=False, default="long")
    rank: Mapped[int | None] = mapped_column(Integer)
    score: Mapped[float | None] = mapped_column(Numeric(asdecimal=False))
    horizon_days: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=20)
    meta: Mapped[dict | None] = mapped_column(JsonType())
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending", server_default="pending")
    entry_date: Mapped[date | None] = mapped_column(Date)
    entry_price: Mapped[float | None] = mapped_column(Numeric(asdecimal=False))
    data_flag: Mapped[str | None] = mapped_column(Text)  # e.g. "price_jump": excluded from scorecard and books
    ret_1d: Mapped[float | None] = _ret()
    ret_5d: Mapped[float | None] = _ret()
    ret_10d: Mapped[float | None] = _ret()
    ret_20d: Mapped[float | None] = _ret()
    vt_1d: Mapped[float | None] = _ret()
    vt_5d: Mapped[float | None] = _ret()
    vt_10d: Mapped[float | None] = _ret()
    vt_20d: Mapped[float | None] = _ret()
    spy_1d: Mapped[float | None] = _ret()
    spy_5d: Mapped[float | None] = _ret()
    spy_10d: Mapped[float | None] = _ret()
    spy_20d: Mapped[float | None] = _ret()
    iwm_1d: Mapped[float | None] = _ret()
    iwm_5d: Mapped[float | None] = _ret()
    iwm_10d: Mapped[float | None] = _ret()
    iwm_20d: Mapped[float | None] = _ret()
    harvested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), server_default=func.now())
    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint("source", "signal_date", "ticker", "side", name="uq_signal_ledger_key"),
        CheckConstraint("side IN ('long','short')", name="side"),
        CheckConstraint("status IN ('pending','complete','no_data')", name="status"),
        Index("idx_signal_ledger_source_date", "source", "signal_date"),
        Index("idx_signal_ledger_status", "status"),
    )


class SignalBookNav(Base):
    """Daily NAV of the simulated paper book of one signal stream."""

    __tablename__ = "signal_book_nav"

    source: Mapped[str] = mapped_column(Text, primary_key=True)
    nav_date: Mapped[date] = mapped_column(Date, primary_key=True)
    nav: Mapped[float] = mapped_column(Numeric(asdecimal=False), nullable=False)
    benchmark_nav: Mapped[float] = mapped_column(Numeric(asdecimal=False), nullable=False)
    invested: Mapped[float] = mapped_column(Numeric(asdecimal=False), nullable=False)
    positions: Mapped[int] = mapped_column(Integer, nullable=False)
    opened: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    closed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
