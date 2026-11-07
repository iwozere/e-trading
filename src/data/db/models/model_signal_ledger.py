"""
SQLAlchemy ORM models for the signal ledger (src/ml/pipeline/signal_ledger).

``signal_ledger`` holds every buy/sell recommendation any pipeline emitted, with
its forward returns filled in once they mature. ``signal_book_nav`` holds the
daily NAV of the simulated paper book of each signal stream and exit policy,
``signal_book_trades`` every paper trade of those books and ``signal_book_stats``
their summaries (all fully rebuilt by the nightly job).
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    BigInteger,
    ForeignKey,
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
    text,
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
    ret_60d: Mapped[float | None] = _ret()
    ret_120d: Mapped[float | None] = _ret()
    ret_250d: Mapped[float | None] = _ret()
    vt_60d: Mapped[float | None] = _ret()
    vt_120d: Mapped[float | None] = _ret()
    vt_250d: Mapped[float | None] = _ret()
    spy_60d: Mapped[float | None] = _ret()
    spy_120d: Mapped[float | None] = _ret()
    spy_250d: Mapped[float | None] = _ret()
    iwm_60d: Mapped[float | None] = _ret()
    iwm_120d: Mapped[float | None] = _ret()
    iwm_250d: Mapped[float | None] = _ret()
    mae_20d: Mapped[float | None] = _ret()  # lowest low over the first 20 sessions vs entry
    mfe_20d: Mapped[float | None] = _ret()  # highest high over the first 20 sessions vs entry
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
    """Daily NAV of the simulated paper book of one signal stream under one exit policy."""

    __tablename__ = "signal_book_nav"

    source: Mapped[str] = mapped_column(Text, primary_key=True)
    policy: Mapped[str] = mapped_column(Text, primary_key=True, default="horizon", server_default="horizon")
    nav_date: Mapped[date] = mapped_column(Date, primary_key=True)
    nav: Mapped[float] = mapped_column(Numeric(asdecimal=False), nullable=False)
    benchmark_nav: Mapped[float] = mapped_column(Numeric(asdecimal=False), nullable=False)
    invested: Mapped[float] = mapped_column(Numeric(asdecimal=False), nullable=False)
    positions: Mapped[int] = mapped_column(Integer, nullable=False)
    opened: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    closed: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))


class SignalBookTrade(Base):
    """One paper trade of a stream's book under one exit policy (closed, or still open at the last session)."""

    __tablename__ = "signal_book_trades"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    policy: Mapped[str] = mapped_column(Text, nullable=False)
    signal_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("signal_ledger.id", ondelete="CASCADE"))
    ticker: Mapped[str] = mapped_column(Text, nullable=False)
    signal_date: Mapped[date] = mapped_column(Date, nullable=False)
    entry_date: Mapped[date] = mapped_column(Date, nullable=False)
    entry_price: Mapped[float] = mapped_column(Numeric(asdecimal=False), nullable=False)
    exit_date: Mapped[date | None] = mapped_column(Date)
    exit_price: Mapped[float | None] = mapped_column(Numeric(asdecimal=False))
    exit_reason: Mapped[str] = mapped_column(Text, nullable=False)  # time/stop/target/trail/native/decay/data/open
    sessions_held: Mapped[int] = mapped_column(Integer, nullable=False)
    net_return: Mapped[float | None] = _ret()  # after costs; marked to the last close for open trades
    last_price: Mapped[float | None] = _ret()  # exit price, or the last close for open trades
    mae: Mapped[float | None] = _ret()
    mfe: Mapped[float | None] = _ret()

    __table_args__ = (Index("idx_signal_book_trades_source_policy", "source", "policy"),)


class SignalBookStats(Base):
    """
    Nightly summary of one stream/policy: capped-book summary and trade stats, plus uncapped trade stats.

    ``stats`` = ``{"book": {...}, "trades": {...}, "all_signals": {...}}`` (see ``portfolio.summarize_*``).
    """

    __tablename__ = "signal_book_stats"

    source: Mapped[str] = mapped_column(Text, primary_key=True)
    policy: Mapped[str] = mapped_column(Text, primary_key=True)
    stats: Mapped[dict] = mapped_column(JsonType(), nullable=False)
    computed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), server_default=func.now())
