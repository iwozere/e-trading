"""
SQLAlchemy ORM models for all k20_* tables (P20 Kestrel pipeline).

Only k20_* tables are defined here. No existing tables are modified.
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
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.data.db.core.base import Base
from src.data.db.core.json_types import JsonType


class K20Universe(Base):
    """Active ticker universe — one row per ticker."""

    __tablename__ = "k20_universe"

    ticker: Mapped[str] = mapped_column(Text, primary_key=True)
    exchange: Mapped[str | None] = mapped_column(Text)
    sector: Mapped[str | None] = mapped_column(Text)
    industry: Mapped[str | None] = mapped_column(Text)
    mcap: Mapped[float | None] = mapped_column(Numeric)
    adv_20d: Mapped[float | None] = mapped_column(Numeric)
    revenue_yoy_growth: Mapped[float | None] = mapped_column(Numeric)
    gross_margin: Mapped[float | None] = mapped_column(Numeric)
    net_debt_ebitda: Mapped[float | None] = mapped_column(Numeric)
    interest_coverage: Mapped[float | None] = mapped_column(Numeric)
    status: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="active",
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('active','delisted','suspended')",
            name="ck_k20_universe_status",
        ),
        Index("idx_k20_universe_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<K20Universe(ticker={self.ticker!r}, status={self.status!r})>"


class K20CompanyAlias(Base):
    """Normalized company name aliases used by the GDELT processor."""

    __tablename__ = "k20_company_aliases"

    ticker: Mapped[str] = mapped_column(Text, primary_key=True)
    alias: Mapped[str] = mapped_column(Text, primary_key=True)
    alias_type: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_alias: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint(
            "alias_type IN ('legal_name','short_name','brand','former_name')",
            name="ck_k20_alias_type",
        ),
        Index("idx_k20_aliases_normalized", "normalized_alias"),
    )


class K20AliasBlocklist(Base):
    """Collision-prone aliases with per-alias match policy."""

    __tablename__ = "k20_alias_blocklist"

    alias: Mapped[str] = mapped_column(Text, primary_key=True)
    ticker: Mapped[str | None] = mapped_column(Text)
    match_policy: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    added_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    __table_args__ = (
        CheckConstraint(
            "match_policy IN ('legal_name_only','name_plus_context','never')",
            name="ck_k20_blocklist_policy",
        ),
    )


class K20Signal(Base):
    """Daily signal values per ticker (technicals, insider flows, crowding, etc.)."""

    __tablename__ = "k20_signals"

    ticker: Mapped[str] = mapped_column(Text, primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    signal_type: Mapped[str] = mapped_column(Text, primary_key=True)
    value: Mapped[float | None] = mapped_column(Numeric)
    sleeve: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        Index("idx_k20_signals_ticker_date", "ticker", "date"),
        Index("idx_k20_signals_type", "signal_type"),
    )

    def __repr__(self) -> str:
        return f"<K20Signal({self.ticker!r} {self.date} {self.signal_type!r}={self.value})>"


class K20SentimentDaily(Base):
    """Per-source daily sentiment aggregates per ticker."""

    __tablename__ = "k20_sentiment_daily"

    ticker: Mapped[str] = mapped_column(Text, primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    source: Mapped[str] = mapped_column(Text, primary_key=True)
    mentions: Mapped[float | None] = mapped_column(Numeric)
    avg_tone: Mapped[float | None] = mapped_column(Numeric)
    tone_std: Mapped[float | None] = mapped_column(Numeric)
    pos_score: Mapped[float | None] = mapped_column(Numeric)
    neg_score: Mapped[float | None] = mapped_column(Numeric)
    bullish_ratio: Mapped[float | None] = mapped_column(Numeric)
    top_domains: Mapped[dict | None] = mapped_column(JsonType())
    mention_z20: Mapped[float | None] = mapped_column(Numeric)
    tone_z20: Mapped[float | None] = mapped_column(Numeric)

    __table_args__ = (
        CheckConstraint(
            "source IN ('gdelt','stocktwits','reddit','apewisdom','trends','av_news')",
            name="ck_k20_sentiment_source",
        ),
        Index("idx_k20_sentiment_ticker_date", "ticker", "date"),
    )


class K20Catalyst(Base):
    """Forward-looking catalyst events (earnings, PDUFA, spin-offs, etc.)."""

    __tablename__ = "k20_catalysts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(Text, nullable=False)
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    event_date: Mapped[date | None] = mapped_column(Date)
    confidence: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)
    catalyst_detail: Mapped[dict | None] = mapped_column(JsonType())
    state: Mapped[str] = mapped_column(Text, nullable=False, default="upcoming")
    t10_alerted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    t3_alerted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    datechange_alerted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint(
            "state IN ('upcoming','date_changed','passed','cancelled')",
            name="ck_k20_catalyst_state",
        ),
        Index("idx_k20_catalysts_ticker", "ticker"),
        Index("idx_k20_catalysts_event_date", "event_date"),
    )


class K20LLMRun(Base):
    """Audit log and cache for every LLM call made by P20."""

    __tablename__ = "k20_llm_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ts: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    ticker: Mapped[str | None] = mapped_column(Text)
    task_type: Mapped[str] = mapped_column(Text, nullable=False)
    input_ref: Mapped[str | None] = mapped_column(Text)
    output_json: Mapped[dict | None] = mapped_column(JsonType())
    model: Mapped[str | None] = mapped_column(Text)
    tokens_in: Mapped[int | None] = mapped_column(Integer)
    tokens_out: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[float | None] = mapped_column(Numeric(10, 6))
    verdict: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        UniqueConstraint("task_type", "input_ref", name="uq_k20_llm_runs_task_ref"),
        Index("idx_k20_llm_runs_ticker", "ticker"),
        Index("idx_k20_llm_runs_task", "task_type"),
        Index("idx_k20_llm_runs_ts", "ts"),
    )


class K20Watchlist(Base):
    """Candidates being actively tracked by a sleeve."""

    __tablename__ = "k20_watchlist"

    ticker: Mapped[str] = mapped_column(Text, primary_key=True)
    sleeve: Mapped[str] = mapped_column(Text, primary_key=True)
    score: Mapped[float | None] = mapped_column(Numeric)
    llm_verdict: Mapped[str | None] = mapped_column(Text)
    dossier_run_id: Mapped[int | None] = mapped_column(BigInteger)
    thesis_short: Mapped[str | None] = mapped_column(Text)
    added_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    state: Mapped[str] = mapped_column(Text, nullable=False, default="screening")

    __table_args__ = (
        CheckConstraint(
            "state IN ('screening','candidate','active_position','rejected','expired')",
            name="ck_k20_watchlist_state",
        ),
        Index("idx_k20_watchlist_state", "state"),
        Index("idx_k20_watchlist_score", "score"),
    )


class K20Position(Base):
    """Open and closed positions managed via /pos commands."""

    __tablename__ = "k20_positions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(Text, nullable=False)
    sleeve: Mapped[str] = mapped_column(Text, nullable=False)
    entry_date: Mapped[date | None] = mapped_column(Date)
    entry_px: Mapped[float | None] = mapped_column(Numeric(12, 4))
    size_pct: Mapped[float | None] = mapped_column(Numeric(5, 2))
    stop_px: Mapped[float | None] = mapped_column(Numeric(12, 4))
    t1_px: Mapped[float | None] = mapped_column(Numeric(12, 4))
    t2_px: Mapped[float | None] = mapped_column(Numeric(12, 4))
    trail_pct: Mapped[float | None] = mapped_column(Numeric(5, 2))
    realized_thirds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="open")
    notes: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        Index("idx_k20_positions_ticker", "ticker"),
        Index("idx_k20_positions_status", "status"),
    )


class K20RequestBudget(Base):
    """Daily per-source API call quota tracker."""

    __tablename__ = "k20_request_budget"

    source: Mapped[str] = mapped_column(Text, primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    quota: Mapped[int] = mapped_column(Integer, nullable=False)
    used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    notes: Mapped[dict | None] = mapped_column(JsonType())


class K20JobRun(Base):
    """One row per (job, run_date) — used for DAG dependency signaling."""

    __tablename__ = "k20_job_runs"

    job: Mapped[str] = mapped_column(Text, primary_key=True)
    run_date: Mapped[date] = mapped_column(Date, primary_key=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="running")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    rows_out: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint(
            "status IN ('running','ok','failed','skipped')",
            name="ck_k20_job_run_status",
        ),
        Index("idx_k20_job_runs_status", "status"),
    )


class K20AlertsLog(Base):
    """Append-only log of every push alert fired by P20."""

    __tablename__ = "k20_alerts_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ts: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    ticker: Mapped[str | None] = mapped_column(Text)
    trigger: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict | None] = mapped_column(JsonType())
    channel: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        Index("idx_k20_alerts_ts", "ts"),
        Index("idx_k20_alerts_ticker", "ticker"),
    )
