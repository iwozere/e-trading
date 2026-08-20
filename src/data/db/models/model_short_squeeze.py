"""
Short Squeeze Detection Pipeline Models

SQLAlchemy models for the short squeeze detection pipeline.
Includes ScreenerSnapshot, DeepScanMetrics, SqueezeAlert, and AdHocCandidateModel models.
"""

from datetime import date, datetime
from datetime import date as DateType
from enum import StrEnum

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.data.db.core.base import Base
from src.data.db.core.json_types import JsonType


class AlertLevel(StrEnum):
    """Alert level enumeration."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class CandidateSource(StrEnum):
    """Candidate source enumeration."""

    SCREENER = "screener"
    ADHOC = "adhoc"
    VOLUME_SCREENER = "volume_screener"
    HYBRID_SCREENER = "hybrid_screener"
    FINRA_SCREENER = "finra_screener"


class ScreenerSnapshot(Base):
    """Weekly screener snapshots with structural metrics."""

    __tablename__ = "ss_snapshot"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True)
    run_date: Mapped[date] = mapped_column(Date, index=True)
    short_interest_pct: Mapped[float | None] = mapped_column(Numeric(5, 4))
    days_to_cover: Mapped[float | None] = mapped_column(Numeric(8, 2))
    float_shares: Mapped[int | None] = mapped_column(BigInteger)
    avg_volume_14d: Mapped[int | None] = mapped_column(BigInteger)
    market_cap: Mapped[int | None] = mapped_column(BigInteger)
    screener_score: Mapped[float | None] = mapped_column(Numeric(5, 4))
    raw_payload: Mapped[dict | None] = mapped_column(JsonType())
    data_quality: Mapped[float | None] = mapped_column(Numeric(3, 2))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())

    __table_args__ = (
        Index("idx_ss_snapshot_ticker_date", "ticker", "run_date"),
        Index("idx_ss_snapshot_run_date_desc", "run_date", postgresql_using="btree"),
        Index("idx_ss_snapshot_screener_score_desc", "screener_score", "run_date", postgresql_using="btree"),
        Index("idx_ss_snapshot_created_at", "created_at"),
        CheckConstraint("short_interest_pct >= 0 AND short_interest_pct <= 1", name="check_short_interest_pct"),
        CheckConstraint("days_to_cover >= 0", name="check_days_to_cover"),
        CheckConstraint("screener_score >= 0 AND screener_score <= 1", name="check_screener_score"),
        CheckConstraint("data_quality >= 0 AND data_quality <= 1", name="check_data_quality"),
    )

    def __repr__(self):
        return f"<ScreenerSnapshot(id={self.id}, ticker='{self.ticker}', run_date={self.run_date}, score={self.screener_score})>"

    def to_structural_metrics(self):
        """Convert to StructuralMetrics dataclass."""
        # Import here to avoid circular imports
        from src.ml.pipeline.p04_short_squeeze.core.models import StructuralMetrics

        if (
            self.short_interest_pct is None
            or self.days_to_cover is None
            or self.float_shares is None
            or self.avg_volume_14d is None
            or self.market_cap is None
        ):
            return None

        return StructuralMetrics(
            short_interest_pct=self.short_interest_pct,
            days_to_cover=self.days_to_cover,
            float_shares=self.float_shares,
            avg_volume_14d=self.avg_volume_14d,
            market_cap=self.market_cap,
        )


class DeepScanMetrics(Base):
    """Daily deep scan metrics with transient data."""

    __tablename__ = "ss_deep_metrics"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True)
    # DateType alias: the column name "date" shadows datetime.date in this scope
    date: Mapped[DateType] = mapped_column(Date, index=True)
    volume_spike: Mapped[float | None] = mapped_column(Numeric(6, 2))
    call_put_ratio: Mapped[float | None] = mapped_column(Numeric(6, 2))
    sentiment_24h: Mapped[float | None] = mapped_column(Numeric(4, 3))
    borrow_fee_pct: Mapped[float | None] = mapped_column(Numeric(5, 4))
    squeeze_score: Mapped[float | None] = mapped_column(Numeric(5, 4))
    alert_level: Mapped[str | None] = mapped_column(String(10))
    raw_payload: Mapped[dict | None] = mapped_column(JsonType())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())

    # Multi-source retail sentiment metrics (added by migration 001_add_sentiment_metrics.sql).
    # These columns existed in the DB before this ORM class mapped them -- _store_results() in
    # daily_deep_scan.py previously computed but never persisted them.
    mentions_24h: Mapped[int | None] = mapped_column(Integer, default=0)
    mentions_growth_7d: Mapped[float | None] = mapped_column(Numeric(8, 4))
    # FLOAT in the DB (migration 001), not NUMERIC -- unsigned reach (Σengagement /
    # sqrt(unique_authors+1)), unbounded above since the Rev 2 redefinition (spec §2.5.5).
    virality_index: Mapped[float | None] = mapped_column(Float, default=0.0)
    bot_pct: Mapped[float | None] = mapped_column(Numeric(4, 3), default=0.0)
    sentiment_data_quality: Mapped[dict | None] = mapped_column(JsonType())

    # Tech-discourse signal class (Hacker News) -- reported separately from retail sentiment
    # above, never blended into it. See sentiment-spec-rev2.md §2.5.6. `tech_coverage_available`
    # distinguishes "ticker absent from the HN entity map" (False) from "covered but zero
    # mentions" (True, tech_mentions_24h=0) -- the other tech_* columns stay NULL, never 0.5,
    # when coverage is unavailable.
    tech_mentions_24h: Mapped[int | None] = mapped_column(Integer)
    tech_sentiment_score_24h: Mapped[float | None] = mapped_column(Numeric(4, 3))
    tech_sentiment_24h: Mapped[float | None] = mapped_column(Numeric(4, 3))
    tech_discussion_depth: Mapped[float | None] = mapped_column(Numeric(8, 2))
    tech_coverage_available: Mapped[bool | None] = mapped_column(Boolean)

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="unique_ticker_date"),
        Index("idx_ss_deep_metrics_ticker_date", "ticker", "date"),
        Index("idx_ss_deep_metrics_date_desc", "date", postgresql_using="btree"),
        Index("idx_ss_deep_metrics_squeeze_score_desc", "squeeze_score", "date", postgresql_using="btree"),
        Index("idx_ss_deep_metrics_alert_level", "alert_level", "date", postgresql_using="btree"),
        Index("idx_ss_deep_metrics_created_at", "created_at"),
        CheckConstraint("volume_spike >= 0", name="check_volume_spike"),
        CheckConstraint("call_put_ratio >= 0", name="check_call_put_ratio"),
        CheckConstraint("sentiment_24h >= -1 AND sentiment_24h <= 1", name="check_sentiment_24h"),
        CheckConstraint("borrow_fee_pct >= 0", name="check_borrow_fee_pct"),
        CheckConstraint("squeeze_score >= 0 AND squeeze_score <= 1", name="check_squeeze_score"),
        CheckConstraint("alert_level IN ('LOW', 'MEDIUM', 'HIGH')", name="check_alert_level"),
        CheckConstraint("tech_mentions_24h >= 0", name="check_tech_mentions_24h"),
        CheckConstraint(
            "tech_sentiment_score_24h >= -1 AND tech_sentiment_score_24h <= 1", name="check_tech_sentiment_score_24h"
        ),
        CheckConstraint("tech_sentiment_24h >= -1 AND tech_sentiment_24h <= 1", name="check_tech_sentiment_24h"),
        CheckConstraint("tech_discussion_depth >= 0", name="check_tech_discussion_depth"),
        CheckConstraint("mentions_24h >= 0", name="check_mentions_positive"),
        # virality_index is unsigned reach (Σengagement / sqrt(unique_authors+1)), unbounded
        # above, since the Rev 2 redefinition (sentiment-spec-rev2.md §2.5.5) -- was 0..1 pre-Rev2.
        CheckConstraint("virality_index >= 0", name="check_virality_range"),
        CheckConstraint("bot_pct >= 0 AND bot_pct <= 1", name="check_bot_pct_range"),
    )

    def __repr__(self):
        return f"<DeepScanMetrics(id={self.id}, ticker='{self.ticker}', date={self.date}, score={self.squeeze_score})>"

    def to_transient_metrics(self):
        """Convert to TransientMetrics dataclass."""
        # Import here to avoid circular imports
        from src.ml.pipeline.p04_short_squeeze.core.models import TransientMetrics

        if self.volume_spike is None or self.sentiment_24h is None:
            return None

        return TransientMetrics(
            volume_spike=self.volume_spike,
            call_put_ratio=self.call_put_ratio,
            sentiment_24h=self.sentiment_24h,
            borrow_fee_pct=self.borrow_fee_pct,
            mentions_24h=self.mentions_24h if self.mentions_24h is not None else 0,
            mentions_growth_7d=self.mentions_growth_7d,
            virality_index=self.virality_index if self.virality_index is not None else 0.0,
            bot_pct=self.bot_pct if self.bot_pct is not None else 0.0,
            sentiment_data_quality=self.sentiment_data_quality if self.sentiment_data_quality is not None else {},
            raw_payload=self.raw_payload if self.raw_payload is not None else {},
            tech_sentiment_24h=self.tech_sentiment_24h,
        )


class HnCorpusItem(Base):
    """
    Shared Hacker News corpus cache.

    Populated once per batch run by the shared-corpus fetch strategy (spec §2.4) and matched
    against every ticker's entity map in-process -- cost is O(corpus size), independent of how
    many tickers are being scanned. Re-running a scan must not re-fetch item IDs already present
    here within the configured TTL.
    """

    __tablename__ = "ss_hn_corpus"

    item_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    item_type: Mapped[str] = mapped_column(String(10))  # "story" | "comment" | "job" | "poll"
    parent_id: Mapped[int | None] = mapped_column(BigInteger)
    story_id: Mapped[int | None] = mapped_column(BigInteger)
    author_hash: Mapped[str | None] = mapped_column(String(64))  # salted SHA-256, see §2.11
    created_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    text_clean: Mapped[str | None] = mapped_column(Text)
    score: Mapped[int | None] = mapped_column(Integer)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())

    __table_args__ = (
        Index("idx_ss_hn_corpus_created_utc", "created_utc"),
        Index("idx_ss_hn_corpus_story_id", "story_id"),
        Index("idx_ss_hn_corpus_fetched_at", "fetched_at"),
        CheckConstraint("item_type IN ('story', 'comment', 'job', 'poll')", name="check_hn_item_type"),
    )

    def __repr__(self):
        return f"<HnCorpusItem(item_id={self.item_id}, type='{self.item_type}', story_id={self.story_id})>"


class SentimentCalibration(Base):
    """
    Per-source, per-day sentiment score distribution used to z-score calibrate raw scores before
    blending (spec §2.5.6).

    Raw scores aren't comparable across platforms -- Bluesky finance chatter skews
    promotional-positive, Hacker News skews critical-negative. One row is written per
    (provider, day) after each batch run; a trailing window (default 30 days) of rows is pooled
    into one distribution at read time (see ``processing/calibration.py``).
    """

    __tablename__ = "ss_sentiment_calibration"

    provider: Mapped[str] = mapped_column(String(32), primary_key=True)
    day: Mapped[DateType] = mapped_column(Date, primary_key=True)
    mean_score: Mapped[float] = mapped_column(Numeric(10, 6))
    std_score: Mapped[float] = mapped_column(Numeric(10, 6))
    n_obs: Mapped[int] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_ss_sentiment_calibration_day", "day"),
        CheckConstraint("n_obs >= 0", name="check_calibration_n_obs_nonneg"),
        CheckConstraint("std_score >= 0", name="check_calibration_std_nonneg"),
    )

    def __repr__(self):
        return f"<SentimentCalibration(provider='{self.provider}', day={self.day}, n_obs={self.n_obs})>"


class SqueezeAlert(Base):
    """Alert history and cooldown tracking."""

    __tablename__ = "ss_alerts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True)
    alert_level: Mapped[str] = mapped_column(String(10))
    reason: Mapped[str | None] = mapped_column(Text)
    squeeze_score: Mapped[float | None] = mapped_column(Numeric(5, 4))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    sent: Mapped[bool] = mapped_column(Boolean, default=False)
    cooldown_expires: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    notification_id: Mapped[str | None] = mapped_column(String(50))

    __table_args__ = (
        Index("idx_ss_alerts_ticker_cooldown", "ticker", "cooldown_expires"),
        Index("idx_ss_alerts_timestamp_desc", "timestamp", postgresql_using="btree"),
        Index("idx_ss_alerts_alert_level_timestamp", "alert_level", "timestamp", postgresql_using="btree"),
        Index("idx_ss_alerts_sent_timestamp", "sent", "timestamp", postgresql_using="btree"),
        CheckConstraint("alert_level IN ('LOW', 'MEDIUM', 'HIGH')", name="check_alert_level"),
        CheckConstraint("squeeze_score >= 0 AND squeeze_score <= 1", name="check_squeeze_score"),
    )

    def __repr__(self):
        return f"<SqueezeAlert(id={self.id}, ticker='{self.ticker}', level='{self.alert_level}', sent={self.sent})>"

    def to_alert(self):
        """Convert to Alert dataclass."""
        # Import here to avoid circular imports
        from src.ml.pipeline.p04_short_squeeze.core.models import Alert

        return Alert(
            ticker=self.ticker,
            alert_level=AlertLevel(self.alert_level),
            reason=self.reason or "",
            squeeze_score=self.squeeze_score if self.squeeze_score is not None else 0.0,
            timestamp=self.timestamp,
            cooldown_expires=self.cooldown_expires,
            sent=self.sent,
            notification_id=self.notification_id,
        )


class AdHocCandidateModel(Base):
    """Ad-hoc candidate management."""

    __tablename__ = "ss_ad_hoc_candidates"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    ticker: Mapped[str] = mapped_column(String(10), unique=True)
    reason: Mapped[str | None] = mapped_column(Text)
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    promoted_by_screener: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (
        UniqueConstraint("ticker", name="unique_ticker"),
        Index("idx_ss_adhoc_active", "active", "expires_at"),
        Index("idx_ss_adhoc_expires_at", "expires_at"),
        Index("idx_ss_adhoc_promoted", "promoted_by_screener", "active"),
    )

    def __repr__(self):
        return f"<AdHocCandidateModel(id={self.id}, ticker='{self.ticker}', active={self.active})>"

    def to_adhoc_candidate(self):
        """Convert to AdHocCandidate dataclass."""
        # Import here to avoid circular imports
        from src.ml.pipeline.p04_short_squeeze.core.models import AdHocCandidate

        return AdHocCandidate(
            ticker=self.ticker,
            reason=self.reason or "",
            first_seen=self.first_seen,
            expires_at=self.expires_at,
            active=self.active,
            promoted_by_screener=self.promoted_by_screener,
        )
