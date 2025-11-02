"""
Short Squeeze Detection Pipeline Models

SQLAlchemy models for the short squeeze detection pipeline.
Includes ScreenerSnapshot, DeepScanMetrics, SqueezeAlert, and AdHocCandidateModel models.
"""

from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Optional

from sqlalchemy import (
    Column, BigInteger, String, Date, DateTime, Numeric, Boolean, Text,
    CheckConstraint, UniqueConstraint, Index, func
)
from sqlalchemy.dialects.postgresql import JSONB
from src.data.db.core.json_types import JsonType

from src.data.db.core.base import Base


class AlertLevel(str, Enum):
    """Alert level enumeration."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class CandidateSource(str, Enum):
    """Candidate source enumeration."""
    SCREENER = "screener"
    ADHOC = "adhoc"
    VOLUME_SCREENER = "volume_screener"
    HYBRID_SCREENER = "hybrid_screener"
    FINRA_SCREENER = "finra_screener"


class ScreenerSnapshot(Base):
    """Weekly screener snapshots with structural metrics."""

    __tablename__ = "ss_snapshot"

    id = Column(BigInteger, primary_key=True, index=True)
    ticker = Column(String(10), nullable=False, index=True)
    run_date = Column(Date, nullable=False, index=True)
    short_interest_pct = Column(Numeric(5, 4), nullable=True)
    days_to_cover = Column(Numeric(8, 2), nullable=True)
    float_shares = Column(BigInteger, nullable=True)
    avg_volume_14d = Column(BigInteger, nullable=True)
    market_cap = Column(BigInteger, nullable=True)
    screener_score = Column(Numeric(5, 4), nullable=True)
    raw_payload = Column(JsonType(), nullable=True)
    data_quality = Column(Numeric(3, 2), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())

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

        if not all([
            self.short_interest_pct is not None,
            self.days_to_cover is not None,
            self.float_shares is not None,
            self.avg_volume_14d is not None,
            self.market_cap is not None
        ]):
            return None

        return StructuralMetrics(
            short_interest_pct=float(self.short_interest_pct),
            days_to_cover=float(self.days_to_cover),
            float_shares=int(self.float_shares),
            avg_volume_14d=int(self.avg_volume_14d),
            market_cap=int(self.market_cap)
        )


class DeepScanMetrics(Base):
    """Daily deep scan metrics with transient data."""

    __tablename__ = "ss_deep_metrics"

    id = Column(BigInteger, primary_key=True, index=True)
    ticker = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    volume_spike = Column(Numeric(6, 2), nullable=True)
    call_put_ratio = Column(Numeric(6, 2), nullable=True)
    sentiment_24h = Column(Numeric(4, 3), nullable=True)
    borrow_fee_pct = Column(Numeric(5, 4), nullable=True)
    squeeze_score = Column(Numeric(5, 4), nullable=True)
    alert_level = Column(String(10), nullable=True)
    raw_payload = Column(JsonType(), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())

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
            volume_spike=float(self.volume_spike),
            call_put_ratio=float(self.call_put_ratio) if self.call_put_ratio is not None else None,
            sentiment_24h=float(self.sentiment_24h),
            borrow_fee_pct=float(self.borrow_fee_pct) if self.borrow_fee_pct is not None else None
        )


class SqueezeAlert(Base):
    """Alert history and cooldown tracking."""

    __tablename__ = "ss_alerts"

    id = Column(BigInteger, primary_key=True, index=True)
    ticker = Column(String(10), nullable=False, index=True)
    alert_level = Column(String(10), nullable=False)
    reason = Column(Text, nullable=True)
    squeeze_score = Column(Numeric(5, 4), nullable=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=func.now())
    sent = Column(Boolean, nullable=False, default=False)
    cooldown_expires = Column(DateTime(timezone=True), nullable=True)
    notification_id = Column(String(50), nullable=True)

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
            squeeze_score=float(self.squeeze_score) if self.squeeze_score is not None else 0.0,
            timestamp=self.timestamp,
            cooldown_expires=self.cooldown_expires,
            sent=self.sent,
            notification_id=self.notification_id
        )


class AdHocCandidateModel(Base):
    """Ad-hoc candidate management."""

    __tablename__ = "ss_ad_hoc_candidates"

    id = Column(BigInteger, primary_key=True, index=True)
    ticker = Column(String(10), nullable=False, unique=True)
    reason = Column(Text, nullable=True)
    first_seen = Column(DateTime(timezone=True), nullable=False, default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)
    active = Column(Boolean, nullable=False, default=True)
    promoted_by_screener = Column(Boolean, nullable=False, default=False)

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
            promoted_by_screener=self.promoted_by_screener
        )


class FINRAShortInterest(Base):
    """FINRA short interest data."""

    __tablename__ = "ss_finra_short_interest"

    id = Column(BigInteger, primary_key=True, index=True)
    ticker = Column(String(10), nullable=False, index=True)
    settlement_date = Column(Date, nullable=False, index=True)
    short_interest_shares = Column(BigInteger, nullable=False)
    total_shares_outstanding = Column(BigInteger, nullable=True)
    float_shares = Column(BigInteger, nullable=True)
    short_interest_pct = Column(Numeric(8, 4), nullable=True)
    days_to_cover = Column(Numeric(8, 2), nullable=True)
    data_source = Column(String(50), nullable=False, default='FINRA')
    data_quality_score = Column(Numeric(3, 2), nullable=False, default=1.0)
    raw_data = Column(JsonType(), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())

    __table_args__ = (
        UniqueConstraint("ticker", "settlement_date", name="unique_ticker_settlement"),
        Index("idx_ss_finra_ticker", "ticker"),
        Index("idx_ss_finra_settlement_date", "settlement_date"),
        Index("idx_ss_finra_ticker_date", "ticker", "settlement_date", postgresql_using="btree"),
        Index("idx_ss_finra_short_interest_pct", "short_interest_pct", postgresql_using="btree"),
        Index("idx_ss_finra_days_to_cover", "days_to_cover", postgresql_using="btree"),
        Index("idx_ss_finra_recent_data", "ticker", "settlement_date", "short_interest_pct",
              postgresql_where="settlement_date >= CURRENT_DATE - INTERVAL '90 days'"),
        CheckConstraint("short_interest_shares >= 0", name="check_short_interest_shares"),
        CheckConstraint("short_interest_pct >= 0 AND short_interest_pct <= 100", name="check_short_interest_pct"),
        CheckConstraint("days_to_cover >= 0", name="check_days_to_cover"),
        CheckConstraint("data_quality_score >= 0 AND data_quality_score <= 1", name="check_data_quality_score"),
    )

    def __repr__(self):
        return f"<FINRAShortInterest(id={self.id}, ticker='{self.ticker}', settlement_date={self.settlement_date}, short_interest_pct={self.short_interest_pct})>"

    def to_finra_metrics(self):
        """Convert to FINRAMetrics dataclass."""
        # Import here to avoid circular imports
        from src.ml.pipeline.p04_short_squeeze.core.models import FINRAMetrics
        from datetime import date

        data_age_days = (date.today() - self.settlement_date).days if self.settlement_date else 0

        return FINRAMetrics(
            short_interest_shares=int(self.short_interest_shares),
            short_interest_pct=float(self.short_interest_pct) if self.short_interest_pct is not None else None,
            settlement_date=self.settlement_date,
            days_to_cover=float(self.days_to_cover) if self.days_to_cover is not None else None,
            data_age_days=data_age_days
        )