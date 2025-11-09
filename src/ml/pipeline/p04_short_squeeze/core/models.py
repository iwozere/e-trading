"""
Short Squeeze Detection Pipeline Core Models

Business logic dataclasses and Pydantic models for the short squeeze detection pipeline.
These models are used for data validation and business logic, separate from database models.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# Import enums from database models to avoid duplication
from src.data.db.models.model_short_squeeze import AlertLevel, CandidateSource
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


# Business logic dataclasses
@dataclass
class StructuralMetrics:
    """Structural metrics for short squeeze analysis."""
    short_interest_pct: float
    days_to_cover: float
    float_shares: int
    avg_volume_14d: int
    market_cap: int

    def __post_init__(self):
        """Validate structural metrics after initialization."""
        if self.short_interest_pct < 0 or self.short_interest_pct > 1:
            raise ValueError("Short interest percentage must be between 0 and 1")
        if self.days_to_cover < 0:
            raise ValueError("Days to cover must be non-negative")
        if self.float_shares <= 0:
            raise ValueError("Float shares must be positive")
        if self.avg_volume_14d <= 0:
            raise ValueError("Average volume must be positive")
        if self.market_cap <= 0:
            raise ValueError("Market cap must be positive")


@dataclass
class TransientMetrics:
    """Transient metrics for short squeeze analysis."""
    volume_spike: float
    call_put_ratio: Optional[float]
    sentiment_24h: float
    borrow_fee_pct: Optional[float]

    def __post_init__(self):
        """Validate transient metrics after initialization."""
        if self.volume_spike < 0:
            raise ValueError("Volume spike must be non-negative")
        if self.call_put_ratio is not None and self.call_put_ratio < 0:
            raise ValueError("Call/put ratio must be non-negative")
        if self.sentiment_24h < -1 or self.sentiment_24h > 1:
            raise ValueError("Sentiment must be between -1 and 1")
        if self.borrow_fee_pct is not None and self.borrow_fee_pct < 0:
            raise ValueError("Borrow fee percentage must be non-negative")


@dataclass
class Candidate:
    """Short squeeze candidate."""
    ticker: str
    screener_score: float
    structural_metrics: StructuralMetrics
    last_updated: datetime
    source: CandidateSource = CandidateSource.SCREENER

    def __post_init__(self):
        """Validate candidate after initialization."""
        if not self.ticker or len(self.ticker.strip()) == 0:
            raise ValueError("Ticker cannot be empty")
        if self.screener_score < 0 or self.screener_score > 1:
            raise ValueError("Screener score must be between 0 and 1")
        self.ticker = self.ticker.upper().strip()


@dataclass
class ScoredCandidate:
    """Candidate with transient metrics and final squeeze score."""
    candidate: Candidate
    transient_metrics: TransientMetrics
    squeeze_score: float
    alert_level: Optional[AlertLevel] = None

    def __post_init__(self):
        """Validate scored candidate after initialization."""
        if self.squeeze_score < 0 or self.squeeze_score > 1:
            raise ValueError("Squeeze score must be between 0 and 1")


@dataclass
class Alert:
    """Short squeeze alert."""
    ticker: str
    alert_level: AlertLevel
    reason: str
    squeeze_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    cooldown_expires: Optional[datetime] = None
    sent: bool = False
    notification_id: Optional[str] = None

    def __post_init__(self):
        """Validate alert after initialization."""
        if not self.ticker or len(self.ticker.strip()) == 0:
            raise ValueError("Ticker cannot be empty")
        if self.squeeze_score < 0 or self.squeeze_score > 1:
            raise ValueError("Squeeze score must be between 0 and 1")
        self.ticker = self.ticker.upper().strip()


@dataclass
class AdHocCandidate:
    """Ad-hoc candidate for manual monitoring."""
    ticker: str
    reason: str
    first_seen: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    active: bool = True
    promoted_by_screener: bool = False

    def __post_init__(self):
        """Validate ad-hoc candidate after initialization."""
        if not self.ticker or len(self.ticker.strip()) == 0:
            raise ValueError("Ticker cannot be empty")
        self.ticker = self.ticker.upper().strip()


# Pydantic models for API validation and serialization
class StructuralMetricsCreate(BaseModel):
    """Pydantic model for creating structural metrics."""
    short_interest_pct: float = Field(..., ge=0, le=1)
    days_to_cover: float = Field(..., ge=0)
    float_shares: int = Field(..., gt=0)
    avg_volume_14d: int = Field(..., gt=0)
    market_cap: int = Field(..., gt=0)


class TransientMetricsCreate(BaseModel):
    """Pydantic model for creating transient metrics."""
    volume_spike: float = Field(..., ge=0)
    call_put_ratio: Optional[float] = Field(None, ge=0)
    sentiment_24h: float = Field(..., ge=-1, le=1)
    borrow_fee_pct: Optional[float] = Field(None, ge=0)


class CandidateCreate(BaseModel):
    """Pydantic model for creating a candidate."""
    ticker: str = Field(..., min_length=1, max_length=10)
    screener_score: float = Field(..., ge=0, le=1)
    structural_metrics: StructuralMetricsCreate
    source: CandidateSource = Field(default=CandidateSource.SCREENER)

    @field_validator('ticker')
    def validate_ticker(cls, v):
        """Validate and normalize ticker."""
        if not v or not v.strip():
            raise ValueError('Ticker cannot be empty')
        return v.upper().strip()


class ScoredCandidateCreate(BaseModel):
    """Pydantic model for creating a scored candidate."""
    candidate: CandidateCreate
    transient_metrics: TransientMetricsCreate
    squeeze_score: float = Field(..., ge=0, le=1)
    alert_level: Optional[AlertLevel] = None


class AlertCreate(BaseModel):
    """Pydantic model for creating an alert."""
    ticker: str = Field(..., min_length=1, max_length=10)
    alert_level: AlertLevel
    reason: str = Field(..., min_length=1)
    squeeze_score: float = Field(..., ge=0, le=1)
    cooldown_expires: Optional[datetime] = None

    @field_validator('ticker')
    def validate_ticker(cls, v):
        """Validate and normalize ticker."""
        if not v or not v.strip():
            raise ValueError('Ticker cannot be empty')
        return v.upper().strip()


class AdHocCandidateCreate(BaseModel):
    """Pydantic model for creating an ad-hoc candidate."""
    ticker: str = Field(..., min_length=1, max_length=10)
    reason: str = Field(..., min_length=1)
    expires_at: Optional[datetime] = None

    @field_validator('ticker')
    def validate_ticker(cls, v):
        """Validate and normalize ticker."""
        if not v or not v.strip():
            raise ValueError('Ticker cannot be empty')
        return v.upper().strip()


# Response models for API serialization
# Note: For API responses, use the SQLAlchemy models directly with Pydantic's from_attributes=True
# These models are kept here only if additional business logic validation is needed beyond the database models