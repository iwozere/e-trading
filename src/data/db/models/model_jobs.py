"""
Job Scheduler Models

SQLAlchemy models for the job scheduling and execution system.
Includes Schedule and Run models with proper relationships and validation.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from uuid import UUID, uuid4

from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, Text, BigInteger,
    CheckConstraint, UniqueConstraint, Index, func, JSON
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PostgresUUID
from sqlalchemy.dialects import postgresql, sqlite
from pydantic import BaseModel, Field, field_validator, ConfigDict

from src.data.db.core.base import Base


# Database-agnostic JSON type
def get_json_type():
    """Get appropriate JSON type based on database dialect."""
    return JSON().with_variant(postgresql.JSONB(), 'postgresql').with_variant(sqlite.JSON(), 'sqlite')




class JobType(str, Enum):
    """Job type enumeration."""
    SCHEDULE = "schedule"
    SCREENER = "screener"
    ALERT = "alert"
    NOTIFICATION = "notification"
    DATA_PROCESSING = "data_processing"
    BACKUP = "backup"


class RunStatus(str, Enum):
    """Run status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Schedule(Base):
    """Schedule model for persistent schedule definitions."""

    __tablename__ = "job_schedules"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    job_type = Column(String(50), nullable=False)
    target = Column(String(255), nullable=False)
    task_params = Column(get_json_type(), nullable=False, default={})
    cron = Column(String(100), nullable=False)
    enabled = Column(Boolean, nullable=False, default=True, index=True)
    next_run_at = Column(DateTime(timezone=True), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())

    # Constraints
    __table_args__ = (
        CheckConstraint("job_type IN ('report', 'screener', 'alert', 'notification', 'data_processing', 'backup')", name="check_job_type"),
        UniqueConstraint("user_id", "name", name="unique_user_schedule_name"),
        Index("idx_schedules_enabled", "enabled"),
        Index("idx_schedules_next_run_at", "next_run_at", postgresql_where="enabled = true"),
    )

    def __repr__(self):
        return f"<Schedule(id={self.id}, name='{self.name}', job_type='{self.job_type}', enabled={self.enabled})>"


class ScheduleRun(Base):
    """Run model for job execution history with snapshots."""

    __tablename__ = "job_schedule_runs"

    id = Column(Integer, primary_key=True, index=True)
    job_type = Column(Text, nullable=False)
    job_id = Column(BigInteger, nullable=True)
    user_id = Column(BigInteger, nullable=True, index=True)
    status = Column(Text, nullable=True, index=True)
    scheduled_for = Column(DateTime(timezone=True), nullable=True, index=True)
    enqueued_at = Column(DateTime(timezone=True), nullable=True, default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    job_snapshot = Column(get_json_type(), nullable=True)
    result = Column(get_json_type(), nullable=True)
    error = Column(Text, nullable=True)
    worker_id = Column(String(255), nullable=True)

    # Constraints
    __table_args__ = (
        UniqueConstraint("job_type", "job_id", "scheduled_for", name="ux_runs_job_scheduled_for"),
    )

    def __repr__(self):
        return f"<ScheduleRun(id={self.id}, job_type='{self.job_type}', status='{self.status}')>"



# Pydantic models for API validation
class ScheduleCreate(BaseModel):
    """Pydantic model for creating a schedule."""
    name: str = Field(..., min_length=1, max_length=255)
    job_type: JobType
    target: str = Field(..., min_length=1, max_length=255)
    task_params: Dict[str, Any] = Field(default_factory=dict)
    cron: str = Field(..., min_length=1, max_length=100)
    enabled: bool = Field(default=True)

    @field_validator('cron')
    def validate_cron(cls, v):
        """Basic cron validation - should be 5 fields separated by spaces."""
        parts = v.strip().split()
        if len(parts) != 5:
            raise ValueError('Cron expression must have exactly 5 fields')
        return v


class ScheduleUpdate(BaseModel):
    """Pydantic model for updating a schedule."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    target: Optional[str] = Field(None, min_length=1, max_length=255)
    task_params: Optional[Dict[str, Any]] = None
    cron: Optional[str] = Field(None, min_length=1, max_length=100)
    enabled: Optional[bool] = None


class ScheduleResponse(BaseModel):
    """Pydantic model for schedule API responses."""
    id: int
    user_id: int
    name: str
    job_type: JobType
    target: str
    task_params: Dict[str, Any]
    cron: str
    enabled: bool
    next_run_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)


class ScheduleRunCreate(BaseModel):
    """Pydantic model for creating a run."""
    job_type: JobType
    job_id: Optional[int] = None
    scheduled_for: datetime
    job_snapshot: Dict[str, Any] = Field(default_factory=dict)


class ScheduleRunUpdate(BaseModel):
    """Pydantic model for updating a run."""
    status: Optional[RunStatus] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    worker_id: Optional[str] = None


class ScheduleRunResponse(BaseModel):
    """Pydantic model for run API responses."""
    run_id: UUID
    job_type: JobType
    job_id: Optional[int]
    user_id: Optional[int]
    status: Optional[RunStatus]
    scheduled_for: Optional[datetime]
    enqueued_at: Optional[datetime]
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    job_snapshot: Optional[Dict[str, Any]]
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    worker_id: Optional[str]
    model_config = ConfigDict(from_attributes=True)

class ReportRequest(BaseModel):
    """Pydantic model for report execution requests."""
    report_type: str = Field(..., min_length=1, max_length=100)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    scheduled_for: Optional[datetime] = None


class ScreenerRequest(BaseModel):
    """Pydantic model for screener execution requests."""
    screener_set: Optional[str] = Field(None, min_length=1, max_length=100)
    tickers: Optional[list[str]] = Field(None, min_length=1)
    filter_criteria: Dict[str, Any] = Field(default_factory=dict)
    top_n: Optional[int] = Field(None, ge=1, le=1000)
    scheduled_for: Optional[datetime] = None

    @field_validator('tickers')
    def validate_tickers_or_set(cls, v, values):
        """Ensure either screener_set or tickers is provided."""
        if not v and not values.get('screener_set'):
            raise ValueError('Either screener_set or tickers must be provided')
        return v


class ScreenerSetInfo(BaseModel):
    """Pydantic model for screener set information."""
    name: str
    description: str
    ticker_count: int
    tickers: list[str]
    categories: list[str]