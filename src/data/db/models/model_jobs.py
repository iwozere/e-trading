"""
Job Scheduler Models

SQLAlchemy models for the job scheduling and execution system.
Includes Schedule and Run models with proper relationships and validation.
"""

from __future__ import annotations
import datetime
from datetime import datetime as dt
from enum import Enum
from typing import Optional, Dict, Any

from sqlalchemy import (
    Integer, String, Boolean, DateTime, Text, BigInteger,
    CheckConstraint, UniqueConstraint, Index, func
)
from sqlalchemy.orm import Mapped, mapped_column
from src.data.db.core.json_types import JsonType
from pydantic import BaseModel, Field, field_validator, ConfigDict

from src.data.db.core.base import Base


## For PostgreSQL we prefer the native JSONB type




class JobType(str, Enum):
    """Job type enumeration."""
    SCHEDULE = "schedule"
    SCREENER = "screener"
    ALERT = "alert"
    NOTIFICATION = "notification"
    DATA_PROCESSING = "data_processing"
    BACKUP = "backup"
    REPORT = "report"


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

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255))
    job_type: Mapped[str] = mapped_column(String(50))
    target: Mapped[str] = mapped_column(String(255))
    task_params: Mapped[dict] = mapped_column(JsonType(), default={})
    cron: Mapped[str] = mapped_column(String(100))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    next_run_at: Mapped[dt | None] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[dt] = mapped_column(DateTime(timezone=True), default=func.now())
    updated_at: Mapped[dt] = mapped_column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    # Constraints
    __table_args__ = (
        CheckConstraint("job_type IN ('report', 'screener', 'alert', 'notification', 'data_processing', 'backup')", name="check_job_type"),
        # Use an explicit unique constraint for user_id + name
        UniqueConstraint("user_id", "name", name="unique_user_schedule_name"),
        Index("idx_schedules_enabled", "enabled"),
        Index("idx_schedules_next_run_at", "next_run_at"),
    )

    def __repr__(self):
        return f"<Schedule(id={self.id}, name='{self.name}', job_type='{self.job_type}', enabled={self.enabled})>"


class ScheduleRun(Base):
    """Run model for job execution history with snapshots."""

    __tablename__ = "job_schedule_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    job_type: Mapped[str] = mapped_column(Text)
    job_id: Mapped[int | None] = mapped_column(BigInteger)
    user_id: Mapped[int | None] = mapped_column(BigInteger)
    status: Mapped[str | None] = mapped_column(Text)
    scheduled_for: Mapped[dt | None] = mapped_column(DateTime(timezone=True))
    enqueued_at: Mapped[dt | None] = mapped_column(DateTime(timezone=True), default=func.now())
    started_at: Mapped[dt | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[dt | None] = mapped_column(DateTime(timezone=True))
    job_snapshot: Mapped[dict | None] = mapped_column(JsonType())
    result: Mapped[dict | None] = mapped_column(JsonType())
    error: Mapped[str | None] = mapped_column(Text)

    # Constraints - using Index instead of UniqueConstraint for better SQLite compatibility
    __table_args__ = (
        # Unique constraint on job_type + job_id + scheduled_for
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
    next_run_at: Optional[dt]
    created_at: dt
    updated_at: dt
    model_config = ConfigDict(from_attributes=True)


class ScheduleRunCreate(BaseModel):
    """Pydantic model for creating a run."""
    job_type: JobType
    job_id: Optional[int] = None
    scheduled_for: dt
    job_snapshot: Dict[str, Any] = Field(default_factory=dict)


class ScheduleRunUpdate(BaseModel):
    """Pydantic model for updating a run."""
    status: Optional[RunStatus] = None
    started_at: Optional[dt] = None
    finished_at: Optional[dt] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ScheduleRunResponse(BaseModel):
    """Pydantic model for run API responses."""
    id: int
    job_type: JobType
    job_id: Optional[int]
    user_id: Optional[int]
    status: Optional[RunStatus]
    scheduled_for: Optional[dt]
    enqueued_at: Optional[dt]
    started_at: Optional[dt]
    finished_at: Optional[dt]
    job_snapshot: Optional[Dict[str, Any]]
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    model_config = ConfigDict(from_attributes=True)

class ReportRequest(BaseModel):
    """Pydantic model for report execution requests."""
    report_type: str = Field(..., min_length=1, max_length=100)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    scheduled_for: Optional[dt] = None


class ScreenerRequest(BaseModel):
    """Pydantic model for screener execution requests."""
    screener_set: Optional[str] = Field(None, min_length=1, max_length=100)
    tickers: Optional[list[str]] = Field(None, min_length=1)
    filter_criteria: Dict[str, Any] = Field(default_factory=dict)
    top_n: Optional[int] = Field(None, ge=1, le=1000)
    scheduled_for: Optional[dt] = None

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