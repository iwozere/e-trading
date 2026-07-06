"""
Job Scheduler Models

SQLAlchemy models for the job scheduling and execution system.
Includes Schedule and Run models with proper relationships and validation.
"""

from __future__ import annotations

from datetime import datetime as dt
from enum import StrEnum

from sqlalchemy import Boolean, CheckConstraint, DateTime, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from src.data.db.core.base import Base
from src.data.db.core.json_types import JsonType

## For PostgreSQL we prefer the native JSONB type


class JobType(StrEnum):
    """Job type enumeration."""

    SCHEDULE = "schedule"
    SCREENER = "screener"
    ALERT = "alert"
    NOTIFICATION = "notification"
    DATA_PROCESSING = "data_processing"
    BACKUP = "backup"
    REPORT = "report"
    SCRIPT = "script"


class RunStatus(StrEnum):
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
    state_json: Mapped[dict] = mapped_column(JsonType(), default={})

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "job_type IN ('report', 'screener', 'alert', 'notification', 'data_processing', 'backup', 'script', 'schedule')",
            name="check_job_type",
        ),
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
    job_id: Mapped[str | None] = mapped_column(String(255))
    user_id: Mapped[int | None] = mapped_column(Integer)
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


# ---------------------------------------------------------------------------
# Pydantic schemas — defined in src.data.db.schemas.schema_jobs
# Re-exported here for backward compatibility with existing imports.
# ---------------------------------------------------------------------------
from src.data.db.schemas.schema_jobs import (  # noqa: E402
    ReportRequest,
    ScheduleCreate,
    ScheduleResponse,
    ScheduleRunCreate,
    ScheduleRunResponse,
    ScheduleRunUpdate,
    ScheduleUpdate,
    ScreenerRequest,
    ScreenerSetInfo,
)

__all__ = [
    # ORM models
    "Schedule",
    "ScheduleRun",
    # Enums
    "JobType",
    "RunStatus",
    # Pydantic schemas (re-exported from schemas.schema_jobs)
    "ScheduleCreate",
    "ScheduleUpdate",
    "ScheduleResponse",
    "ScheduleRunCreate",
    "ScheduleRunUpdate",
    "ScheduleRunResponse",
    "ReportRequest",
    "ScreenerRequest",
    "ScreenerSetInfo",
]
