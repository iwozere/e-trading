"""
Job Scheduler Models

SQLAlchemy models for the job scheduling and execution system.
Includes Schedule and Run models with proper relationships and validation.
"""

from __future__ import annotations

from datetime import datetime as dt
from enum import StrEnum

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql.elements import conv

from src.data.db.core.base import Base
from src.data.db.core.json_types import JsonText, JsonType


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

    id: Mapped[int] = mapped_column(Integer, Identity(always=True), primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(255))
    job_type: Mapped[str] = mapped_column(String(50))  # validated by JobType on the Python side
    target: Mapped[str] = mapped_column(String(255))
    task_params: Mapped[dict] = mapped_column(JsonType(), default=dict, server_default=text("'{}'::jsonb"))
    cron: Mapped[str] = mapped_column(String(100))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, server_default=text("true"))
    next_run_at: Mapped[dt | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[dt] = mapped_column(DateTime(timezone=True), default=func.now(), server_default=func.now())
    updated_at: Mapped[dt] = mapped_column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), server_default=func.now()
    )
    # TEXT in prod (pre-JSONB legacy column); JsonText stores dicts as JSON text and reads them back as dicts
    state_json: Mapped[dict | None] = mapped_column(JsonText(), default=dict, server_default=text("'{}'::text"))

    # Constraints
    __table_args__ = (UniqueConstraint("user_id", "name", name="unique_user_schedule_name"),)

    def __repr__(self):
        return f"<Schedule(id={self.id}, name='{self.name}', job_type='{self.job_type}', enabled={self.enabled})>"


class ScheduleRun(Base):
    """Run model for job execution history with snapshots."""

    __tablename__ = "job_schedule_runs"

    id: Mapped[int] = mapped_column(Integer, Identity(always=True), primary_key=True)
    job_type: Mapped[str] = mapped_column(Text)
    # The schedule this run belongs to; NULL for ad-hoc runs (API-triggered reports / screeners)
    job_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("job_schedules.id", ondelete="CASCADE", name=conv("job_schedule_runs_job_id_fkey"))
    )
    user_id: Mapped[int | None] = mapped_column(BigInteger)
    status: Mapped[str | None] = mapped_column(Text)
    scheduled_for: Mapped[dt | None] = mapped_column(DateTime(timezone=True))
    enqueued_at: Mapped[dt | None] = mapped_column(DateTime(timezone=True), default=func.now(), server_default=func.now())
    started_at: Mapped[dt | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[dt | None] = mapped_column(DateTime(timezone=True))
    job_snapshot: Mapped[dict | None] = mapped_column(JsonType())
    result: Mapped[dict | None] = mapped_column(JsonType())
    error: Mapped[str | None] = mapped_column(Text)

    # Prod has this as a unique index (not a constraint): one run per job and scheduled time
    __table_args__ = (Index("ux_runs_job_scheduled_for", "job_type", "job_id", "scheduled_for", unique=True),)

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
