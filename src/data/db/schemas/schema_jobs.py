"""
Job Scheduler Pydantic Schemas
------------------------------

Request/response schemas for the job scheduling API.
SQLAlchemy ORM models live in ``src.data.db.models.model_jobs``.

Keeping schemas separate from ORM models follows the single-responsibility
principle: ORM models describe the DB layer; Pydantic schemas describe the
API/service contract.
"""

from __future__ import annotations

import json
from datetime import datetime as dt
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Enums are owned by the ORM model module; imported here so consumers only
# need to import from one place.
from src.data.db.models.model_jobs import JobType, RunStatus

# ---------------------------------------------------------------------------
# Schedule schemas
# ---------------------------------------------------------------------------


class ScheduleCreate(BaseModel):
    """Pydantic model for creating a schedule."""

    name: str = Field(..., min_length=1, max_length=255)
    job_type: JobType
    target: str = Field(..., min_length=1, max_length=255)
    task_params: Dict[str, Any] = Field(default_factory=dict)
    cron: str = Field(..., min_length=1, max_length=100)
    enabled: bool = Field(default=True)

    @field_validator("cron")
    @classmethod
    def validate_cron(cls, v: str) -> str:
        """Basic cron validation — must be exactly 5 space-separated fields."""
        parts = v.strip().split()
        if len(parts) != 5:
            raise ValueError("Cron expression must have exactly 5 fields")
        return v


class ScheduleUpdate(BaseModel):
    """Pydantic model for updating a schedule."""

    name: str | None = Field(None, min_length=1, max_length=255)
    target: str | None = Field(None, min_length=1, max_length=255)
    task_params: Dict[str, Any] | None = None
    cron: str | None = Field(None, min_length=1, max_length=100)
    enabled: bool | None = None


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
    next_run_at: dt | None
    created_at: dt
    updated_at: dt
    state_json: Dict[str, Any] = {}
    model_config = ConfigDict(from_attributes=True)

    @field_validator("task_params", "state_json", mode="before")
    @classmethod
    def parse_json_strings(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                pass
        return v


# ---------------------------------------------------------------------------
# Schedule run schemas
# ---------------------------------------------------------------------------


class ScheduleRunCreate(BaseModel):
    """Pydantic model for creating a run."""

    job_type: JobType
    job_id: str | None = None
    scheduled_for: dt
    job_snapshot: Dict[str, Any] = Field(default_factory=dict)


class ScheduleRunUpdate(BaseModel):
    """Pydantic model for updating a run."""

    status: RunStatus | None = None
    started_at: dt | None = None
    finished_at: dt | None = None
    result: Dict[str, Any] | None = None
    error: str | None = None


class ScheduleRunResponse(BaseModel):
    """Pydantic model for run API responses."""

    id: int
    job_type: JobType
    job_id: str | None
    user_id: int | None
    status: RunStatus | None
    scheduled_for: dt | None
    enqueued_at: dt | None
    started_at: dt | None
    finished_at: dt | None
    job_snapshot: Dict[str, Any] | None
    result: Dict[str, Any] | None
    error: str | None
    model_config = ConfigDict(from_attributes=True)

    @field_validator("job_id", mode="before")
    @classmethod
    def coerce_job_id_to_str(cls, v: Any) -> str | None:
        """DB column is BigInteger; coerce to str until a migration converts it to VARCHAR."""
        if v is None:
            return None
        return str(v)

    @field_validator("job_snapshot", "result", mode="before")
    @classmethod
    def parse_json_strings(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                pass
        return v


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------


class ReportRequest(BaseModel):
    """Pydantic model for report execution requests."""

    report_type: str = Field(..., min_length=1, max_length=100)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    scheduled_for: dt | None = None


class ScreenerRequest(BaseModel):
    """Pydantic model for screener execution requests."""

    screener_set: str | None = Field(None, min_length=1, max_length=100)
    tickers: List[str] | None = Field(None, min_length=1)
    filter_criteria: Dict[str, Any] = Field(default_factory=dict)
    top_n: int | None = Field(None, ge=1, le=1000)
    scheduled_for: dt | None = None

    @field_validator("tickers")
    @classmethod
    def validate_tickers_or_set(cls, v: List[str] | None, info: Any) -> List[str] | None:
        """Ensure either screener_set or tickers is provided."""
        if not v and not (info.data or {}).get("screener_set"):
            raise ValueError("Either screener_set or tickers must be provided")
        return v


class ScreenerSetInfo(BaseModel):
    """Pydantic model for screener set information."""

    name: str
    description: str
    ticker_count: int
    tickers: List[str]
    categories: List[str]
