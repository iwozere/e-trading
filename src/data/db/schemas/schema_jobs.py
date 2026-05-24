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
from typing import Any, Dict, List, Optional

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
    job_id: Optional[str] = None
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
    job_id: Optional[str]
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
    scheduled_for: Optional[dt] = None


class ScreenerRequest(BaseModel):
    """Pydantic model for screener execution requests."""

    screener_set: Optional[str] = Field(None, min_length=1, max_length=100)
    tickers: Optional[List[str]] = Field(None, min_length=1)
    filter_criteria: Dict[str, Any] = Field(default_factory=dict)
    top_n: Optional[int] = Field(None, ge=1, le=1000)
    scheduled_for: Optional[dt] = None

    @field_validator("tickers")
    @classmethod
    def validate_tickers_or_set(cls, v: Optional[List[str]], info: Any) -> Optional[List[str]]:
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
