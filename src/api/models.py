"""
Web UI Backend Models
--------------------

Re-exports of data models used by the web UI backend.
This module provides a centralized import location for models
used throughout the web UI backend.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Re-export models from the main data layer
from src.data.db.models.model_jobs import (
    JobType,
    ReportRequest,
    RunStatus,
    Schedule,
    ScheduleCreate,
    ScheduleResponse,
    ScheduleRun,
    ScheduleRunCreate,
    ScheduleRunResponse,
    ScheduleRunUpdate,
    ScheduleUpdate,
    ScreenerRequest,
    ScreenerSetInfo,
)
from src.data.db.models.model_users import User
from src.data.db.models.model_webui import WebUIAuditLog

# Export all models
__all__ = [
    "User",
    "WebUIAuditLog",
    "Schedule",
    "ScheduleRun",
    "JobType",
    "RunStatus",
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
