"""
P20 Kestrel — Job schedule registration (Python fallback).

Prefer the canonical SQL script for initial deploy:
    psql -d <db> < bin/scheduler/insert_p20_schedules.sql

This Python version is kept as a fallback for environments where psql is
unavailable. It uses ON CONFLICT DO NOTHING so it is safe to re-run.

Each job targets a run_*.py script under src/ml/pipeline/p20_kestrel/jobs/.
Scheduler runs scripts as subprocesses via _execute_data_processing_job.
Scripts must print __SCHEDULER_RESULT__:{json} on success.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.data.db.core.database import session_scope
from src.data.db.models.model_jobs import Schedule
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# Scheduler owner user_id for P20 system jobs
_SYSTEM_USER_ID = int(os.getenv("SCHEDULER_SYSTEM_USER_ID", "1"))

_SCRIPT_BASE = "src/ml/pipeline/p20_kestrel/jobs"

# Each entry: (name, cron_utc, script_filename, enabled)
_JOB_SPECS: List[Dict[str, Any]] = [
    # Morning chain
    {"name": "p20_data_health", "cron": "0 6 * * 1-5", "script": "run_data_health.py", "enabled": True},
    {"name": "p20_gdelt_process", "cron": "15 6 * * 1-5", "script": "run_gdelt_process.py", "enabled": True},
    {"name": "p20_social_poll", "cron": "30 6 * * 1-5", "script": "run_social_poll.py", "enabled": True},
    {"name": "p20_av_sentiment", "cron": "45 6 * * 1-5", "script": "run_av_sentiment.py", "enabled": True},
    {"name": "p20_sentiment_aggregate", "cron": "0 7 * * 1-5", "script": "run_sentiment_aggregate.py", "enabled": True},
    {"name": "p20_digest_send", "cron": "30 6 * * 1-5", "script": "run_digest_send.py", "enabled": True},
    # EOD ingest
    {"name": "p20_ingest_eod", "cron": "0 20 * * 1-5", "script": "run_ingest_eod.py", "enabled": True},
    {"name": "p20_ingest_filings", "cron": "30 20 * * 1-5", "script": "run_ingest_filings.py", "enabled": True},
    {"name": "p20_calendar_sync", "cron": "45 20 * * 1-5", "script": "run_catalyst_sync.py", "enabled": True},
    # Screening
    {"name": "p20_screen_turnaround", "cron": "0 21 * * 1-5", "script": "run_screen_turnaround.py", "enabled": True},
    {"name": "p20_screen_spinoffs", "cron": "15 21 * * 1-5", "script": "run_screen_spinoffs.py", "enabled": True},
    {"name": "p20_momentum_rank", "cron": "30 21 * * 1-5", "script": "run_momentum_rank.py", "enabled": True},
    # LLM
    {
        "name": "p20_llm_classify_filings",
        "cron": "0 22 * * 1-5",
        "script": "run_llm_classify_filings.py",
        "enabled": True,
    },
    {"name": "p20_llm_dossiers", "cron": "30 22 * * 1-5", "script": "run_llm_dossiers.py", "enabled": True},
    # Risk
    {"name": "p20_risk_check", "cron": "*/30 9-17 * * 1-5", "script": "run_risk_check.py", "enabled": True},
    # LLM (weekly — 10-K/Q filings change quarterly)
    {"name": "p20_llm_risk_diff", "cron": "0 18 * * 0", "script": "run_llm_risk_diff.py", "enabled": True},
    # Maintenance
    {"name": "p20_weekly_maintenance", "cron": "0 5 * * 1", "script": "run_weekly_maintenance.py", "enabled": True},
    {"name": "p20_trends_watchlist", "cron": "0 3 * * 1-5", "script": "run_trends_watchlist.py", "enabled": True},
    {"name": "p20_weekly_report", "cron": "0 17 * * 0", "script": "run_weekly_report.py", "enabled": True},
]


def run() -> Dict[str, Any]:
    """
    Insert all P20 job schedule rows idempotently.

    Returns:
        Summary dict with inserted count.
    """
    _logger.info("Registering %d P20 job schedules", len(_JOB_SPECS))  # expect 19
    count = 0

    with session_scope() as s:
        for spec in _JOB_SPECS:
            script_path = f"{_SCRIPT_BASE}/{spec['script']}"
            existing = s.query(Schedule).filter_by(user_id=_SYSTEM_USER_ID, name=spec["name"]).first()

            if existing:
                existing.target = script_path
                existing.task_params = {"script_path": script_path}
                existing.cron = spec["cron"]
                existing.enabled = spec["enabled"]
                _logger.debug("Updated existing schedule: %s (%s)", spec["name"], spec["cron"])
            else:
                new_schedule = Schedule(
                    user_id=_SYSTEM_USER_ID,
                    name=spec["name"],
                    job_type="script",
                    target=script_path,
                    task_params={"script_path": script_path},
                    cron=spec["cron"],
                    enabled=spec["enabled"],
                    state_json={},
                )
                s.add(new_schedule)
                _logger.debug("Registered new schedule: %s (%s)", spec["name"], spec["cron"])
            count += 1

    _logger.info("Job registration complete: %d rows", count)
    return {"jobs_registered": count}


if __name__ == "__main__":
    result = run()
    print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
