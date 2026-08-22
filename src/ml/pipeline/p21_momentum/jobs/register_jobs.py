"""
P21 Momentum — Job schedule registration.

Idempotent INSERT (ON CONFLICT DO NOTHING via SQLAlchemy upsert-or-update
below) into job_schedules, same pattern as P20 Kestrel's
jobs/register_jobs.py. Each job targets a run_*.py script under
src/ml/pipeline/p21_momentum/jobs/; the scheduler runs scripts as
subprocesses and expects __SCHEDULER_RESULT__:{json} on success.

**Open Decision #2 resolution (docs/implementation-plan.md §10):**
job_schedules.cron is a plain 5-field croniter expression (confirmed via
src/scheduler/scheduler_service.py) with no native "last/first trading day
of month" concept. All three P21 jobs therefore run on a **daily** cron and
self-guard internally via src.ml.pipeline.p21_momentum.calendar — each
job's run() no-ops immediately unless today is the specific day it cares
about (see each job module's docstring). This is the safer option per the
plan's stated default: no dependency on undocumented scheduler internals.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.core.database import session_scope
from src.data.db.models.model_jobs import Schedule
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_SYSTEM_USER_ID = int(os.getenv("SCHEDULER_SYSTEM_USER_ID", "1"))
_SCRIPT_BASE = "src/ml/pipeline/p21_momentum/jobs"

# Each entry: (name, cron_utc, script_filename). All three run daily and
# self-guard — see module docstring.
_JOB_SPECS: List[Dict[str, Any]] = [
    # 16:30 ET == 20:30 UTC (EST) / 21:30 UTC (EDT); scheduled at 20:30 UTC,
    # the winter-time value — DST drift is a known limitation shared with
    # every other cron-based ET job in this repo (see P20's jobs for the
    # same convention) and is not solved here.
    {"name": "p21_monthly_rebalance", "cron": "30 20 * * 1-5", "script": "run_monthly_rebalance.py"},
    {"name": "p21_monthly_execute", "cron": "45 13 * * 1-5", "script": "run_monthly_execute.py"},  # 09:45 ET
    {"name": "p21_daily_mark", "cron": "30 20 * * 1-5", "script": "run_daily_mark.py"},
]


def run() -> Dict[str, Any]:
    """
    Insert or update all P21 job schedule rows idempotently.

    Returns:
        Summary dict with inserted/updated count.
    """
    _logger.info("Registering %d P21 job schedules", len(_JOB_SPECS))
    count = 0

    with session_scope() as s:
        for spec in _JOB_SPECS:
            script_path = f"{_SCRIPT_BASE}/{spec['script']}"
            task_params = {"script_path": script_path}

            existing = s.query(Schedule).filter_by(user_id=_SYSTEM_USER_ID, name=spec["name"]).first()
            if existing:
                existing.target = script_path
                existing.task_params = task_params
                existing.cron = spec["cron"]
                existing.enabled = True
                _logger.debug("Updated existing schedule: %s (%s)", spec["name"], spec["cron"])
            else:
                s.add(
                    Schedule(
                        user_id=_SYSTEM_USER_ID,
                        name=spec["name"],
                        job_type="script",
                        target=script_path,
                        task_params=task_params,
                        cron=spec["cron"],
                        enabled=True,
                        state_json={},
                    )
                )
                _logger.debug("Registered new schedule: %s (%s)", spec["name"], spec["cron"])
            count += 1

    _logger.info("P21 job registration complete: %d rows", count)
    return {"jobs_registered": count}


if __name__ == "__main__":
    result = run()
    print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
