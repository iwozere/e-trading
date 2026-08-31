"""
P22 Biotech M&A — Job schedule registration.

Mirrors src/ml/pipeline/p20_kestrel/jobs/register_jobs.py exactly: upserts
rows into job_schedules keyed on (user_id, name), idempotent. Each job
targets a run_*.py script under src/ml/pipeline/p22_biotech_ma/jobs/.
Scripts must print __SCHEDULER_RESULT__:{json} on success.
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

# Scheduler owner user_id for P22 system jobs — matches the P20 convention
# (see that module's docstring for the (user_id, name) collision history to
# avoid repeating).
_SYSTEM_USER_ID = int(os.getenv("SCHEDULER_SYSTEM_USER_ID", "2"))

_SCRIPT_BASE = "src/ml/pipeline/p22_biotech_ma/jobs"

# Each entry: (name, cron_utc, script_filename, enabled, optional task_params)
_JOB_SPECS: List[Dict[str, Any]] = [
    # SEC DERA universe must land before any per-company job runs — nothing
    # to iterate over otherwise (see ingest/universe_snapshot.py).
    {
        "name": "P22 SEC Universe Ingest",
        "cron": "0 4 1 */3 *",  # quarterly: 1st of every 3rd month, 04:00 UTC
        "script": "run_sec_universe_ingest.py",
        "enabled": True,
        # Full 2010-present backfill on first run; incremental thereafter via
        # raw-zone content-hash dedup, but still a multi-quarter download.
        "task_params": {"timeout_seconds": 3600},
    },
    {
        "name": "P22 Entity Resolution",
        "cron": "30 4 1 */3 *",  # quarterly, 30 min after the universe ingest it depends on
        "script": "run_entity_resolution.py",
        "enabled": True,
        "task_params": {"timeout_seconds": 1800},
    },
    {
        "name": "P22 Acquirer Roster Load",
        "cron": "45 4 1 */3 *",  # quarterly, after Entity Resolution so ticker-merge finds DERA-resolved rows
        "script": "run_acquirer_load.py",
        "enabled": True,
        "task_params": {"timeout_seconds": 600},
    },
    {
        "name": "P22 SEC Filings Ingest",
        "cron": "0 5 * * 1-5",
        "script": "run_sec_ingest.py",
        "enabled": True,
        "task_params": {"timeout_seconds": 3600},
    },
    {
        "name": "P22 Financial Facts Normalization",
        "cron": "15 5 * * 1-5",  # after SEC Filings Ingest lands the day's companyfacts payloads
        "script": "run_financial_facts_normalization.py",
        "enabled": True,
        "task_params": {"timeout_seconds": 1800},
    },
    {
        "name": "P22 ClinicalTrials Ingest",
        "cron": "30 5 * * 1-5",
        "script": "run_clinicaltrials_ingest.py",
        "enabled": True,
        # Per-company pagination + per-NCT-ID version-history fetch across the
        # full universe — the most request-heavy M1 job by a wide margin.
        "task_params": {"timeout_seconds": 7200},
    },
    {
        "name": "P22 openFDA Ingest",
        "cron": "0 6 * * 1-5",
        "script": "run_openfda_ingest.py",
        "enabled": True,
        "task_params": {"timeout_seconds": 3600},
    },
    {
        "name": "P22 Alias Matching",
        "cron": "30 6 * * 1-5",  # after both CT.gov and openFDA ingest have landed today's data
        "script": "run_alias_matching.py",
        "enabled": True,
        "task_params": {"timeout_seconds": 1800},
    },
    {
        "name": "P22 Trial Normalization",
        "cron": "45 6 * * 1-5",  # after Alias Matching so newly-confirmed companies' trials land too
        "script": "run_trial_normalization.py",
        "enabled": True,
        "task_params": {"timeout_seconds": 1800},
    },
    {
        "name": "P22 Orange Book Ingest",
        "cron": "0 4 2 */3 *",  # quarterly, one day after the universe ingest
        "script": "run_orange_book_ingest.py",
        "enabled": True,
    },
    {
        "name": "P22 Patent Expiry Normalization",
        "cron": "30 4 2 */3 *",  # quarterly, after both Orange Book Ingest and Acquirer Roster Load
        "script": "run_patent_expiry_normalization.py",
        "enabled": True,
        "task_params": {"timeout_seconds": 1800},
    },
    {
        "name": "P22 Purple Book Ingest",
        "cron": "15 4 2 */3 *",
        "script": "run_purple_book_ingest.py",
        "enabled": True,
    },
]


def run() -> Dict[str, Any]:
    """
    Insert all P22 job schedule rows idempotently.

    Returns:
        Summary dict with inserted count.
    """
    _logger.info("Registering %d P22 job schedules", len(_JOB_SPECS))
    count = 0

    with session_scope() as s:
        for spec in _JOB_SPECS:
            script_path = f"{_SCRIPT_BASE}/{spec['script']}"
            module_target = f"src.ml.pipeline.p22_biotech_ma.jobs.{spec['script'][:-3]}"
            task_params: Dict[str, Any] = {"script_path": script_path, "script_args": []}
            if "task_params" in spec:
                task_params.update(spec["task_params"])

            existing = s.query(Schedule).filter_by(user_id=_SYSTEM_USER_ID, name=spec["name"]).first()

            if existing:
                existing.target = module_target
                existing.task_params = task_params
                existing.cron = spec["cron"]
                existing.enabled = spec["enabled"]
                _logger.debug("Updated existing schedule: %s (%s)", spec["name"], spec["cron"])
            else:
                new_schedule = Schedule(
                    user_id=_SYSTEM_USER_ID,
                    name=spec["name"],
                    job_type="data_processing",
                    target=module_target,
                    task_params=task_params,
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
