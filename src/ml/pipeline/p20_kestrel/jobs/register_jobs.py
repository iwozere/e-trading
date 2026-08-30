"""
P20 Kestrel — Job schedule registration (Python fallback).

Prefer the canonical SQL script for initial deploy:
    psql -d <db> < bin/scheduler/insert_p20_schedules.sql

This Python version is kept as a fallback for environments where psql is
unavailable. Names, `user_id`, and `job_type` are kept identical to the SQL
script on purpose: the lookup below is keyed on (user_id, name), so as long
as those match it updates the same row the SQL script would have created
instead of inserting a second one. Two jobs (`P20 GDELT Download`,
`P20 Revisions Ingest`) have no SQL counterpart yet and only exist via this
path — do not rename or renumber them without also checking there isn't a
stray row left at the old name.

Historical note: this used to default user_id to 1 and use snake_case names
(e.g. "p20_llm_risk_diff") while the SQL script used user_id=2 and Title
Case names (e.g. "P20 LLM Risk Diff"). Neither the mismatched user_id nor
the mismatched name collided with the SQL rows' (user_id, name) uniqueness,
so a single run of this script silently created a full second, unmanaged
set of 21 jobs (fixed 2026-08-30; duplicates removed from prod).

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
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.core.database import session_scope
from src.data.db.models.model_jobs import Schedule
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# Scheduler owner user_id for P20 system jobs. Must match the SQL script
# (bin/scheduler/insert_p20_schedules.sql hardcodes 2) — see module docstring.
_SYSTEM_USER_ID = int(os.getenv("SCHEDULER_SYSTEM_USER_ID", "2"))

_SCRIPT_BASE = "src/ml/pipeline/p20_kestrel/jobs"

# Each entry: (name, cron_utc, script_filename, enabled)
# `name` must match the SQL script's Title Case naming exactly — the lookup
# below is keyed on (user_id, name), so a mismatch here creates a duplicate
# row instead of updating the canonical one.
_JOB_SPECS: List[Dict[str, Any]] = [
    # Morning chain
    # GKG download must precede data_health (06:00) and gdelt_process (06:15)
    {"name": "P20 GDELT Download", "cron": "30 5 * * 1-5", "script": "run_gdelt_download.py", "enabled": True},
    {"name": "P20 Data Health Check", "cron": "0 6 * * 1-5", "script": "run_data_health.py", "enabled": True},
    {"name": "P20 GDELT Process", "cron": "15 6 * * 1-5", "script": "run_gdelt_process.py", "enabled": True},
    {"name": "P20 Social Sentiment Poll", "cron": "30 6 * * 1-5", "script": "run_social_poll.py", "enabled": True},
    {"name": "P20 AV Sentiment", "cron": "45 6 * * 1-5", "script": "run_av_sentiment.py", "enabled": True},
    {"name": "P20 Sentiment Aggregate", "cron": "0 7 * * 1-5", "script": "run_sentiment_aggregate.py", "enabled": True},
    {"name": "P20 Daily Digest", "cron": "30 6 * * 1-5", "script": "run_digest_send.py", "enabled": True},
    # EOD ingest
    {
        "name": "P20 EOD Ingest",
        "cron": "0 20 * * 1-5",
        "script": "run_ingest_eod.py",
        "enabled": True,
        # ~3000 universe tickers x 2yr OHLCV + TALib compute on a Raspberry Pi.
        # No override here meant the outer scheduler timeout (job_timeout_seconds,
        # 300s default) capped every run, well before the inner subprocess timeout
        # ever got a chance to apply. Chunked upserts in eod_ingest.py now mean a
        # timeout no longer discards completed work, but the timeout itself still
        # needs enough headroom for the run to actually finish.
        "task_params": {"timeout_seconds": 3600},
    },
    {"name": "P20 Filings Ingest", "cron": "30 20 * * 1-5", "script": "run_ingest_filings.py", "enabled": True},
    {"name": "P20 Catalyst Sync", "cron": "45 20 * * 1-5", "script": "run_catalyst_sync.py", "enabled": True},
    # Gap 10.1: Sleeve A revisions feed (shadow mode — writes k20_signals,
    # REVISIONS_FEED_AVAILABLE stays False in config.py until reviewed).
    # Must run before p20_screen_turnaround so revisions_score is fresh when
    # sleeve_a.py eventually reads it.
    {"name": "P20 Revisions Ingest", "cron": "50 20 * * 1-5", "script": "run_revisions_ingest.py", "enabled": True},
    # Gap 10.2 (half A): Sleeve B1 FDA calendar (pdufa.bio: PDUFA/AdCom/readout
    # dates in one fetch). PDUFA_CALENDAR_AVAILABLE=True in config.py — unlike
    # revisions_ingest this isn't a scoring-formula input, so there's no
    # shadow-mode review gate: it just makes a screen that was always empty
    # start producing real candidates. Must run before p20_screen_spinoffs
    # (which also runs sleeve_b.py's screen_b1()).
    {
        "name": "P20 PDUFA Calendar Ingest",
        "cron": "52 20 * * 1-5",
        "script": "run_pdufa_calendar_ingest.py",
        "enabled": True,
    },
    # Gap 10.2 (half B): Sleeve B2 spin-off registration monitor (EDGAR Form
    # 10/10-12B). event_date is the filing date, not the confirmed distribution
    # date — see spinoff_ingest.py docstring. Must run before p20_screen_spinoffs.
    {"name": "P20 Spinoff Ingest", "cron": "53 20 * * 1-5", "script": "run_spinoff_ingest.py", "enabled": True},
    # Screening
    {"name": "P20 Screen Turnaround", "cron": "0 21 * * 1-5", "script": "run_screen_turnaround.py", "enabled": True},
    {"name": "P20 Screen Spinoffs", "cron": "15 21 * * 1-5", "script": "run_screen_spinoffs.py", "enabled": True},
    {"name": "P20 Momentum Rank", "cron": "30 21 * * 1-5", "script": "run_momentum_rank.py", "enabled": True},
    # LLM
    {
        "name": "P20 LLM Classify Filings",
        "cron": "0 22 * * 1-5",
        "script": "run_llm_classify_filings.py",
        "enabled": True,
    },
    {"name": "P20 LLM Dossiers", "cron": "30 22 * * 1-5", "script": "run_llm_dossiers.py", "enabled": True},
    # Risk
    {"name": "P20 Risk Check", "cron": "*/30 9-17 * * 1-5", "script": "run_risk_check.py", "enabled": True},
    # LLM (weekly — 10-K/Q filings change quarterly)
    {"name": "P20 LLM Risk Diff", "cron": "0 18 * * 0", "script": "run_llm_risk_diff.py", "enabled": True},
    # Maintenance
    {
        "name": "P20 Weekly Maintenance",
        "cron": "0 5 * * 1",
        "script": "run_weekly_maintenance.py",
        "enabled": True,
        # 3000+ tickers, single-threaded fundamentals fetch: 60min was too tight
        # and got hit in practice. Chunked upserts in universe_loader.py now mean
        # a timeout no longer discards completed work, but the timeout itself
        # still needs enough headroom for the fetch to actually finish.
        "task_params": {"timeout_seconds": 10800},
    },
    {"name": "P20 Trends Poll", "cron": "0 3 * * 1-5", "script": "run_trends_watchlist.py", "enabled": True},
    {"name": "P20 Weekly Report", "cron": "0 17 * * 0", "script": "run_weekly_report.py", "enabled": True},
]


def run() -> Dict[str, Any]:
    """
    Insert all P20 job schedule rows idempotently.

    Returns:
        Summary dict with inserted count.
    """
    _logger.info("Registering %d P20 job schedules", len(_JOB_SPECS))  # expect 23
    count = 0

    with session_scope() as s:
        for spec in _JOB_SPECS:
            script_path = f"{_SCRIPT_BASE}/{spec['script']}"
            # Dotted module path, matching the `target` format the SQL script
            # writes (e.g. "src.ml.pipeline.p20_kestrel.jobs.run_data_health").
            module_target = f"src.ml.pipeline.p20_kestrel.jobs.{spec['script'][:-3]}"
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
