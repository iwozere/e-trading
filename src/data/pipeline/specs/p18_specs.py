"""
P18 Institutional Flow Tracker — plugin specs.

Ported verbatim from ``bin/scheduler/insert_p18_schedules.sql``, which this
module supersedes for the daily scan.

The quarterly full 13F-HR consensus backfill (`backfill_consensus.py`) is
**deliberately not here** — it runs for hours and would be SIGKILLed by any
scheduler timeout, so it stays on Linux crontab via
`bin/scheduler/p18_consensus_backfill.sh`, outside `job_schedules` entirely.
Do not add it to this registry; `runner.py --scope p18` covers only the
daily scan.
"""

from __future__ import annotations

from typing import List

from src.data.pipeline.base_plugin import PluginSpec

SPECS: List[PluginSpec] = [
    PluginSpec(
        name="P18 Institutional Flow Daily",
        category="p18",
        cron="0 7 * * *",
        script_path="src/ml/pipeline/p18_institutional_flow_tracker/run_p18_scan.py",
        timeout_seconds=3600,
        description=(
            "Runs daily incl. weekends (EDGAR filings arrive daily). During each 45-day 13F-HR filing window, "
            "the SQL script's comments note this timeout is sometimes seasonally bumped to 21600s by hand — "
            "see bin/scheduler/insert_p18_schedules.sql for that operational note; not automated here."
        ),
        extra_task_params={
            "notification_rules": {"conditions": [
                {"check_field": "high_score_count", "operator": ">", "threshold": 0, "channels": ["telegram"],
                 "comment": "Telegram alert when any ticker scores >= 60 (institutional distribution signal)"},
                {"check_field": "high_score_count", "operator": ">", "threshold": 2, "channels": ["email", "telegram"],
                 "comment": "Email + Telegram when 3+ tickers simultaneously flagged (broad distribution wave)"},
            ]},
        },
    ),
]
