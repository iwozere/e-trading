"""
P18 Institutional Flow Tracker — plugin specs.

Ported verbatim from ``bin/scheduler/insert_p18_schedules.sql``, which this
module supersedes for the daily scan.

The quarterly full 13F-HR consensus backfill (`backfill_consensus.py`) runs
for hours, so its job carries a 24 h `timeout_seconds` (the scheduler's outer
limit follows the per-job value). It fires once per filing season, on the
15th of the month the 45-day 13F-HR deadline falls in; re-running it later in
the month (Run now) only downloads filers that are not cached yet and rebuilds
the consensus, which picks up late filers.
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
    PluginSpec(
        name="P18 Quarterly Consensus Backfill",
        category="p18",
        cron="0 8 15 2,5,8,11 *",
        script_path="src/ml/pipeline/p18_institutional_flow_tracker/backfill_consensus.py",
        script_args=["--auto-quarter"],
        timeout_seconds=86400,
        description=(
            "Builds consensus/<Y>_Q<N>.csv.gz for the quarter whose 13F-HR deadline just passed. 08:00 UTC so the "
            "07:00 daily scan finishes first; a cold run takes ~8 h on the Pi (~9k filers at ~3 s each)."
        ),
        extra_task_params={"notification_rules": {"conditions": [{"on": "failure", "channel": "telegram"}]}},
    ),
]
