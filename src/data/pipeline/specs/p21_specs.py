"""
P21 Momentum — plugin specs.

Ported verbatim (names, crons, script paths) from
``src/ml/pipeline/p21_momentum/jobs/register_jobs.py``'s ``_JOB_SPECS``, which
this module supersedes. All three jobs run on a daily cron and self-guard
internally via ``src.ml.pipeline.p21_momentum.calendar`` — see that module's
docstring for why (no native "last/first trading day of month" concept in
`job_schedules.cron`).

Note: the original register_jobs.py stored ``job_type="script"`` and
``target=script_path`` (a plain path) for these three rows, versus P20/P22's
``job_type="data_processing"``/dotted ``target``. Both job_types dispatch
through the same executor (`SchedulerService._execute_data_processing_job`),
so this is cosmetic — `register_jobs.py`'s dry-run will surface it as a
one-time target-format normalization for these three rows before it's applied.
"""

from __future__ import annotations

from typing import List

from src.data.pipeline.base_plugin import PluginSpec

_SCRIPT_BASE = "src/ml/pipeline/p21_momentum/jobs"

SPECS: List[PluginSpec] = [
    # 16:30 ET == 20:30 UTC (EST) / 21:30 UTC (EDT); scheduled at the winter-time
    # value — DST drift is a known, shared limitation across every ET-based cron
    # job in this repo, not solved here.
    PluginSpec(name="p21_monthly_rebalance", category="p21", cron="30 20 * * 1-5", script_path=f"{_SCRIPT_BASE}/run_monthly_rebalance.py"),
    PluginSpec(name="p21_monthly_execute", category="p21", cron="45 13 * * 1-5", script_path=f"{_SCRIPT_BASE}/run_monthly_execute.py"),  # 09:45 ET
    PluginSpec(name="p21_daily_mark", category="p21", cron="30 20 * * 1-5", script_path=f"{_SCRIPT_BASE}/run_daily_mark.py"),
]
