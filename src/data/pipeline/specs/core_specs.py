"""
Core / cross-cutting jobs — plugin specs.

Ported from ``bin/scheduler/insert_schedules.sql`` (names, crons,
`notification_rules`), which this module supersedes — with one correction:
that SQL file's `VIX Daily Monitor` row points at ``src/data/vix.py``, which
no longer exists. The live `job_schedules` row (checked directly) has already
been hand-corrected to ``src/data/downloader/vix_downloader.py`` — used here
instead, since the SQL file itself is what's now stale.
"""

from __future__ import annotations

from typing import List

from src.data.pipeline.base_plugin import PluginSpec

SPECS: List[PluginSpec] = [
    PluginSpec(
        name="VIX Daily Monitor",
        category="vix",
        cron="30 9 * * 1-5",
        script_path="src/data/downloader/vix_downloader.py",
        timeout_seconds=600,
        extra_task_params={
            "notification_rules": {"conditions": [
                {"check_field": "vix_current", "operator": ">=", "threshold": 20, "channels": ["email"],
                 "comment": "Email notification when VIX >= 20"},
                {"check_field": "vix_current", "operator": ">=", "threshold": 25, "channels": ["email", "telegram"],
                 "comment": "Email + Telegram when VIX >= 25"},
            ]},
        },
    ),
    PluginSpec(
        name="Fundamentals Cache Refresh",
        category="fundamentals",
        cron="0 14 * * 6,0",
        script_path="src/data/utils/refresh_fundamentals_cache.py",
        script_args=["--chunk-fraction", "0.5"],
        timeout_seconds=21600,
        description=(
            "Each run processes only the stalest 50% of the cached universe — Sat and Sun naturally split the "
            "backlog into two disjoint halves. See incident: 2026-08-22 timeouts scanning all ~7.3k tickers in one run."
        ),
        extra_task_params={
            "notification_rules": {"conditions": [
                {"check_field": "successful_symbols", "operator": ">=", "threshold": 0, "channels": ["email"],
                 "comment": "Email notification on completion"},
            ]},
        },
    ),
]
