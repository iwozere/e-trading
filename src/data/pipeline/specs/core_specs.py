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
        name="EMPS2 Morning Scan",
        category="emps2",
        cron="35 9 * * 1-5",
        script_path="src/ml/pipeline/p06_emps2/run_emps2_scan.py",
        timeout_seconds=14400,
        extra_task_params={
            "notification_rules": {"conditions": [
                {"check_field": "phase1_count", "operator": ">", "threshold": 0, "channels": ["email"],
                 "comment": "Email notification for Phase 1 candidates"},
                {"check_field": "phase2_count", "operator": ">", "threshold": 0, "channels": ["email", "telegram"],
                 "comment": "Email + Telegram for Phase 2 candidates"},
            ]},
        },
    ),
    PluginSpec(
        name="EMPS2 Evening Scan (8PM CET)",
        category="emps2",
        cron="0 14 * * 1-5",
        script_path="src/ml/pipeline/p06_emps2/run_emps2_scan.py",
        timeout_seconds=14400,
        extra_task_params={
            "notification_rules": {"conditions": [
                {"check_field": "phase1_count", "operator": ">", "threshold": 0, "channels": ["email"],
                 "comment": "Email notification for Phase 1 candidates"},
                {"check_field": "phase2_count", "operator": ">", "threshold": 0, "channels": ["email", "telegram"],
                 "comment": "Email + Telegram for Phase 2 candidates"},
            ]},
        },
    ),
    # Live-only job discovered when auditing job_schedules for rows missing from this registry
    # (see docs/Tasks.md) — a straight 1:1 port, cron/timeout/notification_rules copied verbatim.
    PluginSpec(
        name="FINRA TRF Daily Download",
        category="emps2",
        cron="0 7 * * *",
        script_path="src/ml/pipeline/p06_emps2/trf_downloader.py",
        timeout_seconds=1800,
        extra_task_params={
            "notification_rules": {"conditions": [
                {"check_field": "success", "operator": "==", "threshold": False, "channels": ["email"],
                 "comment": "Email notification when download fails"},
                {"check_field": "total_candidates", "operator": "<", "threshold": 2000, "channels": ["telegram"],
                 "comment": "Telegram alert when less than 2000 tickers downloaded"},
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
