"""
P10 (EMPS3) — plugin specs.

Ported verbatim from ``bin/scheduler/insert_p10_schedules.sql``, which this
module supersedes.
"""

from __future__ import annotations

from typing import List

from src.data.pipeline.base_plugin import PluginSpec

_NOTIFICATION_RULES = {
    "conditions": [
        {"check_field": "phase1_count", "operator": ">", "threshold": 0, "channels": ["email"],
         "comment": "Email notification for Phase 1 candidates"},
        {"check_field": "phase2_count", "operator": ">", "threshold": 0, "channels": ["email", "telegram"],
         "comment": "Email + Telegram for Phase 2 candidates"},
    ]
}

SPECS: List[PluginSpec] = [
    PluginSpec(
        name="EMPS3 Morning Scan",
        category="p10",
        cron="0 7 * * 1-6",
        script_path="src/ml/pipeline/p10_emps3/run_emps3_scan.py",
        timeout_seconds=14400,
        extra_task_params={"notification_rules": _NOTIFICATION_RULES},
    ),
    PluginSpec(
        name="EMPS3 Mid-Day Scan",
        category="p10",
        cron="0 18 * * 1-5",
        script_path="src/ml/pipeline/p10_emps3/run_emps3_scan.py",
        timeout_seconds=14400,
        extra_task_params={"notification_rules": _NOTIFICATION_RULES},
    ),
]
