"""
P17 Penny Stock Screener — plugin specs.

Ported verbatim from ``bin/scheduler/insert_p17_schedules.sql``, which this
module supersedes.
"""

from __future__ import annotations

from typing import List

from src.data.pipeline.base_plugin import PluginSpec

SPECS: List[PluginSpec] = [
    PluginSpec(
        name="P17 Penny Stock Screener Daily",
        category="p17",
        cron="0 6 * * 1-5",
        script_path="src/ml/pipeline/p17_penny_stocks/run_p17.py",
        timeout_seconds=7200,
        extra_task_params={
            "notification_rules": {"conditions": [
                {"check_field": "tier_a_count", "operator": ">", "threshold": 0, "channels": ["email"],
                 "comment": "Email when Tier A (elite) candidates found"},
                {"check_field": "explosive_count", "operator": ">", "threshold": 0, "channels": ["email", "telegram"],
                 "comment": "Email + Telegram when explosive candidates found"},
            ]},
        },
    ),
]
