"""
Weekly stock screeners — plugin specs.

Ported verbatim from ``bin/scheduler/insert_screener_schedules.sql``, which
this module supersedes.
"""

from __future__ import annotations

from typing import List

from src.data.pipeline.base_plugin import PluginSpec

_NOTIFICATION_RULES = {
    "conditions": [
        {"check_field": "result_count", "operator": ">=", "threshold": 1, "channels": ["telegram", "email"],
         "comment": "Telegram + Email when at least 1 stock passes all criteria"},
    ]
}

SPECS: List[PluginSpec] = [
    PluginSpec(
        name="S&P 500 Weekly Screener",
        category="screener",
        cron="0 7 * * 6",
        script_path="src/screeners/sp500_stock_screener.py",
        timeout_seconds=3600,
        extra_task_params={"notification_rules": _NOTIFICATION_RULES},
    ),
    PluginSpec(
        name="SIX Weekly Screener",
        category="screener",
        cron="30 7 * * 6",
        script_path="src/screeners/six_stock_screener.py",
        timeout_seconds=1800,
        description="Staggered 30 min after S&P 500 to avoid concurrent yfinance load.",
        extra_task_params={"notification_rules": _NOTIFICATION_RULES},
    ),
]
