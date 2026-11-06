"""
P24 Falling Knife — plugin specs (spec §9.1).

Daily and weekly live scans. Registered **disabled**: the 2024+ holdout failed the go/no-go gate, so the scan runs
as paper + notify only once the owner enables it. No ``notification_rules``: the scan sends its own Telegram/email
messages with charts (``p24_falling_knife/notify.py``), like P05.

The daily scan runs at 22:45 UTC Mon–Fri, after the US close in both EST and EDT (22:30 already hosts three jobs);
the weekly scan at 12:00 UTC Saturday on the completed week. The signal ledger harvests the outputs at 06:00 UTC.
"""

from __future__ import annotations

from typing import List

from src.data.pipeline.base_plugin import PluginSpec

_SCRIPT = "src/ml/pipeline/p24_falling_knife/jobs/run_p24_scan.py"

SPECS: List[PluginSpec] = [
    PluginSpec(
        name="P24 Falling Knife Daily Scan",
        category="p24",
        cron="45 22 * * 1-5",
        script_path=_SCRIPT,
        script_args=["--timeframe", "daily"],
        timeout_seconds=10800,
        enabled=False,
        description="Falling-knife signals on the daily close, forming watchlist, paper exits; self-notifies.",
    ),
    PluginSpec(
        name="P24 Falling Knife Weekly Scan",
        category="p24",
        cron="0 12 * * 6",
        script_path=_SCRIPT,
        script_args=["--timeframe", "weekly"],
        timeout_seconds=10800,
        enabled=False,
        description="Falling-knife signals on the completed week (information only); self-notifies.",
    ),
]
