"""
P05 AI Selector — plugin specs.

Ported verbatim from ``bin/scheduler/insert_p05_schedules.sql``, which this
module supersedes.

No `notification_rules` here, deliberately: the P05 pipeline sends its own
rich Telegram + email notifications from Stage 4 (see run_p05_scan.py /
Stage4Output). Adding scheduler-level notification_rules would make the
scheduler ALSO emit a generic key:value result dump on top of that.
"""

from __future__ import annotations

from typing import List

from src.data.pipeline.base_plugin import PluginSpec

SPECS: List[PluginSpec] = [
    PluginSpec(
        name="P05 AI Selector Daily",
        category="p05",
        cron="0 10 * * 1-5",
        script_path="src/ml/pipeline/p05_ai_selector/run_p05_scan.py",
        timeout_seconds=7200,
        description="Stage 1-4 pipeline; self-notifies (see docstring) — timeout covers cold-start OHLCV fetch.",
    ),
]
