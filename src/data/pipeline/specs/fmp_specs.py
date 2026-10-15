"""
FMP historical data pipeline jobs — plugin specs (``src/data/fmp_pipeline``, spec §10.2.7).

Registered **disabled**: the increment ingestion only makes sense on the one increment writer
(``fmp.increments.writer_host``, owner decision Q4: the Pi) once a vintage has been published there. Enable it in
``job_schedules`` on that machine only; the command itself refuses to run on any other host.
"""

from __future__ import annotations

from typing import List

from src.data.pipeline.base_plugin import PluginSpec

SPECS: List[PluginSpec] = [
    PluginSpec(
        name="FMP Increments Ingest",
        category="fmp",
        # 23:00 UTC Mon-Fri: after the US close in both summer and winter time (spec §10.2.7: daily preferred).
        cron="0 23 * * 1-5",
        script_path="src/data/fmp_pipeline/cli.py",
        script_args=["ingest-increments"],
        timeout_seconds=6 * 3600,
        enabled=False,
        description="Extend the active FMP vintage with raw yfinance bars and events (writer host only)",
    ),
]
