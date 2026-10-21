"""
Signal Ledger — plugin specs.

One nightly job that harvests every pipeline's archived picks into `signal_ledger`,
scores their forward returns and rebuilds the simulated paper books. It runs at
06:00 UTC Tue-Sat so the previous US session's close is final. Backfill is a
manual one-off: ``run_signal_ledger.py --full``.
"""

from __future__ import annotations

from typing import List

from src.data.pipeline.base_plugin import PluginSpec

SPECS: List[PluginSpec] = [
    PluginSpec(
        name="Signal Ledger Nightly",
        category="signal_ledger",
        cron="0 6 * * 2-6",
        script_path="src/ml/pipeline/signal_ledger/jobs/run_signal_ledger.py",
        timeout_seconds=3600,
        description="Harvest pipeline picks, score forward returns vs VT/SPY/IWM, rebuild paper books.",
    ),
]
