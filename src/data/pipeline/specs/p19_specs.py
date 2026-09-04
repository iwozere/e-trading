"""
P19 Penny Intraday — plugin specs.

Ported verbatim from ``bin/scheduler/insert_p19_schedules.sql`` (v1: watchlist
build, shadow poll, EOD backfill) and ``insert_p19_v2_schedules.sql`` (v2:
structural profile, label backfill, intraday filings poll), which this module
supersedes. All six share one script (`run_p19.py`) with different
subcommands — P19 has no `jobs/register_jobs.py` of its own (unlike
P20/P22), these SQL files were its only prior registration path.
"""

from __future__ import annotations

from typing import List

from src.data.pipeline.base_plugin import PluginSpec

_SCRIPT = "src/ml/pipeline/p19_penny_intraday/run_p19.py"

SPECS: List[PluginSpec] = [
    # v1
    PluginSpec(
        name="P19 Intraday Watchlist Build",
        category="p19",
        cron="0 13 * * 1-5",
        script_path=_SCRIPT,
        script_args=["build-watchlist"],
        timeout_seconds=600,
        description="Merges P17 output + IBKR pre-market gappers into watchlist.json.",
    ),
    PluginSpec(
        name="P19 Intraday Shadow Poll",
        category="p19",
        cron="*/15 13-21 * * 1-5",
        script_path=_SCRIPT,
        script_args=["run-once", "--mode", "shadow"],
        timeout_seconds=300,
        description="Delayed IBKR reqMktData snapshot -> shadow.sqlite. Phase 1, no alerts.",
    ),
    PluginSpec(
        name="P19 Intraday EOD Backfill",
        category="p19",
        cron="30 21 * * 1-5",
        script_path=_SCRIPT,
        script_args=["eod-backfill"],
        timeout_seconds=1800,
    ),
    # v2
    PluginSpec(
        name="P19 Structural Profile",
        category="p19",
        cron="10 13 * * 1-5",
        script_path=_SCRIPT,
        script_args=["profile-structural"],
        timeout_seconds=3600,  # live value; widened from the originally-documented 1800 after a real production timeout
        description="Reads watchlist.json (must run after Watchlist Build); EDGAR + yfinance only, no IBKR.",
    ),
    PluginSpec(
        name="P19 Label Backfill",
        category="p19",
        cron="0 12 * * 1-5",
        script_path=_SCRIPT,
        script_args=["label-backfill"],
        timeout_seconds=1800,
        description="T+10 forward-return labels; self-gates on shadow dates old enough to have T+10 data.",
    ),
    PluginSpec(
        name="P19 Intraday Filings Poll",
        category="p19",
        cron="*/30 13-21 * * 1-5",
        script_path=_SCRIPT,
        script_args=["filings-poll"],
        timeout_seconds=600,
        description="EFTS scan of watchlist CIKs for 424B5/S-1/S-3 + 8-K 3.01/3.02, filed intraday.",
    ),
]
