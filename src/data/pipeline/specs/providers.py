"""
Standalone provider downloaders — plugin specs.

These three are the only provider downloaders confirmed to have **no**
existing caller at all (checked by grepping for each class's instantiation
across `src/` before adding anything here — see docs/Tasks.md for the full
trail). Each already ships its own scheduler-compatible CLI
(`__SCHEDULER_RESULT__` output), so no new wrapper script was needed —
unlike `cboe`, `wikipedia`/index_changes, and `russell3000`, which are
deliberately NOT here: `p15_daily.py`/`p15_weekly.py` (see
`specs/p15_specs.py`) already refresh those three caches on a schedule, and a
separate row here would double the network calls against the same files.
`openfigi` is also deliberately excluded: `OpenFigiMapper` has no "download
everything" concept — it resolves whatever CUSIPs a caller (e.g. P18's
backfill) hands it on demand, so there's nothing to schedule.

These are genuinely NEW schedule rows, not a reorganization of anything
existing — confirm cadence/enablement before running `apply()` for this group.
"""

from __future__ import annotations

from typing import List

from src.data.pipeline.base_plugin import PluginSpec

SPECS: List[PluginSpec] = [
    PluginSpec(
        name="FRED Macro Series Daily Update",
        category="provider",
        cron="0 11 * * 1-5",
        script_path="src/data/downloader/fred_downloader.py",
        script_args=["update-all"],
        timeout_seconds=600,
        description=(
            "Incremental per-series fetch (only observations after each series' last cached date) + combined "
            "rebuild — cheap even run daily. No existing caller anywhere in src/ touches FredDownloader; "
            "the 50 macro series it covers were never being refreshed until this."
        ),
    ),
    PluginSpec(
        name="AAII Sentiment Weekly Update",
        category="provider",
        cron="0 12 * * 4",  # Thursday — AAII publishes weekly, every Thursday (see aaii_downloader.py docstring)
        script_path="src/data/downloader/aaii_downloader.py",
        script_args=["download"],
        timeout_seconds=300,
        description="No incremental API — each run replaces the whole cache. No existing caller anywhere in src/.",
    ),
    PluginSpec(
        name="Fear & Greed Weekly Archive Rebuild",
        category="provider",
        cron="0 21 * * 5",  # Friday EOD, per fear_greed_downloader.py's own documented intended schedule
        script_path="src/data/downloader/fear_greed_downloader.py",
        script_args=["download", "--full-rebuild"],
        timeout_seconds=300,
        description=(
            "The class docstring documents this exact Friday full-rebuild schedule, but nothing in src/ actually "
            "calls download(full_rebuild=True) — p17's reporting_agent.py only ever calls the default "
            "(full_rebuild=False), which incrementally refreshes it daily on P17's own cron but never re-syncs "
            "the underlying GitHub archive this rebuild pulls from."
        ),
    ),
]
