"""
P15 (Hidden Dependencies) — plugin specs.

``P15 Options Daily`` was ported from ``bin/scheduler/insert_p15_options_schedules.sql``.

``P15 Pipeline - daily bundle`` and ``P15 Pipeline - weekly bundle`` had **no
file source anywhere** — no SQL script, no register_jobs.py — they exist only
as live `job_schedules` rows (found by directly querying the DB while
investigating whether cboe/wikipedia/russell3000 needed new standalone
schedules; see below). This is the same class of gap as P20's
`register_jobs.py`-only rows, just with no file at all rather than a stale
one. Values here (cron, timeout, notification_rules, `target`) are transcribed
verbatim from that live query.

These two bundles are why `cboe`, `wikipedia`/index_changes, and
`russell3000` downloaders are NOT separately registered in
`specs/providers.py`: `p15_daily.py` already calls `CboeDownloader` and
`WikipediaDownloader.download_index_changes()` on this cron, and
`p15_weekly.py` already calls `Russell3000Downloader` on this cron (Russell
3000 is also touched daily by P05's `universe_loader.py`). Adding separate
schedule rows for those three would just double the network calls against
the same cache files.

Note: like P21 and the Strategy Pack rows, these two use a bare `target`
(`p15_daily` / `p15_weekly`) rather than a dotted module path —
`register_jobs.py`'s dry-run will show a one-time, harmless `target`
normalization.
"""

from __future__ import annotations

from typing import List

from src.data.pipeline.base_plugin import PluginSpec

SPECS: List[PluginSpec] = [
    PluginSpec(
        name="P15 Options Daily",
        category="p15",
        cron="0 6 * * 1-5",
        script_path="src/ml/pipeline/p15_hidden_deps/p15_options_daily.py",
        timeout_seconds=21600,
        description="~3-4k optionable NASDAQ symbols; runs before p15_daily.py's 13:00 UTC options_putcall read.",
        extra_task_params={
            "notification_rules": {"conditions": [
                {"check_field": "ok", "operator": ">=", "threshold": 0, "channels": ["email"],
                 "comment": "Email summary on every completion"},
                {"check_field": "error", "operator": ">", "threshold": 0, "channels": ["email", "telegram"],
                 "comment": "Email + Telegram when any ticker errors are recorded"},
            ]},
        },
    ),
    PluginSpec(
        name="P15 Pipeline – daily bundle",  # en dash (U+2013) — matches the live row's exact name, not a hyphen
        category="p15",
        cron="0 13 * * 2-6",
        script_path="src/ml/pipeline/p15_hidden_deps/p15_daily.py",
        timeout_seconds=600,
        description="Also refreshes the shared CBOE put/call and Wikipedia index-changes caches (see module docstring).",
        extra_task_params={"notification_rules": {"conditions": [{"on": "failure", "channel": "telegram"}]}},
    ),
    PluginSpec(
        name="P15 Pipeline – weekly bundle",  # en dash (U+2013) — matches the live row's exact name, not a hyphen
        category="p15",
        cron="0 14 * * 6",
        script_path="src/ml/pipeline/p15_hidden_deps/p15_weekly.py",
        timeout_seconds=600,
        description="Also refreshes the shared Russell 3000 universe cache (see module docstring).",
        extra_task_params={"notification_rules": {"conditions": [{"on": "failure", "channel": "telegram"}]}},
    ),
]
