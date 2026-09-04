"""
P20 Kestrel — plugin specs.

Names, crons, and script paths ported from
``src/ml/pipeline/p20_kestrel/jobs/register_jobs.py``'s ``_JOB_SPECS`` — which
this module supersedes. `timeout_seconds` and `notification_rules` values,
however, come from the **live `job_schedules` rows** (confirmed via
`register_jobs.py --dry-run` against a DB seeded from the canonical
`bin/scheduler/insert_p20_schedules.sql`), not from `_JOB_SPECS`: that Python
fallback file never carried per-job timeout tuning or `notification_rules`
for most of these jobs, and a naive first apply from it alone would have
silently widened several timeouts to a generic 600s default and dropped
every Telegram `notification_rules` block. `register_jobs.py`'s
merge-on-update (`_MANAGED_TASK_PARAM_KEYS`) now protects against this for
any field this module still gets wrong, but the values below were corrected
to match production rather than relying solely on that safety net.

Two more corrections found the same way: `P20 Daily Digest`'s live cron is
``30 7 * * 1-5``, not the stale ``30 6 * * 1-5`` in `_JOB_SPECS` (the SQL
script — designated canonical by `_JOB_SPECS`'s own docstring — evidently
diverged from the Python fallback at some point). ``P20 GDELT Download`` and
``P20 Revisions Ingest`` have no SQL counterpart (Python-only rows); the
naming/ID-collision incident recorded in the old file's docstring
(2026-08-30, mismatched user_id/name silently created a duplicate set of 21
jobs) is exactly the class of bug this single-registry design closes off —
there is now only one place these names, crons, and task_params live.
"""

from __future__ import annotations

from typing import List

from src.data.pipeline.base_plugin import PluginSpec

_SCRIPT_BASE = "src/ml/pipeline/p20_kestrel/jobs"


def _telegram_rule(comment: str, check_field: str, threshold: int = 0, operator: str = ">") -> dict:
    return {"comment": comment, "channels": ["telegram"], "operator": operator, "threshold": threshold, "check_field": check_field}


SPECS: List[PluginSpec] = [
    # Morning chain — GKG download must precede data_health (06:00) and gdelt_process (06:15)
    PluginSpec(name="P20 GDELT Download", category="p20", cron="30 5 * * 1-5", script_path=f"{_SCRIPT_BASE}/run_gdelt_download.py"),
    PluginSpec(
        name="P20 Data Health Check",
        category="p20",
        cron="0 6 * * 1-5",
        script_path=f"{_SCRIPT_BASE}/run_data_health.py",
        timeout_seconds=300,
        depends_on=["P20 GDELT Download"],
        extra_task_params={
            "notification_rules": {"conditions": [
                _telegram_rule("Telegram warning when any data source is stale or budget exceeded", "alerts_sent"),
            ]},
        },
    ),
    PluginSpec(
        name="P20 GDELT Process", category="p20", cron="15 6 * * 1-5",
        script_path=f"{_SCRIPT_BASE}/run_gdelt_process.py", depends_on=["P20 GDELT Download"],
    ),
    PluginSpec(
        name="P20 Social Sentiment Poll", category="p20", cron="30 6 * * 1-5",
        script_path=f"{_SCRIPT_BASE}/run_social_poll.py", timeout_seconds=900,
    ),
    PluginSpec(name="P20 AV Sentiment", category="p20", cron="45 6 * * 1-5", script_path=f"{_SCRIPT_BASE}/run_av_sentiment.py"),
    PluginSpec(
        name="P20 Sentiment Aggregate", category="p20", cron="0 7 * * 1-5",
        script_path=f"{_SCRIPT_BASE}/run_sentiment_aggregate.py", timeout_seconds=300,
    ),
    PluginSpec(
        name="P20 Daily Digest",
        category="p20",
        cron="30 7 * * 1-5",  # live/canonical value — see module docstring
        script_path=f"{_SCRIPT_BASE}/run_digest_send.py",
        timeout_seconds=300,
    ),
    # EOD ingest
    PluginSpec(
        name="P20 EOD Ingest",
        category="p20",
        cron="0 20 * * 1-5",
        script_path=f"{_SCRIPT_BASE}/run_ingest_eod.py",
        timeout_seconds=3600,
        description="~3000 universe tickers x 2yr OHLCV + TALib compute; chunked upserts survive a timeout.",
    ),
    PluginSpec(
        name="P20 Filings Ingest", category="p20", cron="30 20 * * 1-5",
        script_path=f"{_SCRIPT_BASE}/run_ingest_filings.py", timeout_seconds=1800,
    ),
    PluginSpec(
        name="P20 Catalyst Sync",
        category="p20",
        cron="45 20 * * 1-5",
        script_path=f"{_SCRIPT_BASE}/run_catalyst_sync.py",
        timeout_seconds=600,
        extra_task_params={
            "notification_rules": {"conditions": [
                _telegram_rule("Telegram alert when T-10 or T-3 catalyst countdown triggers", "alerts_fired"),
            ]},
        },
    ),
    PluginSpec(
        name="P20 Revisions Ingest",
        category="p20",
        cron="50 20 * * 1-5",
        script_path=f"{_SCRIPT_BASE}/run_revisions_ingest.py",
        description="Sleeve A revisions feed (shadow mode) — must run before P20 Screen Turnaround.",
    ),
    PluginSpec(
        name="P20 PDUFA Calendar Ingest",
        category="p20",
        cron="52 20 * * 1-5",
        script_path=f"{_SCRIPT_BASE}/run_pdufa_calendar_ingest.py",
        timeout_seconds=300,
        description="Sleeve B1 FDA calendar (pdufa.bio) — must run before P20 Screen Spinoffs.",
    ),
    PluginSpec(
        name="P20 Spinoff Ingest",
        category="p20",
        cron="53 20 * * 1-5",
        script_path=f"{_SCRIPT_BASE}/run_spinoff_ingest.py",
        timeout_seconds=300,
        description="Sleeve B2 spin-off registration monitor (EDGAR Form 10/10-12B) — must run before P20 Screen Spinoffs.",
    ),
    # Screening
    PluginSpec(
        name="P20 Screen Turnaround",
        category="p20",
        cron="0 21 * * 1-5",
        script_path=f"{_SCRIPT_BASE}/run_screen_turnaround.py",
        timeout_seconds=300,
        depends_on=["P20 Revisions Ingest"],
        extra_task_params={
            "notification_rules": {"conditions": [
                _telegram_rule("Telegram alert when new Sleeve A turnaround candidates found", "candidates"),
            ]},
        },
    ),
    PluginSpec(
        name="P20 Screen Spinoffs",
        category="p20",
        cron="15 21 * * 1-5",
        script_path=f"{_SCRIPT_BASE}/run_screen_spinoffs.py",
        timeout_seconds=300,
        depends_on=["P20 PDUFA Calendar Ingest", "P20 Spinoff Ingest"],
        extra_task_params={
            "notification_rules": {"conditions": [
                _telegram_rule("Telegram alert when FDA run-up candidates found (Sleeve B1)", "b1_fda_runups"),
                _telegram_rule("Telegram alert when post-spin entry candidates found (Sleeve B2)", "b2_spinoffs"),
            ]},
        },
    ),
    PluginSpec(
        name="P20 Momentum Rank",
        category="p20",
        cron="30 21 * * 1-5",
        script_path=f"{_SCRIPT_BASE}/run_momentum_rank.py",
        timeout_seconds=300,
        extra_task_params={
            "notification_rules": {"conditions": [
                _telegram_rule("Telegram alert when new Sleeve C momentum candidates added", "watchlist_entries"),
            ]},
        },
    ),
    # LLM
    PluginSpec(
        name="P20 LLM Classify Filings",
        category="p20",
        cron="0 22 * * 1-5",
        script_path=f"{_SCRIPT_BASE}/run_llm_classify_filings.py",
        timeout_seconds=1800,
        extra_task_params={
            "notification_rules": {"conditions": [
                _telegram_rule("Telegram warning when LLM monthly budget reaches 80%", "budget_pct", threshold=80, operator=">="),
            ]},
        },
    ),
    PluginSpec(
        name="P20 LLM Dossiers",
        category="p20",
        cron="30 22 * * 1-5",
        script_path=f"{_SCRIPT_BASE}/run_llm_dossiers.py",
        timeout_seconds=3600,
        extra_task_params={
            "notification_rules": {"conditions": [
                _telegram_rule("Telegram notification when new candidate dossiers are ready", "dossiers_generated"),
                {
                    "comment": "Email + Telegram warning when LLM monthly budget reaches 80%",
                    "channels": ["email", "telegram"], "operator": ">=", "threshold": 80, "check_field": "budget_pct",
                },
            ]},
        },
    ),
    # Risk
    PluginSpec(
        name="P20 Risk Check",
        category="p20",
        cron="*/30 9-17 * * 1-5",
        script_path=f"{_SCRIPT_BASE}/run_risk_check.py",
        timeout_seconds=300,
        extra_task_params={
            "notification_rules": {"conditions": [
                _telegram_rule("Telegram alert when stop-loss, T1, T2, or trailing stop is breached", "alerts_fired"),
            ]},
        },
    ),
    # LLM (weekly — 10-K/Q filings change quarterly)
    PluginSpec(
        name="P20 LLM Risk Diff",
        category="p20",
        cron="0 18 * * 0",
        script_path=f"{_SCRIPT_BASE}/run_llm_risk_diff.py",
        timeout_seconds=3600,
        extra_task_params={
            "notification_rules": {"conditions": [
                _telegram_rule("Telegram alert when new risk red flags are detected in watchlist filings", "red_flags_found"),
            ]},
        },
    ),
    # Maintenance
    PluginSpec(
        name="P20 Weekly Maintenance",
        category="p20",
        cron="0 5 * * 1",
        script_path=f"{_SCRIPT_BASE}/run_weekly_maintenance.py",
        timeout_seconds=10800,
        description="3000+ tickers, single-threaded fundamentals fetch; chunked upserts survive a timeout.",
    ),
    PluginSpec(
        name="P20 Trends Poll", category="p20", cron="0 3 * * 1-5",
        script_path=f"{_SCRIPT_BASE}/run_trends_watchlist.py", timeout_seconds=1800,
    ),
    PluginSpec(
        name="P20 Weekly Report", category="p20", cron="0 17 * * 0",
        script_path=f"{_SCRIPT_BASE}/run_weekly_report.py", timeout_seconds=300,
    ),
]
