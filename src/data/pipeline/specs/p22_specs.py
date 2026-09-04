"""
P22 Biotech M&A — plugin specs.

Ported verbatim (names, crons, timeouts, script paths) from
``src/ml/pipeline/p22_biotech_ma/jobs/register_jobs.py``'s ``_JOB_SPECS``,
which this module supersedes — see ``src/data/pipeline/register_jobs.py``.
"""

from __future__ import annotations

from typing import List

from src.data.pipeline.base_plugin import PluginSpec

_SCRIPT_BASE = "src/ml/pipeline/p22_biotech_ma/jobs"

SPECS: List[PluginSpec] = [
    PluginSpec(
        name="P22 SEC Universe Ingest",
        category="p22",
        cron="0 4 1 */3 *",  # quarterly: 1st of every 3rd month, 04:00 UTC
        script_path=f"{_SCRIPT_BASE}/run_sec_universe_ingest.py",
        timeout_seconds=3600,
        description="SEC DERA structured financial data — universe ingest; must land before per-company jobs.",
    ),
    PluginSpec(
        name="P22 Entity Resolution",
        category="p22",
        cron="30 4 1 */3 *",  # quarterly, 30 min after the universe ingest it depends on
        script_path=f"{_SCRIPT_BASE}/run_entity_resolution.py",
        timeout_seconds=1800,
        depends_on=["P22 SEC Universe Ingest"],
    ),
    PluginSpec(
        name="P22 Acquirer Roster Load",
        category="p22",
        cron="45 4 1 */3 *",  # quarterly, after Entity Resolution
        script_path=f"{_SCRIPT_BASE}/run_acquirer_load.py",
        timeout_seconds=600,
        depends_on=["P22 Entity Resolution"],
    ),
    PluginSpec(
        name="P22 SEC Filings Ingest",
        category="p22",
        cron="0 5 * * 1-5",
        script_path=f"{_SCRIPT_BASE}/run_sec_ingest.py",
        timeout_seconds=3600,
    ),
    PluginSpec(
        name="P22 Financial Facts Normalization",
        category="p22",
        cron="15 5 * * 1-5",  # after SEC Filings Ingest lands the day's companyfacts payloads
        script_path=f"{_SCRIPT_BASE}/run_financial_facts_normalization.py",
        timeout_seconds=1800,
        depends_on=["P22 SEC Filings Ingest"],
    ),
    PluginSpec(
        name="P22 ClinicalTrials Ingest",
        category="p22",
        cron="30 5 * * 1-5",
        script_path=f"{_SCRIPT_BASE}/run_clinicaltrials_ingest.py",
        timeout_seconds=21600,
        description="ClinicalTrials.gov studies + version history — most request-heavy M1 job; see script docstring.",
    ),
    PluginSpec(
        name="P22 openFDA Ingest",
        category="p22",
        cron="0 6 * * 1-5",
        script_path=f"{_SCRIPT_BASE}/run_openfda_ingest.py",
        timeout_seconds=3600,
    ),
    PluginSpec(
        name="P22 Alias Matching",
        category="p22",
        cron="30 6 * * 1-5",  # after both CT.gov and openFDA ingest have landed today's data
        script_path=f"{_SCRIPT_BASE}/run_alias_matching.py",
        timeout_seconds=1800,
        # This is the exact pair whose fixed-offset stagger already failed once in
        # production (see dependency_status.py's module docstring for the incident)
        # — the completion gate wired into run_alias_matching.py exists because of this.
        depends_on=["P22 ClinicalTrials Ingest", "P22 openFDA Ingest"],
    ),
    PluginSpec(
        name="P22 Trial Normalization",
        category="p22",
        cron="45 6 * * 1-5",  # after Alias Matching
        script_path=f"{_SCRIPT_BASE}/run_trial_normalization.py",
        timeout_seconds=1800,
        depends_on=["P22 Alias Matching"],
    ),
    PluginSpec(
        name="P22 Orange Book Ingest",
        category="p22",
        cron="0 4 2 */3 *",  # quarterly, one day after the universe ingest
        script_path=f"{_SCRIPT_BASE}/run_orange_book_ingest.py",
    ),
    PluginSpec(
        name="P22 Patent Expiry Normalization",
        category="p22",
        cron="30 4 2 */3 *",  # quarterly, after Orange Book Ingest and Acquirer Roster Load
        script_path=f"{_SCRIPT_BASE}/run_patent_expiry_normalization.py",
        timeout_seconds=1800,
        depends_on=["P22 Orange Book Ingest", "P22 Acquirer Roster Load"],
    ),
    PluginSpec(
        name="P22 Purple Book Ingest",
        category="p22",
        cron="15 4 2 */3 *",
        script_path=f"{_SCRIPT_BASE}/run_purple_book_ingest.py",
    ),
    PluginSpec(
        name="P22 Daily Price Ingest",
        category="p22",
        cron="0 22 * * 1-5",  # 22:00 UTC weekdays, after US market close
        script_path=f"{_SCRIPT_BASE}/run_price_ingest.py",
        timeout_seconds=3600,
        description="yfinance narrow incremental daily bars — see price_ingest.py split-adjustment note.",
    ),
]
