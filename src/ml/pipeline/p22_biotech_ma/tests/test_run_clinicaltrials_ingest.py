"""Tests for jobs/run_clinicaltrials_ingest.py's _load_last_known_updates helper.

No live DB or network — the job's `run()` itself is thin orchestration over
already-tested pieces (universe_snapshot, ClinicalTrialsClient, raw_zone) and
isn't unit tested directly here, matching this codebase's convention for
jobs/run_*.py scripts. `_load_last_known_updates` is pure raw-zone logic and
is worth testing on its own: it's the fix for the 2026-09-02 production
timeout (see config.py's CLINICALTRIALS_HISTORY_RATE_LIMIT_RPS docstring).
"""

import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest import raw_zone
from src.ml.pipeline.p22_biotech_ma.jobs.run_clinicaltrials_ingest import _load_last_known_updates

_STUDY = {
    "protocolSection": {
        "identificationModule": {"nctId": "NCT00000001"},
        "statusModule": {"lastUpdatePostDateStruct": {"date": "2026-08-15", "type": "ACTUAL"}},
    }
}


def test_load_last_known_updates_no_prior_partition_returns_empty(tmp_path):
    assert _load_last_known_updates(date(2026, 9, 2), root=tmp_path) == {}


def test_load_last_known_updates_reads_prior_partition_not_today(tmp_path):
    raw_zone.write(
        source="clinicaltrials_studies", entity="0001", as_of_date=date(2026, 9, 1), payload=[_STUDY], root=tmp_path
    )

    result = _load_last_known_updates(date(2026, 9, 2), root=tmp_path)

    assert result == {"NCT00000001": "2026-08-15"}


def test_load_last_known_updates_ignores_todays_in_progress_partition(tmp_path):
    """A study landed under TODAY's own partition (this run already wrote it) must not count as
    'previously known' — otherwise every study would always look unchanged and history would
    never be fetched even for genuinely new studies."""
    raw_zone.write(
        source="clinicaltrials_studies", entity="0001", as_of_date=date(2026, 9, 2), payload=[_STUDY], root=tmp_path
    )

    result = _load_last_known_updates(date(2026, 9, 2), root=tmp_path)

    assert result == {}


def test_load_last_known_updates_skips_studies_missing_nct_id_or_date(tmp_path):
    incomplete = {"protocolSection": {"identificationModule": {}, "statusModule": {}}}
    raw_zone.write(
        source="clinicaltrials_studies", entity="0001", as_of_date=date(2026, 9, 1), payload=[incomplete], root=tmp_path
    )

    assert _load_last_known_updates(date(2026, 9, 2), root=tmp_path) == {}


def test_load_last_known_updates_merges_across_multiple_companies_in_same_partition(tmp_path):
    other_study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000002"},
            "statusModule": {"lastUpdatePostDateStruct": {"date": "2026-07-01"}},
        }
    }
    raw_zone.write(
        source="clinicaltrials_studies", entity="0001", as_of_date=date(2026, 9, 1), payload=[_STUDY], root=tmp_path
    )
    raw_zone.write(
        source="clinicaltrials_studies", entity="0002", as_of_date=date(2026, 9, 1), payload=[other_study], root=tmp_path
    )

    result = _load_last_known_updates(date(2026, 9, 2), root=tmp_path)

    assert result == {"NCT00000001": "2026-08-15", "NCT00000002": "2026-07-01"}
