"""Tests for features/lookahead_audit_run.py (spec §8.3) — recording proxy and the three row checks."""

import random
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.features.lookahead_audit import AuditSample
from src.ml.pipeline.p22_biotech_ma.features.lookahead_audit_run import (
    AuditReport,
    RecordingRepo,
    audit_sample,
    check_rows,
    sample_as_of_dates,
)

_AS_OF = date(2024, 6, 1)
_SAMPLE = AuditSample(company_id=7, as_of=_AS_OF, source_category="13d_process_event")


def _ts(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc)


class _Repo:
    def __init__(self, facts: List[Dict[str, Any]], positions: List[Dict[str, Any]]):
        self.facts = facts
        self.positions = positions

    def get_financial_facts_as_of(self, company_id: int, metric: str, as_of: date):
        del company_id, as_of
        return [f for f in self.facts if f["metric"] == metric]

    def get_verified_activist_positions(self, company_id: int, as_of: date):
        del company_id, as_of
        return self.positions

    def get_latest_raw_close_as_of(self, company_id: int, as_of: date):
        del company_id, as_of
        return {"trade_date": date(2024, 5, 31), "close_raw": 3.0}


def test_recording_repo_keeps_rows_with_known_from_and_counts_the_rest():
    repo = _Repo([{"metric": "cash", "value": 1.0, "known_from": _ts(date(2024, 1, 1))}], [])
    recorder = RecordingRepo(repo)
    recorder.get_financial_facts_as_of(7, "cash", _AS_OF)
    recorder.get_latest_raw_close_as_of(7, _AS_OF)
    recorder.get_verified_activist_positions(7, _AS_OF)  # empty: neither audited nor counted
    assert [r["method"] for r in recorder.rows] == ["get_financial_facts_as_of"]
    assert recorder.unaudited_calls == {"get_latest_raw_close_as_of": 1}


def test_check_rows_flags_lookahead_filing_date_and_unverified():
    rows = [
        {"method": "get_financial_facts_as_of", "known_from": _ts(date(2024, 6, 2))},
        {"method": "get_verified_activist_positions", "known_from": _ts(date(2024, 5, 20)), "filed_date": date(2024, 5, 10)},
        {"method": "get_verified_process_events", "known_from": _ts(date(2024, 5, 1)), "event_date": date(2024, 5, 1),
         "is_verified": False},
        {"method": "get_financial_facts_as_of", "known_from": _ts(_AS_OF)},  # same day is allowed
    ]
    report = AuditReport()
    check_rows(rows, _SAMPLE, report)
    assert report.rows_checked == 4
    assert len(report.lookahead_violations) == 1
    assert len(report.filing_date_mismatches) == 1
    assert len(report.unverified_reads) == 1
    assert not report.passed


def test_audit_sample_runs_features_through_recorder(monkeypatch):
    from src.ml.pipeline.p22_biotech_ma.features import lookahead_audit_run as mod

    monkeypatch.setattr(mod, "build_block_g", lambda cid, as_of, ctx: None)
    monkeypatch.setattr(mod, "evaluate_size_floor", lambda repo, cid, as_of: None)
    repo = _Repo([{"metric": "cash", "value": 1.0, "known_from": _ts(date(2024, 7, 1))}], [])

    def reads_cash(company_id, as_of, ctx):
        return ctx.get_latest_fact(company_id, "cash")

    def crashes(company_id, as_of, ctx):
        raise ValueError("boom")

    report = AuditReport()
    audit_sample(repo, _SAMPLE, report, features={"cash": reads_cash, "bad": crashes})
    assert len(report.lookahead_violations) == 1
    assert report.feature_errors[0]["feature"] == "bad"


def test_sample_as_of_dates_straddle_known_from():
    population = [{"company_id": i, "known_from": _ts(date(2024, 1, 10)), "category": "vendor_fact"} for i in range(200)]
    samples = sample_as_of_dates(population, random.Random(3))
    before = [s for s in samples if s.as_of == date(2024, 1, 9)]
    after = [s for s in samples if s.as_of >= date(2024, 1, 10)]
    assert before and after and len(before) + len(after) == 200
    assert max(s.as_of for s in after) <= date(2025, 1, 9)
