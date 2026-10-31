"""Tests for ingest/deal_coverage.py (spec §2.0.1) — disappearance classification against deals and candidates."""

import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.deal_coverage import classify, latest_candidates
from src.ml.pipeline.p22_biotech_ma.ingest.entity_resolution import UniverseCandidate


def _cand(cik: str, last: date, reporting: bool = False) -> UniverseCandidate:
    return UniverseCandidate(cik=cik, name=f"Co {cik}", sic="2834", ticker=None, exchange=None, most_recent_form="10-Q",
                             most_recent_filed=last, eligible_reporting=reporting, eligible_exchange=None, flagged_spac=False)


def test_classify_covers_every_case_and_skips_reporting_and_old():
    candidates = [
        _cand("1", date(2018, 5, 1)),
        _cand("2", date(2019, 5, 1)),
        _cand("3", date(2020, 5, 1)),
        _cand("4", date(2021, 5, 1)),
        _cand("5", date(2026, 8, 1), reporting=True),   # still filing
        _cand("6", date(2008, 1, 1)),                   # before --since
        _cand("7", date(2022, 5, 1)),                   # no p22_company row
    ]
    companies = {cik: {"company_id": int(cik), "ticker": f"T{cik}"} for cik in "123456"}
    deals = [{"target_id": 1, "deal_type": "strategic_acquisition"}]
    items = [
        {"status": "pending", "payload": {"company_id": 2}},
        {"status": "rejected", "payload": {"company_id": 3}},
        {"status": "confirmed", "payload": {"company_id": 1}},
    ]
    rows = classify(candidates, companies, deals, items, since=date(2010, 1, 1))
    by_cik = {r.cik: r for r in rows}
    assert set(by_cik) == {"1", "2", "3", "4", "7"}
    assert by_cik["1"].classification == "deal_entered" and by_cik["1"].deal_types == ["strategic_acquisition"]
    assert by_cik["2"].classification == "candidates_pending"
    assert by_cik["3"].classification == "candidates_reviewed_no_deal"
    assert by_cik["4"].classification == "no_candidate"
    assert by_cik["7"].classification == "no_candidate" and by_cik["7"].company_id is None
    assert by_cik["3"].as_row()["candidate_statuses"] == "rejected=1"


def test_latest_candidates_evaluates_at_latest_quarter_end():
    rows = {
        "2020q1": [{"cik": "1", "name": "A", "form": "10-Q", "filed": "20200215", "sic": "2834"}],
        "2020q4": [{"cik": "2", "name": "B", "form": "10-Q", "filed": "20201115", "sic": "2834"}],
    }
    as_of, candidates = latest_candidates(rows)
    assert as_of == date(2020, 12, 31)
    reporting = {c.cik.lstrip("0"): c.eligible_reporting for c in candidates}
    assert reporting == {"1": False, "2": True}


def test_latest_candidates_empty():
    assert latest_candidates({}) == (None, [])
