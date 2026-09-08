"""Tests for ingest/deal_candidates.py (spec §2.5, M6). No live DB or network — repo/edgar are
MagicMocks/fakes."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.deal_candidates import run


def _hit(ciks, form, adsh, doc="primary_doc.htm", names=None):
    return {
        "_id": f"{adsh}:{doc}",
        "_source": {"ciks": ciks, "form": form, "adsh": adsh, "file_date": "2026-09-01",
                     "display_names": names or []},
    }


def test_run_queues_candidate_for_matched_universe_company():
    repo = MagicMock()
    repo.list_companies_full.return_value = [{"company_id": 7, "cik": "0000000001"}]
    repo.get_pending_review_items.return_value = []
    edgar = MagicMock()
    edgar.efts_filings_search.return_value = [_hit(["0000000001"], "SC 14D9", "0001-26-000001")]

    summary = run(repo, edgar, start_dt="2026-08-01", end_dt="2026-09-01")

    assert summary["candidates_queued"] == 1
    repo.add_review_item.assert_called_once()
    kwargs = repo.add_review_item.call_args.kwargs
    assert kwargs["item_type"] == "deal_candidate"
    assert kwargs["payload"]["company_id"] == 7
    assert kwargs["payload"]["form_type"] == "SC 14D9"
    assert kwargs["payload"]["accession_no"] == "0001-26-000001"


def test_run_skips_filing_with_no_universe_match():
    repo = MagicMock()
    repo.list_companies_full.return_value = [{"company_id": 7, "cik": "0000000001"}]
    repo.get_pending_review_items.return_value = []
    edgar = MagicMock()
    edgar.efts_filings_search.return_value = [_hit(["0000000999"], "S-4", "0001-26-000002")]

    summary = run(repo, edgar, start_dt="2026-08-01", end_dt="2026-09-01")

    assert summary["candidates_queued"] == 0
    repo.add_review_item.assert_not_called()


def test_run_dedupes_hits_across_form_queries():
    repo = MagicMock()
    repo.list_companies_full.return_value = [{"company_id": 7, "cik": "0000000001"}]
    repo.get_pending_review_items.return_value = []
    edgar = MagicMock()
    same_hit = _hit(["0000000001"], "S-4", "0001-26-000003")
    edgar.efts_filings_search.return_value = [same_hit]

    summary = run(repo, edgar, start_dt="2026-08-01", end_dt="2026-09-01")

    assert summary["filings_matched"] == 1  # deduped across the 3 form queries
    assert summary["candidates_queued"] == 1


def test_run_does_not_requeue_already_pending_candidate():
    repo = MagicMock()
    repo.list_companies_full.return_value = [{"company_id": 7, "cik": "0000000001"}]
    repo.get_pending_review_items.return_value = [
        {"payload": {"company_id": 7, "accession_no": "0001-26-000004"}},
    ]
    edgar = MagicMock()
    edgar.efts_filings_search.return_value = [_hit(["0000000001"], "DEFM14A", "0001-26-000004")]

    summary = run(repo, edgar, start_dt="2026-08-01", end_dt="2026-09-01")

    assert summary["candidates_queued"] == 0
    assert summary["already_queued"] == 1
    repo.add_review_item.assert_not_called()


def test_run_queues_both_sides_when_both_are_universe_members():
    """A filing naming two P22 companies (e.g. a target's SC 14D9 that also names an
    already-tracked acquirer) queues one candidate per matched side, not one for the pair."""
    repo = MagicMock()
    repo.list_companies_full.return_value = [
        {"company_id": 7, "cik": "0000000001"}, {"company_id": 8, "cik": "0000000002"},
    ]
    repo.get_pending_review_items.return_value = []
    edgar = MagicMock()
    edgar.efts_filings_search.return_value = [_hit(["0000000001", "0000000002"], "S-4", "0001-26-000005")]

    summary = run(repo, edgar, start_dt="2026-08-01", end_dt="2026-09-01")

    assert summary["candidates_queued"] == 2
    queued_company_ids = {c.kwargs["payload"]["company_id"] for c in repo.add_review_item.call_args_list}
    assert queued_company_ids == {7, 8}


def test_run_no_matches_at_all():
    repo = MagicMock()
    repo.list_companies_full.return_value = []
    repo.get_pending_review_items.return_value = []
    edgar = MagicMock()
    edgar.efts_filings_search.return_value = []

    summary = run(repo, edgar, start_dt="2026-08-01", end_dt="2026-09-01")

    assert summary == {"filings_matched": 0, "candidates_queued": 0, "already_queued": 0}
