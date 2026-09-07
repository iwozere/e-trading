"""Tests for ingest/process_events.py (spec §2.6.1, §4.7). No live DB or network — repo/edgar are
MagicMocks/fakes."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.process_events import (
    classify_filing_text,
    load_strategic_process_phrases,
    run,
)

_PHRASES = {
    "strong": [
        "exploring strategic alternatives",
        "engaged {ADVISOR} as financial advisor",
        "formed a strategic committee",
    ],
    "moderate": ["strategic review"],
    "negative": ["concluded its review of strategic alternatives", "determined to continue as a standalone company"],
}


# ---------------------------------------------------------------------
# load_strategic_process_phrases
# ---------------------------------------------------------------------

def test_load_strategic_process_phrases_reads_real_repo_config():
    """Round-trips the real config/pipeline/p22_strategic_process_phrases.yaml file."""
    phrases = load_strategic_process_phrases()
    assert "exploring strategic alternatives" in phrases["strong"]
    assert "strategic review" in phrases["moderate"]
    assert "determined to continue as a standalone company" in phrases["negative"]


def test_load_strategic_process_phrases_raises_on_missing_category(tmp_path):
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("strategic_process_phrases:\n  strong: ['x']\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required phrase categories"):
        load_strategic_process_phrases(bad_file)


# ---------------------------------------------------------------------
# classify_filing_text
# ---------------------------------------------------------------------

def test_classify_filing_text_strong_match():
    result = classify_filing_text("The Board is exploring strategic alternatives for the Company.", _PHRASES)
    assert result == {
        "state": "disclosed_open", "strength": "strong", "matched_phrase": "exploring strategic alternatives",
    }


def test_classify_filing_text_moderate_match():
    result = classify_filing_text("Management commenced a strategic review of the business.", _PHRASES)
    assert result == {"state": "disclosed_open", "strength": "moderate", "matched_phrase": "strategic review"}


def test_classify_filing_text_negative_match():
    result = classify_filing_text(
        "The Board has concluded its review of strategic alternatives and will continue as an independent company.",
        _PHRASES,
    )
    assert result == {
        "state": "concluded_no_deal", "strength": None,
        "matched_phrase": "concluded its review of strategic alternatives",
    }


def test_classify_filing_text_negative_checked_before_strong_to_avoid_false_positive():
    """A conclusion announcement can literally contain 'strategic alternatives' as a substring —
    must not be misclassified as an open process."""
    text = "The special committee concluded its review of strategic alternatives."
    result = classify_filing_text(text, _PHRASES)
    assert result is not None
    assert result["state"] == "concluded_no_deal"


def test_classify_filing_text_advisor_placeholder_matches_two_markers():
    text = "The Company has engaged Goldman Sachs as financial advisor to assist the Board."
    result = classify_filing_text(text, _PHRASES)
    assert result == {
        "state": "disclosed_open", "strength": "strong", "matched_phrase": "engaged {ADVISOR} as financial advisor",
    }


def test_classify_filing_text_no_match_returns_none():
    assert classify_filing_text("Quarterly results were in line with expectations.", _PHRASES) is None


def test_classify_filing_text_case_insensitive():
    result = classify_filing_text("EXPLORING STRATEGIC ALTERNATIVES for shareholders.", _PHRASES)
    assert result is not None
    assert result["strength"] == "strong"


# ---------------------------------------------------------------------
# run
# ---------------------------------------------------------------------

def _index_df(rows):
    cols = ["cik", "company", "accession_number", "items", "description", "filed_date", "primary_document"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def test_run_skips_filing_without_relevant_item_code():
    repo = MagicMock()
    repo.list_companies_full.return_value = [{"company_id": 1, "cik": "0000000001"}]
    edgar = MagicMock()
    edgar.download_8k_filings.return_value = _index_df([
        {"cik": "1", "company": "X", "accession_number": "0001-24-000001", "items": "1.01",
         "description": "", "filed_date": "2026-09-01", "primary_document": "doc.htm"},
    ])

    summary = run(repo, edgar)

    assert summary["candidates_written"] == 0
    assert summary["rejection_breakdown"] == {"item_not_relevant": 1}
    edgar.fetch_filing_document.assert_not_called()


def test_run_skips_cik_not_in_universe():
    repo = MagicMock()
    repo.list_companies_full.return_value = [{"company_id": 1, "cik": "0000000001"}]
    edgar = MagicMock()
    edgar.download_8k_filings.return_value = _index_df([
        {"cik": "999", "company": "Y", "accession_number": "0001-24-000002", "items": "7.01",
         "description": "", "filed_date": "2026-09-01", "primary_document": "doc.htm"},
    ])

    summary = run(repo, edgar)

    assert summary["rejection_breakdown"] == {"cik_not_in_universe": 1}
    edgar.fetch_filing_document.assert_not_called()


def test_run_writes_candidate_event_and_review_item_on_match():
    repo = MagicMock()
    repo.list_companies_full.return_value = [{"company_id": 42, "cik": "0000000001"}]
    repo.upsert_corporate_process_event.return_value = 555
    edgar = MagicMock()
    edgar.download_8k_filings.return_value = _index_df([
        {"cik": "1", "company": "Target Bio", "accession_number": "0001193125-24-012345", "items": "7.01,8.01",
         "description": "", "filed_date": "2026-09-01", "primary_document": "ex99.htm"},
    ])
    edgar.fetch_filing_document.return_value = "The Board is exploring strategic alternatives."

    summary = run(repo, edgar)

    assert summary == {
        "filings_scanned": 1, "candidates_written": 1, "rejection_breakdown": {},
    }
    repo.upsert_corporate_process_event.assert_called_once()
    kwargs = repo.upsert_corporate_process_event.call_args.kwargs
    assert kwargs["company_id"] == 42
    assert kwargs["state"] == "disclosed_open"
    assert kwargs["strength"] == "strong"
    assert kwargs["is_verified"] is False

    repo.add_review_item.assert_called_once()
    review_kwargs = repo.add_review_item.call_args.kwargs
    assert review_kwargs["item_type"] == "strategic_alternatives_candidate"
    assert review_kwargs["payload"]["event_id"] == 555
    assert review_kwargs["priority"] == 2  # strong match -> higher priority


def test_run_no_match_does_not_write_anything():
    repo = MagicMock()
    repo.list_companies_full.return_value = [{"company_id": 42, "cik": "0000000001"}]
    edgar = MagicMock()
    edgar.download_8k_filings.return_value = _index_df([
        {"cik": "1", "company": "Target Bio", "accession_number": "0001-24-000003", "items": "8.01",
         "description": "", "filed_date": "2026-09-01", "primary_document": "doc.htm"},
    ])
    edgar.fetch_filing_document.return_value = "Routine quarterly update, nothing notable."

    summary = run(repo, edgar)

    assert summary["candidates_written"] == 0
    assert summary["filings_scanned"] == 1
    repo.upsert_corporate_process_event.assert_not_called()
    repo.add_review_item.assert_not_called()


def test_run_document_fetch_failure_is_tracked_not_raised():
    repo = MagicMock()
    repo.list_companies_full.return_value = [{"company_id": 42, "cik": "0000000001"}]
    edgar = MagicMock()
    edgar.download_8k_filings.return_value = _index_df([
        {"cik": "1", "company": "Target Bio", "accession_number": "0001-24-000004", "items": "7.01",
         "description": "", "filed_date": "2026-09-01", "primary_document": "doc.htm"},
    ])
    edgar.fetch_filing_document.return_value = None

    summary = run(repo, edgar)

    assert summary["rejection_breakdown"] == {"document_fetch_failed": 1}
    assert summary["filings_scanned"] == 0
