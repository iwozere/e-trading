"""Tests for ingest/sec_raw_ingest.py — mocked EdgarDownloader, no network calls."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.sec_raw_ingest import land_submissions_and_facts


def test_lands_submissions_and_facts_for_each_cik():
    dl = MagicMock()
    dl.load_submissions.side_effect = lambda cik: {"cik": cik, "filings": {}}
    dl.load_company_facts.side_effect = lambda cik: {"cik": cik, "facts": {}}

    with patch("src.ml.pipeline.p22_biotech_ma.ingest.sec_raw_ingest.raw_zone.write") as mock_write:
        mock_write.return_value = MagicMock(was_new=True)
        outcomes = land_submissions_and_facts(["1", "2"], downloader=dl)

    assert outcomes["1"] == {"submissions": True, "company_facts": True}
    assert outcomes["2"] == {"submissions": True, "company_facts": True}
    assert mock_write.call_count == 4  # 2 CIKs x 2 sources


def test_continues_after_one_cik_fails():
    dl = MagicMock()

    def submissions_side_effect(cik):
        if cik == "bad":
            raise RuntimeError("SEC fetch failed")
        return {"cik": cik}

    dl.load_submissions.side_effect = submissions_side_effect
    dl.load_company_facts.side_effect = lambda cik: {"cik": cik}

    with patch("src.ml.pipeline.p22_biotech_ma.ingest.sec_raw_ingest.raw_zone.write") as mock_write:
        mock_write.return_value = MagicMock(was_new=True)
        outcomes = land_submissions_and_facts(["bad", "good"], downloader=dl)

    assert outcomes["bad"]["submissions"] is False
    assert outcomes["bad"]["company_facts"] is True
    assert outcomes["good"]["submissions"] is True
    assert outcomes["good"]["company_facts"] is True


def test_empty_result_marks_outcome_false():
    dl = MagicMock()
    dl.load_submissions.return_value = None
    dl.load_company_facts.return_value = None

    outcomes = land_submissions_and_facts(["1"], downloader=dl)

    assert outcomes["1"] == {"submissions": False, "company_facts": False}


def test_repo_omitted_does_not_crash_on_failure():
    """The default (no `repo`) path — used by every existing test above — must keep working."""
    dl = MagicMock()
    dl.load_submissions.side_effect = RuntimeError("SEC fetch failed")
    dl.load_company_facts.return_value = None

    outcomes = land_submissions_and_facts(["1"], downloader=dl)

    assert outcomes["1"] == {"submissions": False, "company_facts": False}


def test_fetch_failure_logged_to_repo_when_given():
    """spec §7.2: every failed fetch, after retries, is logged to p22_fetch_failure."""
    dl = MagicMock()
    dl.load_submissions.side_effect = RuntimeError("SEC submissions fetch failed")
    dl.load_company_facts.side_effect = RuntimeError("SEC company facts fetch failed")
    repo = MagicMock()

    with patch("src.ml.pipeline.p22_biotech_ma.ingest.sec_raw_ingest.raw_zone.write"):
        land_submissions_and_facts(["1"], downloader=dl, repo=repo)

    assert repo.log_fetch_failure.call_count == 2
    repo.log_fetch_failure.assert_any_call(
        source="sec_submissions", entity="1", error_message="SEC submissions fetch failed"
    )
    repo.log_fetch_failure.assert_any_call(
        source="sec_company_facts", entity="1", error_message="SEC company facts fetch failed"
    )


def test_no_failure_logged_to_repo_on_success():
    dl = MagicMock()
    dl.load_submissions.side_effect = lambda cik: {"cik": cik}
    dl.load_company_facts.side_effect = lambda cik: {"cik": cik}
    repo = MagicMock()

    with patch("src.ml.pipeline.p22_biotech_ma.ingest.sec_raw_ingest.raw_zone.write") as mock_write:
        mock_write.return_value = MagicMock(was_new=True)
        land_submissions_and_facts(["1"], downloader=dl, repo=repo)

    repo.log_fetch_failure.assert_not_called()
