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
