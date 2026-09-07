"""Tests for ingest/activist_positions.py (spec §2.6.2, §4.7). No live DB or network — repo/edgar
are MagicMocks/fakes."""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.activist_positions import (
    classify_filer_type,
    extract_single_pct_of_class,
    load_activist_filers,
    parse_13dg_header,
    run,
)

_REAL_HEADER = """SUBJECT COMPANY:\t

\tCOMPANY DATA:\t
\t\tCOMPANY CONFORMED NAME:\t\t\tValion Bio, Inc.
\t\tCENTRAL INDEX KEY:\t\t\t0001787740
\t\tSTATE OF INCORPORATION:\t\t\tDE

FILED BY:\t\t

\tCOMPANY DATA:\t
\t\tCOMPANY CONFORMED NAME:\t\t\t3i, LP
\t\tCENTRAL INDEX KEY:\t\t\t0001841619
\t\tSTATE OF INCORPORATION:\t\t\tDE
"""

_MULTI_FILER_HEADER = """SUBJECT COMPANY:\t

\tCOMPANY DATA:\t
\t\tCOMPANY CONFORMED NAME:\t\t\tTarget Biotech Inc
\t\tCENTRAL INDEX KEY:\t\t\t0000000042

FILED BY:\t\t

\tCOMPANY DATA:\t
\t\tCOMPANY CONFORMED NAME:\t\t\tActivist Fund LP
\t\tCENTRAL INDEX KEY:\t\t\t0000000099

REPORTING-OWNER:\t

\tOWNER DATA:\t
\t\tCOMPANY CONFORMED NAME:\t\t\tActivist Fund GP LLC
\t\tCENTRAL INDEX KEY:\t\t\t0000000098
"""


# ---------------------------------------------------------------------
# parse_13dg_header
# ---------------------------------------------------------------------

def test_parse_13dg_header_single_filer():
    result = parse_13dg_header(_REAL_HEADER)
    assert result == {
        "subject_cik": "0001787740",
        "filers": [{"cik": "0001841619", "name": "3i, LP"}],
    }


def test_parse_13dg_header_multiple_filer_blocks():
    result = parse_13dg_header(_MULTI_FILER_HEADER)
    assert result is not None
    assert result["subject_cik"] == "0000000042"
    assert result["filers"] == [
        {"cik": "0000000099", "name": "Activist Fund LP"},
        {"cik": "0000000098", "name": "Activist Fund GP LLC"},
    ]


def test_parse_13dg_header_no_subject_company_returns_none():
    assert parse_13dg_header("not a real filing header") is None


def test_parse_13dg_header_no_filer_block_returns_empty_filers():
    text = "SUBJECT COMPANY:\n\tCOMPANY DATA:\n\t\tCENTRAL INDEX KEY:\t0000000001\n"
    result = parse_13dg_header(text)
    assert result == {"subject_cik": "0000000001", "filers": []}


# ---------------------------------------------------------------------
# extract_single_pct_of_class
# ---------------------------------------------------------------------

def test_extract_single_pct_of_class_one_value():
    text = "<item5><percentOfClass>9.9</percentOfClass></item5>"
    assert extract_single_pct_of_class(text) == 9.9


def test_extract_single_pct_of_class_none_when_multiple_distinct_values():
    text = "<percentOfClass>9.9</percentOfClass><percentOfClass>1.1</percentOfClass>"
    assert extract_single_pct_of_class(text) is None


def test_extract_single_pct_of_class_ok_when_repeated_same_value():
    text = "<percentOfClass>9.9</percentOfClass><percentOfClass>9.9</percentOfClass>"
    assert extract_single_pct_of_class(text) == 9.9


def test_extract_single_pct_of_class_none_when_absent():
    assert extract_single_pct_of_class("no percentages here") is None


# ---------------------------------------------------------------------
# load_activist_filers
# ---------------------------------------------------------------------

def test_load_activist_filers_reads_real_repo_config():
    """Round-trips the real config/activist_filers.yaml file."""
    ciks = load_activist_filers()
    assert "0001577524" in ciks  # Sarissa Capital Management, live-verified in that file's header


def test_load_activist_filers_missing_file_returns_empty_set(tmp_path):
    assert load_activist_filers(tmp_path / "does_not_exist.yaml") == set()


def test_load_activist_filers_zero_pads_ciks(tmp_path):
    f = tmp_path / "filers.yaml"
    f.write_text("activist_filer_ciks:\n  - '1577524'\n", encoding="utf-8")
    assert load_activist_filers(f) == {"0001577524"}


# ---------------------------------------------------------------------
# classify_filer_type
# ---------------------------------------------------------------------

def test_classify_filer_type_activist_from_config_list():
    repo = MagicMock()
    result = classify_filer_type("0001577524", repo, {"0001577524"})
    assert result == "activist"
    repo.get_company_by_cik.assert_not_called()  # activist check short-circuits


def test_classify_filer_type_strategic_corporate_from_acquirer_role():
    repo = MagicMock()
    repo.get_company_by_cik.return_value = {"role": "acquirer"}
    assert classify_filer_type("0000000001", repo, set()) == "strategic_corporate"


def test_classify_filer_type_none_when_unrecognized():
    repo = MagicMock()
    repo.get_company_by_cik.return_value = None
    assert classify_filer_type("0000000001", repo, set()) is None


def test_classify_filer_type_none_when_target_role_not_acquirer():
    repo = MagicMock()
    repo.get_company_by_cik.return_value = {"role": "target"}
    assert classify_filer_type("0000000001", repo, set()) is None


# ---------------------------------------------------------------------
# run
# ---------------------------------------------------------------------

def _hit(ciks, form, adsh, doc="primary_doc.xml", file_date="2026-09-01"):
    return {"_id": f"{adsh}:{doc}", "_source": {"ciks": ciks, "form": form, "adsh": adsh, "file_date": file_date}}


def test_run_writes_activist_position_for_subject_in_universe():
    repo = MagicMock()
    repo.list_companies_full.return_value = [{"company_id": 7, "cik": "0001787740"}]
    repo.get_company_by_cik.return_value = None
    edgar = MagicMock()
    edgar.efts_filings_search.return_value = [_hit(["0001787740", "0001841619"], "SCHEDULE 13D", "0001-26-000001")]
    edgar.fetch_filing_document.return_value = _REAL_HEADER

    summary = run(repo, edgar, as_of_date=date(2026, 9, 1))

    assert summary["positions_written"] == 1
    repo.upsert_activist_position.assert_called_once()
    kwargs = repo.upsert_activist_position.call_args.kwargs
    assert kwargs["company_id"] == 7
    assert kwargs["filer_cik"] == "0001841619"
    assert kwargs["form_type"] == "SC 13D"
    assert kwargs["stated_intent"] is None


def test_run_dedupes_hits_across_form_queries():
    repo = MagicMock()
    repo.list_companies_full.return_value = [{"company_id": 7, "cik": "0001787740"}]
    edgar = MagicMock()
    same_hit = _hit(["0001787740", "0001841619"], "SCHEDULE 13D", "0001-26-000001")
    # Same _id returned for every form query (a MagicMock without side_effect returns it every call).
    edgar.efts_filings_search.return_value = [same_hit]
    edgar.fetch_filing_document.return_value = _REAL_HEADER

    summary = run(repo, edgar, as_of_date=date(2026, 9, 1))

    assert summary["filings_matched"] == 1  # deduped across the 4 form-type queries
    assert summary["positions_written"] == 1


def test_run_skips_subject_not_in_universe():
    repo = MagicMock()
    repo.list_companies_full.return_value = [{"company_id": 7, "cik": "0000000999"}]
    edgar = MagicMock()
    edgar.efts_filings_search.return_value = [_hit(["0001787740", "0001841619"], "SCHEDULE 13D", "0001-26-000002")]
    edgar.fetch_filing_document.return_value = _REAL_HEADER

    summary = run(repo, edgar, as_of_date=date(2026, 9, 1))

    assert summary["positions_written"] == 0
    assert summary["rejection_breakdown"] == {"subject_not_in_universe": 1}


def test_run_skips_document_fetch_failure():
    repo = MagicMock()
    repo.list_companies_full.return_value = [{"company_id": 7, "cik": "0001787740"}]
    edgar = MagicMock()
    edgar.efts_filings_search.return_value = [_hit(["0001787740", "0001841619"], "SCHEDULE 13D", "0001-26-000003")]
    edgar.fetch_filing_document.return_value = None

    summary = run(repo, edgar, as_of_date=date(2026, 9, 1))

    assert summary["rejection_breakdown"] == {"document_fetch_failed": 1}


def test_run_writes_one_row_per_filer_for_multi_filer_filing():
    repo = MagicMock()
    repo.list_companies_full.return_value = [{"company_id": 7, "cik": "0000000042"}]
    repo.get_company_by_cik.return_value = None
    edgar = MagicMock()
    edgar.efts_filings_search.return_value = [
        _hit(["0000000042", "0000000099", "0000000098"], "SCHEDULE 13D", "0001-26-000004"),
    ]
    edgar.fetch_filing_document.return_value = _MULTI_FILER_HEADER

    summary = run(repo, edgar, as_of_date=date(2026, 9, 1))

    assert summary["positions_written"] == 2
    assert repo.upsert_activist_position.call_count == 2
