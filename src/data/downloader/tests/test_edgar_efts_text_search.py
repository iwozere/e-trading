"""
Tests for the P19 Phase 3 EdgarDownloader extensions: ``efts_text_search``,
``efts_filings_search``, ``get_auditor_name``/``_extract_auditor_name``.

Field-mapping correctness (ciks/display_names/adsh) is already locked in by
test_edgar_efts_schema.py; these tests cover the new methods' own logic:
per-phrase/per-form query looping and de-duplication, the ``forms`` exact-match
quirk (never comma-lists), and the EX-23.1 auditor-name extraction heuristic —
verified live against real EDGAR data during development (design-v2.md §9.1/
§9.2), formalised here against mocks so the suite stays network-free.
"""

import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.downloader.edgar_downloader import EdgarDownloader, EftsUnavailableError, _extract_auditor_name


def _hit(cik, adsh, doc="doc.htm"):
    return {"_id": f"{adsh}:{doc}", "_source": {"ciks": [cik], "adsh": adsh, "file_date": "2026-06-01"}}


# ── efts_text_search ─────────────────────────────────────────────────────────


def test_efts_text_search_unions_hits_across_phrases_deduplicated(tmp_path):
    dl = EdgarDownloader(cache_dir=tmp_path)
    hit_a = _hit("0000000123", "0001-26-000001")
    hit_b = _hit("0000000123", "0001-26-000002")
    with patch.object(dl, "_efts_search", side_effect=[[hit_a], [hit_a, hit_b]]) as mock_search:
        hits = dl.efts_text_search(cik="123", phrases=["phrase one", "phrase two"], forms="10-K", start_dt="2026-06-01", end_dt="2026-06-01")
    assert len(hits) == 2  # hit_a deduplicated, not counted twice
    assert mock_search.call_count == 2  # one query per phrase
    # Each call quotes the phrase and zero-pads the CIK.
    kwargs = mock_search.call_args_list[0].kwargs
    assert kwargs["q"] == '"phrase one"'
    assert kwargs["ciks"] == "0000000123"


def test_efts_text_search_invalid_cik_returns_empty(tmp_path):
    dl = EdgarDownloader(cache_dir=tmp_path)
    assert dl.efts_text_search(cik="not-a-number", phrases=["x"], forms="10-K", start_dt="2026-06-01", end_dt="2026-06-01") == []


def test_efts_text_search_one_phrase_unavailable_does_not_abort_the_others(tmp_path):
    dl = EdgarDownloader(cache_dir=tmp_path)
    with patch.object(dl, "_efts_search", side_effect=[EftsUnavailableError("down"), [_hit("0000000123", "0001-26-000001")]]):
        hits = dl.efts_text_search(cik="123", phrases=["a", "b"], forms="10-K", start_dt="2026-06-01", end_dt="2026-06-01")
    assert len(hits) == 1


# ── efts_filings_search ──────────────────────────────────────────────────────


def test_efts_filings_search_chunks_at_100_ciks(tmp_path):
    dl = EdgarDownloader(cache_dir=tmp_path)
    ciks = [str(i) for i in range(1, 151)]  # 150 -> 2 chunks
    with patch.object(dl, "_efts_search", return_value=[]) as mock_search:
        dl.efts_filings_search(ciks=ciks, forms="424B5", start_dt="2026-06-01", end_dt="2026-06-01")
    assert mock_search.call_count == 2
    first_ciks = mock_search.call_args_list[0].kwargs["ciks"].split(",")
    assert len(first_ciks) == 100


def test_efts_filings_search_dedupes_across_chunks(tmp_path):
    dl = EdgarDownloader(cache_dir=tmp_path)
    hit = _hit("0000000123", "0001-26-000001")
    with patch.object(dl, "_efts_search", side_effect=[[hit], [hit]]):
        hits = dl.efts_filings_search(ciks=["1"] * 150, forms="S-1", start_dt="2026-06-01", end_dt="2026-06-01")
    assert len(hits) == 1


def test_efts_filings_search_empty_ciks_short_circuits(tmp_path):
    dl = EdgarDownloader(cache_dir=tmp_path)
    with patch.object(dl, "_efts_search") as mock_search:
        assert dl.efts_filings_search(ciks=[], forms="S-1", start_dt="2026-06-01", end_dt="2026-06-01") == []
    mock_search.assert_not_called()


# ── get_auditor_name ──────────────────────────────────────────────────────────


def test_get_auditor_name_queries_each_form_separately_not_a_comma_list(tmp_path):
    """EFTS treats `forms` as an exact match -- a comma-list paradoxically
    returns only amendments (documented live, design-v2.md §9.1)."""
    dl = EdgarDownloader(cache_dir=tmp_path)
    with (
        patch.object(dl, "efts_text_search", return_value=[]) as mock_text_search,
    ):
        dl.get_auditor_name(cik="123", start_dt="2026-06-01", end_dt="2026-07-01", forms="10-K,20-F")
    forms_queried = [c.kwargs["forms"] for c in mock_text_search.call_args_list]
    assert forms_queried == ["10-K", "20-F"]  # never "10-K,20-F"


def test_get_auditor_name_picks_most_recent_hit_and_extracts(tmp_path):
    dl = EdgarDownloader(cache_dir=tmp_path)
    older = {"_id": "0001-26-000001:ex2301.htm", "_source": {"ciks": ["0000000123"], "adsh": "0001-26-000001", "file_date": "2025-01-01"}}
    newer = {"_id": "0001-26-000002:ex2301.htm", "_source": {"ciks": ["0000000123"], "adsh": "0001-26-000002", "file_date": "2026-06-01"}}
    doc_text = "<p>/S/ Jane Doe</p><p>Marcum LLP</p>"
    with (
        patch.object(dl, "efts_text_search", return_value=[older, newer]),
        patch.object(dl, "_fetch_filing_document", return_value=doc_text) as mock_fetch,
    ):
        name = dl.get_auditor_name(cik="123", start_dt="2026-01-01", end_dt="2026-07-01")
    assert name == "Marcum LLP"
    # Fetched from the newer hit's accession/filename, not the older one.
    assert mock_fetch.call_args.args[1] == "000126000002"


def test_get_auditor_name_no_hits_returns_none(tmp_path):
    dl = EdgarDownloader(cache_dir=tmp_path)
    with patch.object(dl, "efts_text_search", return_value=[]):
        assert dl.get_auditor_name(cik="123", start_dt="2026-01-01", end_dt="2026-07-01") is None


def test_get_auditor_name_document_fetch_failure_returns_none(tmp_path):
    dl = EdgarDownloader(cache_dir=tmp_path)
    hit = _hit("0000000123", "0001-26-000001", doc="ex2301.htm")
    with (
        patch.object(dl, "efts_text_search", return_value=[hit]),
        patch.object(dl, "_fetch_filing_document", return_value=None),
    ):
        assert dl.get_auditor_name(cik="123", start_dt="2026-01-01", end_dt="2026-07-01") is None


# ── _extract_auditor_name (verified live during development, formalised here) ─


def test_extract_auditor_name_real_filing_shape():
    """Real EX-23.1 shape (DeltaSoft Corp, CIK 0002020919, accession
    0001683168-26-005450, fetched live 2026-08-18 during development)."""
    doc = """
    <p>CONSENT OF INDEPENDENT REGISTERED PUBLIC ACCOUNTING FIRM</p>
    <p>To The Shareholders and Board of Directors of Deltasoft, Corp.</p>
    <p>We consent to the use in the Form 10-K ...</p>
    <p>/S/ Boladale lawal</p>
    <p>BOLADALE LAWAL &amp; CO</p>
    <p>Chartered Accountant</p>
    <p>PCAOB No:6993</p>
    """
    assert _extract_auditor_name(doc) == "BOLADALE LAWAL & CO"


def test_extract_auditor_name_mixed_case_llp():
    doc = "<p>/S/ John Smith</p><p>Marcum LLP</p><p>New York, NY</p>"
    assert _extract_auditor_name(doc) == "Marcum LLP"


def test_extract_auditor_name_unusual_separator_not_truncated():
    """A lazy bounded-capture regex would truncate this to 'Brown, PC'."""
    doc = "<p>/s/ Jane Doe, CPA</p><p>WithumSmith+Brown, PC</p>"
    assert _extract_auditor_name(doc) == "WithumSmith+Brown, PC"


def test_extract_auditor_name_no_signature_block_returns_none():
    assert _extract_auditor_name("<p>no signature block here at all</p>") is None


def test_extract_auditor_name_only_searches_after_the_signature_line():
    """A '&' earlier in the addressee line must not false-positive before
    the actual signature block is reached."""
    doc = "<p>To the Shareholders & Board of Directors of X Corp</p><p>no signature here</p>"
    assert _extract_auditor_name(doc) is None
