"""
Regression tests for EFTS _source field parsing across all consumers.

The EFTS endpoint returns a list-oriented _source (ciks / display_names / adsh /
period_ending / form) — NOT entity_id / entity_name / accession_no, which silently
yielded empty results and left P18's Form 4 / 13D-G / 13F-today paths dead. These
tests lock the correct field mapping in place.
"""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.downloader.edgar_downloader import EdgarDownloader


def _hit(ciks, names, adsh, doc="doc.xml", **extra) -> dict:
    src = {"ciks": ciks, "display_names": names, "adsh": adsh, "file_date": "2024-05-15"}
    src.update(extra)
    return {"_id": f"{adsh}:{doc}", "_source": src}


# ── 13F index / 13F-today ───────────────────────────────────────────────────


def test_get_new_13f_today_parses_real_schema(tmp_path):
    dl = EdgarDownloader(cache_dir=tmp_path)
    hits = [
        _hit(["0001234567"], ["Acme Capital  (CIK 0001234567)"], "0001234567-24-000001", period_ending="2024-03-31")
    ]
    with patch.object(dl, "_efts_search", return_value=hits):
        df = dl.get_new_13f_filings_today(as_of_date=date(2024, 5, 15))
    row = df.iloc[0]
    assert row["cik"] == "1234567"
    assert row["institution_name"] == "Acme Capital"
    assert row["accession_number"] == "0001234567-24-000001"
    assert row["period_of_report"] == "2024-03-31"


def test_download_13f_index_parses_real_schema(tmp_path):
    dl = EdgarDownloader(cache_dir=tmp_path)
    hits = [
        _hit(["0001234567"], ["Acme Capital  (CIK 0001234567)"], "0001234567-24-000001", period_ending="2024-03-31")
    ]
    with patch.object(dl, "_efts_search", return_value=hits):
        df = dl.download_13f_index(2024, 1, force=True)
    assert df.iloc[0]["cik"] == "1234567"
    assert df.iloc[0]["accession_number"] == "0001234567-24-000001"


# ── 13D/G ───────────────────────────────────────────────────────────────────


def test_download_13dg_parses_real_schema(tmp_path):
    dl = EdgarDownloader(cache_dir=tmp_path)
    hits = [
        _hit(
            ["0000814052"], ["TELEFONICA S A  (TEF, TEFOF)  (CIK 0000814052)"], "0001493152-24-019672", form="SC 13D/A"
        )
    ]
    with patch.object(dl, "_efts_search", return_value=hits):
        df = dl.download_13dg_filings(as_of_date=date(2024, 5, 15), force=True)
    row = df.iloc[0]
    assert row["cik"] == "814052"
    assert row["entity_name"] == "TELEFONICA S A  (TEF, TEFOF)"
    assert row["accession_number"] == "0001493152-24-019672"
    assert row["form_type"] == "SC 13D/A"


# ── Form 4 (adsh + _id primary document) ────────────────────────────────────


def test_download_form4_uses_adsh_and_id_primary_doc(tmp_path):
    dl = EdgarDownloader(cache_dir=tmp_path)
    hit = {
        "_id": "0002001011-24-000052:edgardoc.xml",
        "_source": {
            "ciks": ["0001649903", "0000886163"],  # reporting owner + issuer
            "adsh": "0002001011-24-000052",
            "display_names": [
                "Korenberg Matthew E  (CIK 0001649903)",
                "LIGAND PHARMACEUTICALS INC  (LGND)  (CIK 0000886163)",
            ],
        },
    }
    captured = {}

    def fake_fetch(cik_int, acc_norm, candidate_names=None):
        captured["cik_int"] = cik_int
        captured["acc_norm"] = acc_norm
        captured["names"] = candidate_names
        return "<ownershipDocument/>"

    sale_row = {
        "ticker": "LGND",
        "issuer_cik": "886163",
        "insider_name": "Korenberg",
        "transaction_code": "S",
        "shares": 100,
        "price_per_share": 10.0,
        "total_value_usd": 1000.0,
        "filed_date": "2024-05-15",
    }
    with (
        patch.object(dl, "_efts_search", return_value=[hit]),
        patch.object(dl, "_fetch_filing_xml", side_effect=fake_fetch),
        patch("src.data.downloader.edgar_downloader._parse_form4_xml", return_value=[sale_row]),
    ):
        df = dl.download_form4_filings(as_of_date=date(2024, 5, 15), force=True)

    # Accession comes from adsh (dashes stripped); CIK from ciks[0]; the EFTS _id
    # primary document is tried first.
    assert captured["acc_norm"] == "000200101124000052"
    assert captured["cik_int"] == 1649903
    assert captured["names"][0] == "edgardoc.xml"
    assert df.iloc[0]["ticker"] == "LGND"


def test_download_form4_skips_hits_without_accession(tmp_path):
    dl = EdgarDownloader(cache_dir=tmp_path)
    hit = {"_id": ":x.xml", "_source": {"ciks": ["0000111000"], "adsh": ""}}
    with patch.object(dl, "_efts_search", return_value=[hit]), patch.object(dl, "_fetch_filing_xml") as fetch:
        df = dl.download_form4_filings(as_of_date=date(2024, 5, 15), force=True)
    fetch.assert_not_called()
    assert df.empty
