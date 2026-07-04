"""Tests for EdgarDownloader.download_8k_filings (daily 8-K index)."""

import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.downloader.edgar_downloader import (
    EdgarDownloader,
    _normalize_8k_items,
    _primary_doc_from_efts_id,
)


def _hit(cik, name, acc, items, doc) -> dict:
    """Build an EFTS hit using the REAL _source schema (ciks/adsh/display_names)."""
    return {
        "_id": f"{acc}:{doc}",
        "_source": {
            "ciks": [cik] if cik else [],
            "display_names": [f"{name}  (CIK {cik})"] if name else [],
            "adsh": acc,
            "items": items,
            "file_description": "8-K",
            "file_type": "8-K",
            "file_date": "2026-06-24",
        },
    }


# ── Module helpers ──────────────────────────────────────────────────────────


def test_normalize_items_list():
    assert _normalize_8k_items(["1.01", "9.01"]) == "1.01,9.01"


def test_normalize_items_string_and_none():
    assert _normalize_8k_items("1.01") == "1.01"
    assert _normalize_8k_items(None) == ""


def test_primary_doc_from_id():
    assert _primary_doc_from_efts_id("0001234567-26-000123:scag_8k.htm") == "scag_8k.htm"
    assert _primary_doc_from_efts_id("no-colon") == ""


# ── download_8k_filings ─────────────────────────────────────────────────────


def test_download_8k_parses_and_caches(tmp_path):
    dl = EdgarDownloader(cache_dir=tmp_path)
    hits = [
        _hit("0001234567", "Scage Future", "0001234567-26-000123", ["1.01", "9.01"], "scag.htm"),
        _hit("0000777000", "Acme Corp", "0000777000-26-000045", ["8.01"], "acme.htm"),
    ]
    with patch.object(dl, "_efts_search", return_value=hits) as efts:
        df = dl.download_8k_filings(as_of_date=__import__("datetime").date(2026, 6, 24))

    efts.assert_called_once()
    assert list(df.columns) == [
        "cik",
        "company",
        "accession_number",
        "items",
        "description",
        "filed_date",
        "primary_document",
    ]
    assert len(df) == 2
    scag = df[df["cik"] == "1234567"].iloc[0]
    assert scag["company"] == "Scage Future"  # parsed from display_names
    assert scag["items"] == "1.01,9.01"
    assert scag["primary_document"] == "scag.htm"
    assert scag["filed_date"] == "2026-06-24"

    # Cached to edgar/8k/index/{date}.csv.gz
    cached = tmp_path / "edgar" / "8k" / "index" / "2026-06-24.csv.gz"
    assert cached.exists()


def test_download_8k_cache_hit_skips_efts(tmp_path):
    import datetime

    dl = EdgarDownloader(cache_dir=tmp_path)
    cached = tmp_path / "edgar" / "8k" / "index" / "2026-06-24.csv.gz"
    cached.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "cik": "111",
                "company": "X",
                "accession_number": "a",
                "items": "8.01",
                "description": "8-K",
                "filed_date": "2026-06-24",
                "primary_document": "x.htm",
            }
        ]
    ).to_csv(cached, index=False, compression="gzip")

    with patch.object(dl, "_efts_search") as efts:
        df = dl.download_8k_filings(as_of_date=datetime.date(2026, 6, 24))

    efts.assert_not_called()
    assert len(df) == 1


def test_download_8k_skips_rows_missing_cik_or_accession(tmp_path):
    import datetime

    dl = EdgarDownloader(cache_dir=tmp_path)
    hits = [
        _hit("", "No CIK", "0000000000-26-000001", ["8.01"], "x.htm"),
        _hit("0000111000", "No Acc", "", ["8.01"], "y.htm"),
        _hit("0000222000", "Good", "0000222000-26-000002", ["7.01"], "z.htm"),
    ]
    with patch.object(dl, "_efts_search", return_value=hits):
        df = dl.download_8k_filings(as_of_date=datetime.date(2026, 6, 24))

    assert len(df) == 1
    assert df.iloc[0]["cik"] == "222000"


def test_download_8k_empty_when_no_hits(tmp_path):
    import datetime

    dl = EdgarDownloader(cache_dir=tmp_path)
    with patch.object(dl, "_efts_search", return_value=[]):
        df = dl.download_8k_filings(as_of_date=datetime.date(2026, 6, 24))
    assert df.empty
    assert list(df.columns) == [
        "cik",
        "company",
        "accession_number",
        "items",
        "description",
        "filed_date",
        "primary_document",
    ]
