"""Tests for ingest/patent_expiry_normalization.py. No live DB — repo is a MagicMock."""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.patent_expiry_normalization import (
    build_product_lookup,
    extract_patent_expiry_records,
    resolve_applicant_to_acquirer,
    write_patent_expiry_records,
)

# Row shapes live-verified 2026-08-30 against the real, current Orange Book ZIP.
_PRODUCTS_ROWS = [
    {
        "Ingredient": "BUDESONIDE", "DF;Route": "AEROSOL, FOAM;RECTAL", "Trade_Name": "BUDESONIDE",
        "Applicant": "PADAGIS ISRAEL", "Strength": "2MG/ACTUATION", "Appl_Type": "A", "Appl_No": "215328",
        "Product_No": "001", "TE_Code": "AB", "Approval_Date": "Apr 12, 2023", "RLD": "No", "RS": "Yes",
        "Type": "RX", "Applicant_Full_Name": "PADAGIS ISRAEL PHARMACEUTICALS LTD",
    },
    {
        "Ingredient": "IBRUTINIB", "DF;Route": "CAPSULE;ORAL", "Trade_Name": "IMBRUVICA",
        "Applicant": "PHARMACYCLICS LLC", "Strength": "140MG", "Appl_Type": "N", "Appl_No": "205552",
        "Product_No": "001", "TE_Code": "", "Approval_Date": "Nov 13, 2013", "RLD": "Yes", "RS": "Yes",
        "Type": "RX", "Applicant_Full_Name": "Pfizer Inc",
    },
]

_PATENT_ROWS = [
    {
        "Appl_Type": "A", "Appl_No": "215328", "Product_No": "001", "Patent_No": "7625884",
        "Patent_Expire_Date_Text": "Aug 24, 2026", "Drug_Substance_Flag": "", "Drug_Product_Flag": "",
        "Patent_Use_Code": "U-141", "Delist_Flag": "", "Submission_Date": "",
    },
    {
        "Appl_Type": "N", "Appl_No": "205552", "Product_No": "001", "Patent_No": "9999999",
        "Patent_Expire_Date_Text": "Jan 05, 2031", "Drug_Substance_Flag": "Y", "Drug_Product_Flag": "",
        "Patent_Use_Code": "", "Delist_Flag": "", "Submission_Date": "",
    },
    # No matching product row (Appl_No 999999 not in _PRODUCTS_ROWS) — must be dropped.
    {
        "Appl_Type": "N", "Appl_No": "999999", "Product_No": "001", "Patent_No": "1111111",
        "Patent_Expire_Date_Text": "Jan 01, 2030", "Drug_Substance_Flag": "", "Drug_Product_Flag": "",
        "Patent_Use_Code": "", "Delist_Flag": "", "Submission_Date": "",
    },
    # Blank expiry date — must be dropped, not written with a guessed date.
    {
        "Appl_Type": "A", "Appl_No": "215328", "Product_No": "001", "Patent_No": "0000000",
        "Patent_Expire_Date_Text": "", "Drug_Substance_Flag": "", "Drug_Product_Flag": "",
        "Patent_Use_Code": "", "Delist_Flag": "", "Submission_Date": "",
    },
]


def test_build_product_lookup_keys_on_appl_type_no_product_no():
    lookup = build_product_lookup(_PRODUCTS_ROWS)
    assert lookup[("A", "215328", "001")]["Applicant_Full_Name"] == "PADAGIS ISRAEL PHARMACEUTICALS LTD"


def test_extract_patent_expiry_records_joins_and_parses_dates():
    records = extract_patent_expiry_records(_PRODUCTS_ROWS, _PATENT_ROWS)

    assert len(records) == 2  # the unmatched and blank-date rows are dropped
    by_appl_no = {r.application_no: r for r in records}
    assert by_appl_no["215328"].loe_date == date(2026, 8, 24)
    assert by_appl_no["215328"].applicant_full_name == "PADAGIS ISRAEL PHARMACEUTICALS LTD"
    assert by_appl_no["215328"].product_name == "BUDESONIDE"
    assert by_appl_no["215328"].exclusivity_type == "patent"
    assert by_appl_no["215328"].source == "orange_book"
    assert by_appl_no["205552"].loe_date == date(2031, 1, 5)


def test_extract_patent_expiry_records_drops_unmatched_product():
    records = extract_patent_expiry_records(_PRODUCTS_ROWS, _PATENT_ROWS)
    assert "999999" not in {r.application_no for r in records}


def test_extract_patent_expiry_records_drops_blank_expiry_date():
    records = extract_patent_expiry_records(_PRODUCTS_ROWS, [_PATENT_ROWS[3]])
    assert records == []


def test_resolve_applicant_to_acquirer_deterministic_match():
    acquirers = {1: "Pfizer Inc", 2: "Merck & Co Inc"}
    assert resolve_applicant_to_acquirer("Pfizer Inc", acquirers) == 1


def test_resolve_applicant_to_acquirer_fuzzy_match_not_written():
    """A fuzzy (not exact) applicant name must NOT resolve to a company_id this pass — see module
    docstring on why fuzzy matches here are logged only, not queued or written."""
    acquirers = {1: "Pfizer Inc"}
    assert resolve_applicant_to_acquirer("Pfizer Incorporated", acquirers) is None


def test_resolve_applicant_to_acquirer_no_match_returns_none():
    acquirers = {1: "Pfizer Inc"}
    assert resolve_applicant_to_acquirer("Some Random Generic Manufacturer LLC", acquirers) is None


def test_write_patent_expiry_records_writes_only_resolved_ones():
    records = extract_patent_expiry_records(_PRODUCTS_ROWS, _PATENT_ROWS)
    acquirers = {42: "Pfizer Inc"}  # only the IMBRUVICA row's applicant resolves
    repo = MagicMock()

    counts = write_patent_expiry_records(records, acquirers, repo)

    assert counts == {"written": 1, "unresolved": 1}
    repo.upsert_patent_expiry.assert_called_once()
    kwargs = repo.upsert_patent_expiry.call_args.kwargs
    assert kwargs["acquirer_id"] == 42
    assert kwargs["application_no"] == "205552"
    assert kwargs["loe_date"] == date(2031, 1, 5)
    assert kwargs["exclusivity_type"] == "patent"
    assert kwargs["therapeutic_area"] is None  # never fabricated — see module docstring
    assert kwargs["ttm_revenue_usd"] is None
