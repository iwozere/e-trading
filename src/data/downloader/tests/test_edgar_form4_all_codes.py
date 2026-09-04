"""
Tests for the generalised Form 4 parser (all transaction codes, not just sales).

Widened for P19 Layer 0 (design-v2.md §3.1): P20 Kestrel's ``filings_ingest.py``
already reads ``edgar/13f/form4/{date}.csv.gz`` expecting buy codes {"P", "A"}
that the old sale-only filter could never produce. These tests lock in the new
schema and the regression that matters most: ``download_form4_filings``'s
default behaviour (still discoverable via sale-code filtering) must not change
for P18's ``Form4Monitor``, which self-filters and is unaffected either way.
"""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.downloader.edgar_downloader import EdgarDownloader, _FORM4_COLS, _footnotes_mentioning_10b5_1, _parse_form4_xml

def _issuer_header(relationship: str = "") -> str:
    return f"""
    <issuer>
        <issuerCik>0000886163</issuerCik>
        <issuerTradingSymbol>LGND</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0001649903</rptOwnerCik>
            <rptOwnerName>Korenberg Matthew E</rptOwnerName>
        </reportingOwnerId>
        {relationship}
    </reportingOwner>
"""


def _txn_xml(
    code: str,
    shares: str = "1000",
    price: str = "10.50",
    acq_disp: str = "A",
    footnote_id: str = "",
    transaction_date: str = "",
) -> str:
    footnote_ref = f'<footnoteId id="{footnote_id}"/>' if footnote_id else ""
    date_elem = f"<transactionDate><value>{transaction_date}</value></transactionDate>" if transaction_date else ""
    return f"""
    <nonDerivativeTransaction>
        <transactionCoding>
            <transactionFormType>4</transactionFormType>
            <transactionCode>{code}</transactionCode>
        </transactionCoding>
        {date_elem}
        <transactionAmounts>
            <transactionShares><value>{shares}</value></transactionShares>
            <transactionPricePerShare><value>{price}</value></transactionPricePerShare>
            <transactionAcquiredDisposedCode><value>{acq_disp}</value></transactionAcquiredDisposedCode>
        </transactionAmounts>
        {footnote_ref}
    </nonDerivativeTransaction>
    """


def _relationship_xml(is_director: str = "0", is_officer: str = "0", is_ten_pct: str = "0", title: str = "") -> str:
    title_elem = f"<officerTitle>{title}</officerTitle>" if title else ""
    return f"""
    <reportingOwnerRelationship>
        <isDirector>{is_director}</isDirector>
        <isOfficer>{is_officer}</isOfficer>
        <isTenPercentOwner>{is_ten_pct}</isTenPercentOwner>
        <isOther>0</isOther>
        {title_elem}
    </reportingOwnerRelationship>
    """


def _doc(*txns: str, footnotes: str = "", relationship: str = "") -> str:
    return (
        f"<ownershipDocument>{_issuer_header(relationship)}"
        f"<nonDerivativeTable>{''.join(txns)}</nonDerivativeTable>{footnotes}</ownershipDocument>"
    )


def test_buy_code_p_is_returned():
    """The exact case that was silently broken: code P must survive parsing now."""
    xml = _doc(_txn_xml("P", shares="5000", price="2.10"))
    rows = _parse_form4_xml(xml, filed_date="2026-08-18")
    assert len(rows) == 1
    row = rows[0]
    assert row["transaction_code"] == "P"
    assert row["ticker"] == "LGND"
    assert row["shares"] == 5000
    assert row["total_value_usd"] == 5000 * 2.10
    assert row["acquired_disposed_code"] == "A"
    assert row["is_derivative"] is False


def test_all_codes_pass_through():
    xml = _doc(_txn_xml("P"), _txn_xml("S"), _txn_xml("A"), _txn_xml("M"), _txn_xml("F"))
    rows = _parse_form4_xml(xml, filed_date="2026-08-18")
    assert {r["transaction_code"] for r in rows} == {"P", "S", "A", "M", "F"}


def test_10b5_1_plan_detected_via_footnote_text():
    footnotes = '<footnotes><footnote id="F1">Sale pursuant to a Rule 10b5-1 trading plan adopted 2026-01-15.</footnote></footnotes>'
    xml = _doc(_txn_xml("S", footnote_id="F1"), footnotes=footnotes)
    rows = _parse_form4_xml(xml, filed_date="2026-08-18")
    assert rows[0]["is_10b5_1_plan"] is True


def test_10b5_1_plan_false_when_no_matching_footnote():
    footnotes = '<footnotes><footnote id="F1">Shares withheld to satisfy tax withholding.</footnote></footnotes>'
    xml = _doc(_txn_xml("F", footnote_id="F1"), footnotes=footnotes)
    rows = _parse_form4_xml(xml, filed_date="2026-08-18")
    assert rows[0]["is_10b5_1_plan"] is False


def test_footnotes_mentioning_10b5_1_helper():
    import xml.etree.ElementTree as ET

    root = ET.fromstring(
        '<root><footnotes>'
        '<footnote id="F1">Pursuant to a 10b5-1 plan.</footnote>'
        '<footnote id="F2">Unrelated note.</footnote>'
        "</footnotes></root>"
    )
    assert _footnotes_mentioning_10b5_1(root) == {"F1"}


def test_malformed_xml_returns_empty_list():
    assert _parse_form4_xml("<not><valid", filed_date="2026-08-18") == []


def test_director_role_and_officer_title_are_parsed():
    xml = _doc(
        _txn_xml("S"),
        relationship=_relationship_xml(is_director="1", is_officer="1", title="Chief Financial Officer"),
    )
    row = _parse_form4_xml(xml, filed_date="2026-08-18")[0]
    assert row["is_director"] is True
    assert row["is_officer"] is True
    assert row["is_ten_percent_owner"] is False
    assert row["officer_title"] == "Chief Financial Officer"


def test_ten_percent_owner_with_no_relationship_element_defaults_false():
    xml = _doc(_txn_xml("P"))
    row = _parse_form4_xml(xml, filed_date="2026-08-18")[0]
    assert row["is_director"] is False
    assert row["is_officer"] is False
    assert row["is_ten_percent_owner"] is False
    assert row["officer_title"] == ""


def test_transaction_date_captured_separately_from_filed_date():
    xml = _doc(_txn_xml("S", transaction_date="2026-08-14"))
    row = _parse_form4_xml(xml, filed_date="2026-08-18")[0]
    assert row["transaction_date"] == "2026-08-14"
    assert row["filed_date"] == "2026-08-18"


def test_transaction_date_falls_back_to_filed_date_when_missing():
    xml = _doc(_txn_xml("S"))
    row = _parse_form4_xml(xml, filed_date="2026-08-18")[0]
    assert row["transaction_date"] == "2026-08-18"


def test_form4_cols_has_no_duplicates_and_matches_row_keys():
    xml = _doc(_txn_xml("P"))
    rows = _parse_form4_xml(xml, filed_date="2026-08-18")
    assert set(rows[0].keys()) == set(_FORM4_COLS)
    assert len(_FORM4_COLS) == len(set(_FORM4_COLS))


def test_download_form4_filings_caches_buy_codes_end_to_end(tmp_path):
    """
    The exact bug this fixes: P20's filings_ingest.py reads this cache file
    directly expecting code P rows. Confirm the file written by
    download_form4_filings actually contains them now (no mocking of
    _parse_form4_xml — this exercises the real parser).
    """
    dl = EdgarDownloader(cache_dir=tmp_path)
    hit = {
        "_id": "0002001011-24-000052:edgardoc.xml",
        "_source": {
            "ciks": ["0001649903", "0000886163"],
            "adsh": "0002001011-24-000052",
            "display_names": ["Korenberg Matthew E  (CIK 0001649903)"],
        },
    }
    xml = _doc(_txn_xml("P", shares="5000", price="2.10"))
    with (
        patch.object(dl, "_efts_search", return_value=[hit]),
        patch.object(dl, "_fetch_filing_xml", return_value=xml),
    ):
        df = dl.download_form4_filings(as_of_date=date(2026, 8, 18), force=True)

    assert len(df) == 1
    assert df.iloc[0]["transaction_code"] == "P"

    cache_file = tmp_path / "edgar" / "13f" / "form4" / "2026-08-18.csv.gz"
    assert cache_file.exists()
    import pandas as pd

    cached = pd.read_csv(cache_file, compression="gzip")
    assert cached.iloc[0]["transaction_code"] == "P"
