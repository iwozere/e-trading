"""
Tests for SwissDownloader (SIX Exchange Regulation "sheldon" JSON API + Zefix).

Network calls are mocked throughout. The significant-shareholder and
management-transaction fixtures below are trimmed but schema-faithful copies
of real responses captured from the live sheldon endpoints on 2026-09-08 (see
swiss_downloader.py's module docstring) — including the DKSH Holding AG /
Kardex Holding AG records used to cross-validate the buySellIndicator /
obligorFunctionCode lookup tables.
"""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.data.downloader.swiss_downloader import (
    SwissDownloader,
    _flatten_management_transaction,
    _flatten_official_notice,
    _flatten_significant_shareholder,
    _yyyymmdd_to_iso,
)


def _sheldon_response(status: str, total_count: int, item_list: list) -> MagicMock:
    """Build a MagicMock standing in for a requests.Response carrying a sheldon JSON body."""
    resp = MagicMock()
    resp.json.return_value = {"status": status, "totalCount": total_count, "itemList": item_list}
    resp.raise_for_status = MagicMock()
    return resp


@pytest.fixture()
def downloader(tmp_path):
    """SwissDownloader with a temp cache dir and no request throttling delay."""
    dl = SwissDownloader(cache_dir=tmp_path)
    with patch("src.data.downloader.swiss_downloader.time.sleep"):
        yield dl


# Real item captured 2026-09-08 (trimmed to the fields the flattener reads).
_SIG_SHAREHOLDER_ITEM = {
    "publication": {
        "notificationId": "ZA01-000000000SLC9",
        "notificationSubmitter": "DKSH Holding AG",
        "notificationSubmitterId": "DKSH",
        "category": "A",
        "publicationDate": 20260908,
        "transactionDate": 20260904,
        "purchaseTotalVotingRate": 3.001,
        "saleTotalVotingRate": 0.022,
        "belowThresholdVotingRate": 0.0,
        "triggerComment": ["The obligation to notify was triggered by an acquisition of shares."],
    },
    "beneficialNames": ["BlackRock, Inc."],
    "shareholderNames": [],
}

# Real items captured 2026-09-08 — cross-validated against the RSS descriptions:
# Kardex: "Purchase ... by a non-executive member of the board of directors"
# Rieter: "Sale ... by an executive member of the board of directors / member of senior management"
_KARDEX_MGMT_TXN_ITEM = {
    "notificationId": "T1Q9700014",
    "notificationSubmitter": "Kardex Holding AG",
    "notificationSubmitterId": "KARDEX",
    "ISIN": "CH0100837282",
    "transactionDate": 20260907,
    "buySellIndicator": "1",
    "obligorFunctionCode": "2",
    "transactionSize": 128.0,
    "transactionAmountPerSecurityCHF": 250.810625,
    "transactionAmountCHF": 32103.76,
    "securityTypeCode": "7",
    "securityDescription": "3 Jahre gesperrt",
}
_RIETER_MGMT_TXN_ITEM = {
    "notificationId": "T1Q9700022",
    "notificationSubmitter": "Rieter Holding AG",
    "notificationSubmitterId": "RIETER",
    "ISIN": "CH0003671440",
    "transactionDate": 20260907,
    "buySellIndicator": "2",
    "obligorFunctionCode": "1",
    "transactionSize": 14885.0,
    "transactionAmountPerSecurityCHF": 3.105000335908633,
    "transactionAmountCHF": 46217.93,
    "securityTypeCode": "7",
    "securityDescription": "",
}

_OFFICIAL_NOTICE_ITEM = {
    "noticeId": 365566,
    "date": 20260908,
    "noticeType": "A",
    "contact": "Zuercher Kantonalbank",
    "title": "Rule based parameter adjustment",
    "isin": None,
}


# ------------------------------------------------------------------
# Flattening / lookup-table correctness
# ------------------------------------------------------------------


def test_yyyymmdd_to_iso():
    assert _yyyymmdd_to_iso(20260907) == "2026-09-07"


def test_yyyymmdd_to_iso_handles_falsy_as_none():
    assert _yyyymmdd_to_iso(0) is None
    assert _yyyymmdd_to_iso(None) is None


def test_flatten_significant_shareholder_extracts_voting_rate():
    row = _flatten_significant_shareholder(_SIG_SHAREHOLDER_ITEM)
    assert row["filing_id"] == "ZA01-000000000SLC9"
    assert row["company"] == "DKSH Holding AG"
    assert row["publication_date"] == "2026-09-08"
    assert row["purchase_total_voting_rate"] == 3.001
    assert row["sale_total_voting_rate"] == 0.022
    assert row["beneficial_names"] == "BlackRock, Inc."


def test_flatten_management_transaction_maps_purchase_and_non_executive():
    row = _flatten_management_transaction(_KARDEX_MGMT_TXN_ITEM)
    assert row["action"] == "Purchase"
    assert row["actor_role"] == "non-executive member of the board of directors"
    assert row["quantity"] == 128.0
    assert row["total_value_chf"] == 32103.76


def test_flatten_management_transaction_maps_sale_and_executive():
    row = _flatten_management_transaction(_RIETER_MGMT_TXN_ITEM)
    assert row["action"] == "Sale"
    assert row["actor_role"] == "executive member of the board of directors / member of senior management"


def test_flatten_management_transaction_unknown_codes_pass_through_raw():
    item = {**_KARDEX_MGMT_TXN_ITEM, "buySellIndicator": "9", "obligorFunctionCode": "9"}
    row = _flatten_management_transaction(item)
    assert row["action"] == "9"
    assert row["actor_role"] == "9"


def test_flatten_official_notice():
    row = _flatten_official_notice(_OFFICIAL_NOTICE_ITEM)
    assert row == {
        "filing_id": 365566,
        "date": "2026-09-08",
        "notice_type": "A",
        "contact": "Zuercher Kantonalbank",
        "title": "Rule based parameter adjustment",
        "isin": None,
    }


# ------------------------------------------------------------------
# Download + per-day caching
# ------------------------------------------------------------------


def test_download_significant_shareholders_caches_per_day(downloader, tmp_path):
    with patch("requests.get", return_value=_sheldon_response("Ok", 1, [_SIG_SHAREHOLDER_ITEM])) as mock_get:
        df = downloader.download_significant_shareholders(as_of_date=date(2026, 9, 8))

    assert len(df) == 1
    assert df.iloc[0]["company"] == "DKSH Holding AG"
    assert df.iloc[0]["purchase_total_voting_rate"] == 3.001

    cache_file = tmp_path / "swiss" / "ser" / "significant_shareholders" / "2026-09-08.csv.gz"
    assert cache_file.exists()

    _, kwargs = mock_get.call_args
    assert kwargs["params"]["fromDate"] == "20260908"
    assert kwargs["params"]["toDate"] == "20260908"


def test_download_management_transactions_caches_per_day(downloader, tmp_path):
    with patch(
        "requests.get",
        return_value=_sheldon_response("Ok", 2, [_KARDEX_MGMT_TXN_ITEM, _RIETER_MGMT_TXN_ITEM]),
    ):
        df = downloader.download_management_transactions(as_of_date=date(2026, 9, 7))

    assert len(df) == 2
    assert set(df["filing_id"]) == {"T1Q9700014", "T1Q9700022"}

    cache_file = tmp_path / "swiss" / "ser" / "management_transactions" / "2026-09-07.csv.gz"
    assert cache_file.exists()


def test_download_official_notices_caches_per_day(downloader, tmp_path):
    with patch("requests.get", return_value=_sheldon_response("Ok", 1, [_OFFICIAL_NOTICE_ITEM])):
        df = downloader.download_official_notices(as_of_date=date(2026, 9, 8))

    assert len(df) == 1
    cache_file = tmp_path / "swiss" / "ser" / "official_notices" / "2026-09-08.csv.gz"
    assert cache_file.exists()


def test_second_call_reads_cache_without_refetching(downloader):
    with patch("requests.get", return_value=_sheldon_response("Ok", 1, [_SIG_SHAREHOLDER_ITEM])) as mock_get:
        downloader.download_significant_shareholders(as_of_date=date(2026, 9, 8))
        downloader.download_significant_shareholders(as_of_date=date(2026, 9, 8))

    assert mock_get.call_count == 1  # second call hit the on-disk cache


def test_force_refetches_even_when_cached(downloader):
    with patch("requests.get", return_value=_sheldon_response("Ok", 1, [_SIG_SHAREHOLDER_ITEM])) as mock_get:
        downloader.download_significant_shareholders(as_of_date=date(2026, 9, 8))
        downloader.download_significant_shareholders(as_of_date=date(2026, 9, 8), force=True)

    assert mock_get.call_count == 2


def test_pagination_follows_total_count(downloader):
    """totalCount larger than one page's itemList must trigger a second page fetch."""
    page0 = _sheldon_response("Ok", 2, [_SIG_SHAREHOLDER_ITEM])
    page1 = _sheldon_response("Ok", 2, [_SIG_SHAREHOLDER_ITEM])
    with patch("requests.get", side_effect=[page0, page1]) as mock_get:
        df = downloader.download_significant_shareholders(as_of_date=date(2026, 9, 8))

    assert mock_get.call_count == 2
    assert len(df) == 2
    calls = mock_get.call_args_list
    assert calls[0].kwargs["params"]["pageNumber"] == 0
    assert calls[1].kwargs["params"]["pageNumber"] == 1


def test_no_results_returns_empty_dataframe_with_expected_columns(downloader):
    with patch("requests.get", return_value=_sheldon_response("Ok", 0, [])):
        df = downloader.download_management_transactions(as_of_date=date(2026, 9, 8))

    assert len(df) == 0
    assert list(df.columns) == [
        "filing_id",
        "company",
        "company_id",
        "isin",
        "transaction_date",
        "action",
        "quantity",
        "price_per_security_chf",
        "total_value_chf",
        "actor_role",
        "security_type_code",
        "security_description",
    ]


def test_unexpected_status_stops_without_raising(downloader):
    with patch("requests.get", return_value=_sheldon_response("Error", 0, [])):
        df = downloader.download_significant_shareholders(as_of_date=date(2026, 9, 8))

    assert len(df) == 0


# ------------------------------------------------------------------
# Zefix company registry
# ------------------------------------------------------------------


def test_zefix_request_without_credentials_raises(downloader):
    with pytest.raises(RuntimeError, match="Zefix credentials not configured"):
        downloader.search_company("Kardex")


def test_search_company_returns_dataframe(tmp_path):
    dl = SwissDownloader(cache_dir=tmp_path, zefix_username="user", zefix_password="pw")
    api_response = MagicMock()
    api_response.json.return_value = [
        {"name": "Kardex Holding AG", "uid": "CHE-106.588.217", "legalForm": "AG", "status": "ACTIVE"}
    ]
    api_response.raise_for_status = MagicMock()
    with (
        patch("src.data.downloader.swiss_downloader.time.sleep"),
        patch("requests.request", return_value=api_response) as mock_request,
    ):
        df = dl.search_company("Kardex")

    assert len(df) == 1
    assert df.iloc[0]["uid"] == "CHE-106.588.217"
    _, kwargs = mock_request.call_args
    assert kwargs["auth"] == ("user", "pw")
    assert kwargs["json"] == {"name": "Kardex", "activeOnly": True}


def test_get_company_by_uid_caches_to_json_and_skips_refetch(tmp_path):
    dl = SwissDownloader(cache_dir=tmp_path, zefix_username="user", zefix_password="pw")
    api_response = MagicMock()
    api_response.json.return_value = {"name": "Kardex Holding AG", "uid": "CHE-106.588.217"}
    api_response.raise_for_status = MagicMock()

    with (
        patch("src.data.downloader.swiss_downloader.time.sleep"),
        patch("requests.request", return_value=api_response) as mock_request,
    ):
        first = dl.get_company_by_uid("CHE-106.588.217")
        second = dl.get_company_by_uid("CHE-106.588.217")  # should hit the on-disk cache, not the API

    assert first == second == {"name": "Kardex Holding AG", "uid": "CHE-106.588.217"}
    assert mock_request.call_count == 1

    cache_file = tmp_path / "swiss" / "zefix" / "company_CHE-106588217.json"
    assert cache_file.exists()


def test_get_company_by_uid_force_refetches(tmp_path):
    dl = SwissDownloader(cache_dir=tmp_path, zefix_username="user", zefix_password="pw")
    api_response = MagicMock()
    api_response.json.return_value = {"name": "Kardex Holding AG"}
    api_response.raise_for_status = MagicMock()

    with (
        patch("src.data.downloader.swiss_downloader.time.sleep"),
        patch("requests.request", return_value=api_response) as mock_request,
    ):
        dl.get_company_by_uid("CHE-106.588.217")
        dl.get_company_by_uid("CHE-106.588.217", force=True)

    assert mock_request.call_count == 2


# ------------------------------------------------------------------
# BaseDataDownloader interface
# ------------------------------------------------------------------


def test_get_ohlcv_returns_none(downloader):
    from datetime import datetime

    assert downloader.get_ohlcv("AZN", "1d", datetime(2026, 1, 1), datetime(2026, 1, 2)) is None


def test_get_supported_intervals_is_empty(downloader):
    assert downloader.get_supported_intervals() == []


def test_get_provider_name(downloader):
    assert downloader.get_provider_name() == "swiss"
