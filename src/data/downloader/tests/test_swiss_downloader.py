"""
Tests for SwissDownloader (SIX Exchange Regulation RSS feeds + Zefix registry).

Network calls are mocked throughout — the RSS fixtures below are trimmed but
schema-faithful copies of real feed items captured from the live SER feeds on
2026-09-07 (see swiss_downloader.py's module docstring), so the parsing
assertions exercise the exact template SER actually publishes, not a
hypothetical one.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytest

from src.data.downloader.swiss_downloader import SwissDownloader, _parse_management_transaction

# Trimmed but schema-faithful copies of real SER feed responses (captured 2026-09-07).
_MGMT_TXN_FEED_XML = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0">
<channel>
<title>SIX Exchange Regulation | Management Transactions</title>
<link>https://www.ser-ag.com/en/resources/notifications-market-participants/management-transactions.html</link>
<description>Management Transactions</description>
<item>
<title>Kardex Holding AG</title>
<link>https://www.ser-ag.com/en/resources/notifications-market-participants/management-transactions.html#/transaction-details/T1Q9700014</link>
<category>Transaction</category>
<description>Purchase of 128 securities amounting to CHF 32,103.76 (CHF 250.81 / security) by a non-executive member of the board of directors</description>
<pubDate>Mon, 07 Sep 2026 12:00:00 +0200</pubDate>
<guid>https://www.ser-ag.com/en/resources/notifications-market-participants/management-transactions.html#/transaction-details/T1Q9700014</guid>
</item>
<item>
<title>Rieter Holding AG</title>
<link>https://www.ser-ag.com/en/resources/notifications-market-participants/management-transactions.html#/transaction-details/T1Q9700022</link>
<category>Transaction</category>
<description>Sale of 14,885 securities amounting to CHF 46,217.93 (CHF 3.11 / security) by an executive member of the board of directors / member of senior management</description>
<pubDate>Mon, 07 Sep 2026 12:00:00 +0200</pubDate>
<guid>https://www.ser-ag.com/en/resources/notifications-market-participants/management-transactions.html#/transaction-details/T1Q9700022</guid>
</item>
</channel>
</rss>"""

_SIG_SHAREHOLDERS_FEED_XML = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0">
<channel>
<title>SIX Exchange Regulation | Significant shareholders</title>
<link>https://www.ser-ag.com/en/resources/notifications-market-participants/significant-shareholders.html</link>
<description>Disclosure of shareholdings</description>
<item>
<title>ARYZTA AG</title>
<link>https://www.ser-ag.com/en/resources/notifications-market-participants/significant-shareholders.html#/shareholder-details/ZA01-000000000SKW3</link>
<category>Notification</category>
<description>Disclosure of shareholdings in ARYZTA AG</description>
<pubDate>Sat, 05 Sep 2026 12:00:00 +0200</pubDate>
<guid>https://www.ser-ag.com/en/resources/notifications-market-participants/significant-shareholders.html#/shareholder-details/ZA01-000000000SKW3</guid>
</item>
</channel>
</rss>"""


def _rss_response(xml_text: str) -> MagicMock:
    """Build a MagicMock standing in for a requests.Response carrying RSS bytes."""
    resp = MagicMock()
    resp.content = xml_text.encode("utf-8")
    resp.raise_for_status = MagicMock()
    return resp


@pytest.fixture()
def downloader(tmp_path):
    """SwissDownloader with a temp cache dir and no request throttling delay."""
    dl = SwissDownloader(cache_dir=tmp_path)
    with patch("src.data.downloader.swiss_downloader.time.sleep"):
        yield dl


# ------------------------------------------------------------------
# Management transaction description parsing
# ------------------------------------------------------------------


def test_parse_management_transaction_purchase():
    fields = _parse_management_transaction(
        "Purchase of 128 securities amounting to CHF 32,103.76 (CHF 250.81 / security) "
        "by a non-executive member of the board of directors"
    )
    assert fields["action"] == "Purchase"
    assert fields["quantity"] == "128"
    assert fields["price_per_security_chf"] == "250.81"
    assert fields["total_value_chf"] == "32103.76"
    assert fields["actor_role"] == "a non-executive member of the board of directors"


def test_parse_management_transaction_sale_with_large_quantity():
    fields = _parse_management_transaction(
        "Sale of 14,885 securities amounting to CHF 46,217.93 (CHF 3.11 / security) "
        "by an executive member of the board of directors / member of senior management"
    )
    assert fields["action"] == "Sale"
    assert fields["quantity"] == "14885"
    assert fields["price_per_security_chf"] == "3.11"
    assert fields["total_value_chf"] == "46217.93"


def test_parse_management_transaction_unrecognized_template_returns_none_fields():
    """A future SER template change must degrade to None fields, never raise."""
    fields = _parse_management_transaction("Some future free-text format SER hasn't used yet")
    assert fields == {
        "action": None,
        "quantity": None,
        "price_per_security_chf": None,
        "total_value_chf": None,
        "actor_role": None,
    }


# ------------------------------------------------------------------
# SER RSS feed downloads
# ------------------------------------------------------------------


def test_download_management_transactions_parses_and_caches(downloader, tmp_path):
    with patch("requests.get", return_value=_rss_response(_MGMT_TXN_FEED_XML)):
        df = downloader.download_management_transactions()

    assert len(df) == 2
    assert set(df["filing_id"]) == {"T1Q9700014", "T1Q9700022"}
    kardex = df[df["filing_id"] == "T1Q9700014"].iloc[0]
    assert kardex["company"] == "Kardex Holding AG"
    assert kardex["action"] == "Purchase"
    assert kardex["quantity"] == "128"

    cache_file = tmp_path / "swiss" / "ser" / "management_transactions.csv"
    assert cache_file.exists()
    cached = pd.read_csv(cache_file)
    assert len(cached) == 2


def test_download_significant_shareholders_caches_company_and_link(downloader, tmp_path):
    with patch("requests.get", return_value=_rss_response(_SIG_SHAREHOLDERS_FEED_XML)):
        df = downloader.download_significant_shareholders()

    assert len(df) == 1
    row = df.iloc[0]
    assert row["company"] == "ARYZTA AG"
    assert row["filing_id"] == "ZA01-000000000SKW3"
    assert "shareholder-details" in row["link"]

    cache_file = tmp_path / "swiss" / "ser" / "significant_shareholders.csv"
    assert cache_file.exists()


def test_incremental_download_dedups_already_cached_items(downloader):
    with patch("requests.get", return_value=_rss_response(_MGMT_TXN_FEED_XML)) as mock_get:
        first = downloader.download_management_transactions()
        second = downloader.download_management_transactions()

    assert mock_get.call_count == 2  # feed is re-polled each call ...
    assert len(first) == 2
    assert len(second) == 2  # ... but no duplicate rows are appended


def test_filing_id_extracted_from_guid_fragment(downloader):
    with patch("requests.get", return_value=_rss_response(_SIG_SHAREHOLDERS_FEED_XML)):
        items = downloader._fetch_ser_feed_items("significant_shareholders")
    assert items[0]["filing_id"] == "ZA01-000000000SKW3"


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
