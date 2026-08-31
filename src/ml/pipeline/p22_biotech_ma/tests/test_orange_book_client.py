"""Tests for ingest/orange_book_client.py — mocked HTTP, no network calls."""

import io
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.ingest.orange_book_client import (
    discover_latest_purple_book_url,
    fetch_and_land_orange_book,
    fetch_and_land_purple_book,
)


def _make_orange_book_zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("products.txt", "Ingredient~Trade_Name~Applicant\nADALIMUMAB~HUMIRA~ABBVIE\n")
        zf.writestr("patent.txt", "Appl_No~Product_No~Patent_No~Patent_Expire_Date_Text\n123456~001~9999999~Jan 1, 2030\n")
        zf.writestr("exclusivity.txt", "Appl_No~Product_No~Exclusivity_Code~Exclusivity_Date\n123456~001~NCE~Jan 1, 2028\n")
    return buf.getvalue()


def _mock_response(content: bytes, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.raise_for_status = MagicMock()
    return resp


def test_fetch_and_land_orange_book_extracts_all_three_files():
    zip_bytes = _make_orange_book_zip_bytes()
    with patch("httpx.Client.get", return_value=_mock_response(zip_bytes)):
        with patch(
            "src.ml.pipeline.p22_biotech_ma.ingest.orange_book_client.raw_zone.write"
        ) as mock_write:
            mock_write.return_value = MagicMock(was_new=True)
            results = fetch_and_land_orange_book()

    assert set(results.keys()) == {"products.txt", "patent.txt", "exclusivity.txt"}
    assert mock_write.call_count == 3

    # Verify the patent.txt payload was parsed correctly (tilde-delimited).
    patent_call = next(c for c in mock_write.call_args_list if c.kwargs["entity"] == "patent.txt")
    rows = patent_call.kwargs["payload"]
    assert rows[0]["Patent_Expire_Date_Text"] == "Jan 1, 2030"


def test_fetch_and_land_orange_book_handles_bad_zip():
    with patch("httpx.Client.get", return_value=_mock_response(b"not a zip")):
        results = fetch_and_land_orange_book()
    assert results == {}


def _make_purple_book_downloads_page_html() -> str:
    return """
    <html><body>
    <a href="https://www.accessdata.fda.gov/drugsatfda_docs/PurpleBook/2026/purplebook-search-january-data-download.csv">Jan</a>
    <a href="https://www.accessdata.fda.gov/drugsatfda_docs/PurpleBook/2026/purplebook-search-August-data-download.csv">Aug</a>
    <a href="https://www.accessdata.fda.gov/drugsatfda_docs/PurpleBook/2025/purplebook-search-december-data-download.csv">Prior Dec</a>
    </body></html>
    """


def test_discover_latest_purple_book_url_picks_max_year_month():
    client = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.text = _make_purple_book_downloads_page_html()
    resp.raise_for_status = MagicMock()
    client.get.return_value = resp

    url = discover_latest_purple_book_url(client)

    assert url == "https://www.accessdata.fda.gov/drugsatfda_docs/PurpleBook/2026/purplebook-search-August-data-download.csv"


def test_fetch_and_land_purple_book_skips_preamble_and_parses_from_real_header():
    # Mirrors the FDA file's actual shape: a title row, a blank row, a
    # section-label row, then the real header starting with N/R/U.
    csv_text = (
        "Purple Book Monthly Historical Data Changes Report - August 2026\n"
        "\n"
        "Newly Approved Products (N) / Products Added in Current Release (R) / Updated Products (U)\n"
        "N/R/U,Applicant,BLA Number,Proprietary Name\n"
        "U,Example Biologics Inc,BLA123,ExampleBio\n"
    )
    downloads_page = MagicMock()
    downloads_page.status_code = 200
    downloads_page.text = _make_purple_book_downloads_page_html()
    downloads_page.raise_for_status = MagicMock()

    csv_response = _mock_response(csv_text.encode("utf-8"))

    with patch("httpx.Client.get", side_effect=[downloads_page, csv_response]):
        with patch(
            "src.ml.pipeline.p22_biotech_ma.ingest.orange_book_client.raw_zone.write"
        ) as mock_write:
            mock_write.return_value = MagicMock(was_new=True)
            result = fetch_and_land_purple_book()

    assert result is not None
    payload = mock_write.call_args.kwargs["payload"]
    assert len(payload) == 1
    assert payload[0]["BLA Number"] == "BLA123"
    assert payload[0]["N/R/U"] == "U"


def test_fetch_and_land_purple_book_returns_none_when_discovery_fails():
    empty_page = MagicMock()
    empty_page.status_code = 200
    empty_page.text = "<html><body>no links here</body></html>"
    empty_page.raise_for_status = MagicMock()

    with patch("httpx.Client.get", return_value=empty_page):
        result = fetch_and_land_purple_book()

    assert result is None
