"""Tests for EdgarDownloader._fetch_filing_xml handling of missing-file responses."""

import sys
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.downloader.edgar_downloader import EdgarDownloader, _is_missing_document

_CIK = 2134841
_ACC = "000213484126000123"  # filer CIK prefix equals _CIK, so no agent-path retries


def _resp(status: int, body: bytes = b"") -> MagicMock:
    """Build a minimal requests-like response."""
    resp = MagicMock()
    resp.status_code = status
    resp.content = body
    resp.text = body.decode()
    return resp


_NO_SUCH_KEY = b"<Error><Code>NoSuchKey</Code><Message>The specified key does not exist.</Message></Error>"


def test_is_missing_document() -> None:
    """404 and a NoSuchKey 503 are missing; a plain 503 is throttling."""
    assert _is_missing_document(_resp(404))
    assert _is_missing_document(_resp(503, _NO_SUCH_KEY))
    assert not _is_missing_document(_resp(503, b"<html>Request Rate Threshold Exceeded</html>"))
    assert not _is_missing_document(_resp(200, b"<xml/>"))


def test_no_such_key_503_moves_to_next_candidate_without_sleeping(tmp_path: Path) -> None:
    """A NoSuchKey 503 for a guessed filename is treated like 404, not retried with backoff."""
    dl = EdgarDownloader(cache_dir=tmp_path)
    responses: Dict[str, MagicMock] = {
        "infotable.xml": _resp(503, _NO_SUCH_KEY),
        "InfoTable.xml": _resp(404),
        "form13fInfoTable.xml": _resp(200, b"<informationTable/>"),
    }
    dl._session.get = MagicMock(side_effect=lambda url, **_kwargs: responses[url.rsplit("/", 1)[1]])  # type: ignore[method-assign]
    dl._throttle = MagicMock()  # type: ignore[method-assign]

    with patch("src.data.downloader.edgar_downloader.time.sleep") as sleep:
        xml = dl._fetch_filing_xml(_CIK, _ACC)

    assert xml == "<informationTable/>"
    assert dl._session.get.call_count == 3
    sleep.assert_not_called()


def test_throttling_503_is_still_retried(tmp_path: Path) -> None:
    """A 503 without NoSuchKey keeps the backoff-and-retry behaviour."""
    dl = EdgarDownloader(cache_dir=tmp_path)
    dl._session.get = MagicMock(side_effect=[_resp(503, b"rate limited"), _resp(200, b"<informationTable/>")])  # type: ignore[method-assign]
    dl._throttle = MagicMock()  # type: ignore[method-assign]

    with patch("src.data.downloader.edgar_downloader.time.sleep") as sleep:
        xml = dl._fetch_filing_xml(_CIK, _ACC)

    assert xml == "<informationTable/>"
    sleep.assert_called_once_with(30)


def test_falls_back_to_folder_index_json_for_nonstandard_filename(tmp_path: Path) -> None:
    """When no guessed name exists, the infotable is found via the folder's index.json."""
    dl = EdgarDownloader(cache_dir=tmp_path)
    index = MagicMock(status_code=200)
    index.json.return_value = {
        "directory": {
            "item": [
                {"name": "0002134841-26-000123-index.html"},
                {"name": "primary_doc.xml"},
                {"name": "SPXGestao13FQ226.xml"},
            ]
        }
    }
    requested = []

    def fake_get(url: str, **_kwargs: object) -> MagicMock:
        requested.append(url)
        name = url.rsplit("/", 1)[1]
        if name == "index.json":
            return index
        if name == "SPXGestao13FQ226.xml":
            return _resp(200, b"<informationTable/>")
        return _resp(404)

    dl._session.get = MagicMock(side_effect=fake_get)  # type: ignore[method-assign]
    dl._throttle = MagicMock()  # type: ignore[method-assign]

    xml = dl._fetch_filing_xml(_CIK, _ACC)

    assert xml == "<informationTable/>"
    assert requested[-1] == f"https://www.sec.gov/Archives/edgar/data/{_CIK}/{_ACC}/SPXGestao13FQ226.xml"
    assert not any(u.endswith("primary_doc.xml") for u in requested)


def test_download_13f_infotable_reads_folder_listing_first(tmp_path: Path) -> None:
    """A custom-named infotable is fetched with two requests (listing + file), no guessing."""
    dl = EdgarDownloader(cache_dir=tmp_path)
    index = MagicMock(status_code=200)
    index.json.return_value = {"directory": {"item": [{"name": "primary_doc.xml"}, {"name": "SPXGestao13FQ226.xml"}]}}
    xml = (
        b'<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">'
        b"<infoTable><nameOfIssuer>APPLE INC</nameOfIssuer><cusip>037833100</cusip><value>1000</value>"
        b"<shrsOrPrnAmt><sshPrnamt>10</sshPrnamt></shrsOrPrnAmt><investmentDiscretion>SOLE</investmentDiscretion>"
        b"</infoTable></informationTable>"
    )
    requested = []

    def fake_get(url: str, **_kwargs: object) -> MagicMock:
        requested.append(url.rsplit("/", 1)[1])
        return index if url.endswith("index.json") else _resp(200, xml)

    dl._session.get = MagicMock(side_effect=fake_get)  # type: ignore[method-assign]
    dl._throttle = MagicMock()  # type: ignore[method-assign]

    path = dl.download_13f_infotable(_CIK, "0002134841-26-000123", 2026, 2)

    assert path is not None and path.exists()
    assert requested == ["index.json", "SPXGestao13FQ226.xml"]
