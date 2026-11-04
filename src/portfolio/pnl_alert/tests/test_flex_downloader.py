"""Unit tests for the IBKR Flex Web Service downloader."""

from pathlib import Path
from typing import List

import pytest
import requests

from src.portfolio.pnl_alert import flex_downloader
from src.portfolio.pnl_alert.flex_downloader import FlexQueryError, download_flex_statement, download_open_positions_xml

_SEND_REQUEST_SUCCESS = b"""<FlexStatementResponse timestamp='now'>
<Status>Success</Status>
<ReferenceCode>REF123</ReferenceCode>
<Url>https://example.invalid/GetStatement</Url>
</FlexStatementResponse>
"""

_SEND_REQUEST_FAIL = b"""<FlexStatementResponse timestamp='now'>
<Status>Fail</Status>
<ErrorCode>1003</ErrorCode>
<ErrorMessage>Invalid request or unable to validate request</ErrorMessage>
</FlexStatementResponse>
"""

_STATEMENT_IN_PROGRESS = b"""<FlexStatementResponse timestamp='now'>
<Status>Warn</Status>
<ErrorCode>1019</ErrorCode>
<ErrorMessage>Statement generation in progress. Please try again shortly.</ErrorMessage>
</FlexStatementResponse>
"""

_STATEMENT_FATAL_ERROR = b"""<FlexStatementResponse timestamp='now'>
<Status>Fail</Status>
<ErrorCode>1012</ErrorCode>
<ErrorMessage>Token has expired</ErrorMessage>
</FlexStatementResponse>
"""

_STATEMENT_SUCCESS = b"""<FlexQueryResponse queryName="Open Positions" type="AF">
<FlexStatements count="1">
<FlexStatement accountId="U123">
<OpenPositions>
<OpenPosition symbol="NVDA" position="20" markPrice="204.87" costBasisPrice="179.19" costBasisMoney="3583.80" />
</OpenPositions>
</FlexStatement>
</FlexStatements>
</FlexQueryResponse>
"""


class _FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        pass


def _queue_get(responses: List[bytes], monkeypatch: pytest.MonkeyPatch, sleeps: List[float] | None = None) -> None:
    """Patch requests.get to return each queued body in order, and no-op time.sleep."""
    calls = iter(responses)

    def _fake_get(*_args: object, **_kwargs: object) -> _FakeResponse:
        return _FakeResponse(next(calls))

    monkeypatch.setattr(flex_downloader.requests, "get", _fake_get)
    monkeypatch.setattr(
        flex_downloader.time,
        "sleep",
        lambda s: sleeps.append(s) if sleeps is not None else None,
    )


def test_download_success_writes_fixed_and_dated_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _queue_get([_SEND_REQUEST_SUCCESS, _STATEMENT_SUCCESS], monkeypatch)

    result = download_open_positions_xml(tmp_path, token="tok", query_id="qid")

    assert result is not None
    assert result.parent == tmp_path
    assert result.name.startswith("Open_Positions-") and result.name.endswith(".xml")

    fixed = tmp_path / "Open_Positions.xml"
    assert fixed.read_bytes() == _STATEMENT_SUCCESS
    assert result.read_bytes() == _STATEMENT_SUCCESS


def test_download_missing_credentials_returns_none_and_writes_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = download_open_positions_xml(tmp_path, token="", query_id="")

    assert result is None
    assert list(tmp_path.iterdir()) == []


def test_download_send_request_failure_returns_none_and_preserves_existing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixed = tmp_path / "Open_Positions.xml"
    fixed.write_bytes(b"<old/>")

    _queue_get([_SEND_REQUEST_FAIL], monkeypatch)

    result = download_open_positions_xml(tmp_path, token="tok", query_id="qid")

    assert result is None
    assert fixed.read_bytes() == b"<old/>"  # untouched — caller falls back to this


def test_get_statement_retries_on_in_progress_then_succeeds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sleeps: List[float] = []
    _queue_get(
        [_SEND_REQUEST_SUCCESS, _STATEMENT_IN_PROGRESS, _STATEMENT_IN_PROGRESS, _STATEMENT_SUCCESS],
        monkeypatch,
        sleeps=sleeps,
    )

    result = download_open_positions_xml(tmp_path, token="tok", query_id="qid")

    assert result is not None
    assert result.read_bytes() == _STATEMENT_SUCCESS
    assert len(sleeps) == 2  # slept once per "in progress" response


def test_get_statement_fatal_error_returns_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _queue_get([_SEND_REQUEST_SUCCESS, _STATEMENT_FATAL_ERROR], monkeypatch)

    result = download_open_positions_xml(tmp_path, token="tok", query_id="qid")

    assert result is None
    assert list(tmp_path.iterdir()) == []


def test_get_statement_gives_up_after_max_attempts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(flex_downloader, "_POLL_MAX_ATTEMPTS", 3)
    sleeps: List[float] = []
    _queue_get(
        [_SEND_REQUEST_SUCCESS, _STATEMENT_IN_PROGRESS, _STATEMENT_IN_PROGRESS, _STATEMENT_IN_PROGRESS],
        monkeypatch,
        sleeps=sleeps,
    )

    result = download_open_positions_xml(tmp_path, token="tok", query_id="qid")

    assert result is None
    assert len(sleeps) == 3


def test_download_flex_statement_returns_raw_xml(monkeypatch: pytest.MonkeyPatch) -> None:
    _queue_get([_SEND_REQUEST_SUCCESS, _STATEMENT_SUCCESS], monkeypatch)

    assert download_flex_statement("tok", "qid") == _STATEMENT_SUCCESS


def test_download_flex_statement_raises_on_ibkr_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _queue_get([_SEND_REQUEST_FAIL], monkeypatch)

    with pytest.raises(FlexQueryError, match="1003"):
        download_flex_statement("tok", "qid")


def test_download_flex_statement_wraps_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_args: object, **_kwargs: object) -> None:
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(flex_downloader.requests, "get", _raise)

    with pytest.raises(FlexQueryError, match="boom"):
        download_flex_statement("tok", "qid")


def test_download_flex_statement_wraps_unparseable_response(monkeypatch: pytest.MonkeyPatch) -> None:
    _queue_get([b"not xml"], monkeypatch)

    with pytest.raises(FlexQueryError):
        download_flex_statement("tok", "qid")
