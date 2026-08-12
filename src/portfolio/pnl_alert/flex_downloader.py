"""
IBKR Flex Web Service downloader.

Pulls the "Open Positions" Flex Query report from IBKR's Flex Web Service
(the 2-step SendRequest / GetStatement REST API) and saves it to disk, so
``ibkr_xml_loader`` always has a same-day export to read instead of a stale
manually-exported file.
"""

import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import requests

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_SEND_REQUEST_URL = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService/SendRequest"
_FLEX_VERSION = "3"
_POLL_INTERVAL_SECONDS = 5
_POLL_MAX_ATTEMPTS = 10  # ~50s total; IBKR statements are usually ready within a few seconds
_REQUEST_TIMEOUT_SECONDS = 30
_STATEMENT_IN_PROGRESS_ERROR_CODE = "1019"

_FILE_STEM = "Open_Positions"


class FlexQueryError(RuntimeError):
    """Raised when IBKR's Flex Web Service returns a non-recoverable error."""


def _send_request(token: str, query_id: str) -> Tuple[str, str]:
    """
    Kick off report generation.

    Args:
        token: Flex Web Service token.
        query_id: Flex Query id for the "Open Positions" report.

    Returns:
        (reference_code, statement_url) to poll for the finished report.

    Raises:
        FlexQueryError: If IBKR rejects the request (e.g. bad token/query id).
    """
    resp = requests.get(
        _SEND_REQUEST_URL,
        params={"t": token, "q": query_id, "v": _FLEX_VERSION},
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    status = (root.findtext("Status") or "").strip()
    if status != "Success":
        error_code = root.findtext("ErrorCode") or "unknown"
        error_message = root.findtext("ErrorMessage") or "no message"
        raise FlexQueryError(f"SendRequest failed ({error_code}): {error_message}")

    reference_code = root.findtext("ReferenceCode")
    statement_url = root.findtext("Url")
    if not reference_code or not statement_url:
        raise FlexQueryError("SendRequest succeeded but response is missing ReferenceCode/Url")

    return reference_code, statement_url


def _get_statement(token: str, reference_code: str, statement_url: str) -> bytes:
    """
    Poll GetStatement until the report is ready.

    Args:
        token: Flex Web Service token.
        reference_code: Reference code returned by `_send_request`.
        statement_url: Statement URL returned by `_send_request`.

    Returns:
        Raw XML bytes of the finished FlexQueryResponse report.

    Raises:
        FlexQueryError: If IBKR returns a non-recoverable error, or the
            report never becomes ready within the retry budget.
    """
    for attempt in range(1, _POLL_MAX_ATTEMPTS + 1):
        resp = requests.get(
            statement_url,
            params={"q": reference_code, "t": token, "v": _FLEX_VERSION},
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()

        root = ET.fromstring(resp.content)
        if root.tag == "FlexQueryResponse":
            return resp.content

        # root.tag == "FlexStatementResponse" with Status Warn/Fail
        error_code = root.findtext("ErrorCode") or ""
        error_message = root.findtext("ErrorMessage") or "no message"
        if error_code == _STATEMENT_IN_PROGRESS_ERROR_CODE:
            _logger.debug(
                "Flex statement not ready yet (attempt %d/%d): %s",
                attempt,
                _POLL_MAX_ATTEMPTS,
                error_message,
            )
            time.sleep(_POLL_INTERVAL_SECONDS)
            continue

        raise FlexQueryError(f"GetStatement failed ({error_code}): {error_message}")

    raise FlexQueryError(f"Flex statement not ready after {_POLL_MAX_ATTEMPTS} attempts")


def download_open_positions_xml(
    out_dir: Path,
    *,
    token: str | None = None,
    query_id: str | None = None,
) -> Path | None:
    """
    Download the "Open Positions" Flex Query report and save it to disk.

    Writes two files into `out_dir`:
    - `Open_Positions.xml` (overwritten every run — fixed name for simple configs)
    - `Open_Positions-YYYY-MM-DD.xml` (date-stamped — for glob-based configs
      and as a same-day audit trail)

    Best-effort: on any failure (missing credentials, network error, IBKR
    error response), this logs and returns None without raising, so callers
    can fall back to whatever `Open_Positions.xml` is already on disk —
    matching this pipeline's existing "IBKR unreachable" tolerance.

    Args:
        out_dir: Directory to write the XML files into.
        token: Flex Web Service token. Defaults to the IBKR_FLEX_TOKEN env var.
        query_id: Flex Query id for "Open Positions". Defaults to the
            IBKR_FLEX_QUERY_ID env var.

    Returns:
        Path to the date-stamped file on success, None on failure or when
        credentials are not configured.
    """
    if token is None or query_id is None:
        from config.donotshare.donotshare import IBKR_FLEX_QUERY_ID, IBKR_FLEX_TOKEN

        token = token or IBKR_FLEX_TOKEN
        query_id = query_id or IBKR_FLEX_QUERY_ID

    if not token or not query_id:
        _logger.info("IBKR_FLEX_TOKEN / IBKR_FLEX_QUERY_ID not configured; skipping Flex Query download")
        return None

    try:
        reference_code, statement_url = _send_request(token, query_id)
        xml_bytes = _get_statement(token, reference_code, statement_url)
    except (FlexQueryError, requests.RequestException, ET.ParseError):
        _logger.exception("Failed to download Open Positions Flex Query report")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    fixed_path = out_dir / f"{_FILE_STEM}.xml"
    dated_path = out_dir / f"{_FILE_STEM}-{datetime.now(timezone.utc).date().isoformat()}.xml"

    fixed_path.write_bytes(xml_bytes)
    dated_path.write_bytes(xml_bytes)
    _logger.info("Saved Flex Query Open Positions report to %s and %s", fixed_path, dated_path)

    return dated_path


if __name__ == "__main__":
    import json

    result_path = download_open_positions_xml(PROJECT_ROOT / "src" / "portfolio" / "pnl_alert" / "config")
    result = {"success": result_path is not None, "path": str(result_path) if result_path else None}
    print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
