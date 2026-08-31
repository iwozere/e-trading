"""
P22 — FDA Orange Book and Purple Book clients (spec §2.3).

Orange Book (`products.txt`, `patent.txt`, `exclusivity.txt`) is the core of
the acquirer-pressure model: `patent.txt`'s `Patent_Expire_Date_Text` per
application/product drives revenue-at-risk once joined to product revenue
(spec §2.3, "the highest-value and highest-effort part of the build").
Purple Book covers the biologic-exclusivity equivalent for large-molecule
products.

Both are quarterly, infrequent, large single downloads — no per-request
pagination or aggressive rate limiting needed, unlike the CT.gov/openFDA
clients.
"""

from __future__ import annotations

import csv
import io
import re
import sys
import zipfile
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.config import (
    ORANGE_BOOK_ZIP_URL,
    PURPLE_BOOK_DOWNLOADS_PAGE,
)
from src.ml.pipeline.p22_biotech_ma.ingest import raw_zone
from src.ml.pipeline.p22_biotech_ma.ingest.http_retry import get_with_retry
from src.ml.pipeline.p22_biotech_ma.ingest.rate_limits import fda_book_limiter
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_ORANGE_BOOK_FILES = ("products.txt", "patent.txt", "exclusivity.txt")

# e.g. "https://www.accessdata.fda.gov/drugsatfda_docs/PurpleBook/2026/purplebook-search-August-data-download.csv"
_PURPLE_BOOK_LINK_RE = re.compile(
    r'href="(https://www\.accessdata\.fda\.gov/drugsatfda_docs/PurpleBook/(\d{4})/'
    r'purplebook-search-([A-Za-z]+)-data-download\.csv)"',
    re.IGNORECASE,
)
_MONTH_NUMBERS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
# The real, current biologics database, despite the filename: FDA publishes
# one dated CSV per month, but each is a full snapshot with that month's
# New/Updated rows flagged (N/R/U column) — not a diff. Verified 2026-08-30
# by row count (>2000 rows, most with a blank N/R/U flag).
_PURPLE_BOOK_HEADER_MARKER = "N/R/U"


def _get_ok(client: httpx.Client, url: str) -> Optional[httpx.Response]:
    """GET with retry, returning the response only on 2xx (None otherwise)."""
    resp = get_with_retry(client, url, rate_limiter=fda_book_limiter, follow_redirects=True)
    if resp is None:
        return None
    if resp.status_code != 200:
        _logger.error("Request to %s failed: status %d", url, resp.status_code)
        return None
    return resp


def fetch_and_land_orange_book(url: str = ORANGE_BOOK_ZIP_URL) -> Dict[str, "raw_zone.RawZoneWriteResult"]:
    """
    Download the current Orange Book ZIP and land `products.txt`, `patent.txt`,
    and `exclusivity.txt` separately in the raw zone (spec §2.3).

    Returns:
        Mapping of filename -> RawZoneWriteResult for each file successfully
        extracted and landed; missing files are logged and omitted.
    """
    today = date.today()
    results: Dict[str, raw_zone.RawZoneWriteResult] = {}

    with httpx.Client(timeout=120.0) as client:
        resp = _get_ok(client, url)
        if resp is None:
            return results

        try:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                for filename in _ORANGE_BOOK_FILES:
                    if filename not in zf.namelist():
                        _logger.warning("Orange Book archive missing expected file: %s", filename)
                        continue
                    with zf.open(filename) as f:
                        text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
                        reader = csv.DictReader(text, delimiter="~")
                        rows: List[Dict[str, Any]] = list(reader)
                    result = raw_zone.write(
                        source="orange_book",
                        entity=filename,
                        as_of_date=today,
                        payload=rows,
                    )
                    results[filename] = result
        except zipfile.BadZipFile as exc:
            _logger.error("Orange Book archive malformed: %s", exc)
            return results

    _logger.info("Landed %d/%d Orange Book files", len(results), len(_ORANGE_BOOK_FILES))
    return results


def discover_latest_purple_book_url(client: Optional[httpx.Client] = None) -> Optional[str]:
    """
    Fetch the Purple Book downloads listing and return the most recent
    month's CSV URL. There is no stable "latest" URL (spec §2.3's Purple
    Book source shape is a dated monthly file, unlike Orange Book's single
    ZIP) — derive it rather than hardcode a month/year, same reasoning as
    the SEC DERA landing page.

    Returns:
        The most recent CSV URL, or None if discovery failed.
    """
    owns_client = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        resp = _get_ok(client, PURPLE_BOOK_DOWNLOADS_PAGE)
        if resp is None:
            return None

        candidates = []
        for url, year, month_name in _PURPLE_BOOK_LINK_RE.findall(resp.text):
            month_num = _MONTH_NUMBERS.get(month_name.lower())
            if month_num is None:
                continue
            candidates.append(((int(year), month_num), url))

        if not candidates:
            _logger.error("No Purple Book CSV links found on downloads page")
            return None

        candidates.sort(key=lambda c: c[0])
        latest_url = candidates[-1][1]
        _logger.info("Discovered latest Purple Book CSV: %s", latest_url)
        return latest_url
    finally:
        if owns_client:
            client.close()


def _parse_purple_book_csv(text: str) -> List[Dict[str, Any]]:
    """
    Skip the report-title preamble rows and parse from the real header row
    (identified by its leading `N/R/U` column, not a fixed row offset, since
    the exact preamble length isn't documented and could drift).
    """
    lines = text.splitlines()
    header_idx = next((i for i, line in enumerate(lines) if line.startswith(_PURPLE_BOOK_HEADER_MARKER)), None)
    if header_idx is None:
        _logger.error("Could not locate Purple Book CSV header row (expected to start with %r)", _PURPLE_BOOK_HEADER_MARKER)
        return []
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_idx:])))
    return list(reader)


def fetch_and_land_purple_book() -> Optional["raw_zone.RawZoneWriteResult"]:
    """
    Discover and download the current Purple Book CSV and land it in the raw
    zone (spec §2.3).

    Returns:
        RawZoneWriteResult, or None on failure.
    """
    today = date.today()

    with httpx.Client(timeout=120.0) as client:
        url = discover_latest_purple_book_url(client)
        if url is None:
            return None

        resp = _get_ok(client, url)
        if resp is None:
            return None

        text = resp.content.decode("utf-8", errors="replace")
        rows = _parse_purple_book_csv(text)

    result = raw_zone.write(
        source="purple_book",
        entity="purple_book_full",
        as_of_date=today,
        payload=rows,
    )
    _logger.info("Landed Purple Book: %d rows", len(rows))
    return result
