"""
HTTP client for the Sharadar API (``https://api.sharadar.com/v1.0``).

Two access paths: paged CSV queries (``query``) for slices and the free sample, and whole-table bulk files
(``bulk_status`` / ``download_bulk``), which need a paid key and answer with a redirect to a pre-signed zip.
"""

from __future__ import annotations

import hashlib
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
import pandas as pd

from src.data.sharadar_pipeline.config import API_BASE
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

PAGE_SIZE = 10_000
_RETRY_STATUS = {429, 500, 502, 503, 504}


class SharadarError(RuntimeError):
    """The API answered with an error."""


class AuthError(SharadarError):
    """No valid API key (HTTP 401)."""


class FreeTierExceeded(SharadarError):
    """The request needs a subscription (HTTP 403 "Exceeds free tier")."""


@dataclass(frozen=True)
class BulkFile:
    """One downloaded bulk file."""

    table: str
    path: Path
    size: int
    sha256: str


class SharadarClient:
    """Thin synchronous client; one instance shares one connection pool."""

    def __init__(self, api_key: str, base_url: str = API_BASE, client: Optional[httpx.Client] = None,
                 max_retries: int = 4, sleep: Callable[[float], None] = time.sleep) -> None:
        """
        Args:
            api_key: Paid key, or the public sample key.
            base_url: API root.
            client: Existing ``httpx.Client`` (tests pass one with a mock transport).
            max_retries: Retries on 429 and 5xx answers and on transport errors.
            sleep: Wait function (tests pass a no-op).
        """
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=httpx.Timeout(120.0, connect=30.0), follow_redirects=True)
        self._max_retries = max_retries
        self._sleep = sleep

    def close(self) -> None:
        """Close the connection pool."""
        self._client.close()

    def _get(self, path: str, params: dict[str, Any]) -> httpx.Response:
        params = {"api_key": self._key, **params}
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.get(f"{self._base}{path}", params=params)
            except httpx.TransportError as exc:
                if attempt == self._max_retries:
                    raise SharadarError(f"transport error on {path}: {exc}") from exc
                self._sleep(2.0 ** attempt)
                continue
            if response.status_code in _RETRY_STATUS and attempt < self._max_retries:
                self._sleep(_retry_wait(response, attempt))
                continue
            _raise_for_error(response, path)
            return response
        raise SharadarError(f"retries exhausted on {path}")

    def query(self, table: str, params: Optional[dict[str, Any]] = None, max_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Read rows of a table with paged CSV queries.

        Args:
            table: Table name (``stocks``, ``tickers``, …).
            params: API filters (``ticker``, ``from``, ``to``, ``fields``, ``table``, ``dimension``, ``sort``, …).
                Pass ``from`` explicitly: the API defaults to one year ago. Give ``sort`` a unique order when a
                result spans several pages; the default order is not stable across pages.
            max_rows: Stop after this many rows.

        Returns:
            All rows as strings (callers parse types); empty frame if nothing matches.

        Raises:
            AuthError, FreeTierExceeded, SharadarError.
        """
        parts: list[pd.DataFrame] = []
        skip = 0
        while True:
            response = self._get(f"/data/{table}", {**(params or {}), "format": "csv", "limit": PAGE_SIZE, "skip": skip})
            text = response.text
            page = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False) if text.strip() else pd.DataFrame()
            if page.empty:
                if not parts and len(page.columns):
                    parts.append(page)
                break
            parts.append(page)
            skip += len(page)
            if len(page) < PAGE_SIZE or (max_rows is not None and skip >= max_rows):
                break
        if not parts:
            return pd.DataFrame()
        out = pd.concat(parts, ignore_index=True)
        return out.head(max_rows) if max_rows is not None else out

    def bulk_status(self, table: str, years: str = "full") -> dict[str, Any]:
        """Return the bulk file's name, size and last-modified time without downloading it (paid key)."""
        payload = self._get(f"/data/{table}", {"years": years, "status": "True"}).json()
        return payload if isinstance(payload, dict) else {"files": payload}

    def download_bulk(self, table: str, dest: Path, years: str = "full") -> BulkFile:
        """
        Download a table's bulk zip to ``dest`` (written to ``dest.part`` first, then renamed).

        Args:
            table: Table name.
            dest: Target file.
            years: ``full``, ``10`` or ``5``; must match the subscription.

        Returns:
            Path, size and SHA-256 of the file.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        digest = hashlib.sha256()
        size = 0
        params = {"api_key": self._key, "years": years}
        with self._client.stream("GET", f"{self._base}/data/{table}", params=params) as response:
            if response.status_code >= 400:
                response.read()
                _raise_for_error(response, f"/data/{table} (bulk)")
            with part.open("wb") as handle:
                for chunk in response.iter_bytes(1 << 20):
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
        part.replace(dest)
        _logger.info("Downloaded %s bulk file: %.1f MB", table, size / 1e6)
        return BulkFile(table, dest, size, digest.hexdigest())

    def schema(self, table: str, fmt: str = "postgres") -> str:
        """Return the public table schema as SQL text."""
        return self._get(f"/schema/{table}", {"format": fmt}).text


def _retry_wait(response: httpx.Response, attempt: int) -> float:
    reset = response.headers.get("X-RateLimit-Reset")
    if response.status_code == 429 and reset and reset.isdigit():
        return float(min(max(int(reset) - time.time(), 1.0), 300.0))
    return 2.0 ** attempt


def _raise_for_error(response: httpx.Response, path: str) -> None:
    if response.status_code < 400:
        return
    detail = response.text[:200]
    if response.status_code == 401:
        raise AuthError(f"{path}: {detail}")
    if response.status_code == 403:
        raise FreeTierExceeded(f"{path}: {detail}")
    raise SharadarError(f"{path}: HTTP {response.status_code} {detail}")
