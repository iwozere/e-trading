"""
Call-budgeted, disk-cached FMP client for the free-key probe (spec §4.2, plan WP-1).

- The API key goes in the ``apikey`` request **header**, never the query string (spec §3), so it cannot reach
  URLs, logs or the cache.
- Every response is cached as gzip JSON keyed by ``sha1(endpoint|params|auth)``. A re-run reads the cache and
  costs zero calls; the probe is idempotent.
- Every live call is appended to ``ledger.jsonl``. The budget (default 200) counts the ledger across runs, so
  repeated runs can never exceed it in total.
- Cached bodies stay under ``data_root`` and are never committed (D5).

This is deliberately a small synchronous client. The production async client with retries, circuit breaker and
bandwidth accounting is WP-2.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import httpx

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

BASE_URL = "https://financialmodelingprep.com/stable/"
INVALID_PROBE_KEY = "invalid-probe-key-000000"
_DROPPED_HEADERS = {"set-cookie", "cookie", "authorization", "apikey"}
_TRANSIENT = {"rate_limited", "server_error"}

AuthMode = Literal["header", "invalid"]


class BudgetExhausted(Exception):
    """Raised when a live call is needed but the call budget is used up."""


class QuotaExhausted(Exception):
    """Raised when FMP reports that the account's call or bandwidth quota is used up."""


@dataclass
class ProbeResponse:
    """One recorded FMP response; ``body`` is the decoded text and stays on local disk only."""

    endpoint: str
    params: dict[str, Any]
    auth: str
    status: int
    classification: str
    headers: dict[str, str]
    bytes_wire: int
    bytes_decoded: int
    elapsed_ms: float
    fetched_at: str
    body: str
    from_cache: bool = field(default=False, compare=False)

    def json(self) -> Any:
        """Return the parsed body, or ``None`` if it is not JSON."""
        try:
            return json.loads(self.body)
        except ValueError:
            return None


def classify(status: int, body: Any, text: str) -> str:
    """
    Classify an FMP response.

    Args:
        status: HTTP status code.
        body: Parsed JSON body, or ``None`` if the body is not JSON.
        text: Raw body text.

    Returns:
        One of ``ok``, ``ok_empty``, ``plan_restricted``, ``quota_exhausted``, ``unauthorized``, ``not_found``,
        ``rate_limited``, ``server_error``, ``api_error``.
    """
    lowered = text.lower()
    if "limit reach" in lowered:
        return "quota_exhausted"
    if status == 401:
        return "unauthorized"
    if status in (402, 403) or (status >= 400 and ("subscription" in lowered or "premium" in lowered)):
        return "plan_restricted"
    if status == 404:
        return "not_found"
    if status == 429:
        return "rate_limited"
    if status >= 500:
        return "server_error"
    if status >= 400:
        return "api_error"
    if isinstance(body, dict) and ("Error Message" in body or "error" in body):
        message = str(body.get("Error Message", body.get("error", ""))).lower()
        if "subscription" in message or "premium" in message or "plan" in message:
            return "plan_restricted"
        if "invalid api key" in message:
            return "unauthorized"
        return "api_error"
    if body is None:
        return "ok_empty" if not text.strip() else "api_error"
    if body in ([], {}):
        return "ok_empty"
    return "ok"


def row_count(body: Any) -> int:
    """Return the number of records in a parsed body: list length, a wrapped list's length, or 1 for an object."""
    if isinstance(body, list):
        return len(body)
    if isinstance(body, dict):
        for key in ("historical", "data", "results"):
            if isinstance(body.get(key), list):
                return len(body[key])
        return 1 if body else 0
    return 0


class ProbeClient:
    """Synchronous FMP GET client with a persistent response cache and a cross-run call budget."""

    def __init__(
        self,
        api_key: Optional[str],
        cache_dir: Path,
        budget: int = 200,
        offline: bool = False,
        min_interval_s: float = 0.4,
        base_url: str = BASE_URL,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        """
        Create a probe client.

        Args:
            api_key: FMP key; may be ``None`` only when ``offline`` is true.
            cache_dir: Directory for cached responses and the call ledger.
            budget: Maximum number of live calls across all runs sharing ``cache_dir``.
            offline: Serve only from the cache; a cache miss raises ``BudgetExhausted``.
            min_interval_s: Minimum delay between live calls.
            base_url: FMP base URL (overridable for tests).
            http_client: Optional preconfigured ``httpx.Client`` (for tests).

        Raises:
            ValueError: If no key is given while online.
        """
        if not offline and not api_key:
            raise ValueError("FMP_API_KEY is required for a live probe")
        self._api_key = api_key or ""
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._ledger = cache_dir / "ledger.jsonl"
        self._budget = budget
        self._offline = offline
        self._min_interval_s = min_interval_s
        self._base_url = base_url
        self._client = http_client or httpx.Client(
            timeout=httpx.Timeout(60.0, connect=10.0), headers={"Accept-Encoding": "gzip"}
        )
        self._last_call = 0.0

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "ProbeClient":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        del _exc_info
        self.close()

    @property
    def calls_used(self) -> int:
        """Number of live calls recorded in the ledger across all runs."""
        if not self._ledger.exists():
            return 0
        with self._ledger.open(encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())

    def get(self, endpoint: str, params: Optional[dict[str, Any]] = None, auth: AuthMode = "header") -> ProbeResponse:
        """
        Return the response for ``endpoint`` with ``params``, from the cache if present, otherwise live.

        Args:
            endpoint: Path relative to the base URL, e.g. ``historical-price-eod/non-split-adjusted``.
            params: Query parameters (never the key).
            auth: ``header`` sends the real key; ``invalid`` sends a deliberately wrong one.

        Returns:
            The recorded response.

        Raises:
            BudgetExhausted: If a live call is needed but the budget is used up, or the client is offline.
            QuotaExhausted: If FMP reports the account quota is used up.
        """
        params = dict(params or {})
        path = self._cache_path(endpoint, params, auth)
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                return ProbeResponse(**json.load(fh), from_cache=True)
        if self._offline:
            raise BudgetExhausted(f"offline and not cached: {endpoint}")
        if self.calls_used >= self._budget:
            raise BudgetExhausted(f"call budget of {self._budget} used up")

        response = self._fetch(endpoint, params, auth)
        if response.classification == "quota_exhausted":
            raise QuotaExhausted(f"FMP quota exhausted at {endpoint}")
        # Transient failures are not cached, so a re-run retries them.
        if response.classification not in _TRANSIENT:
            with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as fh:
                record = asdict(response)
                record.pop("from_cache")
                json.dump(record, fh)
        return response

    def _fetch(self, endpoint: str, params: dict[str, Any], auth: AuthMode) -> ProbeResponse:
        """Issue one live GET, record it in the ledger and return it."""
        wait = self._min_interval_s - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        key = self._api_key if auth == "header" else INVALID_PROBE_KEY
        started = time.monotonic()
        try:
            resp = self._client.get(self._base_url + endpoint, params=params, headers={"apikey": key})
            status, text = resp.status_code, resp.text
            headers = {k.lower(): v for k, v in resp.headers.items() if k.lower() not in _DROPPED_HEADERS}
            bytes_wire, bytes_decoded = resp.num_bytes_downloaded, len(resp.content)
        except httpx.HTTPError as exc:
            _logger.warning("Probe call to %s failed: %s", endpoint, type(exc).__name__)
            status, text, headers, bytes_wire, bytes_decoded = 0, f"transport error: {type(exc).__name__}", {}, 0, 0
        elapsed_ms = (time.monotonic() - started) * 1000
        self._last_call = time.monotonic()
        if self._api_key and self._api_key in text:
            text = text.replace(self._api_key, "<redacted>")

        parsed: Any
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        classification = "server_error" if status == 0 else classify(status, parsed, text)
        fetched_at = datetime.now(timezone.utc).isoformat()
        with self._ledger.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"at": fetched_at, "endpoint": endpoint, "params": params, "auth": auth,
                                 "status": status, "bytes_wire": bytes_wire}) + "\n")
        _logger.info("Probe %s %s -> %s %s (%d bytes wire, %d decoded)", endpoint, params, status, classification,
                     bytes_wire, bytes_decoded)
        return ProbeResponse(endpoint, params, auth, status, classification, headers, bytes_wire, bytes_decoded,
                             round(elapsed_ms, 1), fetched_at, text)

    def _cache_path(self, endpoint: str, params: dict[str, Any], auth: str) -> Path:
        canonical = json.dumps({"endpoint": endpoint, "params": params, "auth": auth}, sort_keys=True)
        return self._cache_dir / f"{hashlib.sha1(canonical.encode('utf-8')).hexdigest()}.json.gz"
