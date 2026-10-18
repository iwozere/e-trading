"""
Production async FMP client (spec §6.4, plan WP-2).

- ``httpx.AsyncClient``; the key only in the ``apikey`` header (spec §3), so it never reaches URLs, logs or the
  manifest. ``Accept-Encoding: gzip``; connect timeout 10 s, read 60 s.
- Bodies are streamed and both sizes counted: ``bytes_wire`` from the socket (gzip) and ``bytes_decoded`` after
  decompression. FMP sends no ``Content-Length`` (probe F10), so nothing relies on it. Every attempt, including
  retries, is reported to ``on_bytes`` because each one counts against the bandwidth cap.
- Retries network errors, timeouts, 429 and 5xx with exponential backoff and jitter, up to 6 attempts, honouring
  ``Retry-After``. Other 4xx are classified, never retried. A quota message raises ``QuotaExhausted`` at once.
- A shared ``CircuitBreaker`` pauses every worker after 20 consecutive retryable failures. ``AuthAbort`` is
  raised after ``auth_abort_after`` consecutive 401/403 responses (revoked key or lapsed plan). When the breaker
  opens three times in a row on 429s alone, ``QuotaExhausted`` is raised too: an over-limit account whose 429
  body lacks the quota message is not hammered all night.
- ``from``/``to``/``date`` parameters are validated before sending: FMP silently ignores a malformed date and
  serves its default window instead (probe F7, D21).
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Optional

import httpx

from src.data.fmp_pipeline.client.circuit_breaker import CircuitBreaker
from src.data.fmp_pipeline.client.classify import RETRYABLE, classify, row_count
from src.data.fmp_pipeline.client.rate_limiter import AsyncRateLimiter
from src.data.fmp_pipeline.errors import AuthAbort, QuotaExhausted
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

BASE_URL = "https://financialmodelingprep.com/stable/"
DATE_PARAMS = ("from", "to", "date")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MAX_RETRY_AFTER_S = 300.0


@dataclass
class FetchResult:
    """The final outcome of one logical request (after retries). ``body`` is the decoded body."""

    endpoint: str
    params: dict[str, Any]
    http_status: int
    classification: str
    body: bytes
    bytes_wire: int
    bytes_decoded: int
    rows: int
    attempts: int
    fetched_at: str
    error: str = ""

    def json(self) -> Any:
        """Return the parsed body, or ``None`` if it is not JSON."""
        try:
            return json.loads(self.body)
        except ValueError:
            return None


def validate_params(params: dict[str, Any]) -> None:
    """
    Reject malformed date parameters before they reach FMP (D21).

    Args:
        params: Request parameters.

    Raises:
        ValueError: If ``from``/``to``/``date`` is not a real ``YYYY-MM-DD`` date, or ``from`` is after ``to``.
    """
    parsed: dict[str, date] = {}
    for name in DATE_PARAMS:
        if name not in params:
            continue
        value = str(params[name])
        try:
            if not _DATE_RE.match(value):
                raise ValueError
            parsed[name] = date.fromisoformat(value)
        except ValueError:
            raise ValueError(f"parameter {name!r} must be YYYY-MM-DD, got {value!r}") from None
    if "from" in parsed and "to" in parsed and parsed["from"] > parsed["to"]:
        raise ValueError(f"'from' {parsed['from']} is after 'to' {parsed['to']}")


def parse_retry_after(value: Optional[str], now: Optional[datetime] = None) -> Optional[float]:
    """
    Parse a ``Retry-After`` header (seconds or an HTTP date) into seconds, capped at ``MAX_RETRY_AFTER_S``.

    Args:
        value: Header value, or ``None``.
        now: Current time for HTTP-date values (defaults to now, UTC).

    Returns:
        Seconds to wait, or ``None`` if absent or unparseable.
    """
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return min(float(value), MAX_RETRY_AFTER_S)
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    seconds = (when - (now or datetime.now(timezone.utc))).total_seconds()
    return min(max(seconds, 0.0), MAX_RETRY_AFTER_S)


class FmpClient:
    """Async FMP GET client shared by all download workers."""

    def __init__(
        self,
        api_key: str,
        base_url: str = BASE_URL,
        rate_limiter: Optional[AsyncRateLimiter] = None,
        breaker: Optional[CircuitBreaker] = None,
        max_attempts: int = 6,
        backoff_base_s: float = 1.0,
        backoff_cap_s: float = 60.0,
        auth_abort_after: int = 10,
        on_bytes: Optional[Callable[[int, int], None]] = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: Optional[random.Random] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """
        Args:
            api_key: FMP key; sent only as the ``apikey`` header.
            base_url: API base URL ending in ``/``.
            rate_limiter: Shared pacer; defaults to 600 requests/min.
            breaker: Shared circuit breaker; defaults to 20 failures / 60 s.
            max_attempts: Attempts per request for retryable failures (spec: 6).
            backoff_base_s: Backoff before the 2nd attempt; doubles per attempt, with ±50% jitter.
            backoff_cap_s: Upper bound for one backoff.
            auth_abort_after: Consecutive 401/403 responses that abort the run.
            on_bytes: Called with ``(bytes_wire, bytes_decoded)`` after every attempt that received a response.
            sleep: Async sleep (injectable for tests).
            rng: Random source for jitter (injectable for tests).
            http_client: Preconfigured ``httpx.AsyncClient`` (for tests); otherwise one is created.

        Raises:
            ValueError: If ``api_key`` is empty.
        """
        if not api_key:
            raise ValueError("FMP API key is required")
        self._api_key = api_key
        self._base_url = base_url
        self.rate_limiter = rate_limiter or AsyncRateLimiter(600)
        self.breaker = breaker or CircuitBreaker()
        self._max_attempts = max_attempts
        self._backoff_base_s = backoff_base_s
        self._backoff_cap_s = backoff_cap_s
        self._auth_abort_after = auth_abort_after
        self.on_bytes = on_bytes
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._auth_failures = 0
        self._http = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0), headers={"Accept-Encoding": "gzip"}
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()

    async def __aenter__(self) -> "FmpClient":
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        del _exc_info
        await self.aclose()

    async def fetch(self, endpoint: str, params: Optional[dict[str, Any]] = None) -> FetchResult:
        """
        GET one endpoint, retrying transient failures.

        Args:
            endpoint: Path relative to the base URL, e.g. ``historical-price-eod/non-split-adjusted``.
            params: Query parameters (never the key). Symbols are URL-encoded by httpx.

        Returns:
            The final result. After exhausted retries its classification is the last retryable one
            (``rate_limited``/``server_error``/``network_error``).

        Raises:
            ValueError: On malformed date parameters (nothing is sent).
            QuotaExhausted: When FMP reports the quota is used up, or keeps answering 429 through repeated
                circuit-breaker pauses.
            AuthAbort: After ``auth_abort_after`` consecutive 401/403 responses.
        """
        params = dict(params or {})
        validate_params(params)
        result: Optional[FetchResult] = None
        for attempt in range(1, self._max_attempts + 1):
            await self.breaker.wait_if_open()
            await self.rate_limiter.acquire()
            result, retry_after = await self._attempt(endpoint, params, attempt)

            if result.classification == "quota_exhausted":
                raise QuotaExhausted(f"FMP quota exhausted at {endpoint}")
            self._track_auth(result)
            if result.classification not in RETRYABLE:
                self.breaker.record_success()
                return result

            self.breaker.record_failure(rate_limited=result.classification == "rate_limited")
            if self.breaker.rate_limit_exhausted:
                _logger.error("FMP kept answering 429 through %d circuit-breaker pauses; stopping the run. Last "
                              "response at %s: HTTP %d %s", self.breaker.rate_limited_opens, endpoint,
                              result.http_status, result.error)
                raise QuotaExhausted(f"FMP kept answering 429 at {endpoint} through "
                                     f"{self.breaker.rate_limited_opens} breaker pauses: {result.error[:200]}")
            if attempt < self._max_attempts:
                delay = self._backoff(attempt, retry_after)
                _logger.warning("FMP %s %s -> %s (attempt %d/%d), retrying in %.1fs", endpoint,
                                _loggable(params), result.classification, attempt, self._max_attempts, delay)
                await self._sleep(delay)

        assert result is not None
        _logger.error("FMP %s %s failed after %d attempts: %s", endpoint, _loggable(params), self._max_attempts,
                      result.classification)
        return result

    async def _attempt(
        self, endpoint: str, params: dict[str, Any], attempt: int
    ) -> tuple[FetchResult, Optional[float]]:
        """Send one request; return its result and any ``Retry-After`` seconds."""
        fetched_at = datetime.now(timezone.utc).isoformat()
        try:
            async with self._http.stream("GET", self._base_url + endpoint, params=params,
                                         headers={"apikey": self._api_key}) as response:
                chunks = [chunk async for chunk in response.aiter_bytes()]
                wire = response.num_bytes_downloaded
                status = response.status_code
                retry_after = parse_retry_after(response.headers.get("Retry-After"))
        except httpx.TransportError as exc:
            # Includes timeouts. No usable response: nothing is counted against the bandwidth cap.
            return (FetchResult(endpoint, params, 0, "network_error", b"", 0, 0, 0, attempt, fetched_at,
                                type(exc).__name__), None)

        body = b"".join(chunks)
        if self.on_bytes is not None:
            self.on_bytes(wire, len(body))
        text = body.decode("utf-8", errors="replace")
        if self._api_key in text:
            text = text.replace(self._api_key, "<redacted>")
            body = text.encode("utf-8")
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        classification = classify(status, parsed, text)
        error = "" if classification in ("ok", "ok_empty") else text[:300]
        return (FetchResult(endpoint, params, status, classification, body, wire, len(body), row_count(parsed),
                            attempt, fetched_at, error), retry_after)

    def _track_auth(self, result: FetchResult) -> None:
        if result.http_status in (401, 403):
            self._auth_failures += 1
            if self._auth_failures >= self._auth_abort_after:
                raise AuthAbort(f"{self._auth_failures} consecutive 401/403 responses (last: {result.endpoint})")
        else:
            self._auth_failures = 0

    def _backoff(self, attempt: int, retry_after: Optional[float]) -> float:
        exponential = min(self._backoff_cap_s, self._backoff_base_s * 2 ** (attempt - 1))
        jittered = exponential * self._rng.uniform(0.5, 1.5)
        return max(jittered, retry_after or 0.0)


def _loggable(params: dict[str, Any]) -> str:
    return ",".join(f"{k}={v}" for k, v in params.items())
