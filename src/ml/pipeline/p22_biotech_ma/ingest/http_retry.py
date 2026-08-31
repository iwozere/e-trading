"""
P22 — shared HTTP GET-with-retry helper (spec §7.2).

Retries only on 429 (rate limited) and 5xx (server error), with exponential
backoff. Any other 4xx (400 bad request, 404 not found, etc.) is a
non-retryable client error — retrying it five times with backoff wastes a
request budget on something that will never succeed and delays surfacing the
real problem (a malformed query, a wrong field name) by up to ~30 seconds.

Used by every P22 HTTP client rather than each hand-rolling its own retry
loop, so this one distinction (retryable vs. not) is enforced consistently.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

MAX_ATTEMPTS = 5
_RETRYABLE_STATUS = {429}


def get_with_retry(
    client: httpx.Client,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    rate_limiter: Optional[Any] = None,
    max_attempts: int = MAX_ATTEMPTS,
    **kwargs: Any,
) -> Optional[httpx.Response]:
    """
    GET with exponential backoff, retrying only on 429 or 5xx.

    Args:
        client: An httpx.Client to issue the request on.
        url: Request URL.
        params: Optional query params.
        headers: Optional headers.
        rate_limiter: Optional object with an `acquire()` method, called
            before each attempt (e.g. `src.data.utils.rate_limiting.RateLimiter`).
        max_attempts: Max attempts before giving up.
        **kwargs: Passed through to `client.get`.

    Returns:
        The response on success (2xx) or on a non-retryable status (so the
        caller can inspect it, e.g. to treat a 404 as "no results"). None if
        every retryable attempt failed, or on a transport-level failure.
    """
    for attempt in range(1, max_attempts + 1):
        if rate_limiter is not None:
            rate_limiter.acquire()
        try:
            resp = client.get(url, params=params, headers=headers, **kwargs)
        except httpx.TransportError as exc:
            if attempt == max_attempts:
                _logger.error("Request to %s failed after %d attempts: %s", url, max_attempts, exc)
                return None
            _backoff(url, attempt, max_attempts, exc)
            continue

        if resp.status_code in _RETRYABLE_STATUS or resp.status_code >= 500:
            if attempt == max_attempts:
                _logger.error(
                    "Request to %s failed after %d attempts: retryable status %d",
                    url,
                    max_attempts,
                    resp.status_code,
                )
                return None
            _backoff(url, attempt, max_attempts, f"status {resp.status_code}")
            continue

        # Success (2xx) or a non-retryable client error (4xx other than 429):
        # return it either way so the caller can decide (e.g. treat 404 as
        # "no results" rather than a failure).
        return resp

    return None


def _backoff(url: str, attempt: int, max_attempts: int, reason: object) -> None:
    backoff = min(2**attempt, 30.0)
    _logger.warning(
        "Request to %s attempt %d/%d failed (%s), backing off %.1fs",
        url,
        attempt,
        max_attempts,
        reason,
        backoff,
    )
    time.sleep(backoff)
