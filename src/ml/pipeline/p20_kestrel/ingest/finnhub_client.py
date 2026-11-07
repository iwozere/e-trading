"""
P20 Kestrel — paced Finnhub GET shared by the ingest jobs.

Finnhub's free tier allows 60 calls/min. An unpaced loop over the ~375-ticker watchlist makes ~7 calls/s, so most
calls get a 429; callers that treated that as "no data" silently lost most of their tickers (revisions 2026-11-05,
catalyst sync until 2026-11-07).
"""

from __future__ import annotations

import time
from typing import Any, Dict

import requests

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

FINNHUB_MIN_INTERVAL_S = 1.1
FINNHUB_429_BACKOFF_S = 30.0


class FinnhubThrottle:
    """Enforce a minimum interval between successive calls (single-threaded)."""

    def __init__(self, min_interval_s: float = FINNHUB_MIN_INTERVAL_S) -> None:
        """
        Create a throttle.

        Args:
            min_interval_s: Minimum seconds between two calls.
        """
        self.min_interval_s = min_interval_s
        self._last = float("-inf")

    def wait(self) -> None:
        """Sleep until at least ``min_interval_s`` has passed since the previous call."""
        remaining = self._last + self.min_interval_s - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        self._last = time.monotonic()


def finnhub_get(url: str, params: Dict[str, Any], throttle: FinnhubThrottle | None, label: str, timeout: float = 15) -> requests.Response:
    """
    Issue one Finnhub GET, paced by ``throttle``; with a throttle a 429 is retried once after a back-off.

    Args:
        url: Endpoint URL.
        params: Query parameters (including the token).
        throttle: Pacer, or None for an unpaced single call without retry.
        label: What is being fetched (for the log line), e.g. the ticker.
        timeout: Request timeout in seconds.

    Returns:
        The final response (callers check ``status_code``).
    """
    if throttle is not None:
        throttle.wait()
    resp = requests.get(url, params=params, timeout=timeout)
    if resp.status_code == 429 and throttle is not None:
        _logger.warning("Finnhub 429 for %s; backing off %.0fs and retrying once", label, FINNHUB_429_BACKOFF_S)
        time.sleep(FINNHUB_429_BACKOFF_S)
        throttle.wait()
        resp = requests.get(url, params=params, timeout=timeout)
    return resp
