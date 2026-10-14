"""Shared helpers for the downloader tests (WP-2): fast clients against ``MockFmpServer``, no real sleeping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from src.data.fmp_pipeline.client.circuit_breaker import CircuitBreaker
from src.data.fmp_pipeline.client.fmp_client import FmpClient
from src.data.fmp_pipeline.client.rate_limiter import AsyncRateLimiter
from src.data.fmp_pipeline.tests.mock_server import MockFmpServer

# Building an SSL context costs ~0.4 s on Windows; with a shared one an httpx client costs ~17 ms.
SSL_CTX = httpx.create_ssl_context()


@dataclass
class FakeSleep:
    """Async sleep that records requested delays instead of waiting."""

    delays: list[float] = field(default_factory=list)

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def make_client(server: MockFmpServer, sleep: FakeSleep, read_timeout: float = 10.0, **kwargs: Any) -> FmpClient:
    """
    Build an ``FmpClient`` pointed at ``server`` with an effectively unlimited rate and recorded (not real) sleeps.

    Args:
        server: A started mock server.
        sleep: Records backoff and breaker pauses.
        read_timeout: Read timeout in seconds.
        **kwargs: Overrides for ``FmpClient``.

    Returns:
        The client; close it with ``await client.aclose()``.
    """
    http = httpx.AsyncClient(verify=SSL_CTX, timeout=httpx.Timeout(read_timeout, connect=5.0),
                             headers={"Accept-Encoding": "gzip"})
    options: dict[str, Any] = {
        "rate_limiter": AsyncRateLimiter(1_000_000),
        "breaker": CircuitBreaker(sleep=sleep),
        "sleep": sleep,
        "http_client": http,
    }
    options.update(kwargs)
    return FmpClient(server.config.api_key, base_url=server.base_url, **options)
