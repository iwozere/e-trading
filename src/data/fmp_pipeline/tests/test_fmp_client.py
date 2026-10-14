"""Tests for ``FmpClient`` against ``MockFmpServer`` (spec §6.4): no network, no real backoff waits."""

from __future__ import annotations

import random

import httpx
import pytest

from src.data.fmp_pipeline.client.circuit_breaker import CircuitBreaker
from src.data.fmp_pipeline.client.fmp_client import FmpClient
from src.data.fmp_pipeline.client.rate_limiter import AsyncRateLimiter
from src.data.fmp_pipeline.errors import AuthAbort, QuotaExhausted
from src.data.fmp_pipeline.tests.helpers import FakeSleep, make_client
from src.data.fmp_pipeline.tests.mock_server import Fault, MockConfig, MockFmpServer

PRICE = "historical-price-eod/non-split-adjusted"
WINDOW = {"symbol": "AAA", "from": "2025-06-02", "to": "2025-06-06"}


@pytest.mark.asyncio
async def test_fetch_ok_counts_wire_and_decoded_bytes():
    seen: list[tuple[int, int]] = []
    with MockFmpServer() as fmp:
        client = make_client(fmp, FakeSleep(), on_bytes=lambda w, d: seen.append((w, d)))
        async with client:
            result = await client.fetch(PRICE, {"symbol": "AAA"})
    assert result.classification == "ok" and result.http_status == 200 and result.attempts == 1
    assert result.rows == len(result.json()) > 1000
    assert result.bytes_wire < result.bytes_decoded == len(result.body)
    assert seen == [(result.bytes_wire, result.bytes_decoded)]
    record = fmp.requests[0]
    assert not record.query_has_apikey and record.bytes_wire == result.bytes_wire


@pytest.mark.asyncio
async def test_symbols_with_special_characters_are_encoded():
    with MockFmpServer() as fmp:
        async with make_client(fmp, FakeSleep()) as client:
            for symbol in ("^GSPC", "BRK.B", "BRK-B", "A&B"):
                assert (await client.fetch(PRICE, {**WINDOW, "symbol": symbol})).classification == "ok"
    assert [r.params["symbol"] for r in fmp.requests] == ["^GSPC", "BRK.B", "BRK-B", "A&B"]


@pytest.mark.asyncio
async def test_429_retries_honouring_retry_after():
    sleep = FakeSleep()
    with MockFmpServer() as fmp:
        fmp.add_fault(Fault(endpoint=PRICE, status=429, body={"message": "too many"},
                            headers={"Retry-After": "7"}, times=2))
        async with make_client(fmp, sleep, rng=random.Random(1)) as client:
            result = await client.fetch(PRICE, WINDOW)
    assert result.classification == "ok" and result.attempts == 3
    assert len(sleep.delays) == 2 and all(d >= 7 for d in sleep.delays)


@pytest.mark.asyncio
async def test_backoff_is_exponential_with_jitter_and_capped():
    sleep = FakeSleep()
    with MockFmpServer() as fmp:
        fmp.add_fault(Fault(status=503, body="down", times=None))
        async with make_client(fmp, sleep, backoff_base_s=1, backoff_cap_s=8) as client:
            result = await client.fetch(PRICE, WINDOW)
    assert result.classification == "server_error" and result.attempts == 6 and result.http_status == 503
    assert len(fmp.requests) == 6 and len(sleep.delays) == 5
    for attempt, delay in enumerate(sleep.delays, start=1):
        base = min(8, 2 ** (attempt - 1))
        assert 0.5 * base <= delay <= 1.5 * base


@pytest.mark.asyncio
@pytest.mark.parametrize("config,endpoint,params,classification,status", [
    (MockConfig(restricted_symbols={"AAA"}), PRICE, WINDOW, "plan_restricted", 402),
    (MockConfig(restricted_endpoints={PRICE}), PRICE, WINDOW, "plan_restricted", 402),
    (MockConfig(), "no-such-endpoint", {}, "not_found", 404),
    (MockConfig(unknown_symbols={"AAA"}), PRICE, WINDOW, "ok_empty", 200),
])
async def test_final_outcomes_are_not_retried(config: MockConfig, endpoint: str, params: dict,
                                              classification: str, status: int):
    sleep = FakeSleep()
    with MockFmpServer(config) as fmp:
        async with make_client(fmp, sleep) as client:
            result = await client.fetch(endpoint, params)
    assert (result.classification, result.http_status, result.attempts) == (classification, status, 1)
    assert sleep.delays == [] and len(fmp.requests) == 1
    if classification == "plan_restricted":
        assert "not available under your current subscription" in result.error


@pytest.mark.asyncio
async def test_200_with_error_body_is_api_error_not_data():
    with MockFmpServer() as fmp:
        fmp.add_fault(Fault(endpoint=PRICE, status=200, body={"Error Message": "boom"}))
        async with make_client(fmp, FakeSleep()) as client:
            result = await client.fetch(PRICE, WINDOW)
    assert result.classification == "api_error" and "boom" in result.error


@pytest.mark.asyncio
async def test_quota_exhaustion_raises_immediately():
    sleep = FakeSleep()
    with MockFmpServer(MockConfig(quota_after=1)) as fmp:
        async with make_client(fmp, sleep) as client:
            await client.fetch(PRICE, WINDOW)
            with pytest.raises(QuotaExhausted):
                await client.fetch(PRICE, WINDOW)
    assert len(fmp.requests) == 2 and sleep.delays == []


@pytest.mark.asyncio
async def test_auth_abort_after_consecutive_401s_and_reset_on_success():
    with MockFmpServer() as fmp:
        async with make_client(fmp, FakeSleep(), auth_abort_after=3) as client:
            fmp.revoke_key()
            for _ in range(2):
                assert (await client.fetch(PRICE, WINDOW)).classification == "unauthorized"
            fmp.key_revoked = False
            await client.fetch(PRICE, WINDOW)  # resets the count
            fmp.revoke_key()
            for _ in range(2):
                await client.fetch(PRICE, WINDOW)
            with pytest.raises(AuthAbort):
                await client.fetch(PRICE, WINDOW)


@pytest.mark.asyncio
async def test_malformed_date_is_rejected_before_sending():
    with MockFmpServer() as fmp:
        async with make_client(fmp, FakeSleep()) as client:
            with pytest.raises(ValueError):
                await client.fetch(PRICE, {"symbol": "AAA", "from": "not-a-date"})
    assert fmp.requests == []


@pytest.mark.asyncio
async def test_network_error_is_retried_then_reported():
    # A transport that refuses every connection (a real closed port takes ~2 s per attempt on Windows).
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    sleep = FakeSleep()
    async with FmpClient("test-key", base_url="http://fmp.invalid/stable/", rate_limiter=AsyncRateLimiter(1_000_000),
                         breaker=CircuitBreaker(sleep=sleep), sleep=sleep, max_attempts=3,
                         http_client=httpx.AsyncClient(transport=httpx.MockTransport(refuse))) as client:
        result = await client.fetch(PRICE, WINDOW)
    assert result.classification == "network_error" and result.http_status == 0 and result.attempts == 3
    assert result.error == "ConnectError" and len(sleep.delays) == 2


@pytest.mark.asyncio
async def test_read_timeout_is_a_retryable_network_error():
    sleep = FakeSleep()
    with MockFmpServer() as fmp:
        fmp.add_fault(Fault(endpoint=PRICE, delay_s=1.0, times=1))
        async with make_client(fmp, sleep, read_timeout=0.3) as client:
            result = await client.fetch(PRICE, WINDOW)
    assert result.classification == "ok" and result.attempts == 2 and len(sleep.delays) == 1


@pytest.mark.asyncio
async def test_circuit_breaker_pauses_after_consecutive_failures():
    sleep = FakeSleep()
    with MockFmpServer() as fmp:
        fmp.add_fault(Fault(status=500, body={}, times=4))
        breaker = CircuitBreaker(threshold=4, pause_s=60, sleep=sleep)
        async with make_client(fmp, sleep, breaker=breaker, max_attempts=6, backoff_base_s=0.001) as client:
            result = await client.fetch(PRICE, WINDOW)
    assert result.classification == "ok" and result.attempts == 5
    assert breaker.times_opened == 1
    assert any(d > 59 for d in sleep.delays)  # the 60 s pause (minus elapsed time) was taken


@pytest.mark.asyncio
async def test_echoed_key_is_redacted():
    with MockFmpServer() as fmp:
        fmp.add_fault(Fault(endpoint=PRICE, status=400, body={"Error Message": "bad key test-key"}))
        async with make_client(fmp, FakeSleep()) as client:
            result = await client.fetch(PRICE, WINDOW)
    assert b"test-key" not in result.body and "test-key" not in result.error
