"""Tests for ``src.data.fmp_pipeline.probe.probe_client`` — no network; ``httpx.MockTransport`` only."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import httpx
import pytest

from src.data.fmp_pipeline.probe.probe_client import (
    BudgetExhausted,
    ProbeClient,
    QuotaExhausted,
    classify,
    row_count,
)

KEY = "real-secret-key-123"


def _client(tmp_path: Path, handler, budget: int = 10, offline: bool = False) -> ProbeClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return ProbeClient(None if offline else KEY, tmp_path, budget=budget, offline=offline, min_interval_s=0,
                       base_url="https://fmp.test/stable/", http_client=http)


@pytest.mark.parametrize(
    "status,body,expected",
    [
        (200, [{"a": 1}], "ok"),
        (200, [], "ok_empty"),
        (200, {}, "ok_empty"),
        (200, {"Error Message": "Invalid API KEY."}, "unauthorized"),
        (200, {"Error Message": "Special Endpoint : this endpoint is not available under your subscription"},
         "plan_restricted"),
        (200, {"Error Message": "something else"}, "api_error"),
        (200, {"Error Message": "Limit Reach . Please upgrade your plan"}, "quota_exhausted"),
        (401, {"Error Message": "x"}, "unauthorized"),
        (402, "Premium Query Parameter", "plan_restricted"),
        (403, {}, "plan_restricted"),
        (404, {}, "not_found"),
        (429, {}, "rate_limited"),
        (503, {}, "server_error"),
        (400, {"Error Message": "bad date"}, "api_error"),
    ],
)
def test_classify(status: int, body, expected: str):
    text = body if isinstance(body, str) else json.dumps(body)
    parsed = None if isinstance(body, str) else body
    assert classify(status, parsed, text) == expected


def test_classify_non_json():
    assert classify(200, None, "") == "ok_empty"
    assert classify(200, None, "<html>oops</html>") == "api_error"


def test_row_count():
    assert row_count([{"a": 1}, {"a": 2}]) == 2
    assert row_count({"symbol": "X", "historical": [{}, {}, {}]}) == 3
    assert row_count({"a": 1}) == 1
    assert row_count({}) == 0
    assert row_count(None) == 0


def test_key_sent_in_header_only_and_response_cached(tmp_path: Path):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=[{"date": "2025-06-02", "close": 1.0}])

    with _client(tmp_path, handler) as client:
        first = client.get("historical-price-eod/full", {"symbol": "AAPL"})
        second = client.get("historical-price-eod/full", {"symbol": "AAPL"})
        assert client.calls_used == 1

    assert len(seen) == 1
    assert seen[0].headers["apikey"] == KEY
    assert KEY not in str(seen[0].url)
    assert first.classification == "ok" and not first.from_cache
    assert second.from_cache and second.json() == first.json()
    for path in tmp_path.iterdir():
        content = gzip.open(path, "rt").read() if path.suffix == ".gz" else path.read_text()
        assert KEY not in content


def test_invalid_auth_mode_sends_wrong_key(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["apikey"] != KEY
        return httpx.Response(401, json={"Error Message": "Invalid API KEY."})

    with _client(tmp_path, handler) as client:
        assert client.get("x", auth="invalid").classification == "unauthorized"


def test_echoed_key_is_redacted(tmp_path: Path):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"Error Message": f"bad key {KEY}"})

    with _client(tmp_path, handler) as client:
        assert KEY not in client.get("x").body


def test_budget_is_enforced_across_clients(tmp_path: Path):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"a": 1}])

    with _client(tmp_path, handler, budget=2) as client:
        client.get("a")
        client.get("b")
        with pytest.raises(BudgetExhausted):
            client.get("c")
    with _client(tmp_path, handler, budget=2) as again:
        assert again.get("a").from_cache
        with pytest.raises(BudgetExhausted):
            again.get("d")


def test_offline_serves_cache_and_never_calls(tmp_path: Path):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"a": 1}])

    with _client(tmp_path, handler) as client:
        client.get("a", {"p": 1})

    def fail(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("offline client made a call")

    with _client(tmp_path, fail, offline=True) as offline:
        assert offline.get("a", {"p": 1}).from_cache
        with pytest.raises(BudgetExhausted, match="offline"):
            offline.get("a", {"p": 2})


def test_offline_requires_no_key_but_online_does(tmp_path: Path):
    with pytest.raises(ValueError):
        ProbeClient(None, tmp_path)


def test_quota_exhaustion_raises_and_is_not_cached(tmp_path: Path):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"Error Message": "Limit Reach . Please upgrade your plan"})

    with _client(tmp_path, handler) as client:
        with pytest.raises(QuotaExhausted):
            client.get("a")
    assert not list(tmp_path.glob("*.json.gz"))


def test_transient_errors_are_not_cached(tmp_path: Path):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    with _client(tmp_path, handler) as client:
        assert client.get("a").classification == "server_error"
    assert not list(tmp_path.glob("*.json.gz"))


def test_transport_error_is_recorded(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    with _client(tmp_path, handler) as client:
        response = client.get("a")
    assert response.status == 0 and response.classification == "server_error"
    assert response.json() is None
