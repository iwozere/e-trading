"""
Tests for the WP-3 mock FMP server (``tests/mock_server.py``).

Each scenario is checked against ``probe_client.classify`` too, so the mock and the classifier cannot drift
apart from the shapes recorded in ``docs/fmp_probe_report.md``.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any, Optional

import httpx
import pytest

from src.data.fmp_pipeline.probe.probe_client import ProbeClient, classify
from src.data.fmp_pipeline.tests.mock_server import (
    PLAN_ENDPOINT_TEXT,
    PLAN_SYMBOL_TEXT,
    Fault,
    MockConfig,
    MockFmpServer,
)

KEY = "test-key"
# One shared client: building an httpx.Client costs ~0.4 s on Windows (SSL context), a warm request ~1 ms.
HTTP = httpx.Client(timeout=10)


@pytest.fixture
def fmp() -> Iterator[MockFmpServer]:
    with MockFmpServer() as server:
        yield server


def _get(server: MockFmpServer, endpoint: str, key: Optional[str] = KEY, **params: Any) -> httpx.Response:
    headers = {"apikey": key} if key is not None else {}
    return HTTP.get(server.base_url + endpoint, params=params, headers=headers)


def _class(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        body = None
    return classify(resp.status_code, body, resp.text)


# --- auth, plan restrictions, errors ---------------------------------------------------------------------------


def test_missing_or_wrong_key_is_401_json(fmp: MockFmpServer):
    for key in (None, "wrong"):
        resp = _get(fmp, "profile", key=key, symbol="AAA")
        assert resp.status_code == 401 and "Invalid API KEY" in resp.json()["Error Message"]
        assert _class(resp) == "unauthorized"


def test_revoked_key_fails_everything(fmp: MockFmpServer):
    assert _get(fmp, "profile", symbol="AAA").status_code == 200
    fmp.revoke_key()
    assert _get(fmp, "profile", symbol="AAA").status_code == 401


def test_plan_restrictions_are_402_plain_text():
    with MockFmpServer(MockConfig(restricted_symbols={"BRK-B"}, restricted_endpoints={"eod-bulk"})) as fmp:
        sym = _get(fmp, "historical-price-eod/full", symbol="BRK-B")
        ep = _get(fmp, "eod-bulk", date="2025-06-02")
    assert (sym.status_code, sym.text, _class(sym)) == (402, PLAN_SYMBOL_TEXT, "plan_restricted")
    assert sym.headers["content-type"].startswith("text/plain")
    assert (ep.status_code, ep.text, _class(ep)) == (402, PLAN_ENDPOINT_TEXT, "plan_restricted")


def test_unknown_endpoint_is_404_empty_list(fmp: MockFmpServer):
    resp = _get(fmp, "no-such-endpoint")
    assert resp.status_code == 404 and resp.json() == [] and _class(resp) == "not_found"


def test_unknown_symbol_is_ok_empty():
    with MockFmpServer(MockConfig(unknown_symbols={"ZZZZ"})) as fmp:
        resp = _get(fmp, "historical-price-eod/non-split-adjusted", symbol="ZZZZ")
    assert resp.json() == [] and _class(resp) == "ok_empty"


def test_quota_exhaustion_after_n_requests():
    with MockFmpServer(MockConfig(quota_after=2)) as fmp:
        assert _get(fmp, "profile", symbol="A").status_code == 200
        assert _get(fmp, "profile", symbol="B").status_code == 200
        resp = _get(fmp, "profile", symbol="C")
    assert resp.status_code == 429 and _class(resp) == "quota_exhausted"


# --- scripted faults -------------------------------------------------------------------------------------------


def test_429_with_retry_after_then_recovery(fmp: MockFmpServer):
    fmp.add_fault(Fault(endpoint="profile", status=429, body={"message": "slow down"},
                        headers={"Retry-After": "2"}, times=2))
    first, second, third = (_get(fmp, "profile", symbol="AAA") for _ in range(3))
    assert first.status_code == second.status_code == 429
    assert first.headers["retry-after"] == "2" and _class(first) == "rate_limited"
    assert third.status_code == 200


def test_5xx_burst_and_forever_fault(fmp: MockFmpServer):
    fmp.add_fault(Fault(status=503, body="unavailable", times=3))
    statuses = [_get(fmp, "profile", symbol="AAA").status_code for _ in range(4)]
    assert statuses == [503, 503, 503, 200]
    fmp.add_fault(Fault(endpoint="splits", status=500, body={}, times=None))
    assert all(_get(fmp, "splits", symbol="AAA").status_code == 500 for _ in range(3))
    assert _class(_get(fmp, "splits", symbol="AAA")) == "server_error"


def test_200_with_error_body(fmp: MockFmpServer):
    fmp.add_fault(Fault(endpoint="dividends", status=200, body={"Error Message": "something went wrong"}))
    resp = _get(fmp, "dividends", symbol="AAA")
    assert resp.status_code == 200 and _class(resp) == "api_error"


def test_fault_scoped_to_symbol(fmp: MockFmpServer):
    fmp.add_fault(Fault(endpoint="profile", symbol="BAD", status=500, body={}, times=None))
    assert _get(fmp, "profile", symbol="BAD").status_code == 500
    assert _get(fmp, "profile", symbol="GOOD").status_code == 200


def test_slow_response_still_serves_data(fmp: MockFmpServer):
    fmp.add_fault(Fault(endpoint="profile", delay_s=0.3))
    started = time.monotonic()
    resp = _get(fmp, "profile", symbol="AAA")
    assert time.monotonic() - started >= 0.3
    assert resp.status_code == 200 and resp.json()[0]["symbol"] == "AAA"


def test_slow_response_triggers_client_timeout(fmp: MockFmpServer):
    fmp.add_fault(Fault(endpoint="profile", delay_s=1.0))
    with pytest.raises(httpx.ReadTimeout):
        HTTP.get(fmp.base_url + "profile", params={"symbol": "A"}, headers={"apikey": KEY}, timeout=0.2)


# --- daily prices ----------------------------------------------------------------------------------------------


def test_daily_fields_follow_the_endpoint(fmp: MockFmpServer):
    window = {"symbol": "AAA", "from": "2025-06-02", "to": "2025-06-06"}
    raw = _get(fmp, "historical-price-eod/non-split-adjusted", **window).json()
    full = _get(fmp, "historical-price-eod/full", **window).json()
    div = _get(fmp, "historical-price-eod/dividend-adjusted", **window).json()
    assert list(raw[0]) == ["symbol", "date", "adjOpen", "adjHigh", "adjLow", "adjClose", "volume"]
    assert list(div[0]) == list(raw[0])
    assert list(full[0]) == ["symbol", "date", "open", "high", "low", "close", "volume", "change",
                             "changePercent", "vwap"]
    assert [r["date"] for r in raw] == ["2025-06-06", "2025-06-05", "2025-06-04", "2025-06-03", "2025-06-02"]
    assert raw[0]["adjClose"] == full[0]["close"] and raw[0]["symbol"] == "AAA"


def test_daily_values_are_deterministic_and_ohlc_consistent(fmp: MockFmpServer):
    params = {"symbol": "AAA", "from": "2024-01-01", "to": "2024-03-31"}
    first = _get(fmp, "historical-price-eod/full", **params).json()
    assert first == _get(fmp, "historical-price-eod/full", **params).json()
    for bar in first:
        assert bar["low"] <= min(bar["open"], bar["close"]) <= max(bar["open"], bar["close"]) <= bar["high"]


def test_default_window_is_five_years_and_bad_from_is_ignored(fmp: MockFmpServer):
    default = _get(fmp, "historical-price-eod/non-split-adjusted", symbol="AAA").json()
    assert (default[-1]["date"], default[0]["date"]) == ("2021-10-15", "2026-10-14")
    bad = _get(fmp, "historical-price-eod/non-split-adjusted", symbol="AAA", **{"from": "not-a-date"},
               to="2025-06-06").json()
    assert (bad[-1]["date"], bad[0]["date"]) == ("2021-10-15", "2025-06-06")


def test_weekend_window_is_empty(fmp: MockFmpServer):
    resp = _get(fmp, "historical-price-eod/full", symbol="AAA", **{"from": "2025-06-07"}, to="2025-06-08")
    assert resp.json() == [] and _class(resp) == "ok_empty"


def test_listing_range_limits_history():
    listings = {"GONE": (date(2010, 3, 1), date(2015, 6, 30))}
    with MockFmpServer(MockConfig(listings=listings)) as fmp:
        rows = _get(fmp, "historical-price-eod/non-split-adjusted", symbol="GONE", **{"from": "2000-01-01"},
                    to="2026-01-01").json()
    assert (rows[-1]["date"], rows[0]["date"]) == ("2010-03-01", "2015-06-30")


# --- intraday --------------------------------------------------------------------------------------------------


def test_intraday_1hour_has_seven_bars_per_day_newest_first(fmp: MockFmpServer):
    rows = _get(fmp, "historical-chart/1hour", symbol="AAA", **{"from": "2025-06-02"}, to="2025-06-03").json()
    assert len(rows) == 14
    assert rows[0]["date"] == "2025-06-03 15:30:00" and rows[-1]["date"] == "2025-06-02 09:30:00"
    times = sorted({r["date"][11:16] for r in rows})
    assert times == ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]
    four = _get(fmp, "historical-chart/4hour", symbol="AAA", **{"from": "2025-06-02"}, to="2025-06-02").json()
    assert [r["date"][11:16] for r in four] == ["13:30", "09:30"]


def test_intraday_row_cap_truncates_the_oldest_rows():
    with MockFmpServer(MockConfig(intraday_max_rows=10)) as fmp:
        rows = _get(fmp, "historical-chart/1hour", symbol="AAA", **{"from": "2025-06-02"}, to="2025-06-06").json()
    assert len(rows) == 10
    assert rows[0]["date"] == "2025-06-06 15:30:00"
    assert rows[-1]["date"] > "2025-06-05"  # the window's start was silently lost


def test_intraday_default_window_and_unknown_interval(fmp: MockFmpServer):
    assert _get(fmp, "historical-chart/1hour", symbol="AAA").json()
    assert _get(fmp, "historical-chart/7min", symbol="AAA").status_code == 404


# --- paging and fixture-backed endpoints -----------------------------------------------------------------------


def test_paging_until_empty_page():
    with MockFmpServer(MockConfig(paged_totals={"delisted-companies": 230}, max_page_size=100)) as fmp:
        pages = [_get(fmp, "delisted-companies", page=p, limit=100).json() for p in range(4)]
        capped = _get(fmp, "delisted-companies", page=0, limit=1000).json()
        bad = _get(fmp, "delisted-companies", page="x").json()
    assert [len(p) for p in pages] == [100, 100, 30, 0]
    symbols = [r["symbol"] for p in pages for r in p]
    assert len(set(symbols)) == 230
    assert pages[0][0]["delistedDate"] > pages[0][-1]["delistedDate"]  # newest first
    assert set(pages[0][0]) == {"symbol", "companyName", "exchange", "ipoDate", "delistedDate"}
    assert len(capped) == 100 and len(bad) == 100


def test_fixture_endpoint_echoes_symbol(fmp: MockFmpServer):
    rows = _get(fmp, "dividends", symbol="XYZ").json()
    assert rows and all(r["symbol"] == "XYZ" for r in rows)
    assert set(rows[0]) >= {"date", "dividend", "adjDividend"}
    assert _get(fmp, "treasury-rates").json()


# --- transport -------------------------------------------------------------------------------------------------


def test_gzip_chunked_without_content_length_by_default(fmp: MockFmpServer):
    resp = HTTP.get(fmp.base_url + "historical-price-eod/full", params={"symbol": "AAA"},
                    headers={"apikey": KEY, "Accept-Encoding": "gzip"})
    assert resp.headers["content-encoding"] == "gzip"
    assert "content-length" not in resp.headers and resp.headers["transfer-encoding"] == "chunked"
    assert len(resp.json()) > 1000
    assert resp.num_bytes_downloaded < len(resp.content)
    record = fmp.requests[-1]
    assert record.bytes_wire == resp.num_bytes_downloaded and record.bytes_decoded == len(resp.content)
    assert fmp.bytes_wire < fmp.bytes_decoded


def test_content_length_mode_and_small_bodies_uncompressed():
    with MockFmpServer(MockConfig(send_content_length=True)) as fmp:
        gzip_ok = {"apikey": KEY, "Accept-Encoding": "gzip"}
        big = HTTP.get(fmp.base_url + "historical-price-eod/full", params={"symbol": "A"}, headers=gzip_ok)
        small = HTTP.get(fmp.base_url + "no-such-endpoint", headers=gzip_ok)
    assert int(big.headers["content-length"]) == big.num_bytes_downloaded
    assert "content-encoding" not in small.headers


def test_gzip_disabled_or_not_accepted():
    with MockFmpServer(MockConfig(gzip_enabled=False)) as fmp:
        resp = _get(fmp, "historical-price-eod/full", symbol="A")
    assert "content-encoding" not in resp.headers
    with MockFmpServer() as fmp:
        resp = HTTP.get(fmp.base_url + "historical-price-eod/full", params={"symbol": "A"},
                        headers={"apikey": KEY, "Accept-Encoding": "identity"})
    assert "content-encoding" not in resp.headers


def test_request_log_never_contains_the_key(fmp: MockFmpServer):
    _get(fmp, "profile", symbol="AAA")
    HTTP.get(fmp.base_url + "profile", params={"symbol": "AAA", "apikey": KEY})
    first, second = fmp.requests
    assert first.params == {"symbol": "AAA"} and not first.query_has_apikey
    assert second.query_has_apikey and "apikey" not in second.params and "apikey" not in second.headers
    assert first.status == 200 and second.status == 401


def test_base_url_requires_started_server():
    with pytest.raises(RuntimeError):
        _ = MockFmpServer().base_url


def test_concurrent_requests(fmp: MockFmpServer):
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(lambda i: _get(fmp, "profile", symbol=f"S{i}").status_code, range(32)))
    assert statuses == [200] * 32 and len(fmp.requests) == 32


def test_probe_client_against_mock(tmp_path: Path):
    with MockFmpServer() as fmp:
        with ProbeClient(KEY, tmp_path, budget=5, min_interval_s=0, base_url=fmp.base_url) as client:
            ok = client.get("historical-price-eod/non-split-adjusted", {"symbol": "AAA"})
            restricted = client.get("historical-price-eod/full", {"symbol": "AAA"}, auth="invalid")
    assert ok.classification == "ok" and ok.bytes_wire < ok.bytes_decoded
    assert ok.headers["content-encoding"] == "gzip"
    assert restricted.classification == "unauthorized"
