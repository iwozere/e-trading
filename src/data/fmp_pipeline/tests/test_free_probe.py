"""
End-to-end test of ``fmp probe-free`` against a fake FMP (``httpx.MockTransport``): plan, budget, report,
fixtures and offline re-run. All responses are synthetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from src.data.fmp_pipeline import cli
from src.data.fmp_pipeline.probe.free_probe import ENTITLEMENT_CANDIDATES, FreeProbe
from src.data.fmp_pipeline.probe.probe_client import ProbeClient
from src.data.fmp_pipeline.probe.report import AUTO_MARKER, render_report, write_fixtures, write_report

ENTITLED = {"AAPL", "NVDA", "MSFT", "OLDCO"}


def _price_rows(symbol: str, start: str, end: str, raw: bool) -> list[dict]:
    """Daily rows with a 4:1 (AAPL 2020) or 10:1 (NVDA 2024) split, raw or split-adjusted."""
    rows = []
    split = {"AAPL": ("2020-08-31", 4.0), "NVDA": ("2024-06-10", 10.0)}.get(symbol, ("1900-01-01", 1.0))
    for day in ("2014-06-06", "2014-06-09", "2020-08-28", "2020-08-31", "2024-06-07", "2024-06-10",
                "2025-06-02", "2025-06-03"):
        if not start <= day <= end:
            continue
        price = 100.0 * (split[1] if raw and day < split[0] else 1.0)
        rows.append({"symbol": symbol, "date": day, "open": price, "high": price, "low": price, "close": price,
                     "volume": 1000})
    return rows


def _handler(request: httpx.Request) -> httpx.Response:
    endpoint = request.url.path.removeprefix("/stable/")
    params = dict(request.url.params)
    symbol = params.get("symbol", "")
    if request.headers.get("apikey") != "k":
        return httpx.Response(401, json={"Error Message": "Invalid API KEY."})
    if endpoint == "eod-bulk":
        return httpx.Response(402, text="Premium Query Parameter: special endpoint")
    if endpoint == "this-endpoint-does-not-exist":
        return httpx.Response(404, json={})
    if symbol and symbol not in ENTITLED and endpoint != "profile":
        return httpx.Response(402, text="Premium Query Parameter: this value set for 'symbol' is not available")
    if endpoint.startswith("historical-price-eod/"):
        if params.get("from") == "not-a-date":
            return httpx.Response(400, json={"Error Message": "Invalid date"})
        raw = endpoint.endswith("non-split-adjusted")
        return httpx.Response(200, json=_price_rows(symbol, params.get("from", "2000-01-01"),
                                                    params.get("to", "2099-01-01"), raw))
    if endpoint == "delisted-companies":
        if int(params.get("page", 0)) > 10:
            return httpx.Response(200, json=[])
        return httpx.Response(200, json=[
            {"symbol": "OLDCO", "companyName": "Old Co A", "exchange": "NYSE", "ipoDate": "1990-01-01",
             "delistedDate": "2023-01-01"},
            {"symbol": "OLDCO", "companyName": "Old Co B", "exchange": "NYSE", "ipoDate": "2005-01-01",
             "delistedDate": "2024-01-01"},
        ])
    if endpoint == "actively-trading-list":
        return httpx.Response(200, json=[{"symbol": "AAPL", "name": "Apple"}])
    if endpoint == "dividends":
        return httpx.Response(200, json=[
            {"symbol": symbol, "date": "2020-08-07", "dividend": 0.82, "adjDividend": 0.205},
            {"symbol": symbol, "date": "2024-03-05", "dividend": 0.04, "adjDividend": 0.004},
        ])
    if endpoint == "splits":
        return httpx.Response(200, json=[{"symbol": symbol, "date": "2020-08-31", "numerator": 4, "denominator": 1}])
    if endpoint == "profile":
        return httpx.Response(200, json=[{"symbol": symbol, "companyName": f"{symbol} Inc", "cik": "0001"}])
    if endpoint.startswith("historical-chart/"):
        return httpx.Response(402, text="Premium Query Parameter")
    return httpx.Response(200, json=[{"symbol": "X", "date": "2025-06-02", "value": 1.5}])


def _client(cache: Path, offline: bool = False) -> ProbeClient:
    return ProbeClient(None if offline else "k", cache, budget=200, offline=offline, min_interval_s=0,
                       base_url="https://fmp.test/stable/",
                       http_client=httpx.Client(transport=httpx.MockTransport(_handler)))


def test_probe_run_report_fixtures_and_offline_rerun(tmp_path: Path):
    cache = tmp_path / "cache"
    with _client(cache) as client:
        run = FreeProbe(client).run()
        used = client.calls_used

    assert used <= 200
    assert run.entitled == ["AAPL", "MSFT", "NVDA"]
    assert run.primary_symbol == "AAPL"
    assert [c.symbol for c in run.reused] == ["OLDCO"]
    assert not run.stopped
    assert len([r for r in run.results if r.call.group == "entitlement"]) == len(ENTITLEMENT_CANDIDATES)

    report = render_report(run, used, 200)
    assert report.startswith(AUTO_MARKER)
    assert "| AAPL | 2020-08-07 | 0.82 | True | dividend |" in report
    assert "| raw |" in report and "| split_adjusted |" in report
    assert "### OLDCO — delisted more than once" in report
    assert "| `error:invalid-key` | 401 | unauthorized |" in report
    assert "Premium Query Parameter" in report

    path = tmp_path / "report.md"
    path.write_text("# Title\n\nMy findings.\n\n" + AUTO_MARKER + "\nold\n", encoding="utf-8")
    write_report(path, report)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Title\n\nMy findings.") and "old\n" not in text

    fixtures = write_fixtures(run, tmp_path / "fixtures")
    names = {p.name for p in fixtures}
    assert "historical-price-eod__non-split-adjusted.json" in names and "_error_shapes.json" in names
    synthetic = json.loads((tmp_path / "fixtures" / "dividends.json").read_text(encoding="utf-8"))
    assert synthetic[0]["dividend"] != 0.82 and set(synthetic[0]) == {"symbol", "date", "dividend", "adjDividend"}
    errors = json.loads((tmp_path / "fixtures" / "_error_shapes.json").read_text(encoding="utf-8"))
    assert {"plan_restricted", "unauthorized", "not_found", "api_error"} <= set(errors)

    with _client(cache, offline=True) as offline:
        rerun = FreeProbe(offline).run()
        assert offline.calls_used == used
    assert all(r.response is not None and r.response.from_cache for r in rerun.results)


def test_budget_exhaustion_skips_remaining_calls(tmp_path: Path):
    client = ProbeClient("k", tmp_path, budget=5, min_interval_s=0, base_url="https://fmp.test/stable/",
                         http_client=httpx.Client(transport=httpx.MockTransport(_handler)))
    with client:
        run = FreeProbe(client).run()
    assert client.calls_used == 5
    assert sum(1 for r in run.results if r.response is None and "budget" in r.skipped) > 50
    assert "skipped (call budget of 5 used up)" in render_report(run, 5, 5)


def test_quota_exhaustion_stops_the_run(tmp_path: Path):
    def quota(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"Error Message": "Limit Reach . Please upgrade your plan"})

    client = ProbeClient("k", tmp_path, budget=200, min_interval_s=0, base_url="https://fmp.test/stable/",
                         http_client=httpx.Client(transport=httpx.MockTransport(quota)))
    with client:
        run = FreeProbe(client).run()
    assert run.stopped and client.calls_used == 1
    assert all(r.response is None for r in run.results)


def test_cli_offline_writes_report(tmp_path: Path):
    data_root = tmp_path / "root"
    with _client(data_root / "probe-free") as client:
        FreeProbe(client).run()
    report, fixtures = tmp_path / "r.md", tmp_path / "fx"
    code = cli.main(["probe-free", "--offline", "--data-root", str(data_root), "--report", str(report),
                     "--fixtures", str(fixtures)])
    assert code == 0
    assert AUTO_MARKER in report.read_text(encoding="utf-8")
    assert (fixtures / "_error_shapes.json").exists()
