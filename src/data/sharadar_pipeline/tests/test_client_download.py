"""Tests for the HTTP client, the downloads, the manifest and the purge. No network: httpx mock transport only."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import httpx
import pytest

from src.data.sharadar_pipeline import config
from src.data.sharadar_pipeline.client import PAGE_SIZE, AuthError, FreeTierExceeded, SharadarClient, SharadarError
from src.data.sharadar_pipeline.download import bulk_status, download_bulk, download_sample
from src.data.sharadar_pipeline.manifest import Manifest
from src.data.sharadar_pipeline.purge import PurgeRefused, inventory, purge

ZIP_BYTES = b"PK-not-a-real-zip-but-bytes"


def _client(handler, **kwargs) -> SharadarClient:
    http = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    return SharadarClient("k", client=http, sleep=lambda _: None, **kwargs)


def _csv(header: str, rows: list[str]) -> httpx.Response:
    return httpx.Response(200, text="\n".join([header, *rows]) + ("\n" if rows else ""))


def test_query_pages_until_a_short_page_and_sends_the_key():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.url.params))
        skip = int(request.url.params["skip"])
        count = PAGE_SIZE if skip == 0 else 3
        return _csv("ticker,date", [f"T{skip + i},2024-01-02" for i in range(count)])

    frame = _client(handler).query("stocks", {"ticker": "T", "from": "1990-01-01"})
    assert len(frame) == PAGE_SIZE + 3 and frame["ticker"].iloc[-1] == f"T{PAGE_SIZE + 2}"
    assert [p["skip"] for p in seen] == ["0", str(PAGE_SIZE)]
    assert seen[0]["api_key"] == "k" and seen[0]["from"] == "1990-01-01" and seen[0]["format"] == "csv"


def test_query_returns_an_empty_frame_with_columns_and_honours_max_rows():
    assert list(_client(lambda r: _csv("ticker,date", [])).query("stocks").columns) == ["ticker", "date"]
    assert _client(lambda r: httpx.Response(200, text="")).query("stocks").empty
    assert len(_client(lambda r: _csv("a", ["1", "2", "3"])).query("x", max_rows=2)) == 2


@pytest.mark.parametrize("status, error", [(401, AuthError), (403, FreeTierExceeded), (404, SharadarError)])
def test_errors_are_typed(status, error):
    with pytest.raises(error):
        _client(lambda r: httpx.Response(status, json={"error": "x"})).query("stocks")


def test_retries_on_429_and_5xx_then_gives_up():
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, headers={"X-RateLimit-Reset": "1"}) if calls["n"] < 3 else _csv("a", ["1"])

    assert len(_client(flaky).query("x")) == 1 and calls["n"] == 3
    with pytest.raises(SharadarError):
        _client(lambda r: httpx.Response(503), max_retries=1).query("x")


def test_transport_errors_are_retried_then_raised():
    def broken(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    with pytest.raises(SharadarError):
        _client(broken, max_retries=1).query("x")


def _bulk_handler(request: httpx.Request) -> httpx.Response:
    if request.url.host == "files.example":
        return httpx.Response(200, content=ZIP_BYTES)
    if request.url.params.get("status") == "True":
        return httpx.Response(200, json={"name": "t.csv.zip", "size": len(ZIP_BYTES)})
    if "years" in request.url.params:
        return httpx.Response(302, headers={"Location": "https://files.example/t.csv.zip?sig=abc"})
    return httpx.Response(200, text="-- schema")


def test_download_bulk_follows_the_redirect_and_hashes(tmp_path: Path):
    client = _client(_bulk_handler)
    done = client.download_bulk("stocks", tmp_path / "raw" / "stocks.csv.zip")
    assert done.path.read_bytes() == ZIP_BYTES and done.size == len(ZIP_BYTES)
    assert done.sha256 == hashlib.sha256(ZIP_BYTES).hexdigest()
    assert not done.path.with_name("stocks.csv.zip.part").exists()
    assert client.bulk_status("stocks")["size"] == len(ZIP_BYTES)
    assert client.schema("stocks") == "-- schema"
    client.close()


def test_download_bulk_reports_a_missing_subscription(tmp_path: Path):
    with pytest.raises(FreeTierExceeded):
        _client(lambda r: httpx.Response(403, json={"error": "Exceeds free tier"})).download_bulk("stocks", tmp_path / "s.zip")
    assert not (tmp_path / "s.zip").exists()


def test_download_bulk_all_tables_resumes_and_records_the_deadline(tmp_path: Path):
    root = tmp_path / config.DATA_DIR_NAME
    calls = {"files": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "files.example":
            calls["files"] += 1
        return _bulk_handler(request)

    manifest = download_bulk(_client(handler), root, ("tickers", "actions"), subscription_start="2026-12-01")
    assert manifest.mode == "paid" and set(manifest.files) == {"tickers", "actions"} and calls["files"] == 2
    assert manifest.deletion_deadline() == date(2027, 1, 30)  # 30 days of subscription + 30 days to delete
    assert manifest.days_left(date(2027, 1, 20)) == 10 and manifest.days_left(date(2027, 2, 1)) == -2
    download_bulk(_client(handler), root, ("tickers", "actions"))
    assert calls["files"] == 2  # nothing downloaded twice
    download_bulk(_client(handler), root, ("tickers",), force=True)
    assert calls["files"] == 3
    reloaded = Manifest.load(root)
    assert reloaded.subscription_start == "2026-12-01"
    reloaded.subscription_end = "2026-12-20"
    assert reloaded.deletion_deadline() == date(2027, 1, 19)
    assert bulk_status(_client(handler), ("tickers",))["tickers"]["size"] == len(ZIP_BYTES)
    assert "error" in bulk_status(_client(lambda r: httpx.Response(401)), ("tickers",))["tickers"]


def test_download_sample_skips_tickers_outside_the_free_tier(tmp_path: Path):
    root = tmp_path / config.DATA_DIR_NAME
    asked = []

    def handler(request: httpx.Request) -> httpx.Response:
        table, params = request.url.path.rsplit("/", 1)[-1], request.url.params
        asked.append((table, params.get("table"), params.get("sort"), params.get("ticker")))
        if table == "descriptions":
            return _csv("table,indicator", ["stocks,close"])
        if table == "tickers":
            return _csv("table,permaticker,ticker", [f"{params['table']},1,AAA", f"{params['table']},1,AAA"])
        if params["ticker"] == "PAID":
            return httpx.Response(403, json={"error": "Exceeds free tier"})
        return _csv("ticker,date", ["AAA,2024-01-02"]) if table != "actions" else _csv("ticker,date", [])

    manifest = download_sample(_client(handler), root, ("AAA", "PAID"))
    assert manifest.mode == "sample" and manifest.deletion_deadline() is None and manifest.days_left() is None
    assert manifest.files["tickers"].rows == len(config.TICKER_TABLES)  # exact duplicates dropped
    assert manifest.files["stocks"].rows == 1 and manifest.files["actions"].rows == 0
    assert ("tickers", "stocks", "permaticker.asc", None) in asked
    assert (root / config.RAW_DIR / "stocks.csv").read_text(encoding="utf-8").startswith("ticker,date")
    with pytest.raises(SharadarError):
        download_sample(_client(handler), root, ("PAID",))


def test_manifest_round_trip_and_empty_default(tmp_path: Path):
    assert Manifest.load(tmp_path).files == {} and Manifest.load(tmp_path).mode == "sample"
    manifest = Manifest(mode="paid")
    manifest.record("stocks", "stocks.csv.zip", 10, "abc", rows=5)
    manifest.save(tmp_path)
    loaded = Manifest.load(tmp_path)
    assert loaded.files["stocks"].size == 10 and loaded.files["stocks"].rows == 5 and loaded.deletion_deadline() is None


def test_purge_is_a_dry_run_without_confirm_and_deletes_with_it(tmp_path: Path):
    root = tmp_path / config.DATA_DIR_NAME
    (root / "raw").mkdir(parents=True)
    (root / "raw" / "stocks.csv").write_text("abc", encoding="utf-8")
    (root / "manifest.json").write_text("{}", encoding="utf-8")
    assert [f["path"] for f in inventory(root)] == ["manifest.json", "raw/stocks.csv"]
    dry = purge(root)
    assert dry["deleted"] is False and dry["files"] == 2 and dry["bytes"] == 5 and root.exists()
    report_path = tmp_path / "report.json"
    done = purge(root, confirm=True, report_path=report_path)
    assert done["deleted"] is True and not root.exists()
    assert json.loads(report_path.read_text(encoding="utf-8"))["files"] == 2
    assert purge(root, confirm=True)["files"] == 0 and inventory(root) == []


def test_purge_refuses_other_folders_and_reports_inside_the_root(tmp_path: Path):
    other = tmp_path / "fmp-data"
    other.mkdir()
    with pytest.raises(PurgeRefused):
        purge(other, confirm=True)
    root = tmp_path / config.DATA_DIR_NAME
    root.mkdir()
    with pytest.raises(PurgeRefused):
        purge(root, confirm=True, report_path=root / "report.json")
    assert other.exists() and root.exists()
