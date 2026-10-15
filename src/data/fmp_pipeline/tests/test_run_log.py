"""Tests for the structured run log of ``fmp run`` (spec §13): request records, progress summaries, no secrets."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from src.data.fmp_pipeline.manifest.manifest import Manifest, TaskSpec
from src.data.fmp_pipeline.run_log import RunLog, run_log_path
from src.data.fmp_pipeline.runner import BandwidthGuard, Downloader
from src.data.fmp_pipeline.tests.helpers import FakeSleep, make_client
from src.data.fmp_pipeline.tests.mock_server import Fault, MockConfig, MockFmpServer

PRICE = "historical-price-eod/non-split-adjusted"


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_run_log_path():
    started = datetime(2026, 10, 15, 9, 5, 7, tzinfo=timezone.utc)
    assert run_log_path(Path("root"), "2026-10", started) == Path("root/logs/2026-10/run-20261015T090507Z.jsonl")


@pytest.mark.asyncio
async def test_every_finished_task_is_logged_without_secrets(tmp_path: Path):
    config = MockConfig(restricted_symbols={"R1"}, api_key="secret-key-123")
    log = RunLog(tmp_path / "logs" / "run.jsonl", "d", interval_s=3600)
    with MockFmpServer(config) as fmp, Manifest(tmp_path / "m.sqlite", "d") as manifest:
        manifest.add_tasks([TaskSpec(PRICE, s, {"from": "2025-01-01", "to": "2025-03-31"}, tier="B")
                            for s in ("A1", "A2", "R1")] + [TaskSpec(PRICE, "BAD", {"from": "nope"})])
        async with make_client(fmp, FakeSleep()) as client:
            await Downloader(manifest, client, workers=2, run_log=log).run()
    log.close()
    log.write("ignored")  # closed: no error, nothing written
    text = (tmp_path / "logs" / "run.jsonl").read_text(encoding="utf-8")
    assert "secret-key-123" not in text and "apikey" not in text.lower()
    requests = [r for r in _records(tmp_path / "logs" / "run.jsonl") if r["event"] == "request"]
    assert sorted((r["symbol"], r["status"]) for r in requests) == [
        ("A1", "ok"), ("A2", "ok"), ("BAD", "failed"), ("R1", "plan_restricted")]
    a1 = next(r for r in requests if r["symbol"] == "A1")
    assert a1["endpoint"] == PRICE and a1["http_status"] == 200 and a1["rows"] > 0 and a1["tier"] == "B"
    assert 0 < a1["bytes_wire"] < a1["bytes_decoded"] and a1["attempts"] == 1 and a1["duration_ms"] >= 0
    assert a1["params"] == {"from": "2025-01-01", "to": "2025-03-31"} and len(a1["request_id"]) == 40
    assert a1["dump_id"] == "d" and a1["ts"].endswith("+00:00") and a1["error"] is None
    bad = next(r for r in requests if r["symbol"] == "BAD")
    assert bad["http_status"] is None and bad["error"].startswith("invalid request")
    assert log.finished == {"ok": 2, "plan_restricted": 1, "failed": 1}


@pytest.mark.asyncio
async def test_progress_is_written_while_the_run_is_stalled(tmp_path: Path):
    log = RunLog(tmp_path / "run.jsonl", "d", interval_s=0.05)
    with MockFmpServer() as fmp, Manifest(tmp_path / "m.sqlite", "d") as manifest:
        fmp.add_fault(Fault(endpoint=PRICE, delay_s=0.4))  # one slow response: no task finishes for a while
        manifest.add_tasks([TaskSpec(PRICE, "SLOW")])
        async with make_client(fmp, FakeSleep()) as client:
            await Downloader(manifest, client, workers=1, run_log=log).run()
    log.close()
    events = [r["event"] for r in _records(tmp_path / "run.jsonl")]
    assert events.count("progress") >= 2 and events[-1] == "request"


@pytest.mark.asyncio
async def test_stops_are_logged(tmp_path: Path):
    log = RunLog(tmp_path / "run.jsonl", "d", interval_s=3600)
    with MockFmpServer(MockConfig(quota_after=1)) as fmp, Manifest(tmp_path / "m.sqlite", "d") as manifest:
        manifest.add_tasks([TaskSpec(PRICE, s) for s in ("A", "B", "C")])
        async with make_client(fmp, FakeSleep()) as client:
            await Downloader(manifest, client, workers=1, run_log=log).run()
    log.close()
    stops = [r for r in _records(tmp_path / "run.jsonl") if r["event"] == "stop"]
    assert [s["reason"] for s in stops] == ["quota"]


def test_progress_fields_rates_and_eta(tmp_path: Path):
    clock = FakeClock()
    log = RunLog(tmp_path / "run.jsonl", "d", interval_s=60, clock=clock)
    guard = BandwidthGuard(cap_bytes=1000, safety=0.5, basis="wire", window_days=30)
    with Manifest(tmp_path / "m.sqlite", "d") as manifest:
        manifest.add_tasks([TaskSpec(PRICE, f"S{i}") for i in range(30)])
        manifest.record_bandwidth(100, 700)
        assert log.progress(manifest, guard)["eta_hours"] is None  # nothing finished yet: no rate
        log.finished.update({"ok": 9, "failed": 1})
        clock.now = 120.0  # two minutes in
        fields = log.progress(manifest, guard)
        assert fields["finished"] == 10 and fields["avg_rate_per_min"] == 5.0 and fields["rate_per_min"] == 5.0
        assert fields["pending"] == 30 and fields["eta_hours"] == 0.1  # 30 tasks at 5/min
        assert fields["window_bytes"] == 100 and fields["window_limit_bytes"] == 500 and fields["basis"] == "wire"
        clock.now = 180.0
        log.finished["ok"] += 6
        assert log.progress(manifest, guard, final=True)["rate_per_min"] == 6.0  # the last minute only
    log.close()
    progress = [r for r in _records(tmp_path / "run.jsonl") if r["event"] == "progress"]
    assert len(progress) == 3 and progress[-1]["final"] is True and progress[1]["by_status"] == {"ok": 9, "failed": 1}
