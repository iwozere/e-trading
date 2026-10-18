"""
Integration tests for ``Downloader`` against ``MockFmpServer`` (spec §6.2–§6.4, §13): plan restrictions, quota and
auth stops, bandwidth hard stop, sink failures, and kill-and-resume (in-process cancel and a real process kill).
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import pytest

from src.data.fmp_pipeline.archive.archive import RawArchive
from src.data.fmp_pipeline.archive.reader import iter_records, latest_successful
from src.data.fmp_pipeline.client.circuit_breaker import CircuitBreaker
from src.data.fmp_pipeline.client.fmp_client import FetchResult
from src.data.fmp_pipeline.errors import ArchiveUnavailable
from src.data.fmp_pipeline.manifest.manifest import Manifest, Task, TaskSpec
from src.data.fmp_pipeline.runner import BandwidthGuard, Downloader, sha256_sink
from src.data.fmp_pipeline.tests.helpers import FakeSleep, make_client
from src.data.fmp_pipeline.tests.kill_resume_child import DUMP_ID, file_sink
from src.data.fmp_pipeline.tests.mock_server import Fault, MockConfig, MockFmpServer

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PRICE = "historical-price-eod/non-split-adjusted"


def _price_tasks(symbols: list[str]) -> list[TaskSpec]:
    return [TaskSpec(PRICE, s, {"from": "2025-01-01", "to": "2025-03-31"}) for s in symbols]


@pytest.mark.asyncio
async def test_full_run_records_every_outcome(tmp_path: Path):
    config = MockConfig(restricted_symbols={"R1"}, unknown_symbols={"GONE"})
    with MockFmpServer(config) as fmp, Manifest(tmp_path / "m.sqlite", "d") as manifest:
        manifest.add_tasks(_price_tasks(["A1", "A2", "A3", "R1", "GONE"]) + [TaskSpec("no-such-endpoint")])
        async with make_client(fmp, FakeSleep()) as client:
            summary = await Downloader(manifest, client, workers=3).run()
        assert summary.stop_reason == "done" and summary.processed == 6
        assert manifest.counts() == {"ok": 3, "plan_restricted": 1, "ok_empty": 1, "api_error": 1}
        wire, decoded = manifest.bandwidth_used()
        assert 0 < wire < decoded and decoded == fmp.bytes_decoded
        ok_rows = [manifest.task(t) for t in _ids(manifest, "ok")]
        assert all(r is not None and r["sha256"] and r["rows"] > 0 for r in ok_rows)
        assert [e[1] for e in manifest.events()] == ["run_start", "run_stop"]


def _ids(manifest: Manifest, status: str) -> list[str]:
    db = sqlite3.connect(manifest.path)
    try:
        return [r[0] for r in db.execute("SELECT task_id FROM tasks WHERE status=?", (status,))]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_priority_order_and_retry_bandwidth_is_counted(tmp_path: Path):
    with MockFmpServer() as fmp, Manifest(tmp_path / "m.sqlite", "d") as manifest:
        fmp.add_fault(Fault(endpoint=PRICE, symbol="LATE", status=503, body="x" * 500, times=2))
        manifest.add_tasks([TaskSpec(PRICE, "LATE", priority=50), TaskSpec(PRICE, "EARLY", priority=10)])
        async with make_client(fmp, FakeSleep()) as client:
            await Downloader(manifest, client, workers=1).run()
        assert [r.params["symbol"] for r in fmp.requests] == ["EARLY", "LATE", "LATE", "LATE"]
        assert manifest.bandwidth_used()[1] == fmp.bytes_decoded  # the two 503 bodies count too


@pytest.mark.asyncio
async def test_quota_stop_leaves_remaining_tasks_pending(tmp_path: Path):
    with MockFmpServer(MockConfig(quota_after=3)) as fmp, Manifest(tmp_path / "m.sqlite", "d") as manifest:
        manifest.add_tasks(_price_tasks([f"S{i}" for i in range(10)]))
        async with make_client(fmp, FakeSleep()) as client:
            summary = await Downloader(manifest, client, workers=1).run()
        assert summary.stop_reason == "quota"
        assert manifest.counts() == {"ok": 3, "pending": 7}
        assert len(fmp.requests) == 4  # stopped at the first quota response, no retry loop
        assert any(e[1] == "stop_quota" for e in manifest.events())


@pytest.mark.asyncio
async def test_auth_abort_stops_the_run(tmp_path: Path):
    with MockFmpServer() as fmp, Manifest(tmp_path / "m.sqlite", "d") as manifest:
        fmp.revoke_key()
        manifest.add_tasks(_price_tasks([f"S{i}" for i in range(20)]))
        async with make_client(fmp, FakeSleep(), auth_abort_after=5) as client:
            summary = await Downloader(manifest, client, workers=2).run()
        assert summary.stop_reason == "auth"
        counts = manifest.counts()
        assert counts.get("pending", 0) >= 14 and "running" not in counts


@pytest.mark.asyncio
async def test_endless_429_stops_the_run_as_quota_and_keeps_tasks_pending(tmp_path: Path):
    sleep = FakeSleep()
    with MockFmpServer() as fmp, Manifest(tmp_path / "m.sqlite", "d") as manifest:
        fmp.add_fault(Fault(status=429, body={"message": "bandwidth used up"}, times=None))
        manifest.add_tasks(_price_tasks([f"S{i}" for i in range(20)]))
        breaker = CircuitBreaker(threshold=4, pause_s=60, stop_after_opens=3, sleep=sleep)
        async with make_client(fmp, sleep, breaker=breaker, backoff_base_s=0.001) as client:
            summary = await Downloader(manifest, client, workers=2).run()
        assert summary.stop_reason == "quota"
        counts = manifest.counts()
        assert counts.get("pending", 0) >= 18 and "running" not in counts  # nothing marked failed or done
        assert len(fmp.requests) <= 14
        stops = [e[2] for e in manifest.events() if e[1] == "stop_quota"]
        assert len(stops) == 1 and "bandwidth used up" in stops[0]


@pytest.mark.asyncio
async def test_bandwidth_hard_stop(tmp_path: Path):
    with MockFmpServer() as fmp, Manifest(tmp_path / "m.sqlite", "d") as manifest:
        manifest.add_tasks(_price_tasks([f"S{i}" for i in range(10)]))
        manifest.record_bandwidth(0, 8_000)  # already used earlier in the window
        guard = BandwidthGuard(cap_bytes=40_000, safety=0.5, basis="decoded", refresh_s=3600)
        async with make_client(fmp, FakeSleep()) as client:
            summary = await Downloader(manifest, client, workers=1, guard=guard).run()
        assert summary.stop_reason == "bandwidth"
        done = summary.processed
        assert 0 < done < 10 and manifest.counts()["pending"] == 10 - done
        # Never exceeded by more than the one response in flight when the limit was crossed.
        assert manifest.bandwidth_used()[1] - 8_000 < 20_000 + fmp.requests[-1].bytes_decoded


@pytest.mark.asyncio
async def test_bandwidth_guard_on_wire_basis(tmp_path: Path):
    with Manifest(tmp_path / "m.sqlite", "d") as manifest:
        manifest.record_bandwidth(950, 9_000)
        assert BandwidthGuard(cap_bytes=1_000, safety=0.9, basis="wire").exceeded(manifest)
        assert BandwidthGuard(cap_bytes=100_000, safety=0.9, basis="decoded").exceeded(manifest) is None


def test_bandwidth_guard_scales_local_bytes_to_what_fmp_counts(tmp_path: Path):
    with Manifest(tmp_path / "m.sqlite", "d") as manifest:
        manifest.record_bandwidth(0, 10_000)
        # 10,000 local bytes are 7,500 FMP bytes at scale 0.75: under 90% of a 10,000 cap, unlike unscaled.
        assert BandwidthGuard(cap_bytes=10_000, safety=0.9, basis="decoded", scale=0.75).exceeded(manifest) is None
        assert BandwidthGuard(cap_bytes=10_000, safety=0.9, basis="decoded").exceeded(manifest)
        message = BandwidthGuard(cap_bytes=10_000, safety=0.9, basis="decoded", scale=1.25).exceeded(manifest)
        assert message is not None and "1.25 dashboard bytes per local byte" in message


@pytest.mark.asyncio
async def test_sink_failure_fails_only_that_task_and_async_sinks_work(tmp_path: Path):
    sunk: list[str] = []

    async def sink(task: Task, result: FetchResult) -> str:
        if task.symbol == "BAD":
            raise OSError("disk full")
        sunk.append(task.symbol or "")
        return sha256_sink(task, result)

    with MockFmpServer() as fmp, Manifest(tmp_path / "m.sqlite", "d") as manifest:
        manifest.add_tasks(_price_tasks(["A", "BAD", "C"]))
        async with make_client(fmp, FakeSleep()) as client:
            summary = await Downloader(manifest, client, sink=sink, workers=2).run()
        assert summary.stop_reason == "done" and sorted(sunk) == ["A", "C"]
        assert manifest.counts() == {"ok": 2, "failed": 1}
        failed = manifest.task(_ids(manifest, "failed")[0])
        assert failed is not None and "disk full" in failed["error"]


@pytest.mark.asyncio
async def test_invalid_task_params_fail_without_a_request(tmp_path: Path):
    with MockFmpServer() as fmp, Manifest(tmp_path / "m.sqlite", "d") as manifest:
        manifest.add_tasks([TaskSpec(PRICE, "A", {"from": "bad"})])
        async with make_client(fmp, FakeSleep()) as client:
            summary = await Downloader(manifest, client, workers=1).run()
        assert summary.by_status == {"failed": 1} and fmp.requests == []


@pytest.mark.asyncio
async def test_cancel_and_resume_in_process(tmp_path: Path):
    path = tmp_path / "m.sqlite"
    sink_file = tmp_path / "sink.txt"
    symbols = [f"S{i:03d}" for i in range(40)]
    with MockFmpServer() as fmp:
        fmp.add_fault(Fault(delay_s=0.02, times=None))
        with Manifest(path, "d") as manifest:
            manifest.add_tasks(_price_tasks(symbols))
            async with make_client(fmp, FakeSleep()) as client:
                run = asyncio.create_task(Downloader(manifest, client, sink=file_sink(sink_file), workers=4).run())
                while len(_ids(manifest, "ok")) < 10:
                    await asyncio.sleep(0.01)
                run.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await run
        with Manifest(path, "d") as manifest:
            assert manifest.recovered > 0
            async with make_client(fmp, FakeSleep()) as client:
                summary = await Downloader(manifest, client, sink=file_sink(sink_file), workers=4).run()
            assert summary.stop_reason == "done" and manifest.counts() == {"ok": 40}
            _assert_at_least_once(manifest, sink_file, in_flight=4)


def _assert_at_least_once(manifest: Manifest, sink_file: Path, in_flight: int) -> None:
    """Every task reached the sink; duplicates are bounded by the requests in flight at the kill."""
    lines = sink_file.read_text(encoding="utf-8").split()
    ids = set(_ids(manifest, "ok"))
    assert set(lines) == ids
    assert len(lines) - len(ids) <= in_flight


@pytest.mark.parametrize("with_archive", [False, True])
def test_kill_and_resume_across_processes(tmp_path: Path, with_archive: bool):
    path = tmp_path / "manifest.sqlite"
    sink_file = tmp_path / "sink.txt"
    raw_root: Optional[Path] = tmp_path / "raw" if with_archive else None
    with Manifest(path, DUMP_ID) as manifest:
        manifest.add_tasks(_price_tasks([f"K{i:03d}" for i in range(120)]))

    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    args = [str(path), str(sink_file)] + ([str(raw_root)] if raw_root is not None else [])
    child = subprocess.Popen([sys.executable, "-m", "src.data.fmp_pipeline.tests.kill_resume_child", *args],
                             cwd=PROJECT_ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and child.poll() is None:
            if sink_file.exists() and len(sink_file.read_text(encoding="utf-8").split()) >= 20:
                break
            time.sleep(0.05)
        assert child.poll() is None, child.stderr.read().decode(errors="replace") if child.stderr else ""
        child.kill()  # TerminateProcess on Windows, SIGKILL elsewhere: no cleanup runs
    finally:
        child.wait(timeout=30)
        if child.stderr:
            child.stderr.close()

    async def resume() -> None:
        with MockFmpServer() as fmp, Manifest(path, DUMP_ID) as manifest:
            counts_before = manifest.counts()
            assert counts_before.get("pending", 0) > 0 and "running" not in counts_before
            archive = RawArchive(raw_root, DUMP_ID) if raw_root is not None else None
            async with make_client(fmp, FakeSleep()) as client:
                summary = await Downloader(manifest, client, sink=file_sink(sink_file, archive), workers=4).run()
            assert summary.stop_reason == "done" and manifest.counts() == {"ok": 120}
            _assert_at_least_once(manifest, sink_file, in_flight=4)
            if archive is not None:
                _assert_archive_complete(manifest, archive)

    asyncio.run(resume())


def _assert_archive_complete(manifest: Manifest, archive: RawArchive) -> None:
    """Every file reads strictly end to end, and every ok task's newest line matches the manifest's sha256."""
    records = [r for f in sorted(archive.dump_dir.rglob("*.jsonl.gz")) for r in iter_records(f)]
    by_task = {r["task_id"]: r for r in latest_successful(records).values()}
    for tid in _ids(manifest, "ok"):
        row = manifest.task(tid)
        assert row is not None and by_task[tid]["sha256"] == row["sha256"]


@pytest.mark.asyncio
async def test_archive_is_the_sink(tmp_path: Path):
    config = MockConfig(restricted_symbols={"R1"})
    with MockFmpServer(config) as fmp, Manifest(tmp_path / "m.sqlite", "d") as manifest:
        manifest.add_tasks(_price_tasks(["A1", "A2", "R1"]) + [TaskSpec("delisted-companies", None, {"page": 0})])
        archive = RawArchive(tmp_path / "raw", "d")
        async with make_client(fmp, FakeSleep()) as client:
            summary = await Downloader(manifest, client, sink=archive, workers=2).run()
        assert summary.stop_reason == "done" and archive.appended == 4
        _assert_archive_complete(manifest, archive)
        [restricted] = iter_records(archive.path_for(PRICE, "R1"))
        assert restricted["http_status"] == 402 and restricted["body_text"]


@pytest.mark.asyncio
async def test_archive_unavailable_pauses_the_run(tmp_path: Path):
    calls = 0

    def sink(task: Task, result: FetchResult) -> str:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ArchiveUnavailable("archive root D:/fmp-data/raw is not available")
        return sha256_sink(task, result)

    with MockFmpServer() as fmp, Manifest(tmp_path / "m.sqlite", "d") as manifest:
        manifest.add_tasks(_price_tasks([f"S{i}" for i in range(10)]))
        async with make_client(fmp, FakeSleep()) as client:
            summary = await Downloader(manifest, client, sink=sink, workers=1).run()
        assert summary.stop_reason == "archive"
        assert manifest.counts() == {"ok": 2, "pending": 8}  # nothing marked failed
        assert any(e[1] == "stop_archive" for e in manifest.events())
