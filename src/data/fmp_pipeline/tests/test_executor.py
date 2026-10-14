"""Tests for ``fmp run`` / ``fmp status`` / ``fmp retry-failed`` against the mock server (spec §6.1, §6.3, §6.8)."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from src.data.fmp_pipeline import cli, power
from src.data.fmp_pipeline.archive.archive import RawArchive
from src.data.fmp_pipeline.archive.reader import iter_records
from src.data.fmp_pipeline.errors import InsufficientDiskSpace
from src.data.fmp_pipeline.executor import DumpExecutor, RunScope
from src.data.fmp_pipeline.fsutil import exclusive_lock
from src.data.fmp_pipeline.manifest.manifest import Manifest, TaskSpec
from src.data.fmp_pipeline.planner.config import IntradayConfig, PlanConfig, UniverseConfig
from src.data.fmp_pipeline.planner.planner import REMAINING_KEY, Planner
from src.data.fmp_pipeline.runner import GB, BandwidthGuard
from src.data.fmp_pipeline.status import render_status
from src.data.fmp_pipeline.tests.helpers import FakeSleep, make_client
from src.data.fmp_pipeline.tests.mock_server import MockConfig, MockFmpServer

AS_OF = date(2026, 10, 14)
PROFILE = [{"symbol": "X", "exchange": "NASDAQ", "isEtf": False, "isFund": False, "isAdr": False,
            "isActivelyTrading": True, "ipoDate": "2010-01-04"}]
CONFIG = PlanConfig(dump_id="r", priorities=("P0",), daily_from=date(2025, 1, 1), workers=4,
                    universe=UniverseConfig(tier_a=("SPY",), indices=(), include_exchanges=("NASDAQ", "SYNTH0")),
                    intraday=IntradayConfig(max_rows=1000))


def _mock(**changes: Any) -> MockConfig:
    options: dict[str, Any] = {"extra_fixtures": {"profile": PROFILE}, "paged_totals": {"delisted-companies": 2}}
    options.update(changes)
    return MockConfig(**options)


async def _planned(tmp_path: Path, manifest: Manifest, fmp: MockFmpServer) -> RawArchive:
    archive = RawArchive(tmp_path / "raw", CONFIG.dump_id)
    planner = Planner(manifest, archive, CONFIG, tmp_path / "reports", AS_OF)
    async with make_client(fmp, FakeSleep()) as client:
        await planner.fetch_sources(client)
    planner.plan()
    return archive


def test_run_scope():
    assert RunScope().blocks() == [0, 1, 2, 3, 4, 5, 6]
    assert RunScope(phase=3).blocks() == [3]
    assert RunScope(tier="B").blocks() == [6] == RunScope(phase=6, tier="B").blocks()
    with pytest.raises(ValueError):
        RunScope(phase=9)
    with pytest.raises(ValueError):
        RunScope(tier="Z")
    with pytest.raises(ValueError):
        RunScope(phase=3, tier="A")


def test_guard_per_run_limit(tmp_path: Path):
    guard = BandwidthGuard(run_limit_bytes=100, basis="wire")
    with Manifest(tmp_path / "m.sqlite", "d") as manifest:
        assert guard.exceeded(manifest) is None
        guard.add(60, 600)
        assert guard.exceeded(manifest) is None
        guard.add(40, 400)
        message = guard.exceeded(manifest)
    assert message is not None and "--max-gb" in message and guard.run_used == 100


@pytest.mark.asyncio
async def test_run_executes_all_blocks_in_order_then_replan_adds_intraday(tmp_path: Path):
    with MockFmpServer(_mock()) as fmp, Manifest(tmp_path / "m.sqlite", "r") as manifest:
        archive = await _planned(tmp_path, manifest, fmp)
        executor = DumpExecutor(manifest, archive, CONFIG)
        async with make_client(fmp, FakeSleep()) as client:
            report = await executor.run(client, RunScope())
        assert report.stop_reason == "done" and report.processed > 0 and report.run_bytes > 0
        assert "pending" not in manifest.counts()
        starts = [e for e in manifest.events() if e[1] == "run_start"]
        assert len(starts) == 1 + 7  # the plan's source fetch, then one Downloader per block
        order = [r["priority"] // 100 for r in sorted(manifest.task_rows(), key=lambda r: r["fetched_at"] or "")
                 if r["status"] == "ok"]
        assert order == sorted(order)  # strict block order
        assert list(iter_records(archive.path_for("historical-price-eod/non-split-adjusted", "SPY")))

        replan = Planner(manifest, archive, CONFIG, tmp_path / "reports").plan()
        assert replan.context.added > 0  # intraday windows, now that daily bounds exist
        async with make_client(fmp, FakeSleep()) as client:
            report = await executor.run(client, RunScope(tier="A"))
        assert report.stop_reason == "done" and set(report.blocks) == {6}
        tiers = {r["tier"] for r in manifest.task_rows() if r["status"] == "pending"}
        assert "A" not in tiers and tiers  # only tier A ran

        text = render_status(manifest, CONFIG)
    assert "6 intraday" in text and "  tier A" in text and "Bandwidth, last 30 days" in text
    assert "tasks pending" in text and "bandwidth windows" in text


@pytest.mark.asyncio
async def test_max_gb_stops_the_run(tmp_path: Path):
    with MockFmpServer(_mock()) as fmp, Manifest(tmp_path / "m.sqlite", "r") as manifest:
        archive = await _planned(tmp_path, manifest, fmp)
        executor = DumpExecutor(manifest, archive, CONFIG, max_gb=1 / GB, workers=1)  # one byte
        async with make_client(fmp, FakeSleep()) as client:
            report = await executor.run(client, RunScope())
        assert report.stop_reason == "bandwidth" and report.processed == 1
        assert manifest.counts()["pending"] > 0
        assert any(e[1] == "stop_bandwidth" and "--max-gb" in e[2] for e in manifest.events())


@pytest.mark.asyncio
async def test_disk_check_and_unplanned_dump(tmp_path: Path):
    with MockFmpServer(_mock()) as fmp, Manifest(tmp_path / "m.sqlite", "r") as manifest:
        archive = RawArchive(tmp_path / "raw", "r")
        executor = DumpExecutor(manifest, archive, CONFIG)
        async with make_client(fmp, FakeSleep()) as client:
            with pytest.raises(RuntimeError, match="fmp plan"):
                await executor.run(client, RunScope())
            manifest.meta_set(REMAINING_KEY, json.dumps({"3": {"wire": 10**18, "decoded": 10**19}}))
            assert executor.check_disk([0, 1, 2]) > 0  # nothing projected for these blocks
            with pytest.raises(InsufficientDiskSpace):
                await executor.run(client, RunScope(phase=3))
    with Manifest(tmp_path / "m.sqlite", "other") as manifest, pytest.raises(ValueError, match="dump id"):
        DumpExecutor(manifest, archive, CONFIG)


def test_status_reader_does_not_disturb_a_live_run(tmp_path: Path):
    with Manifest(tmp_path / "m.sqlite", "r") as writer:
        writer.add_tasks([TaskSpec("splits", "A", {}, 101), TaskSpec("splits", "B", {}, 101)])
        writer.claim(1)
        with Manifest(tmp_path / "m.sqlite", "r", recover=False) as reader:
            assert reader.recovered == 0 and reader.counts() == {"pending": 1, "running": 1}
            text = render_status(reader, CONFIG)
    assert "1 events" in text and "not planned" in text and "2 tasks pending" in text


def test_status_lists_failures(tmp_path: Path):
    from src.data.fmp_pipeline.tests.test_manifest import _result

    with Manifest(tmp_path / "m.sqlite", "r") as manifest:
        manifest.add_tasks([TaskSpec("splits", "A", {}, 101)])
        [task] = manifest.claim()
        manifest.complete(task, _result("server_error", 503, "HTTP 503 after 6 attempts"))
        text = render_status(manifest, CONFIG)
    assert "Recent failures" in text and "splits A: HTTP 503 after 6 attempts" in text


def _args(tmp_path: Path, *extra: str) -> list[str]:
    config = tmp_path / "fmp.yaml"
    config.write_text("fmp:\n  dump_id: c\n  priorities: [P0]\n  daily_from: 2025-01-01\n  workers: 2\n"
                      "  universe: {tier_a: [SPY], indices: []}\n", encoding="utf-8")
    return ["--config", str(config), "--data-root", str(tmp_path), *extra]


def test_cli_run_status_and_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    with MockFmpServer(_mock()) as fmp:
        monkeypatch.setattr(cli, "_api_key", lambda: fmp.config.api_key)
        monkeypatch.setattr(cli, "_client", lambda key, config: make_client(fmp, FakeSleep()))
        assert cli.main(["run", *_args(tmp_path)]) == cli.EXIT_USAGE  # not planned yet
        assert cli.main(["status", *_args(tmp_path)]) == cli.EXIT_USAGE
        assert cli.main(["retry-failed", *_args(tmp_path)]) == cli.EXIT_USAGE
        assert cli.main(["plan", *_args(tmp_path, "--as-of", "2026-10-14")]) == cli.EXIT_OK
        assert cli.main(["run", *_args(tmp_path, "--phase", "0")]) == cli.EXIT_OK
        assert cli.main(["run", *_args(tmp_path, "--phase", "4", "--tier", "A")]) == cli.EXIT_USAGE
        assert cli.main(["run", *_args(tmp_path, "--max-gb", "0.0000000001")]) == cli.EXIT_STOPPED
        assert cli.main(["status", *_args(tmp_path)]) == cli.EXIT_OK
        assert cli.main(["retry-failed", *_args(tmp_path)]) == cli.EXIT_OK
        with exclusive_lock(tmp_path / "locks" / "c.lock"):
            assert cli.main(["run", *_args(tmp_path)]) == cli.EXIT_USAGE
            assert cli.main(["retry-failed", *_args(tmp_path)]) == cli.EXIT_USAGE
            assert cli.main(["status", *_args(tmp_path)]) == cli.EXIT_OK  # readers need no lock
        with Manifest(tmp_path / "manifests" / "c.sqlite", "c") as manifest:
            manifest.meta_set(REMAINING_KEY, json.dumps({str(b): {"wire": 10**18, "decoded": 1} for b in range(7)}))
        assert cli.main(["run", *_args(tmp_path)]) == cli.EXIT_DISK
        manifest_path = tmp_path / "manifests" / "c.sqlite"
        with Manifest(manifest_path, "c") as manifest:
            manifest.meta_set(REMAINING_KEY, json.dumps({}))
        assert cli.main(["run", *_args(tmp_path)]) == cli.EXIT_OK
        with Manifest(manifest_path, "c") as manifest:
            assert "pending" not in manifest.counts()
            assert any(e[1] == "retry_failed" for e in manifest.events())


def test_cli_run_without_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cli, "_api_key", lambda: None)
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "c.sqlite").write_bytes(b"")
    assert cli.main(["run", *_args(tmp_path)]) == cli.EXIT_USAGE


@pytest.mark.skipif(sys.platform != "win32", reason="SetThreadExecutionState is Windows-only")
def test_keep_awake_sets_and_resets_execution_state(monkeypatch: pytest.MonkeyPatch):
    import ctypes

    calls: list[int] = []

    class Kernel32:
        def SetThreadExecutionState(self, flags: int) -> int:  # noqa: N802 - Win32 name
            calls.append(flags)
            return 0 if len(calls) == 1 else 1  # first call "fails": logged, not raised

    class WinDll:
        kernel32 = Kernel32()

    monkeypatch.setattr(ctypes, "windll", WinDll())
    with power.keep_awake():
        assert calls == [power.ES_CONTINUOUS | power.ES_SYSTEM_REQUIRED]
    assert calls[-1] == power.ES_CONTINUOUS


def test_keep_awake_is_a_no_op_elsewhere(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(power.sys, "platform", "linux")
    with power.keep_awake():
        pass
