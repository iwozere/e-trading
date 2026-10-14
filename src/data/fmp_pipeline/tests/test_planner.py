"""
End-to-end tests of ``fmp plan`` against the mock server (spec §6.3, §7.2, plan WP-4 / G1 "plan --dry-run").

The universe lists the free key could not fetch (``stock-list``, constituents, …) are injected as synthetic
``MockConfig.extra_fixtures``; their field names are a [VERIFY] item until the paid probe.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from src.data.fmp_pipeline import cli
from src.data.fmp_pipeline.archive.archive import RawArchive
from src.data.fmp_pipeline.fsutil import exclusive_lock
from src.data.fmp_pipeline.manifest.manifest import Manifest
from src.data.fmp_pipeline.planner.catalog import DAILY_RAW
from src.data.fmp_pipeline.planner.config import IntradayConfig, PlanConfig, UniverseConfig
from src.data.fmp_pipeline.planner.follow_ups import FollowUps
from src.data.fmp_pipeline.planner.planner import Planner, bandwidth_guard
from src.data.fmp_pipeline.runner import Downloader
from src.data.fmp_pipeline.tests.helpers import FakeSleep, make_client
from src.data.fmp_pipeline.tests.mock_server import MockConfig, MockFmpServer

AS_OF = date(2026, 10, 14)
LISTS: dict[str, Any] = {
    "stock-list": [{"symbol": "AAA", "companyName": "A"}, {"symbol": "BBB", "companyName": "B"}],
    "actively-trading-list": [{"symbol": "AAA"}, {"symbol": "BBB"}],
    "etf-list": [{"symbol": "SPY", "name": "S"}],
    "sp500-constituent": [{"symbol": "AAA"}],
    "historical-sp500-constituent": [{"symbol": "AAA", "removedTicker": "GONE"}],
    "symbol-change": [{"oldSymbol": "OLDB", "newSymbol": "BBB", "date": "2019-05-06"}],
    # The probe-derived profile fixture has every flag true (ETF, fund and ADR at once); use a plain stock.
    "profile": [{"symbol": "X", "exchange": "NASDAQ", "isEtf": False, "isFund": False, "isAdr": False,
                 "isActivelyTrading": True, "ipoDate": "2010-01-04"}],
}
CONFIG = PlanConfig(dump_id="t", priorities=("P0",), daily_from=date(2024, 1, 1), workers=4,
                    universe=UniverseConfig(tier_a=("SPY",), indices=(), include_exchanges=("NASDAQ", "SYNTH0")),
                    intraday=IntradayConfig(max_rows=1000))


def _mock(**changes: Any) -> MockConfig:
    # symbol-change is served unpaged (same body for every page): the repeated-page guard ends its paging.
    options: dict[str, Any] = {"extra_fixtures": LISTS, "paged_totals": {"delisted-companies": 3}}
    options.update(changes)
    return MockConfig(**options)


def _planner(tmp_path: Path, manifest: Manifest, config: PlanConfig = CONFIG) -> Planner:
    return Planner(manifest, RawArchive(tmp_path / "raw", config.dump_id), config, tmp_path / "reports", AS_OF)


async def _run_all(manifest: Manifest, client: Any, archive_root: Path, config: PlanConfig = CONFIG) -> None:
    archive = RawArchive(archive_root, config.dump_id)
    await Downloader(manifest, client, sink=archive, workers=4, guard=bandwidth_guard(config),
                     reviewer=FollowUps(manifest, config)).run()


@pytest.mark.asyncio
async def test_plan_run_replan_cycle(tmp_path: Path):
    with MockFmpServer(_mock()) as fmp, Manifest(tmp_path / "m.sqlite", "t") as manifest:
        planner = _planner(tmp_path, manifest)
        async with make_client(fmp, FakeSleep()) as client:
            assert (await planner.fetch_sources(client)).stop_reason == "done"
            calls = len(fmp.requests)
            await planner.fetch_sources(client)
            assert len(fmp.requests) == calls  # bootstrap lists are fetched once per dump

            dry = planner.plan(dry_run=True)
            assert dry.context.added == 0 and set(manifest.counts()) <= {"ok", "ok_empty", "api_error"}
            universe = dry.universe
            assert {s.symbol for s in universe.unresolved} == {"AAA", "BBB", "GONE", "OLDB"}
            assert {s.symbol for s in universe.resolved} == {"SPY", "SYM00000", "SYM00001", "SYM00002"}

            first = planner.plan()
            assert first.context.added > 0 and planner.plan().context.added == 0  # idempotent
            pending = {(r["endpoint"], r["symbol"]) for r in manifest.task_rows() if r["status"] == "pending"}
            assert ("profile", "AAA") in pending and (DAILY_RAW, "AAA") not in pending
            assert (DAILY_RAW, "SPY") in pending

            await _run_all(manifest, client, tmp_path / "raw")
            second = planner.plan()
            resolved = {s.symbol: s.tier for s in second.universe.resolved}
            assert resolved["AAA"] == "B" and resolved["GONE"] == "B" and resolved["BBB"] == "C"
            assert second.context.added > 0 and not second.universe.unresolved

            await _run_all(manifest, client, tmp_path / "raw")
            third = planner.plan()  # daily raw done everywhere: every intraday window is now written
            assert not third.projection.deferred_reasons and third.context.added > 0
            assert "historical-chart/1hour" in {r["endpoint"] for r in manifest.task_rows() if r["status"] == "pending"}
            assert len(manifest.bounds()) == len(third.universe.resolved)

    budget = third.budget_path.read_text(encoding="utf-8")
    assert "## Projection per block" in budget and "| intraday tier A |" in budget and "**enough**" in budget
    report = third.universe_path.read_text(encoding="utf-8")
    assert "| `stock-list` | ok (1 request) | 2 |" in report and "| NASDAQ | 8 | yes |" in report
    assert "| `symbol-change` | ok (2 requests) | 1 |" in report  # the repeated page 1 is not counted twice


@pytest.mark.asyncio
async def test_free_key_like_plan_reports_restricted_sources(tmp_path: Path):
    restricted = {"stock-list", "actively-trading-list", "etf-list", "symbol-change", "available-exchanges",
                  "financial-statement-symbol-list", "sp500-constituent", "nasdaq-constituent",
                  "dowjones-constituent", "historical-sp500-constituent", "historical-nasdaq-constituent",
                  "historical-dowjones-constituent"}
    config = replace(CONFIG, intraday=IntradayConfig())  # row cap unknown, as before the paid probe
    with MockFmpServer(_mock(restricted_endpoints=restricted)) as fmp, \
            Manifest(tmp_path / "m.sqlite", "t") as manifest:
        planner = _planner(tmp_path, manifest, config)
        async with make_client(fmp, FakeSleep()) as client:
            assert (await planner.fetch_sources(client)).stop_reason == "done"
        result = planner.plan(dry_run=True)
    assert result.universe.source_status["stock-list"] == "plan_restricted"
    assert result.universe.source_status["delisted-companies"] == "ok (2 requests)"
    assert len(result.universe.resolved) == 4
    assert result.projection.deferred_reasons == {"row cap unknown (paid probe)": result.projection.blocks[6].deferred}
    budget = result.budget_path.read_text(encoding="utf-8")
    assert "Dry run: nothing was added" in budget and "row cap **unknown**" in budget
    assert "| row cap unknown (paid probe) |" in budget


@pytest.mark.asyncio
async def test_symbols_dropped_from_the_universe_are_skipped(tmp_path: Path):
    with MockFmpServer(_mock()) as fmp, Manifest(tmp_path / "m.sqlite", "t") as manifest:
        planner = _planner(tmp_path, manifest)
        async with make_client(fmp, FakeSleep()) as client:
            await planner.fetch_sources(client)
        planner.plan()
        narrower = replace(CONFIG, universe=replace(CONFIG.universe, include_exchanges=("NASDAQ",)))
        result = _planner(tmp_path, manifest, narrower).plan()
        skipped = {r["symbol"] for r in manifest.task_rows() if r["status"] == "skipped"}
    assert result.skipped > 0 and result.universe.excluded["exchange"] == 3
    assert skipped == {"SYM00000", "SYM00001", "SYM00002"}


def test_as_of_is_fixed_by_the_first_plan(tmp_path: Path):
    with Manifest(tmp_path / "m.sqlite", "t") as manifest:
        assert _planner(tmp_path, manifest).as_of == AS_OF
        later = Planner(manifest, RawArchive(tmp_path / "raw", "t"), CONFIG, tmp_path, date(2027, 1, 1))
        assert later.as_of == AS_OF


def test_dump_id_mismatch_is_rejected(tmp_path: Path):
    with Manifest(tmp_path / "m.sqlite", "other") as manifest, pytest.raises(ValueError, match="dump id mismatch"):
        _planner(tmp_path, manifest)


def test_plan_reads_profiles_and_ignores_foreign_files(tmp_path: Path):
    """A stray file in the profile folder is skipped with a warning, not fatal."""
    with Manifest(tmp_path / "m.sqlite", "t") as manifest:
        planner = _planner(tmp_path, manifest)
        folder = tmp_path / "raw" / "t" / "profile"
        folder.mkdir(parents=True)
        (folder / "_ABC.jsonl.gz").write_bytes(b"")  # not canonical: ABC would encode without "_"
        assert planner.plan(dry_run=True).universe.securities.keys() == {"SPY"}


def test_cli_plan_offline_and_lock(tmp_path: Path):
    config = tmp_path / "fmp.yaml"
    config.write_text("fmp:\n  dump_id: cli\n  priorities: [P0]\n  daily_from: 2025-01-01\n", encoding="utf-8")
    args = ["plan", "--offline", "--config", str(config), "--data-root", str(tmp_path), "--as-of", "2026-10-14"]
    assert cli.main(args) == 0
    assert (tmp_path / "reports" / "cli" / "budget_plan.md").exists()
    assert (tmp_path / "manifests" / "cli.sqlite").exists()
    assert cli.main([*args, "--dump-id", "cli2", "--dry-run"]) == 0
    with exclusive_lock(tmp_path / "locks" / "cli.lock"):
        assert cli.main(args) == 2


def test_cli_plan_fetches_with_the_configured_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, Any] = {}

    async def fake_fetch(planner: Planner, api_key: str, config: PlanConfig) -> str:
        seen.update(key=api_key, dump=config.dump_id, as_of=planner.as_of)
        return "quota"

    monkeypatch.setattr(cli, "_fetch_sources", fake_fetch)
    monkeypatch.setattr("src.config.provider_config.get_api_key", lambda name: f"key-for-{name}")
    code = cli.main(["plan", "--data-root", str(tmp_path), "--dump-id", "f", "--as-of", "2026-10-14"])
    assert code == 1 and seen == {"key": "key-for-FMP_API_KEY", "dump": "f", "as_of": AS_OF}


@pytest.mark.asyncio
async def test_large_plans_are_inserted_in_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import src.data.fmp_pipeline.planner.planner as module

    monkeypatch.setattr(module, "INSERT_BATCH", 3)
    with MockFmpServer(_mock()) as fmp, Manifest(tmp_path / "m.sqlite", "t") as manifest:
        planner = _planner(tmp_path, manifest)
        async with make_client(fmp, FakeSleep()) as client:
            await planner.fetch_sources(client)
        before = sum(manifest.counts().values())
        result = planner.plan()
        assert result.context.added > 3 and sum(manifest.counts().values()) == before + result.context.added


def test_source_status_texts():
    assert _status([]) == "not fetched"
    assert _status(["ok"]) == "ok (1 request)"
    assert _status(["ok", "ok_empty"]) == "ok (2 requests)"
    assert _status(["ok", "pending"]) == "incomplete"
    assert _status(["plan_restricted", "failed", "failed"]) == "failed, plan_restricted"


def _status(statuses: list[str]) -> str:
    from src.data.fmp_pipeline.planner.planner import _source_status

    return _source_status(statuses)


def test_cli_plan_without_a_key_fails_clearly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("src.config.provider_config.get_api_key", lambda name: None)
    assert cli.main(["plan", "--data-root", str(tmp_path), "--dump-id", "k", "--as-of", "2026-10-14"]) == 2
