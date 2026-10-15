"""End-to-end tests for ``fmp build`` (spec §7.4): archive in, validated vintage out, ``current.txt`` switched."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.fmp_pipeline import cli
from src.data.fmp_pipeline.archive.archive import RawArchive
from src.data.fmp_pipeline.builders import builder as builder_module
from src.data.fmp_pipeline.builders.builder import BuildFailed, VintageBuilder, archived_endpoints, read_current
from src.data.fmp_pipeline.builders.prices import DAILY_ENDPOINT, bucket_path
from src.data.fmp_pipeline.executor import DumpExecutor, RunScope
from src.data.fmp_pipeline.fsutil import exclusive_lock
from src.data.fmp_pipeline.manifest.manifest import Manifest
from src.data.fmp_pipeline.planner.config import PlanConfig, UniverseConfig
from src.data.fmp_pipeline.planner.planner import Planner
from src.data.fmp_pipeline.tests.archive_fixtures import daily_rows, put
from src.data.fmp_pipeline.tests.helpers import FakeSleep, make_client
from src.data.fmp_pipeline.tests.mock_server import MockConfig, MockFmpServer

AS_OF = date(2026, 10, 14)
CONFIG = PlanConfig(dump_id="v1", universe=UniverseConfig(tier_a=("SPY",), indices=()))


def _profile(symbol: str, name: str, **extra: Any) -> list[dict[str, Any]]:
    return [{"symbol": symbol, "companyName": name, "exchange": "NASDAQ", "isEtf": False, "isFund": False,
             "isAdr": False, "isActivelyTrading": True, "ipoDate": "2010-01-04", "cik": "1", **extra}]


def _archive(data_root: Path) -> RawArchive:
    archive = RawArchive(data_root / "raw", CONFIG.dump_id)
    put(archive, "stock-list", None, [{"symbol": "AAA", "exchange": "NASDAQ"}, {"symbol": "BBB", "exchange": "NYSE"},
                                      {"symbol": "OTCX", "exchange": "OTC"}])
    put(archive, "actively-trading-list", None, [{"symbol": "AAA"}, {"symbol": "BBB"}, {"symbol": "SPY"}])
    put(archive, "symbol-change", None, [{"date": "2018-01-02", "oldSymbol": "OLDA", "newSymbol": "AAA"}],
        {"page": 0, "limit": 100})
    put(archive, "profile", "AAA", _profile("AAA", "Alpha Inc"))
    put(archive, "profile", "SPY", _profile("SPY", "SPDR", isEtf=True))
    closes = {f"2025-01-{d:02d}": 100.0 + d for d in range(2, 31)}
    put(archive, DAILY_ENDPOINT, "AAA", daily_rows("AAA", closes), {"from": "2025-01-01", "to": "2025-12-31"})
    put(archive, DAILY_ENDPOINT, "SPY", daily_rows("SPY", {"2025-01-02": 500.0}),
        {"from": "2025-01-01", "to": "2025-12-31"})
    put(archive, "splits", "AAA", [{"symbol": "AAA", "date": "2025-01-15", "numerator": 2, "denominator": 1},
                                   {"symbol": "AAA", "date": "2019-01-15", "numerator": 1, "denominator": 0}])
    put(archive, "dividends", "AAA", [{"symbol": "AAA", "date": "2025-01-10", "recordDate": "2025-01-11",
                                       "paymentDate": "2025-01-20", "declarationDate": "", "dividend": 0.5,
                                       "adjDividend": 0.25, "yield": 1.1, "frequency": "Quarterly"}])
    put(archive, "income-statement", "AAA", [{"symbol": "AAA", "date": "2024-12-31", "period": "FY",
                                              "filingDate": "2025-02-01", "acceptedDate": "2025-02-01 16:01:02",
                                              "revenue": 10}], {"period": "annual", "limit": 200})
    put(archive, "some-future-endpoint", None, [{"x": 1}])
    return archive


def test_build_produces_a_valid_vintage_and_switches_current(tmp_path: Path):
    _archive(tmp_path)
    result = VintageBuilder(tmp_path, CONFIG, AS_OF).build()
    vintage = tmp_path / "vintages" / "v1"
    assert result.vintage_dir == vintage and read_current(tmp_path) == "v1" and result.switched
    securities = pq.read_table(vintage / "reference/securities.parquet").to_pylist()
    by_symbol = {s["symbol"]: s for s in securities}
    assert set(by_symbol) == {"AAA", "BBB", "SPY"}  # OTC excluded; OLDA is rename-only without prices
    aaa = by_symbol["AAA"]
    assert aaa["first_price_date"] == date(2025, 1, 2) and aaa["last_price_date"] == date(2025, 1, 30)
    assert by_symbol["SPY"]["instrument_type"] == "etf" and by_symbol["BBB"]["first_price_date"] is None
    history = pq.read_table(vintage / "reference/symbol_history.parquet").to_pylist()
    assert {(h["symbol"], h["reason"]) for h in history if h["security_id"] == aaa["security_id"]} == {
        ("AAA", "current"), ("OLDA", "symbol-change")}

    prices = pq.read_table(vintage / bucket_path("1d_raw", aaa["security_id"])).to_pylist()
    assert len(prices) == 29 and prices[0]["close"] == 102.0 and prices[0]["vintage"] == "v1"
    splits = pq.read_table(vintage / "events/splits.parquet").to_pylist()
    assert [(s["date"], s["ratio"]) for s in splits] == [(date(2019, 1, 15), None), (date(2025, 1, 15), 2.0)]
    [dividend] = pq.read_table(vintage / "events/dividends.parquet").to_pylist()
    assert (dividend["amount_raw"], dividend["amount_split_adjusted"], dividend["declaration_date"]) == (
        0.5, 0.25, None)
    [statement] = pq.read_table(vintage / "fundamentals/income_statement.parquet").to_pylist()
    assert statement["accepted_date"] == "2025-02-01 16:01:02" and statement["revenue"] == 10.0

    info = json.loads((vintage / "_meta/build_info.json").read_text(encoding="utf-8"))
    assert info["securities"] == 3 and info["invalid_split_ratios"] == 1
    assert info["unbuilt_endpoints"] == ["some-future-endpoint"] == result.unbuilt
    assert set(info["outputs"]) == set(result.outputs)
    assert all(len(o["sha256"]) == 64 for o in info["outputs"].values())
    assert (tmp_path / "reports/v1/missing_splits_candidates.csv").exists()
    assert (tmp_path / "reports/v1/identity_uncertain.csv").read_text(encoding="utf-8").startswith("symbol,")
    registry = json.loads((tmp_path / "security_ids.json").read_text(encoding="utf-8"))
    assert registry["next_id"] == 4  # OLDA is unresolved (no exchange, no profile): no id


def test_rebuild_is_deterministic_and_keeps_ids(tmp_path: Path):
    _archive(tmp_path)
    first = VintageBuilder(tmp_path, CONFIG, AS_OF).build()
    hashes = {k: v["sha256"] for k, v in first.info["outputs"].items()}
    second = VintageBuilder(tmp_path, CONFIG, AS_OF).build(switch=False)
    assert {k: v["sha256"] for k, v in second.info["outputs"].items()} == hashes  # V-REPRO
    assert not (tmp_path / "vintages" / ".v1.old").exists() and not (tmp_path / "vintages" / ".v1.building").exists()


def test_failed_validation_keeps_the_previous_vintage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _archive(tmp_path)
    VintageBuilder(tmp_path, CONFIG, AS_OF).build()
    (tmp_path / "current.txt").write_text("older\n", encoding="utf-8")
    monkeypatch.setattr(builder_module, "validate_structure", lambda out, outputs: ["broken"])
    with pytest.raises(BuildFailed, match="broken") as info:
        VintageBuilder(tmp_path, CONFIG, AS_OF).build()
    assert info.value.kept == tmp_path / "vintages" / ".v1.rejected" and info.value.kept.is_dir()
    assert read_current(tmp_path) == "older" and (tmp_path / "vintages" / "v1" / "_meta" / "build_info.json").exists()
    with pytest.raises(BuildFailed):  # an earlier rejected build is replaced
        VintageBuilder(tmp_path, CONFIG, AS_OF).build()


def test_validation_catches_structural_errors(tmp_path: Path):
    from src.data.fmp_pipeline.builders.validate import validate_structure

    _archive(tmp_path)
    result = VintageBuilder(tmp_path, CONFIG, AS_OF).build()
    vintage = result.vintage_dir
    assert validate_structure(vintage, result.outputs) == []
    assert validate_structure(vintage, {}) == ["reference/securities.parquet missing"]
    sid = int(Path(next(p for p in result.outputs if p.startswith("prices/1d_raw/"))).stem)
    rel = bucket_path("1d_raw", sid)
    errors = validate_structure(vintage, {**result.outputs, rel: 1, "nope.parquet": 3})
    assert any("rows, recorded 1" in e for e in errors) and any("nope.parquet: unreadable" in e for e in errors)
    table = pq.read_table(vintage / rel)
    swapped = table.take(list(range(table.num_rows - 1, -1, -1)))
    pq.write_table(swapped.set_column(0, "security_id", pa.array([777] * table.num_rows, pa.int64())), vintage / rel)
    stray = bucket_path("1d_raw", 999)
    (vintage / stray).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.slice(0, 1).drop_columns(["source"]), vintage / stray)
    errors = validate_structure(vintage, {**result.outputs, stray: 1})
    assert any("not strictly increasing" in e for e in errors)
    assert any("999.parquet: unexpected schema" in e for e in errors)
    assert any("rows of another security" in e for e in errors) and any("not in securities" in e for e in errors)


def test_missing_archive_and_helpers(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        VintageBuilder(tmp_path, CONFIG, AS_OF).build()
    assert read_current(tmp_path) is None
    (tmp_path / "current.txt").write_text("\n", encoding="utf-8")
    assert read_current(tmp_path) is None
    archive = _archive(tmp_path)
    assert "historical-price-eod/non-split-adjusted" in archived_endpoints(archive.dump_dir)


def test_plan_run_build_against_the_mock(tmp_path: Path):
    """The whole Phase-1 chain on the mock server: plan, run blocks 0-3, build."""
    import asyncio

    config = PlanConfig(dump_id="m", priorities=("P0",), daily_from=date(2025, 1, 1), workers=4,
                        universe=UniverseConfig(tier_a=("SPY",), indices=(), include_exchanges=("NASDAQ", "SYNTH0")))
    profile = [{"symbol": "X", "exchange": "NASDAQ", "isEtf": False, "isFund": False, "isAdr": False,
                "isActivelyTrading": True, "ipoDate": "2010-01-04"}]
    mock = MockConfig(extra_fixtures={"profile": profile}, paged_totals={"delisted-companies": 2})

    async def chain() -> None:
        with MockFmpServer(mock) as fmp, Manifest(tmp_path / "m.sqlite", "m") as manifest:
            archive = RawArchive(tmp_path / "raw", "m")
            planner = Planner(manifest, archive, config, tmp_path / "reports", AS_OF)
            async with make_client(fmp, FakeSleep()) as client:
                await planner.fetch_sources(client)
                planner.plan()
                report = await DumpExecutor(manifest, archive, config).run(client, RunScope())
            assert report.stop_reason == "done"

    asyncio.run(chain())
    result = VintageBuilder(tmp_path, config, AS_OF).build()
    assert result.daily.securities >= 1 and result.daily.bars > 200 and result.daily.out_of_window == 0
    spy = [s for s in pq.read_table(result.vintage_dir / "reference/securities.parquet").to_pylist()
           if s["symbol"] == "SPY"]
    assert spy and spy[0]["first_price_date"] == date(2025, 1, 1)


def test_cli_build(tmp_path: Path):
    args = ["--data-root", str(tmp_path), "--dump-id", "v1"]
    assert cli.main(["build", *args]) == cli.EXIT_USAGE  # no manifest
    with Manifest(tmp_path / "manifests" / "v1.sqlite", "v1"):
        pass
    assert cli.main(["build", *args]) == cli.EXIT_USAGE  # never planned
    with Manifest(tmp_path / "manifests" / "v1.sqlite", "v1") as manifest:
        manifest.meta_set("as_of", AS_OF.isoformat())
    assert cli.main(["build", *args]) == cli.EXIT_USAGE  # no archive
    _archive(tmp_path)
    assert cli.main(["build", *args, "--no-switch"]) == cli.EXIT_OK and read_current(tmp_path) is None
    with exclusive_lock(tmp_path / "locks" / "v1.lock"):
        assert cli.main(["build", *args]) == cli.EXIT_USAGE
    assert cli.main(["build", *args]) == cli.EXIT_OK and read_current(tmp_path) == "v1"


def test_cli_build_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    with Manifest(tmp_path / "manifests" / "v1.sqlite", "v1") as manifest:
        manifest.meta_set("as_of", AS_OF.isoformat())
    _archive(tmp_path)
    monkeypatch.setattr(builder_module, "validate_structure", lambda out, outputs: ["broken"])
    assert cli.main(["build", "--data-root", str(tmp_path), "--dump-id", "v1"]) == cli.EXIT_STOPPED
