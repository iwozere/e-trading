"""Tests for the drift report and increment archiving when a new vintage arrives (spec §10.3)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from src.data.fmp_pipeline import cli
from src.data.fmp_pipeline.archive.archive import RawArchive
from src.data.fmp_pipeline.builders.builder import VintageBuilder, read_current
from src.data.fmp_pipeline.builders.prices import DAILY_ENDPOINT
from src.data.fmp_pipeline.datamanager_adapter.fmp_store import FmpStore
from src.data.fmp_pipeline.increments.drift import archive_increments, drift_report, has_earlier_increments
from src.data.fmp_pipeline.increments.store import IncrementStore
from src.data.fmp_pipeline.manifest.manifest import Manifest
from src.data.fmp_pipeline.planner.config import PlanConfig, UniverseConfig
from src.data.fmp_pipeline.tests.archive_fixtures import daily_rows, put
from src.data.fmp_pipeline.tests.test_increments import AAA_RAW, CONFIG, SPLIT, FakeYahoo, _ingester, yf_daily
from src.data.fmp_pipeline.tests.vintage_fixtures import make_vintage

AS_OF = date(2026, 10, 14)
V2 = PlanConfig(dump_id="v2", universe=UniverseConfig(tier_a=(), indices=()))


def _new_vintage_archive(root: Path) -> None:
    """Dump v2: AAA through 2025-03-07, its 2025-03-06 close 1% off the yfinance value, the split but no dividend."""
    archive = RawArchive(root / "raw", "v2")
    put(archive, "stock-list", None, [{"symbol": "AAA", "exchange": "NYSE"}])
    put(archive, "actively-trading-list", None, [{"symbol": "AAA"}])
    closes = {d: v for d, v in AAA_RAW.items() if d >= "2025-02-24"}
    closes["2025-03-06"] = 77.0 * 1.01
    put(archive, DAILY_ENDPOINT, "AAA", daily_rows("AAA", closes), {"from": "2025-01-01", "to": "2025-12-31"})
    put(archive, "splits", "AAA", [{"symbol": "AAA", "date": SPLIT, "numerator": 2, "denominator": 1}])


@pytest.fixture
def root(tmp_path: Path) -> Path:
    make_vintage(tmp_path)
    yahoo = FakeYahoo({"AAA": yf_daily(AAA_RAW, (SPLIT, 2.0), {"2025-03-04": 0.5})})
    _ingester(tmp_path, yahoo).run(timeframes=["1d"])  # increments on vintage v: 2025-03-03 … 03-07
    _new_vintage_archive(tmp_path)
    return tmp_path


def test_drift_report_and_archiving(root: Path):
    assert not has_earlier_increments(root, "v")
    VintageBuilder(root, V2, AS_OF).build()
    assert read_current(root) == "v2" and has_earlier_increments(root, "v2")
    report = drift_report(root)
    [aaa] = [s for s in report.securities if s.symbol == "AAA"]
    assert aaa.old_vintage == "v" and aaa.overlap_bars == 5
    assert aaa.max_rel_diff == pytest.approx(0.01 / 1.01, rel=1e-3) and aaa.median_rel_diff == 0.0
    assert aaa.missing_in_vintage == 0 and aaa.missing_in_increments == 0
    assert aaa.event_mismatches == ["dividend 2025-03-04 only in increments"]
    md, js = report.write(root / "reports" / "v2")
    assert "| AAA | v | 5 |" in md.read_text(encoding="utf-8")
    assert json.loads(js.read_text(encoding="utf-8"))["new_vintage"] == "v2"

    store = IncrementStore(root)
    sid = FmpStore(root).resolve("AAA").security_id  # type: ignore[union-attr]
    seq = store.meta()["increment_seq"]
    result = archive_increments(root)
    assert (result.bars, result.events, result.files) == (5, 2, 1)
    assert not store.path("1d", sid).exists() and not store.path("events", sid).exists()
    archived = store.root / "archive" / "v" / store.path("1d", sid).relative_to(store.root)
    assert archived.exists() and store.meta()["increment_seq"] == seq + 1
    assert (store.root / "archive" / "before-v2" / store.path("events", sid).relative_to(store.root)).exists()
    assert not has_earlier_increments(root, "v2")
    assert archive_increments(root).bars == 0  # idempotent


def test_partial_archive_keeps_the_newer_bars(root: Path):
    # A newer response for the same request replaces v2's AAA history: it now ends on 2025-03-04.
    put(RawArchive(root / "raw", "v2"), DAILY_ENDPOINT, "AAA",
        daily_rows("AAA", {"2025-03-03": 150.0, "2025-03-04": 152.0}), {"from": "2025-01-01", "to": "2025-12-31"},
        fetched_at="2026-10-15T10:00:00+00:00")
    VintageBuilder(root, V2, AS_OF).build()
    store = IncrementStore(root)
    sid = FmpStore(root).resolve("AAA").security_id  # type: ignore[union-attr]
    result = archive_increments(root)
    left = store.bars(sid, "AAA", "1d", None)
    assert result.bars == 2 and left is not None and [str(d.date()) for d in left.index] == [
        "2025-03-05", "2025-03-06", "2025-03-07"]


def test_cli_build_writes_the_drift_report_and_archive_command(
        root: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    with Manifest(root / "manifests" / "v2.sqlite", "v2") as manifest:
        manifest.meta_set("as_of", AS_OF.isoformat())
    config = tmp_path_factory.mktemp("cfg") / "fmp.yaml"
    config.write_text("fmp:\n  dump_id: v2\n  universe: {tier_a: [], indices: []}\n"
                      "  increments: {writer_host: elsewhere}\n", encoding="utf-8")
    assert cli.main(["build", "--config", str(config), "--data-root", str(root)]) == cli.EXIT_OK
    assert (root / "reports" / "v2" / "drift_report.md").exists()
    args = ["archive-increments", "--config", str(config), "--data-root", str(root)]
    assert cli.main(args) == cli.EXIT_USAGE  # not the writer
    from src.data.fmp_pipeline.increments import ingest

    monkeypatch.setattr(ingest.socket, "gethostname", lambda: "elsewhere")
    assert cli.main(args) == cli.EXIT_OK
    assert CONFIG.increments.writer_host  # the shared ingest config stays untouched
