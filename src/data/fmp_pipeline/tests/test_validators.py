"""Tests for the validation suite and ``fmp verify --final`` (spec §11, §14) on vintages built by the real builder."""

from __future__ import annotations

import asyncio
import gzip
import json
import shutil
import stat
import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.fmp_pipeline import cli
from src.data.fmp_pipeline.archive.archive import RawArchive
from src.data.fmp_pipeline.builders.builder import VintageBuilder
from src.data.fmp_pipeline.builders.prices import bucket_path
from src.data.fmp_pipeline.datamanager_adapter.adapter import FmpOhlcvAdapter
from src.data.fmp_pipeline.datamanager_adapter.fmp_store import FmpStore
from src.data.fmp_pipeline.executor import DumpExecutor, RunScope
from src.data.fmp_pipeline.manifest.manifest import Manifest, TaskSpec
from src.data.fmp_pipeline.planner.config import PlanConfig, UniverseConfig
from src.data.fmp_pipeline.planner.planner import Planner
from src.data.fmp_pipeline.tests.helpers import FakeSleep, make_client
from src.data.fmp_pipeline.tests.mock_server import MockConfig, MockFmpServer
from src.data.fmp_pipeline.tests.vintage_fixtures import make_vintage
from src.data.fmp_pipeline.validators import checks_events, checks_pipeline, checks_reference, verify
from src.data.fmp_pipeline.validators.checksums import gzip_test, list_files, verify_checksums, write_checksums
from src.data.fmp_pipeline.validators.framework import (
    CHECKS,
    FAIL,
    INFO,
    PASS,
    WARN,
    CheckResult,
    ValidationContext,
    run_checks,
)

AS_OF = date(2026, 10, 14)
MOCK_CONFIG = PlanConfig(dump_id="m", priorities=("P0", "P1"), daily_from=date(2024, 1, 1), workers=4,
                         universe=UniverseConfig(tier_a=("SPY",), indices=(), include_exchanges=("NASDAQ", "SYNTH0")))
VINTAGE_CONFIG = PlanConfig(dump_id="v")


def _mock_chain(root: Path) -> None:
    profile = [{"symbol": "X", "exchange": "NASDAQ", "isEtf": False, "isFund": False, "isAdr": False,
                "isActivelyTrading": True, "ipoDate": "2010-01-04"}]
    mock = MockConfig(extra_fixtures={"profile": profile}, paged_totals={"delisted-companies": 2})

    async def chain() -> None:
        with MockFmpServer(mock) as fmp, Manifest(root / "manifests" / "m.sqlite", "m") as manifest:
            archive = RawArchive(root / "raw", "m")
            planner = Planner(manifest, archive, MOCK_CONFIG, root / "reports" / "m", AS_OF)
            async with make_client(fmp, FakeSleep()) as client:
                await planner.fetch_sources(client)
                for _ in range(2):  # plan -> run -> re-plan (resolved universe) -> run
                    planner.plan()
                    await DumpExecutor(manifest, archive, MOCK_CONFIG).run(client, RunScope())

    asyncio.run(chain())
    VintageBuilder(root, MOCK_CONFIG, AS_OF).build()


@pytest.fixture(scope="module")
def mock_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("mock-chain")
    _mock_chain(root)
    return root


@pytest.fixture(scope="module")
def vintage_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("vintage")
    make_vintage(root)
    return root


@pytest.fixture
def mock_root(mock_template: Path, tmp_path: Path) -> Path:
    shutil.copytree(mock_template, tmp_path / "root")
    return tmp_path / "root"


@pytest.fixture
def vintage_root(vintage_template: Path, tmp_path: Path) -> Path:
    shutil.copytree(vintage_template, tmp_path / "root")
    return tmp_path / "root"


def _run(root: Path, config: PlanConfig, *ids: str, **kwargs: Any) -> dict[str, CheckResult]:
    report = run_checks(ValidationContext(root, config, **kwargs), list(ids))
    return {r.id: r for r in report.results}


def _rewrite(path: Path, change: Any) -> None:
    table = pq.read_table(path)
    pq.write_table(change(table.to_pandas()).pipe(lambda df: pa.Table.from_pandas(df, schema=table.schema,
                                                                                  preserve_index=False)), path)


# --- the whole suite on a mock-built dump ---------------------------------------------------------------------


def test_full_suite_on_a_mock_dump(mock_root: Path):
    report = run_checks(ValidationContext(mock_root, MOCK_CONFIG, sample_size=10))
    status = {r.id: r.status for r in report.results}
    assert list(status) == list(checks_pipeline_order())
    for check_id in ("V-SCHEMA", "V-DUP", "V-ORDER", "V-OHLC", "V-CAL", "V-ADJ-RECON", "V-COMPLETE",
                     "V-RAW-ARCHIVE", "V-REPRO", "V-ADJ-UNIT", "V-FUND-PIT", "V-IDENT"):
        assert status[check_id] == PASS, (check_id, report.result(check_id))
    assert status["V-SPLIT-KNOWN"] == FAIL  # no real AAPL/TSLA/... in a mock dump
    assert status["V-DELIST"] == WARN and status["V-INDEX"] == WARN and status["V-YF-XCHECK"] == INFO
    assert status["V-INTRADAY-ALIGN"] == INFO == status["V-INTRADAY-DAILY"]  # no hourly data
    assert report.result("V-REPRO").examined > 0  # type: ignore[union-attr]
    md, js = report.write(mock_root / "reports" / "m")
    assert "| V-SCHEMA | PASS |" in md.read_text(encoding="utf-8")
    assert json.loads(js.read_text(encoding="utf-8"))["summary"]["FAIL"] == 1


def checks_pipeline_order() -> tuple[str, ...]:
    from src.data.fmp_pipeline.validators.checks import ORDER

    return ORDER


def test_completeness_and_archive_integrity_failures(mock_root: Path):
    with Manifest(mock_root / "manifests" / "m.sqlite", "m") as manifest:
        manifest.add_tasks([TaskSpec("splits", "ZZZ", {}, 101)])
    first = next(p for p in sorted((mock_root / "raw" / "m").rglob("*.jsonl.gz")))
    with first.open("ab") as fh:
        fh.write(gzip.compress(b"x")[:10])  # torn tail
    results = _run(mock_root, MOCK_CONFIG, "V-COMPLETE", "V-RAW-ARCHIVE")
    assert results["V-COMPLETE"].status == FAIL and "pending: splits ZZZ" in results["V-COMPLETE"].examples[0]
    assert results["V-RAW-ARCHIVE"].status == FAIL and "unreadable" in results["V-RAW-ARCHIVE"].examples[0]
    (mock_root / "manifests" / "m.sqlite").unlink()
    assert _run(mock_root, MOCK_CONFIG, "V-COMPLETE")["V-COMPLETE"].message.startswith("no manifest")


def test_repro_detects_a_changed_file(mock_root: Path):
    info_path = FmpStore(mock_root).dir / "_meta" / "build_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    for rel in info["outputs"]:
        if rel.startswith("prices/1d_raw/"):
            info["outputs"][rel]["sha256"] = "0" * 64
    info_path.write_text(json.dumps(info), encoding="utf-8")
    result = _run(mock_root, MOCK_CONFIG, "V-REPRO")["V-REPRO"]
    assert result.status == FAIL and result.violations == result.examined > 0


def test_recon_detects_a_systematic_offset(mock_root: Path):
    _rewrite(FmpStore(mock_root).dir / checks_events.RECON_TABLE, lambda df: df.assign(adjClose=df["adjClose"] * 1.05))
    result = _run(mock_root, MOCK_CONFIG, "V-ADJ-RECON")["V-ADJ-RECON"]
    assert result.status == FAIL and result.details["per_symbol"]
    (FmpStore(mock_root).dir / checks_events.RECON_TABLE).unlink()
    assert _run(mock_root, MOCK_CONFIG, "V-ADJ-RECON")["V-ADJ-RECON"].status == WARN


def test_fund_pit_flags_filings_before_period_end(mock_root: Path):
    path = FmpStore(mock_root).dir / "fundamentals/income_statement.parquet"
    _rewrite(path, lambda df: df.assign(filing_date=date(2000, 1, 3), accepted_date=None))
    result = _run(mock_root, MOCK_CONFIG, "V-FUND-PIT")["V-FUND-PIT"]
    assert result.status == WARN and "filed 2000-01-03" in result.examples[0]


# --- price checks on the synthetic vintage ----------------------------------------------------------------------


def test_price_corruption_is_caught(vintage_root: Path):
    store = FmpStore(vintage_root)
    sid = store.resolve("BRK.B").security_id  # type: ignore[union-attr]
    path = store.dir / bucket_path("1d_raw", sid)

    def corrupt(df: pd.DataFrame) -> pd.DataFrame:
        df = pd.concat([df, df.iloc[[3]]]).reset_index(drop=True)  # a duplicate, out of order
        df.loc[0:9, "low"] = df.loc[0:9, "high"] * 2  # 10 of 43 bars break OHLC
        return df

    _rewrite(path, corrupt)
    results = _run(vintage_root, VINTAGE_CONFIG, "V-SCHEMA", "V-DUP", "V-ORDER", "V-OHLC")
    assert results["V-SCHEMA"].status == FAIL and any("rows, recorded" in e for e in results["V-SCHEMA"].examples)
    assert results["V-DUP"].violations == 1 and results["V-ORDER"].status == FAIL
    assert results["V-OHLC"].status == FAIL and "BRK.B (1d)" in results["V-OHLC"].examples[0]


def test_calendar_gaps_warn_and_fail_for_tier_a(vintage_root: Path):
    store = FmpStore(vintage_root)
    sid = store.resolve("AAA").security_id  # type: ignore[union-attr]
    _rewrite(store.dir / bucket_path("1d_raw", sid), lambda df: df.iloc[::2])  # half the sessions gone
    assert _run(vintage_root, VINTAGE_CONFIG, "V-CAL")["V-CAL"].status == WARN
    _rewrite(store.dir / "reference/securities.parquet",
             lambda df: df.assign(tier=df["tier"].where(df["security_id"] != sid, "A")))
    result = _run(vintage_root, VINTAGE_CONFIG, "V-CAL")["V-CAL"]
    assert result.status == FAIL and "AAA (tier A)" in result.examples[0]


def test_intraday_checks(vintage_root: Path):
    results = _run(vintage_root, VINTAGE_CONFIG, "V-INTRADAY-ALIGN", "V-INTRADAY-DAILY", "V-FLAGS")
    assert results["V-INTRADAY-ALIGN"].status == PASS and results["V-INTRADAY-ALIGN"].examined == 2
    daily = results["V-INTRADAY-DAILY"]
    assert daily.status == WARN and daily.examined == 2  # synthetic hourly prices differ from the daily ones
    store = FmpStore(vintage_root)
    sid = store.resolve("AAA").security_id  # type: ignore[union-attr]
    _rewrite(store.dir / bucket_path("1h_raw", sid), lambda df: df.iloc[1:])  # first bar of a day missing
    result = _run(vintage_root, VINTAGE_CONFIG, "V-INTRADAY-ALIGN")["V-INTRADAY-ALIGN"]
    assert result.status == WARN and "1 days not starting 09:30 ET" in result.examples[0]


# --- event checks -----------------------------------------------------------------------------------------------


def test_split_known_consistency_and_dividend_sanity(vintage_root: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(checks_events, "KNOWN_SPLITS", (("AAA", date(2025, 2, 3), 4.0),))
    monkeypatch.setattr(checks_events, "KNOWN_DIVIDENDS", (("AAA", date(2025, 1, 21), 2.0),))
    results = _run(vintage_root, VINTAGE_CONFIG, "V-SPLIT-KNOWN", "V-SPLIT-CONSISTENCY", "V-DIV-SANITY")
    assert results["V-SPLIT-KNOWN"].status == PASS
    assert results["V-SPLIT-CONSISTENCY"].status == PASS and results["V-SPLIT-CONSISTENCY"].examined == 1
    assert results["V-DIV-SANITY"].status == PASS and results["V-DIV-SANITY"].examined == 1
    monkeypatch.setattr(checks_events, "KNOWN_SPLITS", (("AAA", date(2025, 2, 3), 2.0),))
    assert "wrong ratio" in _run(vintage_root, VINTAGE_CONFIG, "V-SPLIT-KNOWN")["V-SPLIT-KNOWN"].examples[0]
    store = FmpStore(vintage_root)
    _rewrite(store.dir / "events/dividends.parquet", lambda df: df.assign(amount_raw=300.0))
    _rewrite(store.dir / "events/splits.parquet", lambda df: df.assign(ratio=10.0))
    results = _run(vintage_root, VINTAGE_CONFIG, "V-DIV-SANITY", "V-SPLIT-CONSISTENCY")
    assert results["V-DIV-SANITY"].status == WARN and "of the previous close" in results["V-DIV-SANITY"].examples[0]
    assert results["V-SPLIT-CONSISTENCY"].status == WARN


def test_events_of_a_security_without_daily_bars_are_skipped(vintage_root: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(FmpStore, "prices", lambda *_: None)
    results = _run(vintage_root, VINTAGE_CONFIG, "V-SPLIT-CONSISTENCY", "V-DIV-SANITY")
    assert results["V-SPLIT-CONSISTENCY"].status == PASS and results["V-SPLIT-CONSISTENCY"].examined == 0
    assert results["V-DIV-SANITY"].status == PASS and results["V-DIV-SANITY"].examined == 1


def test_yf_xcheck_against_a_legacy_cache(vintage_root: Path, tmp_path: Path):
    store = FmpStore(vintage_root)
    ours = FmpOhlcvAdapter(store).get_ohlcv("AAA", "1d", datetime(2025, 1, 1), datetime(2025, 3, 1), "total")
    assert ours is not None
    folder = tmp_path / "cache" / "ohlcv" / "AAA" / "1d"
    folder.mkdir(parents=True)
    legacy = ours.reset_index().rename(columns={"timestamp": "timestamp"})
    legacy.to_csv(folder / "2025.csv.gz", index=False, compression="gzip")
    result = _run(vintage_root, VINTAGE_CONFIG, "V-YF-XCHECK", legacy_cache_dir=tmp_path / "cache")["V-YF-XCHECK"]
    assert result.status == PASS and result.examined == 1
    legacy.assign(close=legacy["close"] * 1.02).to_csv(folder / "2025.csv.gz", index=False, compression="gzip")
    result = _run(vintage_root, VINTAGE_CONFIG, "V-YF-XCHECK", legacy_cache_dir=tmp_path / "cache")["V-YF-XCHECK"]
    assert result.status == WARN and "AAA" in result.examples[0]


# --- reference checks -------------------------------------------------------------------------------------------


def test_delist_and_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    make_vintage(tmp_path, segment=True)
    monkeypatch.setattr(checks_reference, "KNOWN_DELISTED", (("XYZ", 2024, False), ("GONE", 2020, True)))
    results = _run(tmp_path, VINTAGE_CONFIG, "V-DELIST", "V-IDENT", "V-UNIVERSE")
    assert results["V-DELIST"].examples == ["GONE: not in securities (soft check)"]
    assert results["V-IDENT"].status == WARN and results["V-IDENT"].details["name_mismatch"] == 1
    assert "delisted as 'Old Widgets', profile 'New Robotics'" in results["V-IDENT"].examples[0]
    assert results["V-UNIVERSE"].details["by_tier"]


def test_index_rollback_and_coverage(vintage_root: Path):
    changes = pd.DataFrame({"date": [date(2020, 6, 1), date(2010, 3, 1)], "symbol": ["NEW", "MID"],
                            "_extra": [json.dumps({"removedTicker": "OLD"}), json.dumps({"removedTicker": "OLDER"})]})
    members = checks_reference._members_as_of({"NEW", "MID", "KEEP"}, changes, date(2016, 1, 4))
    assert members == {"OLD", "MID", "KEEP"}
    assert checks_reference._members_as_of({"NEW", "MID", "KEEP"}, changes, date(2008, 1, 2)) == {"OLD", "OLDER",
                                                                                                  "KEEP"}
    store = FmpStore(vintage_root)
    for rel, frame in (("index_membership/index_changes.parquet",
                        pd.DataFrame({"security_id": [None], "index": ["sp500"], "date": [date(2020, 1, 2)],
                                      "symbol": ["AAA"], "_extra": [None]})),
                       ("index_membership/index_members_current.parquet",
                        pd.DataFrame({"security_id": [None], "index": ["sp500"], "symbol": ["AAA"],
                                      "date_first_added": [None], "_extra": [None]}))):
        (store.dir / rel).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), store.dir / rel)
    result = _run(vintage_root, VINTAGE_CONFIG, "V-INDEX")["V-INDEX"]
    assert result.status == FAIL and result.details["sizes"]["2024-01-02"] == 1


# --- framework, checksums, verify ------------------------------------------------------------------------------


def test_framework_guards(vintage_root: Path, monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(ValueError, match="unknown check"):
        run_checks(ValidationContext(vintage_root, VINTAGE_CONFIG), ["V-NOPE"])

    def boom(ctx: ValidationContext) -> CheckResult:
        raise RuntimeError("bad check")

    monkeypatch.setitem(CHECKS, "V-DUP", CHECKS["V-DUP"].__class__("V-DUP", FAIL, "x", boom))
    result = _run(vintage_root, VINTAGE_CONFIG, "V-DUP")["V-DUP"]
    assert result.status == FAIL and "bad check" in result.message
    many = CheckResult("X", WARN)
    for i in range(30):
        many.add_example(str(i))
    assert len(many.examples) == 20
    monkeypatch.setattr(checks_pipeline, "UNIT_TESTS", ("src/data/fmp_pipeline/tests/nope.py",))
    assert _run(vintage_root, VINTAGE_CONFIG, "V-ADJ-UNIT")["V-ADJ-UNIT"].status == FAIL
    monkeypatch.setattr(checks_pipeline, "UNIT_TESTS", ())
    assert _run(vintage_root, VINTAGE_CONFIG, "V-ADJ-UNIT")["V-ADJ-UNIT"].message == "unit test modules missing"
    monkeypatch.setattr(checks_pipeline, "UNIT_TESTS", ("src/data/fmp_pipeline/tests/test_adjust.py",))
    monkeypatch.setattr(checks_pipeline.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout="1 failed\n", stderr=""))
    result = _run(vintage_root, VINTAGE_CONFIG, "V-ADJ-UNIT")["V-ADJ-UNIT"]
    assert result.status == FAIL and result.examples == ["1 failed"]


def test_checksums_round_trip(tmp_path: Path):
    tree = tmp_path / "tree"
    (tree / "a").mkdir(parents=True)
    (tree / "a" / "x.bin").write_bytes(b"x")
    (tree / "y.bin").write_bytes(b"yy")
    summary = write_checksums(tree, tmp_path / "SUMS")
    assert (summary.files, summary.bytes) == (2, 3) and list_files(tree) == ["a/x.bin", "y.bin"]
    assert (tmp_path / "SUMS").read_text(encoding="utf-8").splitlines()[0].endswith("  a/x.bin")
    assert verify_checksums(tree, tmp_path / "SUMS", summary) == []
    (tree / "y.bin").write_bytes(b"zz")
    (tree / "new.bin").write_bytes(b"")
    (tree / "a" / "x.bin").unlink()
    errors = verify_checksums(tree, tmp_path / "SUMS", summary)
    assert any("missing: a/x.bin" in e for e in errors) and any("mismatch: y.bin" in e for e in errors)
    assert any("not in SUMS: new.bin" in e for e in errors) and any("expected 2 / 3" in e for e in errors)
    assert verify_checksums(tree, tmp_path / "nope") and verify_checksums(tmp_path / "nope", tmp_path / "SUMS")
    good = tmp_path / "g.jsonl.gz"
    good.write_bytes(gzip.compress(b"{}\n") * 2)
    assert gzip_test(tmp_path)[:2] == (1, 2)


def _writable(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode | stat.S_IWUSR)


def test_verify_final_needs_a_backup_then_is_ready(mock_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(checks_events, "KNOWN_SPLITS", ())
    monkeypatch.setattr(checks_events, "KNOWN_DIVIDENDS", ())
    items, _ = verify.FinalVerifier(mock_root, MOCK_CONFIG, run_tests=False).run()
    by_number = {i.number: i for i in items}
    assert [n for n, i in by_number.items() if not i.ok] == [3, 7]
    assert "no --backup" in by_number[3].detail and "--confirm-postponed" in by_number[7].detail
    assert (mock_root / "checksums" / "m" / "ARCHIVE.sha256").exists()
    assert (mock_root / "reports" / "m" / "coverage_report.md").exists()

    backup = tmp_path / "backup"
    for part in ("raw/m", "vintages/m", "checksums/m"):
        shutil.copytree(mock_root / part, backup / part)
    items, _ = verify.FinalVerifier(mock_root, MOCK_CONFIG, backup, True, run_tests=False).run()
    try:
        assert all(i.ok for i in items), [i.line() for i in items if not i.ok]
        archive_file = next((mock_root / "raw" / "m").rglob("*.jsonl.gz"))
        assert not archive_file.stat().st_mode & stat.S_IWUSR  # read-only after a verified backup
    finally:
        _writable(mock_root)

    (backup / "raw" / "m" / "corrupt.jsonl.gz").write_bytes(b"x")
    items, _ = verify.FinalVerifier(mock_root, MOCK_CONFIG, backup, True, run_tests=False).run()
    _writable(mock_root)
    assert "not in ARCHIVE.sha256: corrupt.jsonl.gz" in next(i for i in items if i.number == 3).detail


def test_verify_adapter_item_runs_the_suites(mock_root: Path, monkeypatch: pytest.MonkeyPatch):
    calls = []

    def fake_pytest(tests: tuple[str, ...]) -> tuple[bool, str]:
        calls.append(tests)
        return False, "1 failed"

    monkeypatch.setattr(verify, "_run_pytest", fake_pytest)
    item = verify.FinalVerifier(mock_root, MOCK_CONFIG)._adapter()
    assert not item.ok and item.detail == "1 failed" and calls == [verify.ADAPTER_TESTS]


def test_cli_validate_and_verify(mock_root: Path, monkeypatch: pytest.MonkeyPatch):
    args = ["--data-root", str(mock_root), "--sample", "5"]
    assert cli.main(["validate", *args, "--checks", "V-DUP", "V-ORDER"]) == cli.EXIT_OK
    assert cli.main(["validate", *args, "--checks", "V-SPLIT-KNOWN"]) == cli.EXIT_STOPPED
    assert cli.main(["validate", *args, "--checks", "V-NOPE"]) == cli.EXIT_USAGE
    assert cli.main(["validate", "--data-root", str(mock_root / "nothing")]) == cli.EXIT_USAGE
    assert cli.main(["verify", *args]) == cli.EXIT_USAGE  # --final is required
    monkeypatch.setattr(verify, "_run_pytest", lambda tests: (True, "ok"))
    monkeypatch.setattr(checks_pipeline.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="2 passed\n", stderr=""))
    assert cli.main(["verify", "--final", *args]) == cli.EXIT_STOPPED  # no backup, nothing confirmed
    assert (mock_root / "reports" / "m" / "validation_report.md").exists()
