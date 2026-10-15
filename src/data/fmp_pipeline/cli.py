"""
``fmp`` command-line entry point (spec §6.1): ``probe-free``, ``plan``, ``run``, ``status``, ``retry-failed``,
``build``, ``golden``, ``ingest-increments``, ``validate``, ``verify``.

Usage (PowerShell, from the project root)::

    .venv\\Scripts\\python.exe -m src.data.fmp_pipeline.cli probe-free
    .venv\\Scripts\\python.exe -m src.data.fmp_pipeline.cli probe-free --offline   # rebuild report from cache
    .venv\\Scripts\\python.exe -m src.data.fmp_pipeline.cli plan --dry-run   # budget_plan.md, no new tasks
    .venv\\Scripts\\python.exe -m src.data.fmp_pipeline.cli plan --offline   # re-plan from the archive only
    .venv\\Scripts\\python.exe -m src.data.fmp_pipeline.cli run [--phase N] [--tier T] [--max-gb G]
    .venv\\Scripts\\python.exe -m src.data.fmp_pipeline.cli status
    .venv\\Scripts\\python.exe -m src.data.fmp_pipeline.cli retry-failed
    .venv\\Scripts\\python.exe -m src.data.fmp_pipeline.cli build [--no-switch]
    .venv\\Scripts\\python.exe -m src.data.fmp_pipeline.cli golden capture|compare
    .venv\\Scripts\\python.exe -m src.data.fmp_pipeline.cli ingest-increments [--timeframes 1d 1h] [--symbols ...]
    .venv\\Scripts\\python.exe -m src.data.fmp_pipeline.cli validate [--sample N] [--checks V-OHLC ...]
    .venv\\Scripts\\python.exe -m src.data.fmp_pipeline.cli verify --final --backup E:\\fmp-backup [--confirm-postponed]

Exit codes: 0 done, 1 stopped (quota, auth, bandwidth, archive) or build rejected by structural validation,
2 usage, lock or not planned, 3 not enough disk space.

Layout under ``data_root`` (``DATA_CACHE_DIR/fmp-data``): ``raw/<dump_id>/`` archive, ``manifests/<dump_id>.sqlite``,
``reports/<dump_id>/`` plan and build reports, ``locks/<dump_id>.lock`` (one writer per dump), ``vintages/<dump_id>/``
Parquet vintages, ``current.txt`` (active vintage) and ``security_ids.json`` (stable id registry).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fmp_pipeline.archive.archive import RawArchive  # noqa: E402
from src.data.fmp_pipeline.builders.builder import BuildFailed, VintageBuilder  # noqa: E402
from src.data.fmp_pipeline.client.fmp_client import FmpClient  # noqa: E402
from src.data.fmp_pipeline.client.rate_limiter import AsyncRateLimiter  # noqa: E402
from src.data.fmp_pipeline.errors import DumpLocked, InsufficientDiskSpace  # noqa: E402
from src.data.fmp_pipeline.executor import DumpExecutor, RunReport, RunScope  # noqa: E402
from src.data.fmp_pipeline.fsutil import exclusive_lock  # noqa: E402
from src.data.fmp_pipeline.manifest.manifest import Manifest  # noqa: E402
from src.data.fmp_pipeline.planner.config import PlanConfig, load_config  # noqa: E402
from src.data.fmp_pipeline.planner.planner import Planner  # noqa: E402
from src.data.fmp_pipeline.planner.projection import GB  # noqa: E402
from src.data.fmp_pipeline.power import keep_awake  # noqa: E402
from src.data.fmp_pipeline.probe.free_probe import FreeProbe  # noqa: E402
from src.data.fmp_pipeline.probe.probe_client import ProbeClient  # noqa: E402
from src.data.fmp_pipeline.probe.report import render_report, write_fixtures, write_report  # noqa: E402
from src.data.fmp_pipeline.status import render_status  # noqa: E402
from src.notification.logger import setup_logger  # noqa: E402

_logger = setup_logger(__name__)

DEFAULT_REPORT = PROJECT_ROOT / "docs" / "fmp_probe_report.md"
DEFAULT_FIXTURES = PROJECT_ROOT / "src" / "data" / "fmp_pipeline" / "tests" / "fixtures"


def data_cache_dir() -> Path:
    """Return ``DATA_CACHE_DIR`` (the legacy cache root)."""
    from config.donotshare.donotshare import DATA_CACHE_DIR

    return Path(DATA_CACHE_DIR)


def default_data_root() -> Path:
    """Return ``DATA_CACHE_DIR/fmp-data`` (owner decision Q2)."""
    return data_cache_dir() / "fmp-data"


def _probe_free(args: argparse.Namespace) -> int:
    from src.config.provider_config import get_api_key

    cache_dir = (args.data_root or default_data_root()) / "probe-free"
    api_key = None if args.offline else get_api_key("FMP_API_KEY")
    with ProbeClient(api_key, cache_dir, budget=args.budget, offline=args.offline) as client:
        run = FreeProbe(client).run()
        calls_used = client.calls_used
    write_report(args.report, render_report(run, calls_used, args.budget))
    fixtures = write_fixtures(run, args.fixtures)
    _logger.info("Probe done: %d live calls used of %d; report %s; %d fixtures in %s",
                 calls_used, args.budget, args.report, len(fixtures), args.fixtures)
    return 1 if run.stopped else 0


class DumpPaths:
    """Where one dump's files live under ``data_root``."""

    def __init__(self, data_root: Path, dump_id: str) -> None:
        """
        Args:
            data_root: Pipeline data root.
            dump_id: Dump id.
        """
        self.data_root = data_root
        self.raw_root = data_root / "raw"
        self.manifest = data_root / "manifests" / f"{dump_id}.sqlite"
        self.reports = data_root / "reports" / dump_id
        self.lock = data_root / "locks" / f"{dump_id}.lock"


#: Exit codes of the dump commands.
EXIT_OK, EXIT_STOPPED, EXIT_USAGE, EXIT_DISK = 0, 1, 2, 3


def _setup(args: argparse.Namespace) -> tuple[PlanConfig, DumpPaths]:
    config = load_config(args.config)
    if args.dump_id:
        config = replace(config, dump_id=args.dump_id)
    return config, DumpPaths(args.data_root or default_data_root(), config.dump_id)


def _api_key() -> Optional[str]:
    from src.config.provider_config import get_api_key

    return get_api_key("FMP_API_KEY")


def _client(api_key: str, config: PlanConfig) -> FmpClient:
    return FmpClient(api_key, base_url=config.base_url, rate_limiter=AsyncRateLimiter(config.rate_limit_per_min))


async def _fetch_sources(planner: Planner, api_key: str, config: PlanConfig) -> str:
    async with _client(api_key, config) as client:
        summary = await planner.fetch_sources(client)
    return summary.stop_reason


def _plan(args: argparse.Namespace) -> int:
    config, paths = _setup(args)
    try:
        with exclusive_lock(paths.lock), Manifest(paths.manifest, config.dump_id) as manifest:
            planner = Planner(manifest, RawArchive(paths.raw_root, config.dump_id), config,
                              args.report_dir or paths.reports, args.as_of)
            stop_reason = "done"
            if not args.offline:
                api_key = _api_key()
                if not api_key:
                    _logger.error("FMP_API_KEY is not set; use --offline to plan from the archive only")
                    return EXIT_USAGE
                stop_reason = asyncio.run(_fetch_sources(planner, api_key, config))
                if stop_reason != "done":
                    _logger.error("Source lists incomplete (run stopped: %s); planning with what was fetched",
                                  stop_reason)
            result = planner.plan(dry_run=args.dry_run)
    except DumpLocked as exc:
        _logger.error("%s", exc)
        return EXIT_USAGE
    wire, decoded = result.projection.remaining_bytes()
    free = result.context.free_bytes
    _logger.info("Remaining: %.2f GB wire / %.2f GB decoded; disk need %.1f GB, free %s. Reports: %s, %s",
                 wire / GB, decoded / GB, result.context.need_bytes / GB,
                 "unknown" if free is None else f"{free / GB:.1f} GB", result.budget_path, result.universe_path)
    return EXIT_OK if stop_reason == "done" else EXIT_STOPPED


async def _execute(executor: DumpExecutor, api_key: str, config: PlanConfig, scope: RunScope) -> RunReport:
    async with _client(api_key, config) as client:
        return await executor.run(client, scope)


def _run(args: argparse.Namespace) -> int:
    config, paths = _setup(args)
    try:
        scope = RunScope(args.phase, args.tier)
    except ValueError as exc:
        _logger.error("%s", exc)
        return EXIT_USAGE
    if not paths.manifest.exists():
        _logger.error("No manifest for dump %s at %s; run `fmp plan` first", config.dump_id, paths.manifest)
        return EXIT_USAGE
    api_key = _api_key()
    if not api_key:
        _logger.error("FMP_API_KEY is not set")
        return EXIT_USAGE
    try:
        with exclusive_lock(paths.lock), Manifest(paths.manifest, config.dump_id) as manifest, keep_awake():
            executor = DumpExecutor(manifest, RawArchive(paths.raw_root, config.dump_id), config,
                                    max_gb=args.max_gb, workers=args.workers)
            report = asyncio.run(_execute(executor, api_key, config, scope))
            _logger.info("Status after the run:\n%s", render_status(manifest, config))
    except DumpLocked as exc:
        _logger.error("%s", exc)
        return EXIT_USAGE
    except RuntimeError as exc:
        _logger.error("%s", exc)
        return EXIT_USAGE
    except InsufficientDiskSpace as exc:
        _logger.error("Not starting the next block: %s", exc)
        return EXIT_DISK
    if report.stop_reason == "done":
        _logger.info("Scope finished. Re-run `fmp plan` to write tasks that were waiting on this data (D36).")
        return EXIT_OK
    return EXIT_STOPPED


def _status(args: argparse.Namespace) -> int:
    config, paths = _setup(args)
    if not paths.manifest.exists():
        _logger.error("No manifest for dump %s at %s", config.dump_id, paths.manifest)
        return EXIT_USAGE
    with Manifest(paths.manifest, config.dump_id, recover=False) as manifest:
        _logger.info("\n%s", render_status(manifest, config))
    return EXIT_OK


def _retry_failed(args: argparse.Namespace) -> int:
    config, paths = _setup(args)
    if not paths.manifest.exists():
        _logger.error("No manifest for dump %s at %s", config.dump_id, paths.manifest)
        return EXIT_USAGE
    try:
        with exclusive_lock(paths.lock), Manifest(paths.manifest, config.dump_id) as manifest:
            count = manifest.requeue_failed()
            manifest.log_event("retry_failed", f"requeued={count}")
    except DumpLocked as exc:
        _logger.error("%s", exc)
        return EXIT_USAGE
    _logger.info("Re-queued %d failed tasks of dump %s; run `fmp run` to fetch them", count, config.dump_id)
    return EXIT_OK


def _build(args: argparse.Namespace) -> int:
    config, paths = _setup(args)
    if not paths.manifest.exists():
        _logger.error("No manifest for dump %s at %s; run `fmp plan` first", config.dump_id, paths.manifest)
        return EXIT_USAGE
    with Manifest(paths.manifest, config.dump_id, recover=False) as manifest:
        as_of = manifest.meta("as_of")
    if as_of is None:
        _logger.error("Dump %s was never planned (no as_of); run `fmp plan` first", config.dump_id)
        return EXIT_USAGE
    try:
        with exclusive_lock(paths.lock), keep_awake():
            result = VintageBuilder(paths.data_root, config, date.fromisoformat(as_of)).build(switch=not args.no_switch)
    except (DumpLocked, FileNotFoundError) as exc:
        _logger.error("%s", exc)
        return EXIT_USAGE
    except BuildFailed as exc:
        _logger.error("Build rejected, current vintage unchanged: %s", exc)
        return EXIT_STOPPED
    _logger.info("Vintage %s built at %s: %d securities (%d identity-uncertain), %d daily bars in %d files, "
                 "%d output files%s; unbuilt endpoints: %s", config.dump_id, result.vintage_dir, result.securities,
                 result.uncertain, result.daily.bars, result.daily.securities, len(result.outputs),
                 ", current.txt switched" if result.switched else "", ", ".join(result.unbuilt) or "none")
    return EXIT_OK


def _validation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help="Default: config/fmp/fmp.yaml")
    parser.add_argument("--data-root", type=Path, default=None, help="Default: DATA_CACHE_DIR/fmp-data")
    parser.add_argument("--sample", type=int, default=100, help="Random symbols for cross-source checks")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Legacy yfinance cache root (DATA_CACHE_DIR) for V-YF-XCHECK; omitted = skipped")


def _golden(args: argparse.Namespace) -> int:
    from src.data.data_manager import DataManager
    from src.data.fmp_pipeline.datamanager_adapter.golden import (
        Snapshot,
        cached_series,
        capture,
        compare,
        offline_fetch,
    )

    cache_dir = args.cache_dir or data_cache_dir()
    path = args.snapshot or (args.data_root or default_data_root()) / "reports" / "golden" / "ohlcv_hashes.json"
    if args.action == "compare" and not path.exists():
        _logger.error("No baseline snapshot at %s; run `fmp golden capture` first", path)
        return EXIT_USAGE
    pairs = cached_series(cache_dir / "ohlcv")
    if args.symbols:
        wanted = set(args.symbols)
        pairs = [p for p in pairs if p[0] in wanted]
    data_manager = DataManager(str(cache_dir))
    meta = {"cache_dir": str(cache_dir), "series": len(pairs),
            "fmp_adapter": getattr(data_manager, "fmp_adapter", None) is not None}
    _logger.info("Hashing get_ohlcv for %d cached series (offline)", len(pairs))
    snapshot = capture(pairs, offline_fetch(data_manager), meta)
    if args.action == "capture":
        snapshot.save(path)
        errors = sum(1 for e in snapshot.entries.values() if "error" in e)
        _logger.info("Golden snapshot of %d series (%d raise) written to %s", len(snapshot.entries), errors, path)
        return EXIT_OK
    result = compare(Snapshot.load(path), snapshot)
    latest = path.with_name(path.stem + "_latest.json")
    snapshot.save(latest)
    _logger.info("Golden compare: %d same, %d changed, %d missing, %d added (this run saved to %s)",
                 result.same, len(result.changed), len(result.missing), len(result.added), latest)
    for key in (result.changed + result.missing + result.added)[:50]:
        _logger.warning("Differs: %s", key)
    return EXIT_OK if result.clean else EXIT_STOPPED


def _ingest_increments(args: argparse.Namespace) -> int:
    from src.data.fmp_pipeline.increments.ingest import IncrementIngester, NotTheWriter

    config = load_config(args.config)
    data_root = args.data_root or default_data_root()
    try:
        ingester = IncrementIngester(data_root, config)
        report = ingester.run(args.symbols, args.timeframes, args.limit)
    except (NotTheWriter, DumpLocked, FileNotFoundError) as exc:
        _logger.error("%s", exc)
        return EXIT_USAGE
    return EXIT_OK if report.stop_reason == "done" else EXIT_STOPPED


def _validate(args: argparse.Namespace) -> int:
    from src.data.fmp_pipeline.validators.framework import ValidationContext, run_checks
    from src.data.fmp_pipeline.validators.verify import write_reports

    config = load_config(args.config)
    data_root = args.data_root or default_data_root()
    try:
        ctx = ValidationContext(data_root, config, args.sample, legacy_cache_dir=args.cache_dir)
        report = run_checks(ctx, args.checks)
    except (FileNotFoundError, ValueError) as exc:
        _logger.error("%s", exc)
        return EXIT_USAGE
    paths = write_reports(ctx, report)
    _logger.info("Validation of %s: %s. Reports: %s", report.vintage, report.summary(), ", ".join(map(str, paths)))
    return EXIT_STOPPED if report.failed else EXIT_OK


def _verify(args: argparse.Namespace) -> int:
    from src.data.fmp_pipeline.validators.verify import FinalVerifier

    if not args.final:
        _logger.error("Only `fmp verify --final` exists (spec §14)")
        return EXIT_USAGE
    config = load_config(args.config)
    data_root = args.data_root or default_data_root()
    try:
        verifier = FinalVerifier(data_root, config, args.backup, args.confirm_postponed, args.cache_dir, args.sample)
        lock = data_root / "locks" / f"{verifier.ctx.store.vintage}.lock"
        with exclusive_lock(lock), keep_awake():
            items, _ = verifier.run()
    except (FileNotFoundError, DumpLocked) as exc:
        _logger.error("%s", exc)
        return EXIT_USAGE
    return EXIT_OK if all(item.ok for item in items) else EXIT_STOPPED


def _dump_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help="Default: config/fmp/fmp.yaml")
    parser.add_argument("--dump-id", default=None, help="Overrides fmp.dump_id from the config")
    parser.add_argument("--data-root", type=Path, default=None, help="Default: DATA_CACHE_DIR/fmp-data")


def main(argv: Optional[list[str]] = None) -> int:
    """
    Parse arguments and run the chosen command.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(prog="fmp", description="FMP historical data pipeline")
    commands = parser.add_subparsers(dest="command", required=True)

    probe = commands.add_parser("probe-free", help="Free-key API probe (spec §4.2) -> docs/fmp_probe_report.md")
    probe.add_argument("--budget", type=int, default=200, help="Max live calls across all runs (default 200)")
    probe.add_argument("--offline", action="store_true", help="Use cached responses only; make no calls")
    probe.add_argument("--data-root", type=Path, default=None, help="Default: DATA_CACHE_DIR/fmp-data")
    probe.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    probe.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    probe.set_defaults(handler=_probe_free)

    plan = commands.add_parser("plan", help="Fetch universe lists, plan tasks, project bandwidth -> budget_plan.md")
    plan.add_argument("--dry-run", action="store_true", help="Write the reports but add no tasks to the manifest")
    plan.add_argument("--offline", action="store_true", help="Make no calls; plan from what is already archived")
    _dump_options(plan)
    plan.add_argument("--as-of", type=date.fromisoformat, default=None,
                      help="Dump date for the first plan of a dump (default: today UTC); fixed afterwards")
    plan.add_argument("--report-dir", type=Path, default=None, help="Default: <data_root>/reports/<dump_id>")
    plan.set_defaults(handler=_plan)

    run = commands.add_parser("run", help="Download the planned tasks, block by block in spec order")
    _dump_options(run)
    run.add_argument("--phase", type=int, default=None, help="Run only this block (0-6, see budget_plan.md)")
    run.add_argument("--tier", default=None, help="Run only this intraday tier (A-E); implies phase 6")
    run.add_argument("--max-gb", type=float, default=None,
                     help="Stop this run after this many GB (on the configured bandwidth basis)")
    run.add_argument("--workers", type=int, default=None, help="Default: fmp.workers from the config")
    run.set_defaults(handler=_run)

    status = commands.add_parser("status", help="Progress, bandwidth used (rolling window), ETA, failures")
    _dump_options(status)
    status.set_defaults(handler=_status)

    retry = commands.add_parser("retry-failed", help="Re-queue failed tasks for the next `fmp run`")
    _dump_options(retry)
    retry.set_defaults(handler=_retry_failed)

    build = commands.add_parser("build", help="Build the Parquet vintage from the raw archive (no network)")
    _dump_options(build)
    build.add_argument("--no-switch", action="store_true", help="Build and validate, but leave current.txt alone")
    build.set_defaults(handler=_build)

    golden = commands.add_parser("golden", help="Hash DataManager.get_ohlcv over the legacy cache (offline) and "
                                                "compare with a baseline (WP-8 regression check)")
    golden.add_argument("action", choices=("capture", "compare"))
    golden.add_argument("--snapshot", type=Path, default=None,
                        help="Default: <data_root>/reports/golden/ohlcv_hashes.json")
    golden.add_argument("--cache-dir", type=Path, default=None, help="Default: DATA_CACHE_DIR")
    golden.add_argument("--data-root", type=Path, default=None, help="Default: DATA_CACHE_DIR/fmp-data")
    golden.add_argument("--symbols", nargs="*", default=None, help="Only these symbols")
    golden.set_defaults(handler=_golden)

    ingest = commands.add_parser("ingest-increments",
                                 help="Extend the active vintage with raw yfinance bars and events (spec §10; "
                                      "only on fmp.increments.writer_host)")
    ingest.add_argument("--config", type=Path, default=None, help="Default: config/fmp/fmp.yaml")
    ingest.add_argument("--data-root", type=Path, default=None, help="Default: DATA_CACHE_DIR/fmp-data")
    ingest.add_argument("--timeframes", nargs="+", default=["1d", "1h"], choices=("1d", "1h"))
    ingest.add_argument("--symbols", nargs="*", default=None, help="Only these symbols (default: every alive one)")
    ingest.add_argument("--limit", type=int, default=None, help="At most this many securities (trial runs)")
    ingest.set_defaults(handler=_ingest_increments)

    validate = commands.add_parser("validate", help="Run the spec §11 validation suite on the active vintage")
    _validation_options(validate)
    validate.add_argument("--checks", nargs="+", default=None, help="Only these check IDs (e.g. V-OHLC V-CAL)")
    validate.set_defaults(handler=_validate)

    verify = commands.add_parser("verify", help="Gate G3 checklist; prints READY TO CANCEL when everything holds")
    _validation_options(verify)
    verify.add_argument("--final", action="store_true", help="Run the spec §14 checklist")
    verify.add_argument("--backup", type=Path, default=None,
                        help="Root of the backup copy (raw/, vintages/, checksums/ as under data_root)")
    verify.add_argument("--confirm-postponed", action="store_true",
                        help="Confirm that blocks not downloaded are intentionally postponed or abandoned")
    verify.set_defaults(handler=_verify)

    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
