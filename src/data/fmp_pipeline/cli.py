"""
``fmp`` command-line entry point (spec §6.1): ``probe-free``, ``plan``, ``run``, ``status``, ``retry-failed``,
``build``.

Usage (PowerShell, from the project root)::

    .venv\\Scripts\\python.exe -m src.data.fmp_pipeline.cli probe-free
    .venv\\Scripts\\python.exe -m src.data.fmp_pipeline.cli probe-free --offline   # rebuild report from cache
    .venv\\Scripts\\python.exe -m src.data.fmp_pipeline.cli plan --dry-run   # budget_plan.md, no new tasks
    .venv\\Scripts\\python.exe -m src.data.fmp_pipeline.cli plan --offline   # re-plan from the archive only
    .venv\\Scripts\\python.exe -m src.data.fmp_pipeline.cli run [--phase N] [--tier T] [--max-gb G]
    .venv\\Scripts\\python.exe -m src.data.fmp_pipeline.cli status
    .venv\\Scripts\\python.exe -m src.data.fmp_pipeline.cli retry-failed
    .venv\\Scripts\\python.exe -m src.data.fmp_pipeline.cli build [--no-switch]

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


def default_data_root() -> Path:
    """Return ``DATA_CACHE_DIR/fmp-data`` (owner decision Q2)."""
    from config.donotshare.donotshare import DATA_CACHE_DIR

    return Path(DATA_CACHE_DIR) / "fmp-data"


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

    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
