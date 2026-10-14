"""
``fmp`` command-line entry point (spec §6.1): ``probe-free`` (WP-1) and ``plan`` (WP-4) so far.

Usage (PowerShell, from the project root)::

    .venv\\Scripts\\python.exe -m src.data.fmp_pipeline.cli probe-free
    .venv\\Scripts\\python.exe -m src.data.fmp_pipeline.cli probe-free --offline   # rebuild report from cache
    .venv\\Scripts\\python.exe -m src.data.fmp_pipeline.cli plan --dry-run   # budget_plan.md, no new tasks
    .venv\\Scripts\\python.exe -m src.data.fmp_pipeline.cli plan --offline   # re-plan from the archive only

Layout under ``data_root`` (``DATA_CACHE_DIR/fmp-data``): ``raw/<dump_id>/`` archive, ``manifests/<dump_id>.sqlite``,
``reports/<dump_id>/`` plan reports, ``locks/<dump_id>.lock`` (one writer per dump).
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
from src.data.fmp_pipeline.client.fmp_client import FmpClient  # noqa: E402
from src.data.fmp_pipeline.client.rate_limiter import AsyncRateLimiter  # noqa: E402
from src.data.fmp_pipeline.errors import DumpLocked  # noqa: E402
from src.data.fmp_pipeline.fsutil import exclusive_lock  # noqa: E402
from src.data.fmp_pipeline.manifest.manifest import Manifest  # noqa: E402
from src.data.fmp_pipeline.planner.config import PlanConfig, load_config  # noqa: E402
from src.data.fmp_pipeline.planner.planner import Planner  # noqa: E402
from src.data.fmp_pipeline.planner.projection import GB  # noqa: E402
from src.data.fmp_pipeline.probe.free_probe import FreeProbe  # noqa: E402
from src.data.fmp_pipeline.probe.probe_client import ProbeClient  # noqa: E402
from src.data.fmp_pipeline.probe.report import render_report, write_fixtures, write_report  # noqa: E402
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
        self.raw_root = data_root / "raw"
        self.manifest = data_root / "manifests" / f"{dump_id}.sqlite"
        self.reports = data_root / "reports" / dump_id
        self.lock = data_root / "locks" / f"{dump_id}.lock"


async def _fetch_sources(planner: Planner, api_key: str, config: PlanConfig) -> str:
    client = FmpClient(api_key, base_url=config.base_url, rate_limiter=AsyncRateLimiter(config.rate_limit_per_min))
    async with client:
        summary = await planner.fetch_sources(client)
    return summary.stop_reason


def _plan(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.dump_id:
        config = replace(config, dump_id=args.dump_id)
    paths = DumpPaths(args.data_root or default_data_root(), config.dump_id)
    try:
        with exclusive_lock(paths.lock), Manifest(paths.manifest, config.dump_id) as manifest:
            planner = Planner(manifest, RawArchive(paths.raw_root, config.dump_id), config,
                              args.report_dir or paths.reports, args.as_of)
            stop_reason = "done"
            if not args.offline:
                from src.config.provider_config import get_api_key

                api_key = get_api_key("FMP_API_KEY")
                if not api_key:
                    _logger.error("FMP_API_KEY is not set; use --offline to plan from the archive only")
                    return 2
                stop_reason = asyncio.run(_fetch_sources(planner, api_key, config))
                if stop_reason != "done":
                    _logger.error("Source lists incomplete (run stopped: %s); planning with what was fetched",
                                  stop_reason)
            result = planner.plan(dry_run=args.dry_run)
    except DumpLocked as exc:
        _logger.error("%s", exc)
        return 2
    wire, decoded = result.projection.remaining_bytes()
    free = result.context.free_bytes
    _logger.info("Remaining: %.2f GB wire / %.2f GB decoded; disk need %.1f GB, free %s. Reports: %s, %s",
                 wire / GB, decoded / GB, result.context.need_bytes / GB,
                 "unknown" if free is None else f"{free / GB:.1f} GB", result.budget_path, result.universe_path)
    return 0 if stop_reason == "done" else 1


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
    plan.add_argument("--config", type=Path, default=None, help="Default: config/fmp/fmp.yaml")
    plan.add_argument("--dump-id", default=None, help="Overrides fmp.dump_id from the config")
    plan.add_argument("--as-of", type=date.fromisoformat, default=None,
                      help="Dump date for the first plan of a dump (default: today UTC); fixed afterwards")
    plan.add_argument("--data-root", type=Path, default=None, help="Default: DATA_CACHE_DIR/fmp-data")
    plan.add_argument("--report-dir", type=Path, default=None, help="Default: <data_root>/reports/<dump_id>")
    plan.set_defaults(handler=_plan)

    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
