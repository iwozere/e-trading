"""
``fmp`` command-line entry point (spec §6.1). Only ``probe-free`` exists so far (plan WP-1).

Usage (PowerShell, from the project root)::

    .venv\\Scripts\\python.exe -m src.data.fmp_pipeline.cli probe-free
    .venv\\Scripts\\python.exe -m src.data.fmp_pipeline.cli probe-free --offline   # rebuild report from cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fmp_pipeline.probe.free_probe import FreeProbe
from src.data.fmp_pipeline.probe.probe_client import ProbeClient
from src.data.fmp_pipeline.probe.report import render_report, write_fixtures, write_report
from src.notification.logger import setup_logger

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

    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
