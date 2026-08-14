"""
Command-line interface for the portfolio management (stop-loss reminder) pipeline.

Usage::

    python -m src.portfolio.management [--config PATH] [--dry-run] [--as-of-date YYYY-MM-DD]
"""

import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger  # noqa: E402
from src.portfolio.management.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from src.portfolio.management.runner import run_once, summary_to_dict  # noqa: E402

_logger = setup_logger(__name__)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m src.portfolio.management",
        description="Earnings-triggered stop-loss coverage reminder for IBKR holdings.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to pipeline YAML (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Format the notification and log it but do not send",
    )
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="Reference date (YYYY-MM-DD) for the earnings-window lookup; defaults to today (UTC)",
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    """Load config and run once; return an exit code."""
    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        _logger.error("Config error: %s", exc)
        return 2

    as_of_date = date.fromisoformat(args.as_of_date) if args.as_of_date else None

    summary = await run_once(cfg, dry_run=args.dry_run, as_of_date=as_of_date)

    print(json.dumps(summary_to_dict(summary), indent=2, sort_keys=True))

    if summary.errors:
        return 1
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
