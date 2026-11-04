"""
Command-line interface of the trade ledger.

Usage::

    python -m src.portfolio.trade_ledger run [--month YYYY-MM] [--no-download] [--dry-run] [--config PATH]
    python -m src.portfolio.trade_ledger import PATH [PATH ...]
    python -m src.portfolio.trade_ledger reconcile

`run` builds and emails the report (default: previous month). `import` loads
Flex XML files or directories (e.g. the backfill). `reconcile` checks the
ledger against every stored IBKR position snapshot.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger  # noqa: E402
from src.portfolio.trade_ledger.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from src.portfolio.trade_ledger.reconciler import STATUS_MISMATCH, reconcile_all  # noqa: E402
from src.portfolio.trade_ledger.repository import load_events  # noqa: E402
from src.portfolio.trade_ledger.runner import import_files, parse_month, run_month, summary_to_dict  # noqa: E402
from src.portfolio.trade_ledger.xml_parser import FlexParseError  # noqa: E402

_logger = setup_logger(__name__)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(prog="python -m src.portfolio.trade_ledger", description="Monthly IBKR trade ledger with weighted average cost.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Build and email the monthly report")
    run.add_argument("--month", type=parse_month, default=None, help="Report month YYYY-MM (default: previous month)")
    run.add_argument("--no-download", action="store_true", help="Skip the Flex download; use what is already imported")
    run.add_argument("--dry-run", action="store_true", help="Write XLSX / HTML but send no email")
    run.add_argument("--config", default=DEFAULT_CONFIG_PATH, help=f"Config YAML (default: {DEFAULT_CONFIG_PATH})")

    imp = sub.add_parser("import", help="Import Flex XML files or directories (idempotent)")
    imp.add_argument("paths", nargs="+", type=Path, help="XML files and/or directories")

    sub.add_parser("reconcile", help="Check share counts against every stored IBKR snapshot")
    return parser.parse_args(argv)


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        _logger.error("Config error: %s", exc)
        return 2
    summary = run_month(cfg, args.month, dry_run=args.dry_run, download=False if args.no_download else None)
    _logger.info("Run summary:\n%s", json.dumps(summary_to_dict(summary), indent=2, sort_keys=True))
    return 0 if summary.ok else 1


def _cmd_import(paths: List[Path]) -> int:
    try:
        inserted, skipped = import_files(paths)
    except (FileNotFoundError, FlexParseError) as exc:
        _logger.error("Import failed: %s", exc)
        return 1
    _logger.info("Import done: newly inserted %s, skipped %s", inserted, skipped)
    return 0


def _cmd_reconcile() -> int:
    reports = reconcile_all(load_events())
    for rep in reports:
        _logger.info("Snapshot %s: %s (%d positions)", rep.snapshot_date, rep.status, len(rep.lines))
        for line in rep.mismatches:
            _logger.warning("  %s: ledger %s vs IBKR %s", line.symbol, line.ledger_shares, line.ibkr_shares)
    failed = sum(1 for rep in reports if rep.status == STATUS_MISMATCH)
    _logger.info("Reconciliation: %d snapshot(s), %d with mismatches", len(reports), failed)
    return 1 if failed else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    CLI entry point.

    Args:
        argv: Arguments (defaults to sys.argv).

    Returns:
        Process exit code.
    """
    args = _parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "import":
        return _cmd_import(args.paths)
    return _cmd_reconcile()


if __name__ == "__main__":
    raise SystemExit(main())
