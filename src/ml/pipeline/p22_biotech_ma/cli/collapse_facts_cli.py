"""
P22 — one-off cleanup: merge back-to-back `p22_financial_fact` rows from one source that repeat the
same value (`P22Repo.collapse_repeated_facts`). Before 2026-11-08 the daily domicile job wrote a new
row per company per weekday even when nothing changed.

    python -m src.ml.pipeline.p22_biotech_ma.cli.collapse_facts_cli --dry-run
    python -m src.ml.pipeline.p22_biotech_ma.cli.collapse_facts_cli
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.database_service import DatabaseService
from src.ml.pipeline.p22_biotech_ma.ingest.domicile_normalization import METRIC, SOURCE_ID
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class _DryRunRollback(Exception):
    """Raised inside the unit of work to roll back a dry run."""


def build_parser() -> argparse.ArgumentParser:
    """CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metric", default=METRIC, help="Fact metric (default %(default)s)")
    parser.add_argument("--source", default=SOURCE_ID, help="Fact source_id (default %(default)s)")
    parser.add_argument("--dry-run", action="store_true", help="Count the repeats, then roll back")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Collapse the repeats; returns the process exit code."""
    args = build_parser().parse_args(argv)
    deleted = 0
    try:
        with DatabaseService().uow() as uow:
            deleted = uow.p22.collapse_repeated_facts(metric=args.metric, source_id=args.source)
            if args.dry_run:
                raise _DryRunRollback()
        _logger.info("Collapsed %s/%s: %d repeated rows deleted", args.metric, args.source, deleted)
    except _DryRunRollback:
        _logger.info("Dry run — rolled back: %s/%s would delete %d repeated rows", args.metric, args.source, deleted)
    return 0


if __name__ == "__main__":
    sys.exit(main())
