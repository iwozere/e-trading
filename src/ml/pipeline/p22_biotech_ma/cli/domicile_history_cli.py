"""
P22 — backfill point-in-time `is_foreign_domiciled` history for the M6 backtest (`ingest/domicile_history.py`).

One-off, rerunnable (inserts only history older than what is stored):
    python -m src.ml.pipeline.p22_biotech_ma.cli.domicile_history_cli --start 2010-01-01 --dry-run
    python -m src.ml.pipeline.p22_biotech_ma.cli.domicile_history_cli --start 2010-01-01
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.database_service import DatabaseService
from src.data.downloader.edgar_downloader import EdgarDownloader
from src.ml.pipeline.p22_biotech_ma.ingest.domicile_history import run as run_domicile_history
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class _DryRunRollback(Exception):
    """Raised inside the unit of work to roll back a dry run."""


def build_parser() -> argparse.ArgumentParser:
    """CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default="2010-01-01", help="First filing date to scan (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Last filing date to scan (default today)")
    parser.add_argument("--dry-run", action="store_true", help="Scan and count, then roll the inserts back")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Run the backfill; returns the process exit code."""
    args = build_parser().parse_args(argv)
    end = args.end or date.today().isoformat()
    try:
        with DatabaseService().uow() as uow:
            summary = run_domicile_history(uow.p22, EdgarDownloader(), start_dt=args.start, end_dt=end)
            if args.dry_run:
                raise _DryRunRollback()
    except _DryRunRollback:
        _logger.info("Dry run — rolled back: %s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
