"""
P22 — one-off cleanup: delete duplicate `p22_financial_fact` rows (`P22Repo.delete_duplicate_financial_facts`).
Until 2026-11-09 the daily Financial Facts Normalization re-inserted every company's full XBRL history on
every run (~169,000 rows a weekday). A duplicate repeats another row's company, metric, period end,
`known_from`, value and source; the newest copy of each is kept.

    python -m src.ml.pipeline.p22_biotech_ma.cli.dedupe_facts_cli --dry-run
    python -m src.ml.pipeline.p22_biotech_ma.cli.dedupe_facts_cli
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.database_service import DatabaseService
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def main(argv: Optional[list[str]] = None) -> int:
    """Count, or delete, the duplicates; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Only count the duplicates")
    args = parser.parse_args(argv)
    with DatabaseService().uow() as uow:
        total, duplicates = uow.p22.count_duplicate_financial_facts()
        _logger.info("p22_financial_fact: %d rows, %d duplicates", total, duplicates)
        if args.dry_run:
            return 0
        deleted = uow.p22.delete_duplicate_financial_facts()
    _logger.info("Deleted %d duplicate rows (%d remain)", deleted, total - deleted)
    return 0


if __name__ == "__main__":
    sys.exit(main())
