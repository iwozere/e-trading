"""
P22 — spec §8.3 lookahead audit against the live store. Read-only; exits 1 on any violation.

    python -m src.ml.pipeline.p22_biotech_ma.cli.lookahead_audit_cli [--samples 200] [--seed 7]

Spec §8.3 wants this to block the build. CI has no database, so run it on the Pi after ingest
changes and before trusting a backtest. See `features/lookahead_audit_run.py` for what is checked.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.database_service import DatabaseService
from src.ml.pipeline.p22_biotech_ma.features.lookahead_audit_run import AuditReport, run_audit
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_SHOW = 10


def log_report(report: AuditReport) -> None:
    """Log the summary and the first few findings of each kind."""
    _logger.info("Lookahead audit: %d samples %s, %d rows checked", report.samples, report.samples_by_category, report.rows_checked)
    _logger.info("Rows checked by repo method: %s", report.rows_by_method)
    _logger.info("Reads without known_from (not auditable): %s", report.unaudited_calls)
    for title, findings in (
        ("lookahead violations (known_from after as_of)", report.lookahead_violations),
        ("13D/8-K rows whose known_from is not the filing date", report.filing_date_mismatches),
        ("unverified Block G rows that reached a feature", report.unverified_reads),
        ("feature errors", report.feature_errors),
    ):
        if findings:
            _logger.error("%d %s, first %d: %s", len(findings), title, _SHOW, findings[:_SHOW])
    if report.passed:
        _logger.info("Lookahead audit PASSED")
    else:
        _logger.error("Lookahead audit FAILED")


def main(argv: Optional[List[str]] = None) -> int:
    """Run the audit; exit code 0 = pass, 1 = fail."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--min-per-category", dest="min_per_category", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    with DatabaseService().uow() as uow:
        report = run_audit(uow.p22, total=args.samples, min_per_category=args.min_per_category, seed=args.seed)
    log_report(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
