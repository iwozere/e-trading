"""
P18 Institutional Flow Tracker — Quarterly consensus backfill.

Downloads all 13F-HR filings for the specified quarter from EDGAR and builds
the consensus cache that the daily pipeline depends on.

This is a HEAVY job (hours: thousands of filers × EDGAR rate-limit + FIGI CUSIP
mapping). It MUST NOT be run through the project scheduler (job_schedules /
APScheduler), whose `timeout_seconds` would SIGKILL it mid-run. Run it detached
(nohup / tmux / systemd one-shot) or via the dedicated crontab wrapper
``bin/scheduler/p18_consensus_backfill.sh`` — see that script and the P18 README.

Usage (from project root, with venv activated):
    # Explicit quarter, always rebuild:
    python src/ml/pipeline/p18_institutional_flow_tracker/backfill_consensus.py --year 2026 --quarter 1
    # Force re-download of every infotable:
    python src/ml/pipeline/p18_institutional_flow_tracker/backfill_consensus.py --year 2026 --quarter 1 --force
    # Idempotent, self-targeting (for cron): pick the most recently completed
    # quarter and only build if the consensus cache is missing:
    python src/ml/pipeline/p18_institutional_flow_tracker/backfill_consensus.py --auto-quarter --if-missing
"""

import argparse
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.ml.pipeline.p18_institutional_flow_tracker.config import P18Config
from src.ml.pipeline.p18_institutional_flow_tracker.pipeline import (
    InstitutionalFlowPipeline,
    _resolve_current_quarter,
)

_logger = setup_logger(__name__)

# A real consensus CSV.gz (header + at least one row) is comfortably larger than
# this; the daily pipeline only ever writes the file when it is non-empty.
_MIN_CONSENSUS_BYTES = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P18 quarterly consensus backfill.")
    parser.add_argument("--year", type=int, default=2026, help="Calendar year (default: 2026)")
    parser.add_argument("--quarter", type=int, default=1, choices=[1, 2, 3, 4], help="Quarter 1-4 (default: 1)")
    parser.add_argument(
        "--auto-quarter",
        action="store_true",
        help="Ignore --year/--quarter and target the most recently completed quarter for today's date.",
    )
    parser.add_argument(
        "--if-missing",
        action="store_true",
        help="Skip the rebuild (exit 0) when a non-empty consensus cache already exists. Safe for cron.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download all infotables even if cached")
    return parser.parse_args()


def _consensus_path(pipeline: InstitutionalFlowPipeline, year: int, quarter: int) -> Path:
    """Return the on-disk path of the consensus cache for the given quarter."""
    return pipeline._edgar._13f_dir / "consensus" / f"{year}_Q{quarter}.csv.gz"


def main() -> int:
    args = parse_args()

    config = P18Config.create_default()
    pipeline = InstitutionalFlowPipeline(config)

    if args.auto_quarter:
        _, year, quarter = _resolve_current_quarter(date.today())
    else:
        year, quarter = args.year, args.quarter

    cache_path = _consensus_path(pipeline, year, quarter)

    if args.if_missing and cache_path.exists() and cache_path.stat().st_size >= _MIN_CONSENSUS_BYTES:
        _logger.info(
            "P18 backfill: consensus cache already present for %d Q%d (%s) — nothing to do.",
            year, quarter, cache_path,
        )
        return 0

    _logger.info("P18 backfill: rebuilding consensus for %d Q%d (force=%s)", year, quarter, args.force)

    consensus_df = pipeline.rebuild_quarterly_consensus(year, quarter, force_download=args.force)

    if consensus_df.empty:
        _logger.error(
            "Backfill produced an empty consensus for %d Q%d — check EDGAR connectivity and "
            "prior-quarter data. If this quarter's 45-day filing window has not closed yet, "
            "few filings may be available; retry after the filing deadline.",
            year, quarter,
        )
        return 1

    _logger.info("Backfill complete: %d tickers in consensus for %d Q%d", len(consensus_df), year, quarter)
    return 0


if __name__ == "__main__":
    sys.exit(main())
