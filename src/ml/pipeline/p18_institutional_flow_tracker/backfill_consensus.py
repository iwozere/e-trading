"""
P18 Institutional Flow Tracker — One-time quarterly consensus backfill.

Downloads all 13F-HR filings for the specified quarter from EDGAR and builds
the consensus cache that the daily pipeline depends on.

Usage (from project root, with venv activated):
    python src/ml/pipeline/p18_institutional_flow_tracker/backfill_consensus.py
    python src/ml/pipeline/p18_institutional_flow_tracker/backfill_consensus.py --year 2026 --quarter 1
    python src/ml/pipeline/p18_institutional_flow_tracker/backfill_consensus.py --year 2026 --quarter 1 --force
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.ml.pipeline.p18_institutional_flow_tracker.config import P18Config
from src.ml.pipeline.p18_institutional_flow_tracker.pipeline import InstitutionalFlowPipeline

_logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P18 quarterly consensus backfill.")
    parser.add_argument("--year", type=int, default=2026, help="Calendar year (default: 2026)")
    parser.add_argument("--quarter", type=int, default=1, choices=[1, 2, 3, 4], help="Quarter 1-4 (default: 1)")
    parser.add_argument("--force", action="store_true", help="Re-download all infotables even if cached")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _logger.info("P18 backfill: rebuilding consensus for %d Q%d (force=%s)", args.year, args.quarter, args.force)

    config = P18Config.create_default()
    pipeline = InstitutionalFlowPipeline(config)
    consensus_df = pipeline.rebuild_quarterly_consensus(args.year, args.quarter, force_download=args.force)

    if consensus_df.empty:
        _logger.error("Backfill produced an empty consensus — check EDGAR connectivity and prior-quarter data")
        return 1

    _logger.info("Backfill complete: %d tickers in consensus for %d Q%d", len(consensus_df), args.year, args.quarter)
    return 0


if __name__ == "__main__":
    sys.exit(main())
