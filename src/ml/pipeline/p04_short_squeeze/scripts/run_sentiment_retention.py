#!/usr/bin/env python3
"""
Sentiment Raw Payload Retention Job for the Short Squeeze Detection Pipeline

Nulls out ss_deep_metrics.raw_payload for rows older than the retention window (default 90
days). sentiment-spec-rev2.md §1.3/§2.11: "raw_payload retained for audit under access control.
Add a retention policy -- default 90 days -- and a purge job. Indefinite retention of
third-party social content has no upside for a signal pipeline with a 7-day feature horizon."

Only the raw_payload column is nulled -- the row itself (squeeze_score, sentiment_24h, and the
other scalar metrics) is left intact for historical backtesting.

Usage:
    python run_sentiment_retention.py [options]

Examples:
    # Purge raw_payload older than the default 90-day window
    python run_sentiment_retention.py

    # Report how many rows WOULD be purged without changing anything
    python run_sentiment_retention.py --dry-run

    # Custom retention window
    python run_sentiment_retention.py --retention-days 30
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.short_squeeze_service import ShortSqueezeService
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

DEFAULT_RETENTION_DAYS = 90


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Purge old sentiment raw_payload data from ss_deep_metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if __doc__ and "Usage:" in __doc__ else "",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=DEFAULT_RETENTION_DAYS,
        help=f"Rows older than this many days have raw_payload nulled (default: {DEFAULT_RETENTION_DAYS})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report the row count without purging anything")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_arguments()

    if args.verbose:
        import logging

        logging.getLogger().setLevel(logging.DEBUG)

    if args.retention_days <= 0:
        _logger.error("--retention-days must be positive")
        return 1

    try:
        service = ShortSqueezeService()

        if args.dry_run:
            count = service.count_sentiment_raw_payload_older_than(args.retention_days)
            _logger.info(
                "DRY RUN: %d ss_deep_metrics rows have raw_payload older than %d days and would be purged",
                count,
                args.retention_days,
            )
            return 0

        purged = service.purge_sentiment_raw_payload_older_than(args.retention_days)
        _logger.info("Purged raw_payload on %d rows older than %d days", purged, args.retention_days)
        return 0

    except KeyboardInterrupt:
        _logger.warning("Interrupted by user")
        return 130
    except Exception:
        _logger.exception("Sentiment retention job failed:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
