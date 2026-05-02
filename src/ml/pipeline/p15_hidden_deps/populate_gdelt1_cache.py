"""
GDELT 1.0 GKG Cache Population Script

Bulk-downloads every available GDELT 1.0 GKG daily zip file
(2013-04-01 through 2015-02-17) and caches them under
DATA_CACHE_DIR/gdelt/gkg/YYYYMMDD.gkg.csv.zip.

Dates supplied outside the available v1 range are clamped automatically:
  - Anything before 2013-04-01 is advanced to 2013-04-01.
  - Anything after 2015-02-17 is pulled back to 2015-02-17.

Already-cached files are skipped unless --force is supplied.

Usage:
    # Full historical backfill (default)
    python populate_gdelt1_cache.py

    # Custom window
    python populate_gdelt1_cache.py --start 2013-04-01 --end 2014-12-31

    # Re-download everything
    python populate_gdelt1_cache.py --force

    # Dry-run: print plan without downloading
    python populate_gdelt1_cache.py --dry-run
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.data.downloader.gdelt_downloader import (
    Gdelt1Downloader,
    _GDELT_1_GKG_END,
    _GDELT_1_GKG_START,
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def _yesterday_utc() -> datetime:
    """Return yesterday's date at midnight, timezone-naive UTC."""
    dt = datetime.now(timezone.utc) - timedelta(days=1)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def main() -> None:
    """
    Entry point for the GDELT 1.0 GKG cache population script.

    Clamps the requested date window to the available v1 range, then
    delegates to Gdelt1Downloader.download_gkg_range().
    """
    parser = argparse.ArgumentParser(
        description="Populate the local GDELT 1.0 GKG cache (2013-04-01 to 2015-02-17).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2013-01-01",
        help="First day to download (default: 2013-01-01; clamped to 2013-04-01 if earlier)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help=(
            "Last day to download inclusive "
            "(default: yesterday or 2015-02-17, whichever is earlier)"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files that are already cached",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the download plan without actually fetching anything",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override the cache root directory",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.5,
        help="Seconds between HTTP requests (default: 0.5)",
    )
    args = parser.parse_args()

    # --- resolve date window ------------------------------------------------
    start = _parse_date(args.start)
    yest = _yesterday_utc()
    end = _parse_date(args.end) if args.end else yest

    # Clamp to available v1 range
    if start < _GDELT_1_GKG_START:
        _logger.info(
            "Start %s is before GDELT 1.0 GKG availability; clamping to %s",
            start.date(), _GDELT_1_GKG_START.date(),
        )
        start = _GDELT_1_GKG_START

    if end > _GDELT_1_GKG_END:
        _logger.info(
            "End %s is past GDELT 1.0 GKG cutoff; clamping to %s",
            end.date(), _GDELT_1_GKG_END.date(),
        )
        end = _GDELT_1_GKG_END

    if start > end:
        _logger.error("Nothing to download: effective start %s > end %s", start.date(), end.date())
        sys.exit(1)

    total_days = (end - start).days + 1
    _logger.info(
        "GDELT 1.0 GKG cache population plan: %s → %s (%d days)",
        start.date(), end.date(), total_days,
    )

    if args.dry_run:
        _logger.info("Dry-run mode — no files will be downloaded.")
        print(
            f"__SCHEDULER_RESULT__:{json.dumps({'success': True, 'dry_run': True, 'total': total_days})}"
        )
        return

    # --- download -----------------------------------------------------------
    dl = Gdelt1Downloader(cache_dir=args.cache_dir, request_delay=args.request_delay)
    summary = dl.download_gkg_range(start, end, force=args.force)

    _logger.info(
        "Complete: downloaded=%d skipped=%d errors=%d / %d total days",
        summary["downloaded"], summary["skipped"], summary["errors"], summary["total"],
    )
    print(f"__SCHEDULER_RESULT__:{json.dumps({'success': True, **summary})}")


if __name__ == "__main__":
    main()
