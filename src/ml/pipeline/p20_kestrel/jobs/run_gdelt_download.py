"""
P20 Kestrel — scheduler entry point: GDELT GKG daily download.

Downloads yesterday's (and, as catch-up, the day before's) GKG file into the
shared DATA_CACHE_DIR/gdelt/gkg/ cache via the central GdeltDownloader.

Why this job exists: p20_data_health (06:00 UTC) and p20_gdelt_process
(06:15 UTC) both expect yesterday's GKG file, but historically the only
downloader was the P15 daily bundle at 13:00 UTC — seven hours too late, so
gdelt_process ran against zero files every day. GDELT 2.0 publishes all of
day D's 15-minute files by ~00:15 UTC on D+1, so an early-morning download
closes the gap. Idempotent: already-cached days are skipped.
"""

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.downloader.gdelt_downloader import GdeltDownloader
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


from src.ml.pipeline.p20_kestrel.jobs.run_common import setup_run_logging


def run(as_of_date: date | None = None) -> dict:
    """
    Download GKG files for yesterday and the day before (catch-up).

    Args:
        as_of_date: Reference date (defaults to today UTC); downloads target
            as_of_date - 1 and as_of_date - 2.

    Returns:
        Summary dict with per-date download outcomes.
    """
    today = as_of_date or date.today()
    downloader = GdeltDownloader()
    results: dict = {}
    for days_back in (2, 1):  # oldest first so gaps heal before the fresh day
        target = today - timedelta(days=days_back)
        try:
            path = downloader.download_gkg_day(datetime(target.year, target.month, target.day))
            results[target.isoformat()] = str(path) if path else "no data"
        except Exception as exc:
            _logger.exception("GKG download failed for %s", target)
            results[target.isoformat()] = f"error: {exc}"
    return results


def main() -> None:
    """Download recent GDELT GKG files and print scheduler result."""
    setup_run_logging()
    result = run()
    _logger.info("GDELT download complete: %s", result)
    print(f"__SCHEDULER_RESULT__:{json.dumps(result, default=str)}")


if __name__ == "__main__":
    main()
