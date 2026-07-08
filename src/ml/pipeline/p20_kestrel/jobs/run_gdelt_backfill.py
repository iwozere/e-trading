"""P20 Kestrel — manual entry point: GDELT GKG multi-day backfill.

Usage:
    python run_gdelt_backfill.py --start 2024-01-01
    python run_gdelt_backfill.py --start 2024-01-01 --end 2024-03-31
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.sentiment.gdelt_processor import run_backfill
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


from src.ml.pipeline.p20_kestrel.jobs.run_common import setup_run_logging


def main() -> None:
    """Parse CLI args and run GDELT backfill for the given date range."""
    setup_run_logging()
    parser = argparse.ArgumentParser(description="GDELT GKG backfill for a date range")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD, inclusive)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD, inclusive); defaults to yesterday")
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end) if args.end else None

    result = run_backfill(start_date, end_date)
    _logger.info("GDELT backfill complete: %s", result)
    print(f"__SCHEDULER_RESULT__:{json.dumps(result, default=str)}")


if __name__ == "__main__":
    main()
