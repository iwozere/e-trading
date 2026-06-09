"""
P18 Institutional Flow Tracker — Scheduler Entry Point

Called daily by the scheduler as a DATA_PROCESSING job.
Follows the same contract as p10_emps3/run_emps3_scan.py:
- accepts --user-id injected by the scheduler
- prints __SCHEDULER_RESULT__:<json> to stdout
- exits 0 on success, 1 on failure
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.ml.pipeline.p18_institutional_flow_tracker.config import P18Config
from src.ml.pipeline.p18_institutional_flow_tracker.pipeline import InstitutionalFlowPipeline

_logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P18 Institutional Flow Tracker — daily scan.")
    parser.add_argument(
        "--user-id",
        type=str,
        default=None,
        help="User ID for alerts (injected by the scheduler).",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="ISO date to run for, e.g. 2024-02-14. Defaults to today.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-download of all cached data.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    as_of_date = None
    if args.as_of_date:
        try:
            as_of_date = date.fromisoformat(args.as_of_date)
        except ValueError:
            _logger.error("Invalid --as-of-date value: %s (expected YYYY-MM-DD)", args.as_of_date)
            result = {"success": False, "error": "Invalid --as-of-date", "high_score_count": 0}
            print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
            return 1

    try:
        config = P18Config.create_default()
        pipeline = InstitutionalFlowPipeline(config)
        result = pipeline.run(
            user_id=args.user_id,
            as_of_date=as_of_date,
            force_refresh=args.force_refresh,
        )
        print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
        return 0 if result.get("success") else 1

    except Exception as exc:
        _logger.exception("Unhandled error in P18 scan: %s", exc)
        result = {
            "success": False,
            "error": "Unhandled pipeline error",
            "high_score_count": 0,
        }
        print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
