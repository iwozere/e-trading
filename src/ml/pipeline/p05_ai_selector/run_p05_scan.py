"""P05 AI Selector — scheduler entry point."""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.ml.pipeline.p05_ai_selector.pipeline import P05Pipeline

_logger = setup_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P05 AI Selector — daily scan.")
    parser.add_argument("--user-id", type=str, default=None,
                        help="User ID injected by the scheduler.")
    parser.add_argument("--as-of-date", type=str, default=None,
                        help="ISO date to run for (e.g. 2026-06-14). Defaults to today.")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Force re-download of all cached data.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    as_of_date = None
    if args.as_of_date:
        try:
            as_of_date = date.fromisoformat(args.as_of_date)
        except ValueError:
            _logger.error("Invalid --as-of-date value: %s (expected YYYY-MM-DD)", args.as_of_date)
            result = {"success": False, "error": "Invalid --as-of-date", "pick_count": 0,
                      "p18_signals_count": 0, "notification_override": 0}
            print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
            return 0

    try:
        result = P05Pipeline().run(
            user_id=args.user_id,
            as_of_date=as_of_date,
            force_refresh=args.force_refresh,
        )
        print(f"__SCHEDULER_RESULT__:{json.dumps(result, default=str)}")
        return 0

    except Exception as exc:
        _logger.exception("Unhandled error in P05 scan: %s", exc)
        result = {
            "success": False,
            "error": "Unhandled pipeline error",
            "pick_count": 0,
            "p18_signals_count": 0,
            "notification_override": 0,
        }
        print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
