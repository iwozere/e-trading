"""
P17 Penny Stock Screener — CLI Entry Point

Usage:
    python run_p17.py
    python run_p17.py --target-date 2025-05-14
    python run_p17.py --force-refresh
    python run_p17.py --tickers ABCD,EFGH,IJKL
    python run_p17.py --user-id telegram_user_123
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p17_penny_stocks.config import P17PipelineConfig
from src.ml.pipeline.p17_penny_stocks.p17_pipeline import P17Pipeline
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P17 Explosive Penny Stock Screener")
    parser.add_argument(
        "--target-date",
        type=str,
        default=None,
        help="Target date YYYY-MM-DD (defaults to yesterday)",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass all caches and re-download everything",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated ticker list (skips NASDAQ FTP download)",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default=None,
        help="Telegram user ID for alert delivery",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = P17PipelineConfig.create_default()
    if args.user_id:
        config.user_id = args.user_id

    tickers = None
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    try:
        pipeline = P17Pipeline(config=config, target_date=args.target_date)
        result = pipeline.run(force_refresh=args.force_refresh, tickers=tickers)

        results_dir = PROJECT_ROOT / "results" / "p17_penny_stocks" / pipeline.target_date

        print(
            f"P17 complete. "
            f"Candidates={result.get('total_candidates', 0)} | "
            f"A={result.get('tier_a', 0)} "
            f"B={result.get('tier_b', 0)} "
            f"C={result.get('tier_c', 0)} | "
            f"Explosive={result.get('explosive', 0)} | "
            f"results_dir={results_dir}"
        )

        scheduler_result = {
            "success": result.get("success", False),
            "total_candidates": result.get("total_candidates", 0),
            "tier_a_count": result.get("tier_a", 0),
            "tier_b_count": result.get("tier_b", 0),
            "tier_c_count": result.get("tier_c", 0),
            "explosive_count": result.get("explosive", 0),
            "results_dir": str(results_dir),
            "timestamp": datetime.now().isoformat(),
        }
        print(f"__SCHEDULER_RESULT__:{json.dumps(scheduler_result)}")

        return 0 if result.get("success") else 1

    except Exception as e:
        _logger.exception("P17 pipeline failed: %s", e)
        print("\n[ERROR] P17 pipeline failed.\n")
        result_err = {
            "success": False,
            "error": "Pipeline execution failed",
            "timestamp": datetime.now().isoformat(),
        }
        print(f"__SCHEDULER_RESULT__:{json.dumps(result_err)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
