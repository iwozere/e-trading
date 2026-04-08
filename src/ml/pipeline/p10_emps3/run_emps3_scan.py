"""
EMPS3 Universe Scanner - CLI Entry Point
Pipeline execution script for Accumulation Phase Detection.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import json

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.ml.pipeline.p10_emps3.config import EMPS3PipelineConfig
from src.ml.pipeline.p10_emps3.emps3_pipeline import EMPS3Pipeline
from src.ml.pipeline.shared.pipeline_summary_generator import PipelineSummaryGenerator

_logger = setup_logger(__name__)


def _csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return len(pd.read_csv(path))
    except Exception:
        _logger.warning("Could not count rows in %s", path, exc_info=True)
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EMPS3 Precursor phase scanner.")
    parser.add_argument('--force-refresh', action='store_true', help='Force refresh caches.')
    parser.add_argument('--tickers', type=str, help='Comma-separated list of tickers to scan.')
    # Scheduler (_execute_data_processing_job) always appends this when job_schedules.user_id is set.
    parser.add_argument(
        '--user-id',
        type=str,
        help='User ID for alerts (same contract as p06 run_emps2_scan)',
    )
    return parser.parse_args()

def main() -> int:
    args = parse_args()

    config = EMPS3PipelineConfig.create_default()
    if args.user_id:
        config.user_id = args.user_id
    force_refresh = args.force_refresh

    try:
        pipeline = EMPS3Pipeline(config)
        
        tickers = None
        if args.tickers:
            tickers = [t.strip().upper() for t in args.tickers.split(',')]
            
        final_df = pipeline.run(force_refresh=force_refresh, tickers=tickers)

        # Same contract as p06: pipeline writes under target_date (default yesterday), not calendar today.
        results_dir = PROJECT_ROOT / "results" / "p10_emps3" / pipeline.target_date

        print(
            f"Pipeline complete. Found {len(final_df)} passing candidates. "
            f"results_dir={results_dir} (target_date={pipeline.target_date})"
        )

        # Generate/Update historical performance summary
        try:
            print("\n" + "="*70 + "\n Historical Summary \n" + "="*70)
            summary_gen = PipelineSummaryGenerator()
            summary_gen.generate_historical_summary(results_dir.parent)
        except Exception as e:
            _logger.warning(f"Failed to generate historical summary: {e}")

        # job_schedules.notification_rules expect phase1_count / phase2_count (insert_p10_schedules.sql).
        prebreakout_n = _csv_row_count(results_dir / "07_prebreakout_watchlist.csv")
        phase1_count = max(len(final_df), prebreakout_n)
        phase2_count = _csv_row_count(results_dir / "07_phase1_5_alerts.csv")

        result = {
            "success": True,
            "total_candidates": len(final_df),
            "phase1_count": phase1_count,
            "phase2_count": phase2_count,
            "results_dir": str(results_dir),
            "timestamp": datetime.now().isoformat(),
        }
        print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")

        return 0

    except Exception as e:
        _logger.exception("Error running pipeline: %s", e)
        print("\n[ERROR] Pipeline failed.\n")
        
        result = {
            "success": False,
            "error": "Pipeline execution failed"
        }
        print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")

        return 1

if __name__ == "__main__":
    sys.exit(main())
