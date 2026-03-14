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

_logger = setup_logger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EMPS3 Precursor phase scanner.")
    parser.add_argument('--force-refresh', action='store_true', help='Force refresh caches.')
    return parser.parse_args()

def main() -> int:
    args = parse_args()

    config = EMPS3PipelineConfig.create_default()
    force_refresh = args.force_refresh

    try:
        pipeline = EMPS3Pipeline(config)
        final_df = pipeline.run(force_refresh=force_refresh)

        today = datetime.now().strftime('%Y-%m-%d')
        results_dir = PROJECT_ROOT / 'results' / 'p10_emps3' / today

        print(f"Pipeline complete. Found {len(final_df)} passing candidates.")
        
        result = {
            "success": True,
            "total_candidates": len(final_df),
            "results_dir": str(results_dir),
            "timestamp": datetime.now().isoformat()
        }
        print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")

        return 0

    except Exception as e:
        _logger.exception("Error running pipeline:", e)
        print("\n[ERROR] Pipeline failed.\n")
        
        result = {
            "success": False,
            "error": "Pipeline execution failed"
        }
        print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")

        return 1

if __name__ == "__main__":
    sys.exit(main())
