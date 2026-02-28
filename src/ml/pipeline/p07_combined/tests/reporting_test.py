import sys
from pathlib import Path
import pandas as pd
import vectorbt as vbt

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p07_combined.pipeline import P07Pipeline
from src.ml.pipeline.p07_combined.json2csv import aggregate_results
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

def test_reporting_functionality():
    """
    Verification script for High-Res Plotting, JSON Logging, and CSV Aggregation.
    """
    p = P07Pipeline()
    _logger.info("Running Persistent Reporting Verification...")

    # Use a known 2025 dataset for full feature coverage (macro data available)
    f = Path("data/BTCUSDT_4h_20250101_20251111.csv")

    if not f.exists():
        _logger.error("Test file %s missing. Cannot run verification.", f)
        return

    try:
        # 1. Pipeline Execution
        df_merged = p.data_loader.get_merged_dataset(f)
        df_enriched = p.enrich_data(df_merged)

        # Run a minimal optimization
        p.run_optimization(
            ticker="BTCUSDT",
            timeframe="4h",
            df_enriched=df_enriched,
            n_trials=2,
            start_date="20250101",
            end_date="20251111"
        )

        # 2. Aggregation Check
        _logger.info("Verifying CSV Aggregation...")
        aggregate_results()

        # 3. Artifact Validation
        res_dir = p.get_result_dir("BTCUSDT", "4h", "20250101", "20251111")
        expected_files = ["best_model.json", "metrics.json", "trades.json", "strategy_overlay.png"]

        missing = [f for f in expected_files if not (res_dir / f).exists()]
        if missing:
            _logger.error("Missing artifacts in %s: %s", res_dir, missing)
        else:
            _logger.info("All reporting artifacts verified successfully in %s", res_dir)

    except Exception as e:
        _logger.exception("Reporting test failed: %s", e)

if __name__ == "__main__":
    test_reporting_functionality()
