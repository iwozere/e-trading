import sys
from pathlib import Path

import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p07_combined.pipeline import P07Pipeline
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def main():
    pipeline = P07Pipeline()

    candidates_path = PROJECT_ROOT / "results" / "p07_combined" / "p07_robustness_candidates.csv"

    if not candidates_path.exists():
        _logger.error(f"Candidates file not found: {candidates_path}")
        return

    _logger.info(f"Loading robustness candidates from {candidates_path}")
    df = pd.read_csv(candidates_path)

    if df.empty:
        _logger.warning("No candidates found in the CSV.")
        return

    for _, row in df.iterrows():
        ticker = row["ticker"]
        timeframe = row["timeframe"]
        _logger.info(f"=== Starting Robustness Check for {ticker} {timeframe} ===")
        try:
            pipeline.run_robustness(ticker, timeframe)
            _logger.info(f"=== Completed Robustness Check for {ticker} {timeframe} ===")
        except Exception as e:
            _logger.error(f"Failed robustness check for {ticker} {timeframe}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
