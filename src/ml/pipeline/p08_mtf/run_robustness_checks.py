import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p08_mtf.pipeline import P08Pipeline
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def run_robustness_batch(candidates_path: Path):
    """Refactored batch logic for automation integration."""
    pipeline = P08Pipeline()
    if not candidates_path.exists():
        _logger.error(f"Candidates file not found: {candidates_path}")
        return

    _logger.info(f"Loading robustness candidates from {candidates_path}")
    df = pd.read_csv(candidates_path)
    if df.empty:
        _logger.warning("No candidates found in the CSV.")
        return

    candidates = df.to_dict("records")

    for cand in candidates:
        ticker = cand["ticker"]
        timeframe = cand["timeframe"]
        _logger.info(f"=== Starting Robustness Check for {ticker} {timeframe} ===")
        try:
            pipeline.run_robustness(ticker, timeframe)
            _logger.info(f"=== Completed Robustness Check for {ticker} {timeframe} ===")
        except Exception as e:
            _logger.error(f"Failed robustness check for {ticker} {timeframe}: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="P08 Robustness Suite Runner")
    parser.add_argument("--ticker", type=str, help="Ticker to check (e.g. ETHUSDT)")
    parser.add_argument("--tf", type=str, help="Timeframe to check (e.g. 30m)")
    parser.add_argument(
        "--candidates", default="results/p08_mtf/p08_robustness_candidates.csv", type=str, help="Path to candidates CSV"
    )

    args = parser.parse_args()

    if args.ticker and args.tf:
        pipeline = P08Pipeline()
        _logger.info(f"=== Starting Single Robustness Check for {args.ticker} {args.tf} ===")
        pipeline.run_robustness(args.ticker, args.tf)
    else:
        candidates_path = PROJECT_ROOT / args.candidates
        run_robustness_batch(candidates_path)


if __name__ == "__main__":
    main()
