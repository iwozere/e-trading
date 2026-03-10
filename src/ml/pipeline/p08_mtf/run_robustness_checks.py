import sys
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p08_mtf.pipeline import P08Pipeline
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="P08 Robustness Suite Runner")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker to check (e.g. ETHUSDT)")
    parser.add_argument("--tf", type=str, required=True, help="Timeframe to check (e.g. 30m)")

    args = parser.parse_args()

    _logger.info("Starting robustness checks for %s %s", args.ticker, args.tf)

    pipeline = P08Pipeline()
    pipeline.run_robustness(args.ticker, args.tf)

    _logger.info("Checks completed.")

if __name__ == "__main__":
    main()
