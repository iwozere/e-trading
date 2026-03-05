import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p07_combined.pipeline import P07Pipeline
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

def main():
    pipeline = P07Pipeline()

    strategies = [
        {"ticker": "ETHUSDT", "timeframe": "30m"},
        {"ticker": "ETHUSDT", "timeframe": "4h"}
    ]

    for strategy in strategies:
        ticker = strategy["ticker"]
        timeframe = strategy["timeframe"]
        _logger.info(f"=== Starting Robustness Check for {ticker} {timeframe} ===")
        try:
            pipeline.run_robustness(ticker, timeframe)
            _logger.info(f"=== Completed Robustness Check for {ticker} {timeframe} ===")
        except Exception as e:
            _logger.error(f"Failed robustness check for {ticker} {timeframe}: {e}", exc_info=True)

if __name__ == "__main__":
    main()
