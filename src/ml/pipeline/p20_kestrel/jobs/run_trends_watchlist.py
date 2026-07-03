"""P20 Kestrel — scheduler entry point: Google Trends poll for watchlist."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.sentiment.trends_poll import run
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def main() -> None:
    """Poll Google Trends for watchlist tickers and print scheduler result."""
    result = run()
    _logger.info("Trends poll complete: %s", result)
    print(f"__SCHEDULER_RESULT__:{json.dumps(result, default=str)}")


if __name__ == "__main__":
    main()
