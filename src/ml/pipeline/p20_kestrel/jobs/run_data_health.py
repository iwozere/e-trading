"""P20 Kestrel — scheduler entry point: data health check."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.reporting.data_health import run
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def main() -> None:
    """Run data-health guard and print scheduler result."""
    result = run()
    _logger.info("Data health complete: %s", result)
    print(f"__SCHEDULER_RESULT__:{json.dumps(result, default=str)}")


if __name__ == "__main__":
    main()
