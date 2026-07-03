"""P20 Kestrel — scheduler entry point: Sleeve C momentum RS ranking."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.screening.sleeve_c import run
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def main() -> None:
    """Run Sleeve C RS rank and print scheduler result."""
    result = run()
    _logger.info("Momentum rank complete: %s", result)
    print(f"__SCHEDULER_RESULT__:{json.dumps(result, default=str)}")


if __name__ == "__main__":
    main()
