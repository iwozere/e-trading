"""P20 Kestrel — scheduler entry point: daily digest send."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.reporting.daily_digest import run
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


from src.ml.pipeline.p20_kestrel.jobs.run_common import setup_run_logging


def main() -> None:
    """Build and send daily digest, print scheduler result."""
    setup_run_logging()
    result = run()
    _logger.info("Digest send complete: %s", result)
    print(f"__SCHEDULER_RESULT__:{json.dumps(result, default=str)}")


if __name__ == "__main__":
    main()
