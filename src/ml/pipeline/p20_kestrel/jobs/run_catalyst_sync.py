"""P20 Kestrel — scheduler entry point: catalyst calendar sync."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.ingest.calendar_sync import run
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


from src.ml.pipeline.p20_kestrel.jobs.run_common import setup_run_logging


def main() -> None:
    """Run catalyst sync and print scheduler result."""
    setup_run_logging()
    result = run()
    _logger.info("Catalyst sync complete: %s", result)
    print(f"__SCHEDULER_RESULT__:{json.dumps(result, default=str)}")


if __name__ == "__main__":
    main()
