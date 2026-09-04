"""P20 Kestrel — scheduler entry point: data health check."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline.dependency_status import deferred_result, require_dependencies_or_defer
from src.ml.pipeline.p20_kestrel.reporting.data_health import run
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


from src.ml.pipeline.p20_kestrel.jobs.run_common import setup_run_logging


def main() -> None:
    """Run data-health guard and print scheduler result."""
    setup_run_logging()
    ready, statuses = require_dependencies_or_defer("P20 Data Health Check")
    if ready:
        result = run()
        _logger.info("Data health complete: %s", result)
    else:
        result = deferred_result(statuses)
    print(f"__SCHEDULER_RESULT__:{json.dumps(result, default=str)}")


if __name__ == "__main__":
    main()
