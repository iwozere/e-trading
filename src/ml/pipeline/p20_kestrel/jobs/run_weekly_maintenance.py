"""P20 Kestrel — scheduler entry point: weekly maintenance (universe + alias refresh)."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.ingest.universe_loader import run as run_universe
from src.ml.pipeline.p20_kestrel.sentiment.alias_builder import run as run_aliases
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


from src.ml.pipeline.p20_kestrel.jobs.run_common import setup_run_logging


def main() -> None:
    """Run universe refresh then alias rebuild, print combined scheduler result."""
    setup_run_logging()
    universe_result = run_universe()
    _logger.info("Universe refresh complete: %s", universe_result)

    alias_result = run_aliases()
    _logger.info("Alias refresh complete: %s", alias_result)

    result = {
        "universe": universe_result,
        "aliases": alias_result,
        "success": universe_result.get("success", True) and alias_result.get("success", True),
    }
    print(f"__SCHEDULER_RESULT__:{json.dumps(result, default=str)}")


if __name__ == "__main__":
    main()
