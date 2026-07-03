"""P20 Kestrel — scheduler entry point: LLM 8-K filing classification."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.llm.classifier_8k import run
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def main() -> None:
    """Classify queued 8-K filings via LLM and print scheduler result."""
    result = run()
    _logger.info("LLM classify filings complete: %s", result)
    print(f"__SCHEDULER_RESULT__:{json.dumps(result, default=str)}")


if __name__ == "__main__":
    main()
