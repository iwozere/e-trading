"""
P03 CNN-XGBoost — deprecated.

The simple CNN1D model and walk-forward validation from P03 have been merged
into P02 (p02_cnn_lstm_xgboost) as configurable options.  Use P02 with
cnn_variant: simple in config/pipeline/p02.yaml instead:

    # config/pipeline/p02.yaml
    cnn_variant: simple
    target_strategy: multi

    python src/ml/pipeline/p02_cnn_lstm_xgboost/run_pipeline.py --config config/pipeline/p02.yaml

This shim is kept for backward-compatibility during the deprecation window.
Remove once no scheduler or caller references this entrypoint directly.
"""

import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_DEPRECATION_MSG = (
    "P03 (p03_cnn_xgboost) is deprecated. "
    "Use P02 (p02_cnn_lstm_xgboost) with cnn_variant: simple in config/pipeline/p02.yaml instead."
)

warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=1)
_logger.warning(_DEPRECATION_MSG)

if __name__ == "__main__":
    _logger.info("Redirecting to P02 pipeline runner...")
    import subprocess

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "ml" / "pipeline" / "p02_cnn_lstm_xgboost" / "run_pipeline.py")]
        + sys.argv[1:],
        check=False,
    )
    sys.exit(result.returncode)
