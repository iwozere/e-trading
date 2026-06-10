"""
P00 HMM-3LSTM — deprecated.

Per-regime LSTM training has been merged into P01 (p01_hmm_lstm) as a
configurable option.  Set lstm.multi_regime: true in config/pipeline/p01.yaml
and run P01 instead:

    python src/ml/pipeline/p01_hmm_lstm/run_pipeline.py

This shim is kept for backward-compatibility during the deprecation window.
Remove once no scheduler or caller references this entrypoint directly.
"""

import warnings
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_DEPRECATION_MSG = (
    "P00 (p00_hmm_3lstm) is deprecated. "
    "Use P01 (p01_hmm_lstm) with lstm.multi_regime: true in config/pipeline/p01.yaml instead."
)

warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=1)
_logger.warning(_DEPRECATION_MSG)

if __name__ == "__main__":
    _logger.info("Redirecting to P01 pipeline runner with multi_regime flag...")
    import subprocess
    result = subprocess.run(
        [sys.executable,
         str(PROJECT_ROOT / "src" / "ml" / "pipeline" / "p01_hmm_lstm" / "run_pipeline.py")] +
        sys.argv[1:],
        check=False
    )
    sys.exit(result.returncode)
