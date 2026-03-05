from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p08_mtf.pipeline import P08Pipeline
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

def run_p08_trial():
    print("\n--- Starting P08 MTF Trial Run (ETHUSDT 30m) ---")

    # Initialize pipeline with a test database
    p = P08Pipeline(db_url="sqlite:///src/ml/pipeline/p08_mtf/test_optuna.db")

    # Select only ETHUSDT 30m files for 2020-2022 to keep it fast
    data_dir = Path("data")
    test_files = [
        data_dir / "ETHUSDT_30m_20200101_20201231.csv",
        data_dir / "ETHUSDT_30m_20210101_20211231.csv",
        data_dir / "ETHUSDT_30m_20220101_20221231.csv",
        # Need anchor files too (4h)
        data_dir / "ETHUSDT_4h_20200101_20201231.csv",
        data_dir / "ETHUSDT_4h_20210101_20211231.csv",
        data_dir / "ETHUSDT_4h_20220101_20221231.csv"
    ]

    # Filter to only execution files
    exec_files = [f for f in test_files if "30m" in f.name]

    # Run a small batch (10 trials per optimization)
    # We'll temporarily override n_trials in run_optimization logic if needed,
    # but let's just use the default or pass it if the class allows.

    _logger.info("Running trial batch for %d files...", len(exec_files))
    p.run_batch(exec_files)

    print("\n✅ Trial run completed. Check results/p08_mtf/ETHUSDT/30m/ for artifacts.")

if __name__ == "__main__":
    run_p08_trial()
