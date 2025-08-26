"""
Configurable HMM Training and Evaluation Pipeline Runner.

This script automates the process of training and evaluating Hidden Markov
Models (HMMs). It acts as a master controller, calling other specialized scripts
in sequence for each input file found in the data directory.

This version is configurable via command-line arguments, allowing the user to
easily change training parameters like the number of optimization trials, the
backend library, and the features used across all runs.

Workflow:
1.  Parses command-line arguments to configure the pipeline run.
2.  Creates the 'results' output directories.
3.  Finds all CSV files in the `data/` directory.
4.  For each CSV file, it performs a two-step process, handling errors gracefully:
    a. **Training:** It dynamically constructs and calls `src/ml/hmm/x_02_train_hmm.py`
       with the parameters specified in the command line.
    b. **Evaluation:** It passes the resulting CSV to `src/ml/hmm/x_03_evaluate_hmm.py`
       to generate a visualization plot.

Assumed Project Structure:
    (Same as Version 1)

Usage:
    # Run with default settings (pomegranate backend, 50 trials)
    python run_pipeline.py

    # Run a quick test with fewer trials and a different backend
    python run_pipeline.py --n_trials 10 --backend gaussian

    # Run with a minimal feature set
    python run_pipeline.py --features log_return volatility
"""

import sys
import subprocess
from glob import glob
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[3] # Go up 3 levels from 'src/ml/hmm'
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

def main(args):
    """
    Main function to orchestrate the HMM training and evaluation pipeline.

    It finds all data files, then loops through them, executing the training
    and evaluation scripts with parameters provided via command-line arguments.
    """
    # Use pathlib for modern path handling
    output_dir_results = Path('results')
    output_dir_results.mkdir(exist_ok=True)

    csv_files = glob('data/*.csv')
    if not csv_files:
        _logger.error("Error: No CSV files found in the 'data/' directory.")
        return

    _logger.info("Found %s files to process.")
    _logger.info("Running with Backend: %s, Trials: %s, Features: %s")

    for csv_path_str in csv_files:
        csv_path = Path(csv_path_str)
        symbol_tf = csv_path.stem  # pathlib's clean way to get filename without extension

        _logger.info("\n--- Processing %s ---")

        try:
            # Step 1: Construct and run the training command dynamically
            _logger.info("  -> Running training for %s...")
            train_cmd = [
                "python", "src/ml/hmm/x_02_train_hmm.py",
                "--csv", str(csv_path),
                "--timeframe", symbol_tf,
                "--optimize",
                "--n_trials", str(args.n_trials),
                "--backend", args.backend,
                "--features", *args.features  # Unpack the list of features
            ]
            subprocess.run(train_cmd, check=True, capture_output=True, text=True)

            # Step 2: Evaluate the results
            _logger.info("  -> Running evaluation for %s...")
            result_csv_path = output_dir_results / f"HMM_{symbol_tf}.csv"
            eval_cmd = [
                "python", "src/ml/hmm/x_03_evaluate_hmm.py",
                "--csv", str(result_csv_path)
            ]
            subprocess.run(eval_cmd, check=True, capture_output=True, text=True)

        except subprocess.CalledProcessError as e:
            # Makes the pipeline robust: if one file fails, it reports and continues
            _logger.error("  -> ERROR processing %s.", csv_path.name)
            _logger.error("  -> Return Code: %s", e.returncode)
            _logger.error("  -> STDOUT: %s", e.stdout)
            _logger.error("  -> STDERR: %s", e.stderr)
            continue # Move to the next file

    _logger.info("\n--- Pipeline finished! ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the HMM training and evaluation pipeline.")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials per file.")
    parser.add_argument("--backend", choices=["gaussian", "pomegranate"], default="gaussian", help="HMM library backend.")
    parser.add_argument("--features", nargs="*", default=["log_return", "volatility", "rsi", "macd", "boll"], help="List of features to use.")

    cli_args = parser.parse_args()
    main(cli_args)
