# File: scripts/run_all.py

import os
import subprocess
from glob import glob


def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    csv_files = glob('data/*.csv')

    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        symbol_tf = os.path.splitext(filename)[0]  # e.g., LTCUSDT_1h_20220101_20250707

        print(f"\n--- Processing {filename} ---")

        # Train with optimization and all features using pomegranate backend
        train_cmd = [
            "python", "scripts/train_hmm.py",
            "--csv", csv_path,
            "--timeframe", symbol_tf,
            "--optimize",
            "--n_trials", "50",
            "--features", "log_return", "volatility", "rsi", "macd", "boll",
            "--backend", "pomegranate"
        ]
        subprocess.run(train_cmd, check=True)

        # Evaluate and plot
        result_csv = f"results/{symbol_tf}.csv"
        eval_cmd = [
            "python", "scripts/evaluate_hmm.py",
            "--csv", result_csv
        ]
        subprocess.run(eval_cmd, check=True)


if __name__ == '__main__':
    main()
