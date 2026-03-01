import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p07_combined.labeling import get_triple_barrier_labels

def run_label_diagnostics():
    data_dir = Path("data")
    # Sample files for different timeframes
    timeframes = ["5m", "15m", "30m", "1h", "4h"]
    ticker = "LTCUSDT"
    year = "2020"

    # Standard params from your Optuna space ends or current "successful" runs
    test_params = [
        {'pt_mult': 2.0, 'sl_mult': 1.0, 'tpl_bars': 12},  # old baseline
        {'pt_mult': 1.0, 'sl_mult': 0.5, 'tpl_bars': 96},  # new potential (long duration, tight barriers)
        {'pt_mult': 0.5, 'sl_mult': 0.25, 'tpl_bars': 24}, # new potential (short duration, very tight barriers)
    ]

    for tf in timeframes:
        pattern = f"{ticker}_{tf}_{year}0101_{year}1231.csv"
        matches = list(data_dir.glob(pattern))
        if not matches:
            print(f"Skipping {tf}: No data file found matching {pattern}")
            continue

        filepath = matches[0]
        print(f"\n--- Analyzing {tf} ({filepath.name}) ---")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)

        for p in test_params:
            labels = get_triple_barrier_labels(
                df,
                pt_mult=p['pt_mult'],
                sl_mult=p['sl_mult'],
                tpl_bars=p['tpl_bars']
            )
            counts = labels.value_counts()
            total = len(labels)
            dist = {k: f"{(v/total)*100:.1f}%" for k, v in counts.items()}
            print(f"Params {p}: {dist} (Total: {total})")

if __name__ == "__main__":
    run_label_diagnostics()
