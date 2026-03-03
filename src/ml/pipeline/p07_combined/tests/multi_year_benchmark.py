import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p07_combined.data_loader import P07DataLoader
from src.ml.pipeline.p07_combined.labeling import get_triple_barrier_labels
from src.ml.pipeline.p07_combined.features import P07FeatureEngine
from src.ml.pipeline.p07_combined.evaluator import P07Evaluator

def run_benchmark():
    loader = P07DataLoader()
    ticker = "LTCUSDT"
    timeframe = "15m"

    # Load all files for this ticker/tf
    data_dir = Path("data")
    files = sorted(list(data_dir.glob(f"{ticker}_{timeframe}_*.csv")))

    if not files:
        print(f"No files found for {ticker} {timeframe}")
        return

    print(f"--- Multi-Year Benchmark for {ticker} {timeframe} ---")

    dfs = []
    for f in files:
        df = loader.get_merged_dataset(f)
        dfs.append(df)

    full_df = pd.concat(dfs).sort_index()
    full_df = full_df.loc[~full_df.index.duplicated(keep='last')]

    # Test different tpl_hours
    tpl_test = [4, 12, 24, 48, 96]

    for hours in tpl_test:
        bars = P07Evaluator.hours_to_bars(hours, timeframe)
        labels = get_triple_barrier_labels(
            full_df,
            pt_mult=2.0,
            sl_mult=1.0,
            tpl_bars=bars
        )
        dist = labels.value_counts(normalize=True).to_dict()
        dist_str = {k: f"{v*100:.1f}%" for k, v in dist.items()}
        print(f"tpl_hours {hours}h ({bars} bars): {dist_str}")

    # Feature Analysis
    print("\n--- Feature Stability (Correlation with labels) ---")
    features = P07FeatureEngine.build_features(full_df, {})

    # Re-calculate labels for a standard 24h window
    bars_24 = P07Evaluator.hours_to_bars(24, timeframe)
    labels_24 = get_triple_barrier_labels(full_df, pt_mult=2.0, sl_mult=1.0, tpl_bars=bars_24)

    common_idx = features.index.intersection(labels_24.index)
    X = features.loc[common_idx]
    y = labels_24.loc[common_idx]

    # Split by year and check correlation
    full_df_with_y = X.copy()
    full_df_with_y['target'] = y

    for year in range(2020, 2026):
        year_str = str(year)
        year_data = full_df_with_y[full_df_with_y.index.year == year]
        if year_data.empty: continue

        # Correlation of log_ret_1 and rsi with target
        corr_ret = year_data['log_ret_1'].corr(year_data['target'])
        corr_rsi = year_data['rsi'].corr(year_data['target'])
        print(f"Year {year_str}: log_ret_1 corr: {corr_ret:.4f}, rsi corr: {corr_rsi:.4f} (Samples: {len(year_data)})")

if __name__ == "__main__":
    run_benchmark()
