import os
import json
import pandas as pd
from pathlib import Path
import re

def aggregate_results(results_root: str = "results/p07_combined"):
    """
    Scans results directory for metrics.json files and aggregates them into a CSV.
    Structure: results/p07_combined/<ticker>/<timeframe>/<start_end>/metrics.json
    """
    all_metrics = []
    root = Path(results_root)

    if not root.exists():
        print(f"Root directory {results_root} not found.")
        return

    # Pattern: ticker / timeframe / start_end / metrics.json
    for metrics_file in root.glob("**/metrics.json"):
        try:
            # Extract metadata from path
            # parts[-2] = start_end, parts[-3] = timeframe, parts[-4] = ticker
            parts = metrics_file.parts
            if len(parts) < 4:
                continue

            ticker = parts[-4]
            timeframe = parts[-3]
            period = parts[-2]

            # Split period if it matches YYYYMMDD_YYYYMMDD
            start_date, end_date = period, period
            if "_" in period:
                start_date, end_date = period.split("_", 1)

            with open(metrics_file, 'r') as f:
                data = json.load(f)

            # Add metadata
            entry = {
                "ticker": ticker,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
                **data
            }
            all_metrics.append(entry)

        except Exception as e:
            print(f"Error processing {metrics_file}: {e}")

    if all_metrics:
        df = pd.DataFrame(all_metrics)
        # Reorder to put metadata first
        cols = ["ticker", "timeframe", "start_date", "end_date"]
        other_cols = [c for c in df.columns if c not in cols]
        df = df[cols + other_cols]

        output_path = root / "p07_aggregated_results.csv"
        df.to_csv(output_path, index=False)
        print(f"Aggregated {len(all_metrics)} results to {output_path}")
    else:
        print("No metrics found to aggregate.")

if __name__ == "__main__":
    aggregate_results()
