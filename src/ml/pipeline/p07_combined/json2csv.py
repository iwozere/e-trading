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

            # --- Formatting Logic ---
            # 1. Start/End (from milliseconds to human readable)
            for key in ["Start", "End"]:
                if key in data and isinstance(data[key], (int, float)):
                    # Convert ms to datetime
                    dt = pd.to_datetime(data[key], unit='ms')
                    data[key] = dt.strftime('%Y-%m-%d %H:%M:%S')

            # 2. Period (from milliseconds to duration string)
            if "Period" in data and isinstance(data["Period"], (int, float)):
                duration = pd.to_timedelta(data["Period"], unit='ms')
                # Format: "X days HH:MM:SS"
                days = duration.days
                hours, remainder = divmod(duration.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                data["Period"] = f"{days} days {hours:02d}:{minutes:02d}:{seconds:02d}"

            # 3. Handle ticker/timeframe/dates
            # file_start/file_end come from the folder name (e.g. 20200101_20250101)
            f_start, f_end = start_date, end_date
            if len(f_start) == 8: f_start = f"{f_start[:4]}-{f_start[4:6]}-{f_start[6:8]}"
            if len(f_end) == 8: f_end = f"{f_end[:4]}-{f_end[4:6]}-{f_end[6:8]}"

            # test_start/test_end come from the JSON (high precision)
            t_start = data.get("Start", f_start)
            t_end = data.get("End", f_end)

            # Add metadata and combine with data
            entry = {
                "ticker": ticker,
                "timeframe": timeframe,
                "file_start": f_start,
                "file_end": f_end,
                "test_start": t_start,
                "test_end": t_end,
                **{k: v for k, v in data.items() if k not in ["Start", "End"]}
            }
            all_metrics.append(entry)

        except Exception as e:
            print(f"Error processing {metrics_file}: {e}")

    if all_metrics:
        df = pd.DataFrame(all_metrics)
        # Reorder to put metadata first
        cols = ["ticker", "timeframe", "file_start", "file_end", "test_start", "test_end", "Period"]
        other_cols = [c for c in df.columns if c not in cols]
        df = df[cols + other_cols]

        output_path = root / "p07_aggregated_results.csv"
        df.to_csv(output_path, index=False)
        print(f"Aggregated {len(all_metrics)} results to {output_path}")
    else:
        print("No metrics found to aggregate.")

if __name__ == "__main__":
    aggregate_results()
