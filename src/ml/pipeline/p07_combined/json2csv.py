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

        # Use the folder name as prefix for the CSV
        prefix = root.name
        output_path = root / f"{prefix}_aggregated_results.csv"
        df.to_csv(output_path, index=False)
        print(f"Aggregated {len(all_metrics)} results to {output_path}")

        # --- New: Select candidates for robustness ---
        try:
            from src.ml.pipeline.p07_combined.select_candidates import select_top_candidates
            select_top_candidates(results_root=results_root)
        except Exception as e:
            print(f"Failed to select candidates: {e}")
    else:
        print("No metrics found to aggregate.")

def aggregate_robustness(results_root: str = "results/p07_combined"):
    """
    Scans for robustness_summary.json files and aggregates them into flattened CSVs.
    """
    summary_data = []
    sensitivity_data = []
    wfa_data = []
    root = Path(results_root)

    for rob_file in root.glob("**/robustness_summary.json"):
        try:
            # parts mapping: .../ticker/timeframe/robustness/robustness_summary.json
            parts = rob_file.parts
            if len(parts) < 4: continue

            ticker = parts[-4]
            timeframe = parts[-3]

            with open(rob_file, 'r') as f:
                data = json.load(f)

            # 1. Summary
            summary_data.append({
                "ticker": ticker,
                "timeframe": timeframe,
                "avg_oos_sharpe": data.get("wfa", {}).get("avg_oos_sharpe"),
                "mc_positivity_rate": data.get("monte_carlo", {}).get("positivity_rate")
            })

            # 2. Sensitivity
            for res in data.get("sensitivity", {}).get("sensitivity_results", []):
                sensitivity_data.append({
                    "ticker": ticker,
                    "timeframe": timeframe,
                    "perturbation": res.get("perturbation"),
                    "sharpe": res.get("sharpe"),
                    "total_return": res.get("return")
                })

            # 3. WFA (if available in summary or from nested artifacts)
            # The summary.json might only have avg, but we want detail.
            # If wfa results were saved individually or nested in the summary:
            # Note: Current robustness.py saves oos_results in a way that needs coordination.
            # Assuming we can find details or they are in the JSON.
            for res in data.get("wfa_details", []): # I will update robustness.py to include this
                wfa_data.append({
                    "ticker": ticker,
                    "timeframe": timeframe,
                    "window": res.get("window"),
                    "sharpe": res.get("sharpe"),
                    "total_return": res.get("return")
                })

        except Exception as e:
            print(f"Error processing robustness {rob_file}: {e}")

    # Write files
    prefix = root.name
    if summary_data:
        pd.DataFrame(summary_data).to_csv(root / f"{prefix}_robustness_summary.csv", index=False)
        print(f"Saved {root / f'{prefix}_robustness_summary.csv'}")

    if sensitivity_data:
        pd.DataFrame(sensitivity_data).to_csv(root / f"{prefix}_robustness_sensitivity.csv", index=False)
        print(f"Saved {root / f'{prefix}_robustness_sensitivity.csv'}")

    if wfa_data:
        pd.DataFrame(wfa_data).to_csv(root / f"{prefix}_robustness_wfa.csv", index=False)
        print(f"Saved {root / f'{prefix}_robustness_wfa.csv'}")

if __name__ == "__main__":
    aggregate_results()
    aggregate_robustness()
