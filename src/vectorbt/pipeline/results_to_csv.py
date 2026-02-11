import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """Recursively flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def aggregate_results(results_dir: str = "results/vectorbt"):
    """
    Recursively find all trial_X.json files and aggregate them into a CSV.
    """
    base_path = Path(results_dir)
    all_data = []

    print(f"🔍 Searching for trial results in {base_path.absolute()}...")

    # Pattern: results/vectorbt/<strategy_name>/<symbol>/<interval>/trial_<id>.json
    for json_file in base_path.glob("**/trial_*.json"):
        # We handle both historical flat reports and the new hierarchical structure
        parts = json_file.relative_to(base_path).parts

        strategy_name = "unknown"
        symbol = "unknown"
        interval = "unknown"

        if len(parts) >= 4:
            # New structure: strategy_name/symbol/interval/trial_X.json
            strategy_name = parts[0]
            symbol = parts[1]
            interval = parts[2]
        elif len(parts) == 3:
            # Old structure: symbol/interval/trial_X.json
            symbol = parts[0]
            interval = parts[1]

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # If strategy_name is in the JSON (newly generated), use that
            if 'strategy_name' in data:
                strategy_name = data['strategy_name']

            # Remove 'trades' list as it's too large for a flat CSV cell
            data.pop('trades', None)

            # Flatten metrics and params
            flat_data = flatten_dict(data)

            # Add metadata (override if already in JSON but ensure it reflects folder structure too)
            flat_data['strategy_name'] = strategy_name
            flat_data['symbol'] = symbol
            flat_data['interval'] = interval
            flat_data['source_file'] = str(json_file)

            all_data.append(flat_data)
        except Exception as e:
            print(f"⚠️ Error processing {json_file}: {e}")

    if not all_data:
        print("❌ No trial results found.")
        return

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Reorder columns to put metadata first
    cols = ['strategy_name', 'symbol', 'interval', 'trial_id']
    actual_cols = [c for c in cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in actual_cols]
    df = df[actual_cols + other_cols]

    output_path = base_path / "comparison_report.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Aggregated {len(all_data)} trials into {output_path.absolute()}")

if __name__ == "__main__":
    aggregate_results()
