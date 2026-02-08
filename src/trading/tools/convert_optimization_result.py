#!/usr/bin/env python3
"""
Optimization Result Converter
----------------------------

This utility converts optimization result JSON files into Strategy Instance JSONs
compatible with the modular configuration architecture.

Features:
- Reads `best_params` from optimization results.
- Automatically adds `e_` and `x_` prefixes to parameters.
- Generates descriptive filenames (Logic + Symbol + Timeframe).
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, Any, Optional

def clean_mixin_name(name: str) -> str:
    """
    Convert CamelCaseMixinName to kebab-case-name without 'Mixin' suffix.
    Example: RSIOrBBEntryMixin -> rsi-or-bb
    """
    # Remove 'Mixin' suffix if present
    name = name.replace("Mixin", "")

    # Remove 'Entry' or 'Exit' suffix if present to keep it short
    name = name.replace("Entry", "").replace("Exit", "")

    # Convert CamelCase to kebab-case
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1-\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1-\2', s1).lower()

def transform_params(params: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """
    Add prefix to keys if not already present.
    Also round float values to 5 decimal places for readability.
    """
    new_params = {}
    for key, value in params.items():
        if not key.startswith(prefix):
            new_key = f"{prefix}{key}"
        else:
            new_key = key

        # Round float values
        if isinstance(value, float):
            new_params[new_key] = round(value, 5)
        else:
            new_params[new_key] = value

    return new_params

def generate_filename(data: Dict[str, Any], entry_name: str, exit_name: str) -> str:
    """
    Generate a descriptive filename based on strategy contents.
    Format: strategy-{entry}+{exit}-{symbol}-{timeframe}.json
    """
    # Extract symbol and timeframe from filename in data (if available) or parsing filename
    # The optimization file typically has `data_file`: "LTCUSDT_30m_..."

    symbol = "UNKNOWN"
    timeframe = "UNKNOWN"

    data_file = data.get("data_file", "")
    if data_file:
        parts = data_file.split("_")
        if len(parts) >= 2:
            symbol = parts[0]
            timeframe = parts[1]

    entry_short = clean_mixin_name(entry_name)
    exit_short = clean_mixin_name(exit_name)

    return f"strategy-{entry_short}+{exit_short}-{symbol}-{timeframe}.json"

def convert_optimization_result(input_path: str, output_dir: str, manual_name: Optional[str] = None):
    try:
        input_file = Path(input_path)
        if not input_file.exists():
            print(f"Error: Input file not found: {input_path}")
            return

        with open(input_file, 'r') as f:
            data = json.load(f)

        best_params = data.get("best_params", {})
        if not best_params:
            print("Error: No 'best_params' found in input file.")
            return

        entry_logic = best_params.get("entry_logic", {})
        exit_logic = best_params.get("exit_logic", {})

        entry_name = entry_logic.get("name", "UnknownEntry")
        exit_name = exit_logic.get("name", "UnknownExit")

        # Transform parameters
        entry_params = transform_params(entry_logic.get("params", {}), "e_")
        exit_params = transform_params(exit_logic.get("params", {}), "x_")

        # Construct new strategy object
        strategy_config = {
            "strategy": {
                "type": "CustomStrategy",
                "parameters": {
                    "entry_logic": {
                        "name": entry_name,
                        "params": entry_params
                    },
                    "exit_logic": {
                        "name": exit_name,
                        "params": exit_params
                    },
                    "use_talib": best_params.get("use_talib", False),
                    "position_size": best_params.get("position_size", 0.1)
                }
            }
        }

        # Determine output filename
        if manual_name:
            filename = manual_name
            if not filename.endswith(".json"):
                filename += ".json"
        else:
            filename = generate_filename(data, entry_name, exit_name)

        output_path = Path(output_dir) / filename

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(strategy_config, f, indent=4)

        print(f"✅ Successfully created strategy config: {output_path}")

    except Exception as e:
        print(f"❌ Error converting file: {e}")






# You can paste the path to your input file here to run the script directly
DEFAULT_INPUT_FILE = r"results/optimization-jan-2026/4h/LTCUSDT_4h_20250101_20251111_RSIBBEntryMixin_MACrossoverExitMixin_20260127_154136.json"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert optimization results to strategy config")
    # Make input_file optional in CLI
    parser.add_argument("input_file", nargs='?', help="Path to optimization result JSON file")
    parser.add_argument("--output-dir", default="config/contracts/instances/strategies", help="Output directory")
    parser.add_argument("--name", help="Manual output filename (optional)")

    args = parser.parse_args()

    input_path = args.input_file

    # Fallback to DEFAULT_INPUT_FILE if not provided via CLI
    if not input_path:
        if DEFAULT_INPUT_FILE:
            input_path = DEFAULT_INPUT_FILE
            print(f"ℹ️  Using default input file from code: {input_path}")
        else:
            parser.error("input_file is required (either via CLI or DEFAULT_INPUT_FILE in code)")

    convert_optimization_result(input_path, args.output_dir, args.name)
