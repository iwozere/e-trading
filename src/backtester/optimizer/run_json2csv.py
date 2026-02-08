"""
Process Results Module
---------------------

This script processes optimization result JSON files in the 'results' directory, extracts relevant trading metrics and parameters, and outputs a summary CSV for further analysis and reporting. It is used to aggregate and summarize the results of strategy optimization runs for easier comparison and visualization.

Main Features:
- Parse and extract key metrics from optimization result files (name like BTCUSDT_1h_20230101_20250501_RSIBBVolumeEntryMixin_RSIBBExitMixin_20250621_095351.json, stored in results/ directory)
  and data files (name like BTCUSDT_1h_20230101_20250501.csv, stored in data/ directory)
- Summarize results into a single CSV file for analysis
- Support for multiple strategies, symbols, and intervals

Functions:
- extract_symbol_interval_dates(filename): Extract metadata from result filenames
- process_json_file(file_path): Parse and extract metrics from a single result file
- main(): Process all result files and generate a summary CSV

Output fields:
- json_filename: filename of the result file
- data_filename: filename of the data file, taken from json_filename, "data_file" field
- total_trades: total trades, taken from result file, "total_trades" field
- total_profit: gross profit, taken from result file, "total_profit" field
- total_profit_with_commission: net profit with commission, taken from result file, "total_profit_with_commission" field
- total_commission: total commission, taken from result file, "total_commission" field
- symbol: symbol, taken from data filename
- interval: interval, taken from data filename
- data_start_date: start date, taken from data filename
- data_end_date: end date, taken from data filename
- entry_logic_name: entry logic name, taken from result file, "best_params" field, "entry_logic" field, "name" field
- exit_logic_name: exit logic name, taken from result file, "best_params" field, "exit_logic" field, "name" field
- win_rate: win rate, taken from result file, "analyzers" field, "winrate" field, "win_rate" field
- profit_factor: profit factor, taken from result file, "analyzers" field, "profit_factor" field, "profit_factor" field
- max_drawdown_pct: max drawdown percentage, taken from result file, "analyzers" field, "drawdown" field, "drawdown" field
- sharpe_ratio: sharpe ratio, taken from result file, "analyzers" field, "sharpe" field, "sharperatio" field
- sqn: sqn, taken from result file, "analyzers" field, "sqn" field, "sqn" field
- cagr: cagr, taken from result file, "analyzers" field, "cagr" field, "cagr" field
- calmar_ratio: calmar ratio, taken from result file, "analyzers" field, "calmar" field, "calmar" field
- sortino_ratio: sortino ratio, taken from result file, "analyzers" field, "sortino" field, "sortino" field
- volatility: portfolio volatility, taken from result file, "analyzers" field, "portfoliovolatility" field, "volatility" field
- vwr: variabilty-weighted return, taken from result file, "analyzers" field, "vwr" field, "vwr" field
- max_consecutive_wins: max consecutive wins, taken from result file, "analyzers" field, "consecutivewinslosses" field, "max_consecutive_wins" field
- max_consecutive_losses: max consecutive losses, taken from result file, "analyzers" field, "consecutivewinslosses" field, "max_consecutive_losses" field
- avg_trade_length: average trade length, taken from result file, "analyzers" field, "trades" field, "len" field, "average" field
- total_return: total return, taken from result file, "analyzers" field, "returns" field, "rtot" field
- avg_return: average return, taken from result file, "analyzers" field, "returns" field, "ravg" field
- normalized_return: normalized return, taken from result file, "analyzers" field, "returns" field, "rnorm" field
- position_size: position size, taken from result file, "best_params" field, "position_size" field
"""

import sys
import json
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))


def extract_symbol_interval_dates(filename):
    """Extract symbol, interval, start_date, end_date from filename like BTCUSDT_1h_20230101_20250501_RSIBBVolumeEntryMixin_RSIBBExitMixin_20250621_095351.json"""
    base = os.path.basename(filename)
    # Remove the .json extension
    if base.endswith(".json"):
        base = base[:-5]

    parts = base.split("_")
    if len(parts) >= 4:
        symbol = parts[0]
        interval = parts[1]
        start_date = parts[2]
        end_date = parts[3]
        return symbol, interval, start_date, end_date
    else:
        return None, None, None, None


def extract_nested_value(data, keys, default=None):
    """Extract value from nested dictionary using a list of keys"""
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def process_json_file(file_path):
    """Process a single JSON file and extract relevant information"""
    with open(file_path, "r") as f:
        data = json.load(f)

    # Extract symbol, interval, start_date, end_date from filename
    symbol, interval, start_date, end_date = extract_symbol_interval_dates(file_path)
    if not all([symbol, interval, start_date, end_date]):
        print(f"Warning: Could not extract symbol/interval/dates from filename: {file_path}")
        return None

    # Get basic information
    best_params = data.get("best_params", {})
    analyzers = data.get("analyzers", {})

    # Extract entry and exit logic names
    entry_logic_name = extract_nested_value(best_params, ["entry_logic", "name"], "unknown")
    exit_logic_name = extract_nested_value(best_params, ["exit_logic", "name"], "unknown")

    # Extract position size
    position_size = best_params.get("position_size", 0.1)

    # Extract metrics from analyzers
    win_rate = extract_nested_value(analyzers, ["winrate", "win_rate"], None)
    profit_factor = extract_nested_value(analyzers, ["profit_factor", "profit_factor"], None)
    max_drawdown_pct = extract_nested_value(analyzers, ["drawdown", "drawdown"], None)
    sharpe_ratio = extract_nested_value(analyzers, ["sharpe", "sharperatio"], None)
    sqn = extract_nested_value(analyzers, ["sqn", "sqn"], None)
    cagr = extract_nested_value(analyzers, ["cagr", "cagr"], None)
    calmar_ratio = extract_nested_value(analyzers, ["calmar", "calmar"], None)
    sortino_ratio = extract_nested_value(analyzers, ["sortino", "sortino"], None)
    volatility = extract_nested_value(analyzers, ["portfoliovolatility", "volatility"], None)
    vwr = extract_nested_value(analyzers, ["vwr", "vwr"], None)
    max_consecutive_wins = extract_nested_value(analyzers, ["consecutivewinslosses", "max_consecutive_wins"], None)
    max_consecutive_losses = extract_nested_value(analyzers, ["consecutivewinslosses", "max_consecutive_losses"], None)
    avg_trade_length = extract_nested_value(analyzers, ["trades", "len", "average"], None)
    total_return = extract_nested_value(analyzers, ["returns", "rtot"], None)
    avg_return = extract_nested_value(analyzers, ["returns", "ravg"], None)
    normalized_return = extract_nested_value(analyzers, ["returns", "rnorm"], None)

    # Create result dictionary
    result = {
        "json_filename": os.path.basename(file_path),
        "data_filename": data.get("data_file", "unknown"),
        "total_trades": data.get("total_trades", 0),
        "total_profit": data.get("total_profit", 0.0),
        "total_profit_with_commission": data.get("total_profit_with_commission", 0.0),
        "total_commission": data.get("total_commission", 0.0),
        "initial_deposit": data.get("initial_deposit"),
        "symbol": symbol,
        "interval": interval,
        "data_start_date": start_date,
        "data_end_date": end_date,
        "entry_logic_name": entry_logic_name,
        "exit_logic_name": exit_logic_name,
        "position_size": position_size,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe_ratio": sharpe_ratio,
        "sqn": sqn,
        "cagr": cagr,
        "calmar_ratio": calmar_ratio,
        "sortino_ratio": sortino_ratio,
        "volatility": volatility,
        "vwr": vwr,
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
        "avg_trade_length": avg_trade_length,
        "total_return": total_return,
        "avg_return": avg_return,
        "normalized_return": normalized_return,
    }

    # Add entry logic parameters
    entry_params = extract_nested_value(best_params, ["entry_logic", "params"], {})
    for param_name, param_value in entry_params.items():
        result[f"entry_{param_name}"] = param_value

    # Add exit logic parameters
    exit_params = extract_nested_value(best_params, ["exit_logic", "params"], {})
    for param_name, param_value in exit_params.items():
        result[f"exit_{param_name}"] = param_value

    return result


def main():
    results_dir = "results"
    all_results = []

    for root, dirs, files in os.walk(results_dir):
        for filename in files:
            if filename.endswith(".json") and not filename.endswith("_summary.csv"):
                file_path = os.path.join(root, filename)
                try:
                    result = process_json_file(file_path)
                    if result is not None:
                        all_results.append(result)
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")

    df = pd.DataFrame(all_results)
    print("Columns in DataFrame:", df.columns.tolist())
    print(f"Processed {len(df)} result files")

    if not df.empty:
        df = df.sort_values(["symbol", "interval", "data_start_date"])
        output_file = os.path.join(results_dir, "optimization_results_summary.csv")
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        print(f"CSV contains {len(df)} rows and {len(df.columns)} columns")
    else:
        print("No valid results to save.")


if __name__ == "__main__":
    main()
