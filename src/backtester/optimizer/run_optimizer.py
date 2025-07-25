"""
Run Optimizer Module

This module provides functionality to run optimizations for trading strategies.
It handles:
1. Loading and preparing data
2. Running optimizations for different entry/exit strategy combinations
3. Saving results and plots
4. Managing visualization settings
5. Resume functionality to skip already processed combinations
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import json
from datetime import datetime as dt

import backtrader as bt
import optuna
import pandas as pd
from src.strategy.entry.entry_mixin_factory import ENTRY_MIXIN_REGISTRY
from src.strategy.exit.exit_mixin_factory import EXIT_MIXIN_REGISTRY
from src.notification.logger import setup_logger
from src.backtester.optimizer.custom_optimizer import CustomOptimizer

_logger = setup_logger(__name__)


def check_if_already_processed(data_file, entry_logic_name, exit_logic_name):
    """
    Check if this combination has already been processed.

    Args:
        data_file: Name of the data file
        entry_logic_name: Name of the entry logic mixin
        exit_logic_name: Name of the exit logic mixin

    Returns:
        bool: True if already processed, False otherwise
    """
    # Generate the base filename (without timestamp)
    base_filename = get_result_filename(
        data_file,
        entry_logic_name=entry_logic_name,
        exit_logic_name=exit_logic_name,
        suffix="",
        include_timestamp=False
    )

    # Look for existing files in results directory
    results_dir = "results"
    if not os.path.exists(results_dir):
        return False

    # Check if any file matches the pattern
    for filename in os.listdir(results_dir):
        if filename.startswith(base_filename) and filename.endswith('.json'):
            # Found a matching file, check if it's valid
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    result_data = json.load(f)

                # Check if the file has complete results
                if (result_data.get('trades') is not None and
                    result_data.get('analyzers') is not None and
                    result_data.get('best_params') is not None and
                    len(result_data.get('trades', [])) >= 0):

                    _logger.info("Skipping %s + %s + %s - already processed", data_file, entry_logic_name, exit_logic_name)
                    _logger.info("   Found existing file: %s", filename)
                    return True
                else:
                    _logger.warning("Found existing file %s but it appears incomplete, will reprocess", filename)
                    return False

            except Exception as e:
                _logger.warning("Found existing file %s but it appears corrupted: %s", filename, e)
                continue

    return False


def prepare_data_frame(data_file) -> pd.DataFrame:
    """Load and prepare data from CSV file"""
    # Load and prepare data
    df = pd.read_csv(os.path.join("data", data_file))
    print("Available columns:", df.columns.tolist())

    # Find datetime column
    df["datetime"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("datetime", ascending=True)
    df.set_index("datetime", inplace=True)

    df = df[["open", "high", "low", "close", "volume"]]

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Extract symbol from data file name
    symbol = "UNKNOWN"
    if "_" in data_file:
        parts = data_file.replace(".csv", "").split("_")
        if len(parts) >= 1:
            symbol = parts[0]

    return df

def prepare_data_feed(df : pd.DataFrame, symbol : str):
    """Prepare data feed from pandas dataframe"""
    return bt.feeds.PandasData(
        dataname=df,
        datetime=None,
        open=df.columns.get_loc("open"),
        high=df.columns.get_loc("high"),
        low=df.columns.get_loc("low"),
        close=df.columns.get_loc("close"),
        volume=df.columns.get_loc("volume"),
        openinterest=None,
        name=symbol,  # Set the symbol as the data feed name
    )


def parse_data_file_name(data_file : str) -> dict:
    """Parse data file name and return a dictionary with symbol, interval, start_date, end_date"""
    parts = data_file.replace(".csv", "").split("_")
    return parts

def get_result_filename(
    data_file, entry_logic_name=None, exit_logic_name=None, suffix="", include_timestamp=True
):
    """Generate a standardized filename for results"""
    # Extract symbol, interval, and dates from data_file
    symbol, interval, start_date, end_date = parse_data_file_name(data_file)

    if "_" in data_file:
        parts = data_file.replace(".csv", "").split("_")
        if len(parts) >= 4:  # We expect at least symbol_interval_startdate_enddate
            symbol = parts[0]
            interval = parts[1]
            start_date = parts[2]
            end_date = parts[3]

    # Only include timestamp if requested
    timestamp = ""
    if include_timestamp:
        timestamp = f"_{dt.now().strftime('%Y%m%d_%H%M%S')}"

    # Include strategy names in filename if provided
    strategy_part = ""
    if entry_logic_name and exit_logic_name:
        strategy_part = f"_{entry_logic_name}_{exit_logic_name}"

    return f"{symbol}_{interval}_{start_date}_{end_date}{strategy_part}{timestamp}{suffix}"


def save_results(result, data_file):
    """Save optimization results to a JSON file"""
    try:
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)

        # Generate filename based on data file
        filename = get_result_filename(
            data_file,
            entry_logic_name=result.get("best_params", {}).get("entry_logic", {}).get("name", ""),
            exit_logic_name=result.get("best_params", {}).get("exit_logic", {}).get("name", ""),
            suffix="",
        )

        # Convert trade records to serializable format
        trades = []
        for trade in result.get("trades", []):
            try:
                # Ensure we have all required fields
                if not all(
                    k in trade
                    for k in ["entry_time", "exit_time", "entry_price", "exit_price"]
                ):
                    _logger.warning(
                        f"Skipping trade with missing required fields: {trade}"
                    )
                    continue

                # Convert datetime objects to ISO format strings
                entry_time = trade["entry_time"]
                exit_time = trade["exit_time"]

                if isinstance(entry_time, pd.Timestamp):
                    entry_time = entry_time.isoformat()
                elif isinstance(entry_time, dt):
                    entry_time = entry_time.isoformat()

                if isinstance(exit_time, pd.Timestamp):
                    exit_time = exit_time.isoformat()
                elif isinstance(exit_time, dt):
                    exit_time = exit_time.isoformat()

                # Create serializable trade record
                serializable_trade = {
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "entry_price": float(trade["entry_price"]),
                    "exit_price": float(trade["exit_price"]),
                    "entry_value": float(trade.get("entry_value", 0.0)),
                    "exit_value": float(trade.get("exit_value", 0.0)),
                    "size": float(trade["size"]),
                    "symbol": str(trade["symbol"]),
                    "trade_type": str(trade["trade_type"]),
                    "commission": float(trade["commission"]),
                    "gross_pnl": float(trade["gross_pnl"]),
                    "net_pnl": float(trade["net_pnl"]),
                    "pnl_percentage": float(trade["pnl_percentage"]),
                    "exit_reason": str(trade["exit_reason"]),
                    "status": str(trade["status"]),
                }
                trades.append(serializable_trade)

                # Log trade details for debugging
                _logger.debug(
                    f"Processed trade: Entry={serializable_trade['entry_price']} @ {serializable_trade['entry_time']}, "
                    f"Exit={serializable_trade['exit_price']} @ {serializable_trade['exit_time']}"
                )
            except Exception as e:
                _logger.error("Error processing trade: %s", e, exc_info=True)
                continue

        # Process analyzer results
        analyzers = {}
        for name, analyzer in result.get("analyzers", {}).items():
            try:
                # Handle different analyzer types
                if isinstance(analyzer, dict):
                    # Already a dictionary, just convert values
                    processed_analysis = {}
                    for k, v in analyzer.items():
                        if isinstance(v, (int, float)):
                            processed_analysis[str(k)] = float(v)
                        elif isinstance(v, dt):
                            processed_analysis[str(k)] = v.isoformat()
                        else:
                            processed_analysis[str(k)] = str(v)
                    analyzers[name] = processed_analysis
                elif hasattr(analyzer, "get_analysis"):
                    # Get analysis using get_analysis method
                    analysis = analyzer.get_analysis()
                    if isinstance(analysis, dict):
                        processed_analysis = {}
                        for k, v in analysis.items():
                            if isinstance(v, (int, float)):
                                processed_analysis[str(k)] = float(v)
                            elif isinstance(v, dt):
                                processed_analysis[str(k)] = v.isoformat()
                            else:
                                processed_analysis[str(k)] = str(v)
                        analyzers[name] = processed_analysis
                    elif isinstance(analysis, (int, float)):
                        analyzers[name] = float(analysis)
                    else:
                        analyzers[name] = str(analysis)
                else:
                    # Direct value or other type
                    if isinstance(analyzer, (int, float)):
                        analyzers[name] = float(analyzer)
                    else:
                        analyzers[name] = str(analyzer)
            except Exception as e:
                _logger.warning("Could not process analyzer %s: %s", name, e, exc_info=True)
                # Store the raw analyzer value if processing fails
                analyzers[name] = str(analyzer)

        # Create the final result dictionary
        result_dict = {
            "data_file": str(data_file),
            "total_trades": len(trades),
            "total_profit": float(result.get("total_profit", 0)),  # Gross profit (before commission)
            "total_profit_with_commission": float(result.get("total_profit_with_commission", 0)),  # Net profit (after commission)
            "total_commission": float(result.get("total_commission", 0)),  # Total commission paid
            "best_params": result.get("best_params", {}),
            "analyzers": analyzers,
            "trades": trades,
        }

        # Save to JSON file
        json_file = os.path.join("results", f"{filename}.json")
        with open(json_file, "w") as f:
            json.dump(result_dict, f, indent=4)

        _logger.info("Results saved to %s", json_file)

    except Exception as e:
        _logger.error("Error saving results: %s", e, exc_info=True)
        raise



if __name__ == "__main__":
    """Run all optimizers with their respective configurations."""

    with open(os.path.join("config", "optimizer", "optimizer.json"), "r",) as f:
        optimizer_config = json.load(f)

    start_time = dt.now()
    _logger.info("Starting optimization at %s", start_time)

    # Get the data files
    data_files = [f for f in os.listdir("data/") if f.endswith(".csv") and not f.startswith(".")]

    # Count total combinations for progress tracking
    total_combinations = len(data_files) * len(ENTRY_MIXIN_REGISTRY) * len(EXIT_MIXIN_REGISTRY)
    processed_combinations = 0
    skipped_combinations = 0

    _logger.info("Found %d data files", len(data_files))
    _logger.info("Found %d entry mixins", len(ENTRY_MIXIN_REGISTRY))
    _logger.info("Found %d exit mixins", len(EXIT_MIXIN_REGISTRY))
    _logger.info("Total combinations to process: %d", total_combinations)

    for data_file in data_files:
        _logger.info("Processing data file: %s", data_file)
        df = prepare_data_frame(data_file)
        symbol, interval, start_date, end_date = parse_data_file_name(data_file)

        for entry_logic_name in ENTRY_MIXIN_REGISTRY.keys():
            # Load entry logic configuration
            with open(os.path.join("config", "optimizer", "entry", f"{entry_logic_name}.json"), "r") as f:
                entry_logic_config = json.load(f)

            for exit_logic_name in EXIT_MIXIN_REGISTRY.keys():
                processed_combinations += 1

                # Check if already processed
                if check_if_already_processed(data_file, entry_logic_name, exit_logic_name):
                    skipped_combinations += 1
                    _logger.info("Progress: %d/%d (Skipped: %d)", processed_combinations, total_combinations, skipped_combinations)
                    continue

                # Load exit logic configuration
                with open(os.path.join("config", "optimizer", "exit", f"{exit_logic_name}.json"), "r") as f:
                    exit_logic_config = json.load(f)

                _logger.info("Running optimization %d/%d: %s + %s + %s", processed_combinations, total_combinations, data_file, entry_logic_name, exit_logic_name)

                # Create optimizer configuration
                try:

                    def objective(trial):
                        """Objective function for optimization"""
                        # Create a new data feed for each trial (important for parallel jobs)
                        data = prepare_data_feed(df, symbol)
                        _optimizer_config = {
                            "data": data,
                            "entry_logic": entry_logic_config,
                            "exit_logic": exit_logic_config,
                            "optimizer_settings": optimizer_config.get("optimizer_settings", {}),
                            "visualization_settings": optimizer_config.get("visualization_settings", {}),
                        }
                        optimizer = CustomOptimizer(_optimizer_config)
                        # Disable analyzers for optimization trials to improve performance
                        _, _, result = optimizer.run_optimization(trial, include_analyzers=False)
                        return result["total_profit_with_commission"]

                    # Create study
                    study = optuna.create_study(direction="maximize")

                    # Run optimization
                    try:
                        study.optimize(
                            objective,
                            n_trials=optimizer_config.get("optimizer_settings", {}).get("n_trials", 100),
                            n_jobs=optimizer_config.get("optimizer_settings", {}).get("n_jobs", -1),
                        )
                    except Exception as e:
                        _logger.error("Error during optimization for %s + %s: %s", entry_logic_name, exit_logic_name, e, exc_info=True)
                        raise

                    # Get best result
                    if len(study.trials) == 0:
                        _logger.warning("No successful trials for %s + %s, skipping", entry_logic_name, exit_logic_name)
                        continue

                    data = prepare_data_feed(df, symbol)
                    _optimizer_config = {
                        "data": data,
                        "entry_logic": entry_logic_config,
                        "exit_logic": exit_logic_config,
                        "optimizer_settings": optimizer_config.get("optimizer_settings", {}),
                        "visualization_settings": optimizer_config.get("visualization_settings", {}),
                    }
                    best_trial = study.best_trial
                    best_optimizer = CustomOptimizer(_optimizer_config)

                    # Run full backtest with best parameters
                    _logger.info("Running full backtest with best parameters")
                    try:
                        strategy, cerebro, best_result = best_optimizer.run_optimization(best_trial, include_analyzers=True)

                        # Save results
                        save_results(best_result, data_file)
                    except Exception as e:
                        _logger.error("Error in final backtest for%s + %s: %s", entry_logic_name, exit_logic_name, e, exc_info=True)
                        raise

                    _logger.info("Completed optimization %d/%d", processed_combinations, total_combinations)

                except Exception as e:
                    _logger.error("Error for %s + %s: %s", entry_logic_name, exit_logic_name, e, exc_info=True)

    end_time = dt.now()
    duration = end_time - start_time

    _logger.info("Optimization completed at %s", end_time)
    _logger.info("Total duration: %s", duration)
    _logger.info("Summary:")
    _logger.info("   - Total combinations: %d", total_combinations)
    _logger.info("   - Processed: %d", processed_combinations - skipped_combinations)
    _logger.info("   - Skipped (already processed): %d", skipped_combinations)
    _logger.info("   - Time saved by resume: %d minutes (estimated)", skipped_combinations * 5)
