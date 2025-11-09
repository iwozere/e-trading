"""
Walk-Forward Validator Module

This module performs out-of-sample (OOS) validation by:
1. Loading optimization results from in-sample periods
2. Extracting best parameters for each strategy
3. Running backtests on OOS data with fixed parameters (no re-optimization)
4. Saving OOS results for comparison

This ensures true forward testing without look-ahead bias.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import json
from datetime import datetime as dt

import backtrader as bt
import pandas as pd
from src.notification.logger import setup_logger
from src.backtester.optimizer.custom_optimizer import CustomOptimizer

# Import utilities from run_optimizer
from src.backtester.optimizer.run_optimizer import (
    prepare_data_frame,
    prepare_data_feed,
    parse_data_file_name,
)

# Import walk-forward utilities
from src.backtester.optimizer.walk_forward_optimizer import (
    load_walk_forward_config,
    filter_data_files,
)

_logger = setup_logger(__name__)


def load_optimization_results(optimization_dir: str) -> dict:
    """
    Load all optimization results from a directory.

    Args:
        optimization_dir: Directory containing optimization result JSON files

    Returns:
        dict: Dictionary mapping strategy keys to result dictionaries
    """
    if not os.path.exists(optimization_dir):
        _logger.warning("Optimization directory not found: %s", optimization_dir)
        return {}

    results = {}
    json_files = [f for f in os.listdir(optimization_dir) if f.endswith('.json')]

    _logger.info("Loading optimization results from %s", optimization_dir)
    _logger.info("  Found %d result files", len(json_files))

    for json_file in json_files:
        try:
            file_path = os.path.join(optimization_dir, json_file)
            with open(file_path, 'r') as f:
                result = json.load(f)

            # Extract strategy key components
            symbol = result.get('symbol', '')
            timeframe = result.get('timeframe', '')
            entry_name = result.get('best_params', {}).get('entry_logic', {}).get('name', '')
            exit_name = result.get('best_params', {}).get('exit_logic', {}).get('name', '')

            if not all([symbol, timeframe, entry_name, exit_name]):
                _logger.warning("Skipping file %s: missing key fields", json_file)
                continue

            strategy_key = f"{symbol}_{timeframe}_{entry_name}_{exit_name}"
            results[strategy_key] = result

            _logger.debug("  Loaded: %s", strategy_key)

        except Exception as e:
            _logger.exception("Error loading %s: %s", json_file, e)
            continue

    _logger.info("  Loaded %d strategy results", len(results))
    return results


def extract_best_params(result: dict) -> dict:
    """
    Extract best parameters from optimization result.

    Args:
        result: Optimization result dictionary

    Returns:
        dict: Best parameters including entry/exit logic and settings
    """
    return result.get('best_params', {})


def run_oos_backtest(
    data_file: str,
    best_params: dict,
    optimizer_config: dict,
    data_dir: str = "data/_all"
) -> dict:
    """
    Run backtest on OOS data with fixed parameters (no optimization).

    Args:
        data_file: OOS data file name
        best_params: Best parameters from IS optimization
        optimizer_config: Optimizer configuration
        data_dir: Directory containing data files

    Returns:
        dict: Backtest results
    """
    # Load and prepare OOS data
    file_path = os.path.join(data_dir, data_file)
    df = prepare_data_frame(file_path)

    # Parse data file to get symbol
    symbol, interval, start_date, end_date = parse_data_file_name(data_file)

    # Prepare data feed
    data = prepare_data_feed(df, symbol)

    # Create entry logic config with fixed params
    entry_logic_config = {
        "name": best_params['entry_logic']['name'],
        "params": {
            param_name: {"default": param_value, "type": "fixed"}
            for param_name, param_value in best_params['entry_logic']['params'].items()
        }
    }

    # Create exit logic config with fixed params
    exit_logic_config = {
        "name": best_params['exit_logic']['name'],
        "params": {
            param_name: {"default": param_value, "type": "fixed"}
            for param_name, param_value in best_params['exit_logic']['params'].items()
        }
    }

    # Create optimizer config
    _optimizer_config = {
        "data": data,
        "entry_logic": entry_logic_config,
        "exit_logic": exit_logic_config,
        "optimizer_settings": optimizer_config.get("optimizer_settings", {}),
        "visualization_settings": optimizer_config.get("visualization_settings", {}),
    }

    # Run backtest with fixed params (no trial)
    optimizer = CustomOptimizer(_optimizer_config)
    strategy, cerebro, result = optimizer.run_optimization(trial=None, include_analyzers=True)

    return result


def save_validation_results(
    results: dict,
    window_name: str,
    test_year: str,
    trained_on_year: str,
    output_dir: str = "results/validation"
):
    """
    Save validation results to appropriate directory.

    Args:
        results: Dictionary of validation results
        window_name: Name of the window
        test_year: Testing year for directory organization
        trained_on_year: Year the model was trained on
        output_dir: Base output directory
    """
    # Create year-specific directory
    year_dir = os.path.join(output_dir, test_year)
    os.makedirs(year_dir, exist_ok=True)

    for strategy_key, result in results.items():
        try:
            # Generate filename
            data_file = result.get('data_file', 'unknown.csv')
            entry_name = result.get('best_params', {}).get('entry_logic', {}).get('name', '')
            exit_name = result.get('best_params', {}).get('exit_logic', {}).get('name', '')

            # Parse data file for filename components
            parts = parse_data_file_name(data_file)
            if len(parts) >= 4:
                symbol, interval, start_date, end_date = parts
                timestamp = dt.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{symbol}_{interval}_{start_date}_{end_date}_{entry_name}_{exit_name}_OOS_{timestamp}"
            else:
                timestamp = dt.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{strategy_key}_OOS_{timestamp}"

            # Convert trades to serializable format
            trades = []
            for trade in result.get("trades", []):
                try:
                    if not all(k in trade for k in ["entry_time", "exit_time", "entry_price", "exit_price"]):
                        continue

                    # Convert datetime objects
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

                    serializable_trade = {
                        "entry_time": entry_time,
                        "exit_time": exit_time,
                        "entry_price": float(trade["entry_price"]),
                        "exit_price": float(trade["exit_price"]),
                        "entry_value": float(trade.get("entry_value", 0.0)),
                        "exit_value": float(trade.get("exit_value", 0.0)),
                        "size": float(trade["size"]),
                        "symbol": str(trade["symbol"]),
                        "direction": str(trade["direction"]),
                        "commission": float(trade["commission"]),
                        "gross_pnl": float(trade["gross_pnl"]),
                        "net_pnl": float(trade["net_pnl"]),
                        "pnl_percentage": float(trade["pnl_percentage"]),
                        "exit_reason": str(trade["exit_reason"]),
                        "status": str(trade["status"]),
                    }
                    trades.append(serializable_trade)
                except Exception:
                    _logger.exception("Error processing trade")
                    continue

            # Process analyzers
            analyzers = {}
            for name, analyzer in result.get("analyzers", {}).items():
                try:
                    if isinstance(analyzer, dict):
                        processed_analysis = {}
                        for k, v in analyzer.items():
                            if isinstance(v, (int, float)):
                                processed_analysis[str(k)] = float(v)
                            elif isinstance(v, dt):
                                processed_analysis[str(k)] = v.isoformat()
                            else:
                                processed_analysis[str(k)] = str(v)
                        analyzers[name] = processed_analysis
                except Exception as e:
                    _logger.warning("Could not process analyzer %s: %s", name, e)
                    analyzers[name] = str(analyzer)

            # Create final result dictionary
            result_dict = {
                "is_out_of_sample": True,
                "data_file": str(data_file),
                "window_name": window_name,
                "test_year": test_year,
                "trained_on_year": trained_on_year,
                "symbol": result.get('symbol', ''),
                "timeframe": result.get('timeframe', ''),
                "total_trades": len(trades),
                "total_profit": float(result.get("total_profit", 0)),
                "total_profit_with_commission": float(result.get("total_profit_with_commission", 0)),
                "total_commission": float(result.get("total_commission", 0)),
                "best_params": result.get("best_params", {}),
                "analyzers": analyzers,
                "trades": trades,
            }

            # Save to JSON file
            json_file = os.path.join(year_dir, f"{filename}.json")
            with open(json_file, "w") as f:
                json.dump(result_dict, f, indent=4)

            _logger.debug("Saved OOS result to %s", json_file)

        except Exception:
            _logger.exception("Error saving OOS result for %s", strategy_key)
            continue


def main():
    """Main orchestrator for walk-forward validation."""
    start_time = dt.now()
    _logger.info("=" * 80)
    _logger.info("Walk-Forward Validation Started")
    _logger.info("=" * 80)
    _logger.info("Start time: %s", start_time)

    # Load configurations
    wf_config_path = os.path.join("config", "walk_forward", "walk_forward_config.json")
    wf_config = load_walk_forward_config(wf_config_path)

    optimizer_config_path = wf_config['optimizer_config_path']
    with open(optimizer_config_path, 'r') as f:
        optimizer_config = json.load(f)

    # Statistics tracking
    total_windows = len(wf_config['windows'])
    total_symbols = len(wf_config['symbols'])
    total_timeframes = len(wf_config['timeframes'])
    processed_count = 0
    total_validations = 0

    _logger.info("=" * 80)
    _logger.info("Validation Plan:")
    _logger.info("  Windows: %d", total_windows)
    _logger.info("  Symbols: %d", total_symbols)
    _logger.info("  Timeframes: %d", total_timeframes)
    _logger.info("=" * 80)

    # Process each window
    for window_idx, window in enumerate(wf_config['windows'], 1):
        _logger.info("")
        _logger.info("=" * 80)
        _logger.info("Processing Window %d/%d: %s", window_idx, total_windows, window['name'])
        _logger.info("  Trained On: %s", window['train_year'])
        _logger.info("  Testing On: %s", window['test_year'])
        _logger.info("=" * 80)

        # Load IS optimization results for this window
        optimization_dir = os.path.join("results", "optimization", window['train_year'])
        is_results = load_optimization_results(optimization_dir)

        if not is_results:
            _logger.warning("No optimization results found in %s, skipping window", optimization_dir)
            continue

        # Process each symbol/timeframe combination
        for symbol in wf_config['symbols']:
            for timeframe in wf_config['timeframes']:
                processed_count += 1

                _logger.info("")
                _logger.info("-" * 80)
                _logger.info(
                    "Validating: %s | %s | %s",
                    window['name'], symbol, timeframe
                )
                _logger.info("-" * 80)

                # Filter test data files for this symbol/timeframe
                test_files = filter_data_files(window['test'], symbol, timeframe)

                if not test_files:
                    _logger.warning(
                        "No test data files found for %s/%s in window %s",
                        symbol, timeframe, window['name']
                    )
                    continue

                if len(test_files) > 1:
                    _logger.warning(
                        "Multiple test files found for %s/%s: %s. Using first file only.",
                        symbol, timeframe, test_files
                    )

                # Use the first matching file
                test_data_file = test_files[0]
                _logger.info("  Using test data: %s", test_data_file)

                # Run OOS validation for relevant strategies
                oos_results = {}

                for strategy_key, is_result in is_results.items():
                    # Check if this strategy matches current symbol/timeframe
                    if not strategy_key.startswith(f"{symbol}_{timeframe}_"):
                        continue

                    _logger.info("    Validating: %s", strategy_key)

                    try:
                        # Extract best params from IS result
                        best_params = extract_best_params(is_result)

                        # Run OOS backtest with fixed params
                        oos_result = run_oos_backtest(
                            test_data_file,
                            best_params,
                            optimizer_config
                        )

                        # Add metadata
                        oos_result['window_name'] = window['name']
                        oos_result['test_year'] = window['test_year']
                        oos_result['trained_on_year'] = window['train_year']
                        oos_result['symbol'] = symbol
                        oos_result['timeframe'] = timeframe

                        oos_results[strategy_key] = oos_result
                        total_validations += 1

                        _logger.info(
                            "    OOS Result: Profit: %.2f | Trades: %d",
                            oos_result['total_profit_with_commission'],
                            oos_result.get('total_trades', 0)
                        )

                    except Exception as e:
                        _logger.exception("Error validating %s: %s", strategy_key, e)
                        continue

                # Save OOS results
                if oos_results:
                    save_validation_results(
                        oos_results,
                        window['name'],
                        window['test_year'],
                        window['train_year']
                    )
                    _logger.info(
                        "  Saved %d OOS results for %s/%s",
                        len(oos_results), symbol, timeframe
                    )
                else:
                    _logger.warning(
                        "  No OOS results to save for %s/%s",
                        symbol, timeframe
                    )

    # Summary
    end_time = dt.now()
    duration = end_time - start_time

    _logger.info("")
    _logger.info("=" * 80)
    _logger.info("Walk-Forward Validation Completed")
    _logger.info("=" * 80)
    _logger.info("End time: %s", end_time)
    _logger.info("Total duration: %s", duration)
    _logger.info("Total OOS validations: %d", total_validations)
    _logger.info("Results saved to: results/validation/")
    _logger.info("=" * 80)


if __name__ == "__main__":
    main()
