"""
Walk-Forward Optimizer Module

This module orchestrates walk-forward optimization by:
1. Loading walk-forward configuration (training/testing windows)
2. Running optimization on in-sample (IS) data for each window
3. Saving IS optimization results for later validation

The optimizer ensures temporal integrity by never training on future data.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import json
from datetime import datetime as dt

import backtrader as bt
import optuna
import pandas as pd
from src.strategy.entry.entry_mixin_factory import ENTRY_MIXIN_REGISTRY
from src.strategy.exit.exit_mixin_factory import EXIT_MIXIN_REGISTRY
from src.notification.logger import setup_logger, setup_multiprocessing_logging
from src.backtester.optimizer.custom_optimizer import CustomOptimizer

# Import utilities from run_optimizer
from src.backtester.optimizer.run_optimizer import (
    prepare_data_frame,
    prepare_data_feed,
    load_mixin_config,
    parse_data_file_name,
    get_result_filename
)

_logger = setup_logger(__name__)


def auto_discover_windows(config: dict, data_dir: str = "data") -> list:
    """
    Automatically discover and generate walk-forward windows from data files.

    This function scans the data directory for CSV files matching the pattern:
    SYMBOL_TIMEFRAME_STARTDATE_ENDDATE.csv

    It then groups files by symbol and timeframe, sorts them chronologically,
    and generates train/test window pairs based on the window_type.

    Args:
        config: Configuration dict with symbols, timeframes, and window_type
        data_dir: Directory containing data files (default: "data")

    Returns:
        list: List of window definitions with train/test file pairs

    Example:
        For files like BTCUSDT_4h_20200101_20201231.csv, generates:
        [
            {
                "name": "2020_train_2021_test",
                "train": ["BTCUSDT_4h_20200101_20201231.csv"],
                "test": ["BTCUSDT_4h_20210101_20211231.csv"],
                "train_year": "2020",
                "test_year": "2021"
            },
            ...
        ]
    """
    from collections import defaultdict
    import glob

    symbols = config.get('symbols', [])
    timeframes = config.get('timeframes', [])
    window_type = config.get('window_type', 'rolling')

    _logger.info("=" * 80)
    _logger.info("AUTO-DISCOVERING DATA FILES")
    _logger.info("=" * 80)
    _logger.info("Data directory: %s", data_dir)
    _logger.info("Symbols filter: %s", symbols)
    _logger.info("Timeframes filter: %s", timeframes)
    _logger.info("Window type: %s", window_type)

    # Scan data directory
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    _logger.info("Found %d CSV files", len(all_files))

    # Group files by symbol and timeframe
    file_groups = defaultdict(list)

    for file_path in all_files:
        filename = os.path.basename(file_path)

        try:
            parts = parse_data_file_name(filename)
            if len(parts) != 4:
                continue

            symbol, timeframe, start_date, end_date = parts

            # Filter by configured symbols and timeframes
            if symbols and symbol not in symbols:
                continue
            if timeframes and timeframe not in timeframes:
                continue

            # Extract year from start_date (YYYYMMDD format)
            year = start_date[:4]

            key = (symbol, timeframe)
            file_groups[key].append({
                'filename': filename,
                'symbol': symbol,
                'timeframe': timeframe,
                'year': year,
                'start_date': start_date,
                'end_date': end_date
            })

        except Exception as e:
            _logger.warning("Failed to parse filename %s: %s", filename, e)
            continue

    _logger.info("Grouped files into %d symbol/timeframe combinations", len(file_groups))

    # Generate windows for each symbol/timeframe combination
    all_windows = []

    for (symbol, timeframe), files in sorted(file_groups.items()):
        _logger.info("  Processing %s/%s: %d files", symbol, timeframe, len(files))

        # Sort files by year
        files.sort(key=lambda x: x['year'])

        # Generate train/test pairs based on window_type
        if window_type == 'rolling':
            # Rolling window: each year trains on previous year, tests on current
            for i in range(len(files) - 1):
                train_file = files[i]
                test_file = files[i + 1]

                window = {
                    'name': f"{train_file['year']}_train_{test_file['year']}_test_{symbol}_{timeframe}",
                    'train': [train_file['filename']],
                    'test': [test_file['filename']],
                    'train_year': train_file['year'],
                    'test_year': test_file['year'],
                    'symbol': symbol,
                    'timeframe': timeframe
                }
                all_windows.append(window)
                _logger.debug("    Window: %s -> %s", train_file['year'], test_file['year'])

        elif window_type == 'expanding':
            # Expanding window: train on all previous years, test on current
            for i in range(1, len(files)):
                train_files = [f['filename'] for f in files[:i]]
                test_file = files[i]

                window = {
                    'name': f"{files[0]['year']}_to_{files[i-1]['year']}_train_{test_file['year']}_test_{symbol}_{timeframe}",
                    'train': train_files,
                    'test': [test_file['filename']],
                    'train_year': f"{files[0]['year']}-{files[i-1]['year']}",
                    'test_year': test_file['year'],
                    'symbol': symbol,
                    'timeframe': timeframe
                }
                all_windows.append(window)
                _logger.debug("    Window: %s-%s -> %s", files[0]['year'], files[i-1]['year'], test_file['year'])

        elif window_type == 'anchored':
            # Anchored window: train on first year, test on subsequent years
            if len(files) < 2:
                continue
            train_file = files[0]
            for i in range(1, len(files)):
                test_file = files[i]

                window = {
                    'name': f"{train_file['year']}_anchored_{test_file['year']}_test_{symbol}_{timeframe}",
                    'train': [train_file['filename']],
                    'test': [test_file['filename']],
                    'train_year': train_file['year'],
                    'test_year': test_file['year'],
                    'symbol': symbol,
                    'timeframe': timeframe
                }
                all_windows.append(window)
                _logger.debug("    Window: %s (anchored) -> %s", train_file['year'], test_file['year'])

    _logger.info("Generated %d windows across all symbol/timeframe combinations", len(all_windows))
    _logger.info("=" * 80)

    return all_windows


def load_walk_forward_config(config_path: str) -> dict:
    """
    Load and validate walk-forward configuration.

    Supports two modes:
    1. Manual mode: 'windows' field contains explicit train/test file definitions
    2. Auto-discovery mode: 'auto_discover_data' is true, windows are generated automatically

    Args:
        config_path: Path to walk_forward_config.json

    Returns:
        dict: Validated configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Walk-forward config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Check if auto-discovery is enabled
    auto_discover = config.get('auto_discover_data', False)
    data_dir = config.get('data_dir', 'data')

    if auto_discover:
        _logger.info("Auto-discovery mode enabled")

        # Validate required fields for auto-discovery
        required_fields = ['symbols', 'timeframes', 'optimizer_config_path']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in config: {field}")

        # Auto-generate windows from data files
        config['windows'] = auto_discover_windows(config, data_dir=data_dir)

        if not config['windows']:
            raise ValueError(f"No data files found matching symbols {config['symbols']} and timeframes {config['timeframes']} in {data_dir}")

    else:
        _logger.info("Manual window configuration mode")

        # Validate required fields for manual mode
        required_fields = ['windows', 'symbols', 'timeframes', 'optimizer_config_path']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in config: {field}")

        # Validate windows
        if not isinstance(config['windows'], list) or len(config['windows']) == 0:
            raise ValueError("Configuration must contain at least one window")

        for window in config['windows']:
            required_window_fields = ['name', 'train', 'test', 'train_year', 'test_year']
            for field in required_window_fields:
                if field not in window:
                    raise ValueError(f"Missing required field in window: {field}")

    # Set default strategy filters (all strategies if not specified)
    if 'entry_strategies' not in config:
        config['entry_strategies'] = list(ENTRY_MIXIN_REGISTRY.keys())
    if 'exit_strategies' not in config:
        config['exit_strategies'] = list(EXIT_MIXIN_REGISTRY.keys())

    _logger.info("Walk-forward configuration loaded successfully")
    _logger.info("  Windows: %d", len(config['windows']))
    _logger.info("  Symbols: %s", config['symbols'])
    _logger.info("  Timeframes: %s", config['timeframes'])
    _logger.info("  Entry strategies: %s", config['entry_strategies'])
    _logger.info("  Exit strategies: %s", config['exit_strategies'])

    return config


def validate_windows(windows: list, data_dir: str = "data") -> bool:
    """
    Validate window definitions (temporal ordering and file existence).

    Args:
        windows: List of window definitions
        data_dir: Directory containing data files

    Returns:
        bool: True if validation passes

    Raises:
        ValueError: If validation fails
    """
    _logger.info("Validating window definitions...")

    # Check temporal ordering
    prev_test_year = None
    for window in windows:
        train_year = window['train_year']
        test_year = window['test_year']

        # Train year should be before test year
        if train_year >= test_year:
            raise ValueError(
                f"Invalid window {window['name']}: train_year ({train_year}) must be before test_year ({test_year})"
            )

        # For rolling window, ensure chronological order
        if prev_test_year is not None:
            if train_year < prev_test_year:
                _logger.warning(
                    "Window %s: train_year (%s) is before previous test_year (%s). This may be intentional for overlapping windows.",
                    window['name'], train_year, prev_test_year
                )

        prev_test_year = test_year

        # Check if data files exist
        for data_file in window['train'] + window['test']:
            file_path = os.path.join(data_dir, data_file)
            if not os.path.exists(file_path):
                raise ValueError(f"Data file not found: {file_path}")

        _logger.info(
            "  Window '%s': Train(%s) -> Test(%s) ✓",
            window['name'], train_year, test_year
        )

    _logger.info("Window validation passed")
    return True


def filter_data_files(data_files: list, symbol: str, timeframe: str) -> list:
    """
    Filter data files by symbol and timeframe.

    Args:
        data_files: List of data file names
        symbol: Symbol to filter (e.g., 'BTCUSDT')
        timeframe: Timeframe to filter (e.g., '1h')

    Returns:
        list: Filtered list of data files
    """
    filtered = []
    for data_file in data_files:
        parts = parse_data_file_name(data_file)
        if len(parts) >= 2:
            file_symbol = parts[0]
            file_timeframe = parts[1]
            if file_symbol == symbol and file_timeframe == timeframe:
                filtered.append(data_file)

    return filtered


def check_result_exists(
    data_file: str,
    entry_logic_name: str,
    exit_logic_name: str,
    train_year: str,
    output_dir: str = "results/walk_forward_reports"
) -> bool:
    """
    Check if a result file already exists for a given strategy combination.

    Args:
        data_file: Data file name
        entry_logic_name: Entry strategy name
        exit_logic_name: Exit strategy name
        train_year: Training year for directory organization
        output_dir: Base output directory

    Returns:
        bool: True if result file exists, False otherwise
    """
    import glob

    # Create year-specific directory path
    year_dir = os.path.join(output_dir, train_year)

    if not os.path.exists(year_dir):
        return False

    # Parse data file to extract symbol, timeframe, dates
    parts = parse_data_file_name(data_file)
    if len(parts) != 4:
        return False

    symbol, timeframe, start_date, end_date = parts

    # Create pattern to match existing result files
    # Pattern: SYMBOL_TIMEFRAME_STARTDATE_ENDDATE_ENTRYNAME_EXITNAME_*.json
    pattern = f"{symbol}_{timeframe}_{start_date}_{end_date}_{entry_logic_name}_{exit_logic_name}_*.json"

    # Search for matching files
    search_pattern = os.path.join(year_dir, pattern)
    matching_files = glob.glob(search_pattern)

    if matching_files:
        _logger.debug("Found existing result file: %s", matching_files[0])
        return True

    return False


def run_window_optimization(
    window: dict,
    symbol: str,
    timeframe: str,
    optimizer_config: dict,
    entry_strategies: list = None,
    exit_strategies: list = None,
    data_dir: str = "data"
) -> tuple:
    """
    Run optimization for a single window, symbol, and timeframe combination.

    Args:
        window: Window definition
        symbol: Trading symbol
        timeframe: Trading timeframe
        optimizer_config: Optimizer configuration from optimizer.json
        entry_strategies: List of entry strategies to test
        exit_strategies: List of exit strategies to test
        data_dir: Directory containing data files

    Returns:
        tuple: (results dict, skipped_count, processed_count)
            - results: Dictionary with optimization outcomes for all strategy combinations
            - skipped_count: Number of strategies skipped (already completed)
            - processed_count: Number of strategies processed (newly optimized)
    """
    _logger.info(
        "Running window optimization: %s | %s | %s",
        window['name'], symbol, timeframe
    )

    # Filter training data files for this symbol/timeframe
    train_files = filter_data_files(window['train'], symbol, timeframe)

    if not train_files:
        _logger.warning(
            "No training data files found for %s/%s in window %s",
            symbol, timeframe, window['name']
        )
        return {}, 0, 0

    if len(train_files) > 1:
        _logger.warning(
            "Multiple training files found for %s/%s: %s. Using first file only.",
            symbol, timeframe, train_files
        )

    # Use the first matching file
    data_file = train_files[0]
    _logger.info("  Using training data: %s", data_file)

    # Load and prepare data
    file_path = os.path.join(data_dir, data_file)
    df = prepare_data_frame(file_path)

    # Parse data file to get symbol
    symbol_from_file, interval, start_date, end_date = parse_data_file_name(data_file)

    results = {}
    skipped_count = 0
    processed_count = 0

    # Use filtered strategies or all if not specified
    entry_strategies_to_test = entry_strategies or list(ENTRY_MIXIN_REGISTRY.keys())
    exit_strategies_to_test = exit_strategies or list(EXIT_MIXIN_REGISTRY.keys())

    # Iterate through filtered entry/exit strategy combinations
    for entry_logic_name in entry_strategies_to_test:
        # Load entry logic configuration
        entry_logic_config = load_mixin_config(entry_logic_name, "entry", timeframe)

        for exit_logic_name in exit_strategies_to_test:
            strategy_key = f"{symbol}_{timeframe}_{entry_logic_name}_{exit_logic_name}"

            # Check if result already exists (resume functionality)
            if check_result_exists(data_file, entry_logic_name, exit_logic_name, window['train_year']):
                _logger.info("    ✓ Skipping (already completed): %s", strategy_key)
                skipped_count += 1
                continue

            _logger.info("    Optimizing: %s", strategy_key)
            processed_count += 1

            # Load exit logic configuration
            exit_logic_config = load_mixin_config(exit_logic_name, "exit", timeframe)

            try:
                # Define objective function for this strategy combination
                def objective(trial):
                    """Objective function for optimization"""
                    # optuna.pruners.MedianPruner(n_warmup_steps=10)
                    # Create a new data feed for each trial
                    data = prepare_data_feed(df, symbol_from_file)

                    _optimizer_config = {
                        "data": data,
                        "entry_logic": entry_logic_config,
                        "exit_logic": exit_logic_config,
                        "optimizer_settings": optimizer_config.get("optimizer_settings", {}),
                        "visualization_settings": optimizer_config.get("visualization_settings", {}),
                        "symbol": symbol_from_file,
                        "timeframe": timeframe,
                    }
                    optimizer = CustomOptimizer(_optimizer_config)
                    # Disable analyzers for optimization trials to improve performance
                    _, _, result = optimizer.run_optimization(trial, include_analyzers=False)
                    return result["total_profit_with_commission"]

                # Create Optuna study
                study = optuna.create_study(direction="maximize")

                # Run optimization
                study.optimize(
                    objective,
                    n_trials=optimizer_config.get("optimizer_settings", {}).get("n_trials", 100),
                    n_jobs=optimizer_config.get("optimizer_settings", {}).get("n_jobs", -1),
                    show_progress_bar=False  # Disable progress bar for cleaner logs
                )

                # Get best result
                if len(study.trials) == 0:
                    _logger.warning("No successful trials for %s, skipping", strategy_key)
                    continue

                # Run final backtest with best parameters and full analyzers
                data = prepare_data_feed(df, symbol_from_file)
                _optimizer_config = {
                    "data": data,
                    "entry_logic": entry_logic_config,
                    "exit_logic": exit_logic_config,
                    "optimizer_settings": optimizer_config.get("optimizer_settings", {}),
                    "visualization_settings": optimizer_config.get("visualization_settings", {}),
                    "symbol": symbol_from_file,
                    "timeframe": timeframe,
                }
                best_trial = study.best_trial
                best_optimizer = CustomOptimizer(_optimizer_config)

                strategy, cerebro, best_result = best_optimizer.run_optimization(
                    best_trial, include_analyzers=True
                )

                # Add metadata
                best_result['window_name'] = window['name']
                best_result['train_year'] = window['train_year']
                best_result['symbol'] = symbol
                best_result['timeframe'] = timeframe
                best_result['data_file'] = data_file

                # Save result immediately to disk
                save_success = save_single_strategy_result(
                    best_result,
                    window['name'],
                    window['train_year']
                )

                if save_success:
                    results[strategy_key] = best_result

                    # Log completion with correct trade count from trades list
                    _logger.info(
                        "    ✓ Completed & Saved: %s | Profit: %.2f | Trades: %d",
                        strategy_key,
                        best_result['total_profit_with_commission'],
                        len(best_result.get('trades', []))  # Count trades from the list
                    )
                else:
                    _logger.error(
                        "    ✗ Failed to save: %s (optimization completed but save failed)",
                        strategy_key
                    )

            except Exception as e:
                _logger.exception("Error optimizing %s: %s", strategy_key, e)
                continue

    return results, skipped_count, processed_count


def save_single_strategy_result(
    result: dict,
    window_name: str,
    train_year: str,
    output_dir: str = "results/walk_forward_reports"
) -> bool:
    """
    Save a single strategy optimization result to disk immediately.

    Args:
        result: Single strategy optimization result
        window_name: Name of the window
        train_year: Training year for directory organization
        output_dir: Base output directory

    Returns:
        bool: True if saved successfully, False otherwise
    """
    # Create year-specific directory
    year_dir = os.path.join(output_dir, train_year)
    os.makedirs(year_dir, exist_ok=True)

    try:
        # Generate filename
        data_file = result.get('data_file', 'unknown.csv')
        entry_name = result.get('best_params', {}).get('entry_logic', {}).get('name', '')
        exit_name = result.get('best_params', {}).get('exit_logic', {}).get('name', '')

        filename = get_result_filename(
            data_file,
            entry_logic_name=entry_name,
            exit_logic_name=exit_name,
            suffix="",
            include_timestamp=True
        )

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
            "data_file": str(data_file),
            "window_name": window_name,
            "train_year": train_year,
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

        _logger.debug("Saved result to %s", json_file)
        return True

    except Exception:
        _logger.exception("Error saving result")
        return False


def save_optimization_results(
    results: dict,
    window_name: str,
    train_year: str,
    output_dir: str = "results/walk_forward_reports"
):
    """
    Save optimization results to appropriate directory.

    NOTE: This function is kept for backwards compatibility but is deprecated.
    Use save_single_strategy_result() instead to save results immediately after optimization.

    Args:
        results: Dictionary of optimization results
        window_name: Name of the window
        train_year: Training year for directory organization
        output_dir: Base output directory
    """
    # Create year-specific directory
    year_dir = os.path.join(output_dir, train_year)
    os.makedirs(year_dir, exist_ok=True)

    for strategy_key, result in results.items():
        try:
            # Generate filename
            data_file = result.get('data_file', 'unknown.csv')
            entry_name = result.get('best_params', {}).get('entry_logic', {}).get('name', '')
            exit_name = result.get('best_params', {}).get('exit_logic', {}).get('name', '')

            filename = get_result_filename(
                data_file,
                entry_logic_name=entry_name,
                exit_logic_name=exit_name,
                suffix="",
                include_timestamp=True
            )

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
                "data_file": str(data_file),
                "window_name": window_name,
                "train_year": train_year,
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

            _logger.debug("Saved result to %s", json_file)

        except Exception:
            _logger.exception("Error saving result for %s", strategy_key)
            continue


def main():
    """Main orchestrator for walk-forward optimization."""
    # Set up multiprocessing-safe logging FIRST
    setup_multiprocessing_logging()
    _logger.info("Multiprocessing-safe logging enabled for walk-forward optimization")

    start_time = dt.now()
    _logger.info("=" * 80)
    _logger.info("Walk-Forward Optimization Started")
    _logger.info("=" * 80)
    _logger.info("Start time: %s", start_time)

    # Load configurations
    wf_config_path = os.path.join("config", "walk_forward", "walk_forward_config.json")
    wf_config = load_walk_forward_config(wf_config_path)

    optimizer_config_path = wf_config['optimizer_config_path']
    with open(optimizer_config_path, 'r') as f:
        optimizer_config = json.load(f)

    # Validate windows
    validate_windows(wf_config['windows'])

    # Statistics tracking
    total_windows = len(wf_config['windows'])
    total_symbols = len(wf_config['symbols'])
    total_timeframes = len(wf_config['timeframes'])
    total_combinations = total_windows * total_symbols * total_timeframes
    processed_count = 0
    total_strategies_skipped = 0
    total_strategies_processed = 0

    # Get strategy filters
    entry_strategies = wf_config.get('entry_strategies', list(ENTRY_MIXIN_REGISTRY.keys()))
    exit_strategies = wf_config.get('exit_strategies', list(EXIT_MIXIN_REGISTRY.keys()))

    _logger.info("=" * 80)
    _logger.info("Optimization Plan:")
    _logger.info("  Windows: %d", total_windows)
    _logger.info("  Symbols: %d", total_symbols)
    _logger.info("  Timeframes: %d", total_timeframes)
    _logger.info("  Total window/symbol/timeframe combinations: %d", total_combinations)
    _logger.info("  Entry strategies to test: %d - %s", len(entry_strategies), entry_strategies)
    _logger.info("  Exit strategies to test: %d - %s", len(exit_strategies), exit_strategies)
    _logger.info("  Strategy combinations per window: %d", len(entry_strategies) * len(exit_strategies))
    _logger.info("=" * 80)

    # Process each window
    for window_idx, window in enumerate(wf_config['windows'], 1):
        _logger.info("")
        _logger.info("=" * 80)
        _logger.info("Processing Window %d/%d: %s", window_idx, total_windows, window['name'])
        _logger.info("  Train Period: %s", window['train_year'])
        _logger.info("  Test Period: %s", window['test_year'])
        _logger.info("=" * 80)

        # Process each symbol/timeframe combination
        for symbol in wf_config['symbols']:
            for timeframe in wf_config['timeframes']:
                processed_count += 1

                _logger.info("")
                _logger.info("-" * 80)
                _logger.info(
                    "Processing [%d/%d]: %s | %s | %s",
                    processed_count, total_combinations,
                    window['name'], symbol, timeframe
                )
                _logger.info("-" * 80)

                # Run optimization for this combination
                # Results are now saved immediately during optimization
                _, skipped, processed = run_window_optimization(
                    window, symbol, timeframe, optimizer_config,
                    entry_strategies=entry_strategies,
                    exit_strategies=exit_strategies
                )

                # Update global statistics
                total_strategies_skipped += skipped
                total_strategies_processed += processed

                # Log summary for this combination
                total_for_combo = skipped + processed
                if total_for_combo > 0:
                    _logger.info(
                        "  Summary for %s/%s: Total=%d, Completed=%d, Skipped=%d",
                        symbol, timeframe, total_for_combo, processed, skipped
                    )
                else:
                    _logger.warning(
                        "  No strategies to process for %s/%s",
                        symbol, timeframe
                    )

    # Summary
    end_time = dt.now()
    duration = end_time - start_time
    total_strategies = total_strategies_skipped + total_strategies_processed

    _logger.info("")
    _logger.info("=" * 80)
    _logger.info("Walk-Forward Optimization Completed")
    _logger.info("=" * 80)
    _logger.info("End time: %s", end_time)
    _logger.info("Total duration: %s", duration)
    _logger.info("Processed: %d/%d window/symbol/timeframe combinations", processed_count, total_combinations)
    _logger.info("")
    _logger.info("Strategy Statistics:")
    _logger.info("  Total strategies: %d", total_strategies)
    _logger.info("  Newly optimized: %d (%.1f%%)", total_strategies_processed,
                 (total_strategies_processed / total_strategies * 100) if total_strategies > 0 else 0)
    _logger.info("  Skipped (already completed): %d (%.1f%%)", total_strategies_skipped,
                 (total_strategies_skipped / total_strategies * 100) if total_strategies > 0 else 0)
    _logger.info("")
    _logger.info("Results saved to: results/walk_forward_reports/")
    _logger.info("=" * 80)


if __name__ == "__main__":
    main()
