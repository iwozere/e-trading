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
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import json
from datetime import datetime as dt

import backtrader as bt
import numpy as np
import optuna
import pandas as pd
from scipy import stats
from src.strategy.entry.entry_mixin_factory import ENTRY_MIXIN_REGISTRY
from src.strategy.exit.exit_mixin_factory import EXIT_MIXIN_REGISTRY
from src.notification.logger import setup_logger, setup_multiprocessing_logging
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
    #df = pd.read_csv(os.path.join("data", data_file))
    df = pd.read_csv(data_file)
    print("Available columns:", df.columns.tolist())

    # Find datetime column and convert properly
    df["datetime"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("datetime", ascending=True)
    df.set_index("datetime", inplace=True)

    # Ensure the index is timezone-naive for Backtrader compatibility
    df.index = df.index.tz_localize(None)

    # Ensure the index is pandas datetime, not numpy float64
    df.index = pd.to_datetime(df.index)

    df = df[["open", "high", "low", "close", "volume"]]

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Remove any rows with NaN values
    df = df.dropna()

    # Ensure we have data
    if len(df) == 0:
        raise ValueError(f"No valid data found in {data_file}")

    print(f"Final data shape: {df.shape}")
    print(f"Data range: {df.index[0]} to {df.index[-1]}")

    # Extract symbol from data file name
    symbol = "UNKNOWN"
    if "_" in data_file:
        parts = data_file.replace(".csv", "").split("_")
        if len(parts) >= 1:
            symbol = parts[0]

    return df

def pre_calculate_htf_data(df: pd.DataFrame, intervals: list) -> pd.DataFrame:
    """
    Pre-calculate HTF indicators using Pandas once per combination.
    This replaces expensive Backtrader resampling inside trials.
    """
    _logger.info("Pre-calculating HTF data for intervals: %s", intervals)
    result_df = df.copy()

    for interval in intervals:
        # intervals are in minutes, e.g., 60, 240
        rule = f"{interval}min"

        # Resample to HTF
        # Note: We use 'closed=right', 'label=right' to match standard crypto bar behavior
        # and prevent look-ahead bias (the 4h bar at 04:00 contains data up to 04:00)
        htf = df.resample(rule, closed='right', label='right').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        # Calculate True Range and ATR
        # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
        htf['prev_close'] = htf['close'].shift(1)
        tr = pd.concat([
            htf['high'] - htf['low'],
            (htf['high'] - htf['prev_close']).abs(),
            (htf['low'] - htf['prev_close']).abs()
        ], axis=1).max(axis=1)

        # Standard periods used in config
        periods = [10, 14, 20, 30]
        for p in periods:
            # We use Simple Moving Average of TR to match Backtrader's bt.indicators.ATR default (Simple)
            # though some use EWMA. Let's use Simple for consistency with our current tests.
            htf[f'atr_{interval}_{p}'] = tr.rolling(window=p).mean()

        # Aligne (Forward-fill) HTF indicators back to LTF
        # We reindex to match LTF and then ffill
        htf_aligned = htf[[f'atr_{interval}_{p}' for p in periods]].reindex(df.index, method='ffill')

        # Add to result_df
        for p in periods:
            col_name = f'atr_{interval}_{p}'
            result_df[col_name] = htf_aligned[col_name]

    return result_df

# Define DynamicPandasData once at module level to avoid redundant class creation
class DynamicPandasData(bt.feeds.PandasData):
    pass

def prepare_data_feed(df: pd.DataFrame, symbol: str):
    """Prepare data feed from pandas dataframe with robust datetime handling"""
    # Validate DataFrame before creating feed
    if df.empty:
        raise ValueError("DataFrame is empty")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"DataFrame index is not DatetimeIndex: {type(df.index)}")

    # Create a shallow copy if possible, or at least avoid deep copy if not needed
    # Backtrader doesn't strictly require a deep copy if we don't modify the source df
    df_copy = df.copy(deep=False)

    # Ensure the required columns are present. Columns are already prepared in prepare_data_frame.
    # Just filter to what's needed for the feed.
    atr_cols = [c for c in df.columns if c.startswith('atr_')]
    cols = ["open", "high", "low", "close", "volume"] + atr_cols
    df_copy = df_copy[cols]

    # Dynamically update the class lines and params for this specific instance if needed,
    # or better, just pass the extra columns to the constructor.
    # We update the class attributes once per data file if they change, but here
    # we can stick to a simpler approach since atr_cols are consistent per data file.

    # We can use a trick with type() to create the subclass with dynamic lines ONLY IF it changed
    # to avoid the overhead of creating it 150 times.

    cache_key = tuple(sorted(atr_cols))
    if not hasattr(prepare_data_feed, "_class_cache"):
        prepare_data_feed._class_cache = {}

    if cache_key not in prepare_data_feed._class_cache:
        prepare_data_feed._class_cache[cache_key] = type(
            'DynamicPandasData',
            (bt.feeds.PandasData,),
            {
                'lines': tuple(atr_cols),
                'params': tuple((col, -1) for col in atr_cols)
            }
        )

    DataClass = prepare_data_feed._class_cache[cache_key]

    data_feed = DataClass(
        dataname=df_copy,
        # Standard OHLCV mapping
        open=0,
        high=1,
        low=2,
        close=3,
        volume=4,
        openinterest=None,
        # HTF ATR mapping
        fromdate=df_copy.index.min(),
        todate=df_copy.index.max(),
        name=symbol,
        **{col: 5 + i for i, col in enumerate(atr_cols)}
    )

    return data_feed


def load_mixin_config(mixin_name: str, config_type: str, timeframe: str) -> dict:
    """
    Load mixin configuration with timeframe-specific fallback logic.

    Args:
        mixin_name: Name of the mixin (e.g., 'RSIBBEntryMixin')
        config_type: Type of config ('entry' or 'exit')
        timeframe: Trading timeframe (e.g., '1h', '4h', '1d')

    Returns:
        Configuration dictionary
    """
    try:
        # Try to load timeframe-specific configuration first
        tf_specific_path = os.path.join("config", "optimizer", config_type, f"{mixin_name}_{timeframe}.json")
        if os.path.exists(tf_specific_path):
            _logger.info("Loading timeframe-specific config: %s", tf_specific_path)
            with open(tf_specific_path, 'r') as f:
                config = json.load(f)
            _logger.debug("Loaded %s config for %s: %s", timeframe, mixin_name, config.get('params', {}))
            return config

        # Fallback to generic configuration
        generic_path = os.path.join("config", "optimizer", config_type, f"{mixin_name}.json")
        if os.path.exists(generic_path):
            _logger.info("Using generic config (timeframe-specific not found): %s", generic_path)
            with open(generic_path, 'r') as f:
                config = json.load(f)
            _logger.debug("Loaded generic config for %s: %s", mixin_name, config.get('params', {}))
            return config

        _logger.warning("No configuration found for %s (neither timeframe-specific nor generic)", mixin_name)
        raise FileNotFoundError(f"No configuration found for {mixin_name}")

    except Exception as e:
        _logger.exception("Error loading configuration for %s: %s", mixin_name, e)
        raise

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
                    "direction": str(trade["direction"]),
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
            except Exception:
                _logger.exception("Error processing trade:")
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

    except Exception:
        _logger.exception("Error saving results: %s")
        raise


def calculate_optimization_metrics(study: optuna.Study) -> dict:
    """
    Calculate comprehensive metrics from optimization study.

    Args:
        study: Optuna study object

    Returns:
        Dictionary with metrics: best, median, mean, std, top_10_avg, etc.
        Returns None if no valid trials found.
    """
    completed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    if len(completed_trials) == 0:
        _logger.warning("No completed trials found in study")
        return None

    trial_values = [t.value for t in completed_trials if t.value is not None]

    if len(trial_values) == 0:
        _logger.warning("No valid trial values found")
        return None

    metrics = {
        'best_value': study.best_value,
        'median_value': float(np.median(trial_values)),
        'mean_value': float(np.mean(trial_values)),
        'std_value': float(np.std(trial_values)),
        'min_value': float(np.min(trial_values)),
        'max_value': float(np.max(trial_values)),
        'top_10_avg': float(np.mean(
            sorted(trial_values, reverse=True)[:min(10, len(trial_values))]
        )),
        'top_20_percent_avg': float(np.mean(
            sorted(trial_values, reverse=True)[:max(1, len(trial_values) // 5)]
        )),
        'bottom_10_avg': float(np.mean(
            sorted(trial_values)[:min(10, len(trial_values))]
        )),
        'total_trials': len(trial_values),
        'completed_trials': len(completed_trials),
        'failed_trials': len(study.trials) - len(completed_trials),
    }

    return metrics


def evaluate_combination_promise(metrics: dict, thresholds: dict) -> tuple:
    """
    Evaluate if combination is promising based on multiple criteria.

    Args:
        metrics: Dictionary of calculated metrics
        thresholds: Dictionary of threshold values

    Returns:
        Tuple of (is_promising: bool, reason: str)
    """
    if metrics is None:
        return False, "No valid metrics"

    threshold_median = thresholds.get('threshold_median', 0.05)
    threshold_best = thresholds.get('threshold_best', 0.15)
    threshold_std = thresholds.get('threshold_std', 1.0)
    min_trials = thresholds.get('min_trials_for_evaluation', 50)

    # Check minimum trials
    if metrics['total_trials'] < min_trials:
        return False, f"Insufficient trials ({metrics['total_trials']} < {min_trials})"

    # Check median return
    if metrics['median_value'] <= threshold_median:
        return False, f"Low median return ({metrics['median_value']:.4f} <= {threshold_median:.4f})"

    # Check best return
    if metrics['best_value'] <= threshold_best:
        return False, f"Low best return ({metrics['best_value']:.4f} <= {threshold_best:.4f})"

    # Check stability (standard deviation)
    if metrics['std_value'] >= threshold_std:
        return False, f"High volatility ({metrics['std_value']:.4f} >= {threshold_std:.4f})"

    return True, "Passed all criteria"


def perform_statistical_validation(
    promising_results: dict,
    unpromising_results: dict
) -> dict:
    """
    Perform statistical tests to validate that promising combinations are significantly better.

    Args:
        promising_results: Dictionary of metrics for promising combinations
        unpromising_results: Dictionary of metrics for unpromising combinations

    Returns:
        Dictionary with statistical test results
    """
    if not promising_results or not unpromising_results:
        return None

    promising_medians = [v['metrics']['median_value'] for v in promising_results.values() if v.get('metrics')]
    unpromising_medians = [v['metrics']['median_value'] for v in unpromising_results.values() if v.get('metrics')]

    if len(promising_medians) < 2 or len(unpromising_medians) < 2:
        return None

    try:
        t_stat, p_value = stats.ttest_ind(promising_medians, unpromising_medians)

        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'is_significant': p_value < 0.05,
            'promising_mean': float(np.mean(promising_medians)),
            'unpromising_mean': float(np.mean(unpromising_medians)),
            'promising_count': len(promising_medians),
            'unpromising_count': len(unpromising_medians),
        }
    except Exception as e:
        _logger.warning("Error performing statistical validation: %s", e)
        return None


def generate_summary_report(
    all_results: dict,
    statistical_validation: dict,
    output_dir: str = "results"
) -> None:
    """
    Generate summary report comparing all combinations.

    Args:
        all_results: Dictionary with all optimization results
        statistical_validation: Results from statistical validation
        output_dir: Directory to save report files
    """
    # Create summary DataFrame
    summary_data = []
    for (entry, exit), data in all_results.items():
        metrics = data.get('metrics')
        if metrics is not None:
            summary_data.append({
                'Entry Logic': entry,
                'Exit Logic': exit,
                'Best': metrics['best_value'],
                'Median': metrics['median_value'],
                'Mean': metrics['mean_value'],
                'Std': metrics['std_value'],
                'Top 10 Avg': metrics['top_10_avg'],
                'Top 20% Avg': metrics['top_20_percent_avg'],
                'Total Trials': metrics['total_trials'],
                'Stage': data.get('stage', 'unknown'),
                'Promising': data.get('is_promising', False),
            })

    if not summary_data:
        _logger.warning("No results to summarize")
        return

    summary_df = pd.DataFrame(summary_data)

    # Sort by median (most reliable metric)
    summary_df = summary_df.sort_values('Median', ascending=False)

    # Save as CSV
    os.makedirs(output_dir, exist_ok=True)
    timestamp = dt.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(output_dir, f"optimization_summary_{timestamp}.csv")
    summary_df.to_csv(csv_path, index=False)

    _logger.info("Summary report saved to %s", csv_path)

    # Print to console
    print("\n" + "=" * 120)
    print("OPTIMIZATION SUMMARY - ALL COMBINATIONS")
    print("=" * 120)
    print(summary_df.to_string(index=False))

    # Show top combinations
    print("\n" + "=" * 120)
    print("TOP 10 COMBINATIONS BY MEDIAN RETURN")
    print("=" * 120)
    print(summary_df.head(10).to_string(index=False))

    # Show statistics by stage
    print("\n" + "=" * 120)
    print("STATISTICS BY STAGE")
    print("=" * 120)

    stage_stats = summary_df.groupby('Stage').agg({
        'Median': ['count', 'mean', 'std', 'min', 'max'],
        'Best': ['mean', 'max'],
        'Std': ['mean']
    }).round(4)
    print(stage_stats)

    # Show promising vs unpromising
    print("\n" + "=" * 120)
    print("PROMISING vs UNPROMISING COMBINATIONS")
    print("=" * 120)

    promising_df = summary_df[summary_df['Promising'] == True]
    unpromising_df = summary_df[summary_df['Promising'] == False]

    print(f"\nPromising combinations: {len(promising_df)}")
    print(f"Unpromising combinations: {len(unpromising_df)}")

    if len(promising_df) > 0:
        print(f"\nPromising - Median stats: mean={promising_df['Median'].mean():.4f}, "
              f"std={promising_df['Median'].std():.4f}, "
              f"min={promising_df['Median'].min():.4f}, "
              f"max={promising_df['Median'].max():.4f}")

    if len(unpromising_df) > 0:
        print(f"Unpromising - Median stats: mean={unpromising_df['Median'].mean():.4f}, "
              f"std={unpromising_df['Median'].std():.4f}, "
              f"min={unpromising_df['Median'].min():.4f}, "
              f"max={unpromising_df['Median'].max():.4f}")

    # Statistical validation results
    if statistical_validation:
        print("\n" + "=" * 120)
        print("STATISTICAL VALIDATION (T-TEST)")
        print("=" * 120)
        print(f"T-statistic: {statistical_validation['t_statistic']:.4f}")
        print(f"P-value: {statistical_validation['p_value']:.6f}")
        print(f"Statistically significant: {statistical_validation['is_significant']} (p < 0.05)")
        print(f"Promising mean median: {statistical_validation['promising_mean']:.4f}")
        print(f"Unpromising mean median: {statistical_validation['unpromising_mean']:.4f}")

    # Overall statistics
    print("\n" + "=" * 120)
    print("OVERALL STATISTICS")
    print("=" * 120)

    stage1_count = len([d for d in all_results.values() if d.get('stage') == 'screening'])
    stage2_count = len([d for d in all_results.values() if d.get('stage') == 'deep_optimization'])
    promising_count = len([d for d in all_results.values() if d.get('is_promising', False)])

    print(f"Total combinations tested: {len(all_results)}")
    print(f"Stage 1 (screening): {stage1_count}")
    print(f"Stage 2 (deep optimization): {stage2_count}")
    print(f"Promising combinations: {promising_count}")
    if stage1_count > 0:
        print(f"Pass rate to Stage 2: {stage2_count / stage1_count * 100:.1f}%")
        print(f"Promising rate: {promising_count / stage1_count * 100:.1f}%")

    print("\n")


if __name__ == "__main__":
    """Run all optimizers with their respective configurations."""

    # Set up multiprocessing-safe logging FIRST
    setup_multiprocessing_logging()
    _logger.info("Multiprocessing-safe logging enabled for optimizer")

    with open(os.path.join("config", "optimizer", "optimizer.json"), "r",) as f:
        optimizer_config = json.load(f)

    start_time = dt.now()
    _logger.info("Starting optimization at %s", start_time)

    # Extract two-stage optimization settings
    two_stage_enabled = optimizer_config.get("optimizer_settings", {}).get("two_stage_optimization", True)
    stage1_trials = optimizer_config.get("optimizer_settings", {}).get("stage1_n_trials", 150)
    stage2_trials = optimizer_config.get("optimizer_settings", {}).get("stage2_n_trials", 500)
    selection_criteria = optimizer_config.get("optimizer_settings", {}).get("selection_criteria", {})
    dry_run_mode = optimizer_config.get("optimizer_settings", {}).get("dry_run_mode", False)

    _logger.info("Two-stage optimization: %s", "ENABLED" if two_stage_enabled else "DISABLED")
    if two_stage_enabled:
        _logger.info("Stage 1 trials: %d", stage1_trials)
        _logger.info("Stage 2 trials: %d", stage2_trials)
        _logger.info("Selection criteria: %s", selection_criteria)
    if dry_run_mode:
        _logger.info("DRY RUN MODE: Will show filtering decisions without running Stage 2")

    # Global results storage
    all_results = {}
    promising_results = {}
    unpromising_results = {}

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
        full_data_path = os.path.join("data", data_file)
        df = prepare_data_frame(full_data_path)

        # Pre-calculate HTF ATRs for the intervals used in configs (60, 120, 240, 480)
        # This is done ONCE per data file instead of 150 times per combination
        df = pre_calculate_htf_data(df, [60, 120, 240, 480])

        symbol, interval, start_date, end_date = parse_data_file_name(data_file)

        for entry_logic_name in ENTRY_MIXIN_REGISTRY.keys():
            # Load entry logic configuration (timeframe-specific with fallback)
            entry_logic_config = load_mixin_config(entry_logic_name, "entry", interval)

            for exit_logic_name in EXIT_MIXIN_REGISTRY.keys():
                processed_combinations += 1

                # Check if already processed
                if check_if_already_processed(data_file, entry_logic_name, exit_logic_name):
                    skipped_combinations += 1
                    _logger.info("Progress: %d/%d (Skipped: %d)", processed_combinations, total_combinations, skipped_combinations)
                    continue

                # Load exit logic configuration (timeframe-specific with fallback)
                exit_logic_config = load_mixin_config(exit_logic_name, "exit", interval)

                _logger.info("Running optimization %d/%d: %s + %s + %s", processed_combinations, total_combinations, data_file, entry_logic_name, exit_logic_name)

                # Create optimizer configuration
                try:

                    def objective(trial):
                        """Objective function for optimization"""
                        # Create a new data feed for each trial
                        data = prepare_data_feed(df, symbol)

                        _optimizer_config = {
                            "data": data,
                            "entry_logic": entry_logic_config,
                            "exit_logic": exit_logic_config,
                            "optimizer_settings": optimizer_config.get("optimizer_settings", {}),
                            "visualization_settings": optimizer_config.get("visualization_settings", {}),
                            "symbol": symbol,
                            "timeframe": interval,
                        }

                        # Log progress every 10 trials
                        if trial.number % 10 == 0:
                            _logger.info(f"Trial {trial.number}/{stage1_trials if study.trials == [] else stage2_trials} in progress...")

                        optimizer = CustomOptimizer(_optimizer_config)
                        # Disable analyzers for optimization trials to improve performance
                        _, _, result = optimizer.run_optimization(trial, include_analyzers=False)
                        return result["total_profit_with_commission"]

                    # ========== STAGE 1: SCREENING OPTIMIZATION ==========
                    _logger.info("STAGE 1: Screening phase (%d trials)", stage1_trials)

                    # Create study
                    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=50, n_warmup_steps=10))

                    # Run Stage 1 optimization
                    try:
                        study.optimize(
                            objective,
                            n_trials=stage1_trials,
                            n_jobs=optimizer_config.get("optimizer_settings", {}).get("n_jobs", -1),
                        )
                    except Exception as e:
                        _logger.exception("Error during Stage 1 optimization for %s + %s: %s", entry_logic_name, exit_logic_name, e)
                        raise

                    # Calculate metrics from Stage 1
                    metrics = calculate_optimization_metrics(study)

                    if metrics is None:
                        _logger.warning("No valid trials for %s + %s, skipping", entry_logic_name, exit_logic_name)
                        continue

                    # Log Stage 1 results
                    _logger.info(
                        "Stage 1 Results - %s + %s: Best=%.4f, Median=%.4f, Mean=%.4f, Std=%.4f, Top10=%.4f",
                        entry_logic_name, exit_logic_name,
                        metrics['best_value'], metrics['median_value'],
                        metrics['mean_value'], metrics['std_value'], metrics['top_10_avg']
                    )

                    # Evaluate if combination is promising
                    is_promising, reason = evaluate_combination_promise(metrics, selection_criteria)

                    # Store Stage 1 results
                    combination_key = (entry_logic_name, exit_logic_name)
                    all_results[combination_key] = {
                        'metrics': metrics,
                        'best_params': study.best_params,
                        'study': study if is_promising else None,  # Only save study for promising combinations
                        'stage': 'screening',
                        'is_promising': is_promising,
                        'reason': reason,
                    }

                    if is_promising:
                        promising_results[combination_key] = all_results[combination_key]
                    else:
                        unpromising_results[combination_key] = all_results[combination_key]

                    # Log decision
                    if is_promising:
                        _logger.info("✓ PROMISING - %s + %s: %s", entry_logic_name, exit_logic_name, reason)
                    else:
                        _logger.info("✗ FILTERED OUT - %s + %s: %s", entry_logic_name, exit_logic_name, reason)

                    # ========== STAGE 2: DEEP OPTIMIZATION (if enabled and promising) ==========
                    if two_stage_enabled and is_promising and not dry_run_mode:
                        _logger.info("STAGE 2: Deep optimization phase (additional %d trials, %d total)",
                                   stage2_trials - stage1_trials, stage2_trials)

                        # Continue from existing study (warm start)
                        try:
                            study.optimize(
                                objective,
                                n_trials=stage2_trials - stage1_trials,  # Additional trials
                                n_jobs=optimizer_config.get("optimizer_settings", {}).get("n_jobs", -1),
                            )
                        except Exception as e:
                            _logger.exception("Error during Stage 2 optimization for %s + %s: %s", entry_logic_name, exit_logic_name, e)
                            raise

                        # Recalculate metrics with all trials
                        metrics = calculate_optimization_metrics(study)

                        if metrics is None:
                            _logger.warning("No valid trials after Stage 2 for %s + %s, skipping", entry_logic_name, exit_logic_name)
                            continue

                        # Log Stage 2 results
                        _logger.info(
                            "Stage 2 Results - %s + %s: Best=%.4f, Median=%.4f, Mean=%.4f, Std=%.4f, Top10=%.4f",
                            entry_logic_name, exit_logic_name,
                            metrics['best_value'], metrics['median_value'],
                            metrics['mean_value'], metrics['std_value'], metrics['top_10_avg']
                        )

                        # Update results with Stage 2 data
                        all_results[combination_key].update({
                            'metrics': metrics,
                            'best_params': study.best_params,
                            'study': study,
                            'stage': 'deep_optimization',
                        })
                        promising_results[combination_key] = all_results[combination_key]

                    # Skip full backtest if dry run mode or not promising
                    if dry_run_mode or (two_stage_enabled and not is_promising):
                        _logger.info("Skipping full backtest for %s + %s", entry_logic_name, exit_logic_name)
                        continue

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
                        "symbol": symbol,
                        "timeframe": interval,
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
                        _logger.exception("Error in final backtest for%s + %s: %s", entry_logic_name, exit_logic_name, e)
                        raise

                    _logger.info("Completed optimization %d/%d", processed_combinations, total_combinations)

                except Exception as e:
                    _logger.exception("Error for %s + %s: %s", entry_logic_name, exit_logic_name, e)

    end_time = dt.now()
    duration = end_time - start_time

    _logger.info("Optimization completed at %s", end_time)
    _logger.info("Total duration: %s", duration)

    # Perform statistical validation
    _logger.info("Performing statistical validation...")
    statistical_validation = perform_statistical_validation(promising_results, unpromising_results)

    if statistical_validation:
        _logger.info("Statistical validation completed: p-value=%.6f, significant=%s",
                    statistical_validation['p_value'],
                    statistical_validation['is_significant'])
    else:
        _logger.info("Statistical validation skipped (insufficient data)")

    # Generate summary report
    _logger.info("Generating summary report...")
    generate_summary_report(all_results, statistical_validation)

    _logger.info("Summary:")
    _logger.info("   - Total combinations: %d", total_combinations)
    _logger.info("   - Processed: %d", processed_combinations - skipped_combinations)
    _logger.info("   - Skipped (already processed): %d", skipped_combinations)
    _logger.info("   - Promising combinations: %d", len(promising_results))
    _logger.info("   - Unpromising combinations: %d", len(unpromising_results))
    if two_stage_enabled:
        stage2_count = len([r for r in all_results.values() if r.get('stage') == 'deep_optimization'])
        _logger.info("   - Deep optimization runs: %d", stage2_count)
        if len(all_results) > 0:
            _logger.info("   - Pass rate to Stage 2: %.1f%%", stage2_count / len(all_results) * 100)
    _logger.info("   - Time saved by resume: %d minutes (estimated)", skipped_combinations * 5)
