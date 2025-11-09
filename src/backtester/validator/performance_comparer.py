"""
Performance Comparer Module

This module compares in-sample (IS) and out-of-sample (OOS) results to:
1. Calculate degradation metrics
2. Identify overfitting and robust strategies
3. Generate comprehensive comparison reports (CSV/JSON)
4. Rank strategies by OOS performance

Output formats:
- performance_comparison.csv: Detailed IS/OOS comparison for all strategies
- degradation_analysis.json: Aggregate statistics and insights
- robustness_summary.csv: Aggregated robustness metrics per strategy
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import json
import csv
from datetime import datetime as dt
from typing import Dict, List, Tuple
from collections import defaultdict

import pandas as pd
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def load_all_results(base_dir: str) -> Dict[str, Dict]:
    """
    Load all result JSON files from subdirectories (organized by year).

    Args:
        base_dir: Base directory (e.g., 'results/optimization' or 'results/validation')

    Returns:
        dict: Nested dictionary mapping year -> strategy_key -> result
    """
    results_by_year = {}

    if not os.path.exists(base_dir):
        _logger.warning("Directory not found: %s", base_dir)
        return results_by_year

    # Iterate through year subdirectories
    for year_dir in os.listdir(base_dir):
        year_path = os.path.join(base_dir, year_dir)

        if not os.path.isdir(year_path):
            continue

        _logger.info("Loading results from %s", year_path)
        year_results = {}

        json_files = [f for f in os.listdir(year_path) if f.endswith('.json')]

        for json_file in json_files:
            try:
                file_path = os.path.join(year_path, json_file)
                with open(file_path, 'r') as f:
                    result = json.load(f)

                # Extract strategy key
                symbol = result.get('symbol', '')
                timeframe = result.get('timeframe', '')
                entry_name = result.get('best_params', {}).get('entry_logic', {}).get('name', '')
                exit_name = result.get('best_params', {}).get('exit_logic', {}).get('name', '')

                if not all([symbol, timeframe, entry_name, exit_name]):
                    _logger.warning("Skipping %s: missing key fields", json_file)
                    continue

                strategy_key = f"{symbol}_{timeframe}_{entry_name}_{exit_name}"
                year_results[strategy_key] = result

            except Exception as e:
                _logger.exception("Error loading %s: %s", json_file, e)
                continue

        results_by_year[year_dir] = year_results
        _logger.info("  Loaded %d results from %s", len(year_results), year_dir)

    return results_by_year


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def calculate_degradation_metrics(is_result: dict, oos_result: dict) -> dict:
    """
    Calculate degradation and robustness metrics between IS and OOS results.

    Args:
        is_result: In-sample optimization result
        oos_result: Out-of-sample validation result

    Returns:
        dict: Calculated metrics
    """
    # Extract IS metrics
    is_profit = is_result.get('total_profit_with_commission', 0.0)
    is_trades = is_result.get('total_trades', 0)
    is_analyzers = is_result.get('analyzers', {})

    # Extract OOS metrics
    oos_profit = oos_result.get('total_profit_with_commission', 0.0)
    oos_trades = oos_result.get('total_trades', 0)
    oos_analyzers = oos_result.get('analyzers', {})

    # Profit degradation
    profit_degradation_ratio = safe_divide(oos_profit, is_profit, default=0.0)
    profit_degradation_pct = ((is_profit - oos_profit) / abs(is_profit) * 100) if is_profit != 0 else 0.0

    # Win rate
    is_win_rate = is_analyzers.get('winrate', {}).get('winrate', 0.0)
    oos_win_rate = oos_analyzers.get('winrate', {}).get('winrate', 0.0)
    win_rate_degradation = abs(is_win_rate - oos_win_rate)

    # Sharpe ratio
    is_sharpe = is_analyzers.get('sharpe', {}).get('sharperatio', 0.0)
    oos_sharpe = oos_analyzers.get('sharpe', {}).get('sharperatio', 0.0)
    sharpe_degradation = safe_divide(is_sharpe - oos_sharpe, abs(is_sharpe), default=0.0) if is_sharpe != 0 else 0.0

    # Drawdown
    is_max_dd = is_analyzers.get('drawdown', {}).get('max', {}).get('drawdown', 0.0)
    oos_max_dd = oos_analyzers.get('drawdown', {}).get('max', {}).get('drawdown', 0.0)
    drawdown_increase = oos_max_dd - is_max_dd

    # Profit factor
    is_profit_factor = is_analyzers.get('profit_factor', {}).get('profit_factor', 0.0)
    oos_profit_factor = oos_analyzers.get('profit_factor', {}).get('profit_factor', 0.0)

    # Trade count consistency
    trade_count_ratio = safe_divide(oos_trades, is_trades, default=0.0)

    # Calculate overfitting score (0-1, higher = more overfitting)
    profit_penalty = max(0, 1.0 - profit_degradation_ratio) if is_profit > 0 else 0.5
    sharpe_penalty = abs(sharpe_degradation)
    trade_inconsistency = abs(1.0 - trade_count_ratio)
    win_rate_inconsistency = win_rate_degradation / 100.0  # Normalize to 0-1

    overfitting_score = (
        0.4 * profit_penalty +
        0.3 * min(sharpe_penalty, 1.0) +
        0.2 * min(trade_inconsistency, 1.0) +
        0.1 * min(win_rate_inconsistency, 1.0)
    )

    # Robustness score (inverse of overfitting)
    robustness_score = 1.0 - overfitting_score

    return {
        'is_profit': is_profit,
        'oos_profit': oos_profit,
        'profit_degradation_ratio': profit_degradation_ratio,
        'profit_degradation_pct': profit_degradation_pct,
        'is_trade_count': is_trades,
        'oos_trade_count': oos_trades,
        'trade_count_ratio': trade_count_ratio,
        'is_win_rate': is_win_rate,
        'oos_win_rate': oos_win_rate,
        'win_rate_degradation': win_rate_degradation,
        'is_sharpe_ratio': is_sharpe,
        'oos_sharpe_ratio': oos_sharpe,
        'sharpe_degradation': sharpe_degradation,
        'is_max_drawdown': is_max_dd,
        'oos_max_drawdown': oos_max_dd,
        'drawdown_increase': drawdown_increase,
        'is_profit_factor': is_profit_factor,
        'oos_profit_factor': oos_profit_factor,
        'overfitting_score': overfitting_score,
        'robustness_score': robustness_score,
    }


def generate_comparison_csv(comparisons: List[dict], output_path: str):
    """
    Generate CSV report with IS/OOS comparison.

    Args:
        comparisons: List of comparison dictionaries
        output_path: Output CSV file path
    """
    if not comparisons:
        _logger.warning("No comparisons to write to CSV")
        return

    # Define CSV columns
    fieldnames = [
        'strategy_id',
        'window_name',
        'symbol',
        'timeframe',
        'entry_mixin',
        'exit_mixin',
        'is_period',
        'oos_period',
        'is_total_profit',
        'oos_total_profit',
        'profit_degradation_ratio',
        'profit_degradation_pct',
        'is_trade_count',
        'oos_trade_count',
        'trade_count_ratio',
        'is_win_rate',
        'oos_win_rate',
        'win_rate_degradation',
        'is_sharpe_ratio',
        'oos_sharpe_ratio',
        'sharpe_degradation',
        'is_max_drawdown',
        'oos_max_drawdown',
        'drawdown_increase',
        'is_profit_factor',
        'oos_profit_factor',
        'overfitting_score',
        'robustness_score',
    ]

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for comparison in comparisons:
            writer.writerow(comparison)

    _logger.info("Saved comparison CSV to %s", output_path)
    _logger.info("  Total rows: %d", len(comparisons))


def generate_degradation_json(comparisons: List[dict], output_path: str):
    """
    Generate JSON report with detailed degradation analysis.

    Args:
        comparisons: List of comparison dictionaries
        output_path: Output JSON file path
    """
    if not comparisons:
        _logger.warning("No comparisons to analyze")
        return

    # Calculate summary statistics
    total_strategies = len(comparisons)
    positive_oos = sum(1 for c in comparisons if c['oos_total_profit'] > 0)
    negative_oos = total_strategies - positive_oos

    avg_profit_degradation = sum(c['profit_degradation_ratio'] for c in comparisons) / total_strategies
    avg_sharpe_degradation = sum(c['sharpe_degradation'] for c in comparisons) / total_strategies
    avg_robustness = sum(c['robustness_score'] for c in comparisons) / total_strategies

    high_robustness = sum(1 for c in comparisons if c['robustness_score'] > 0.7)

    # Group by window
    by_window = defaultdict(list)
    for c in comparisons:
        by_window[c['window_name']].append(c)

    window_stats = {}
    for window_name, window_comps in by_window.items():
        avg_deg = sum(c['profit_degradation_ratio'] for c in window_comps) / len(window_comps)
        sorted_by_oos = sorted(window_comps, key=lambda x: x['oos_total_profit'], reverse=True)

        window_stats[window_name] = {
            'avg_profit_degradation': round(avg_deg, 4),
            'total_strategies': len(window_comps),
            'best_strategy': sorted_by_oos[0]['strategy_id'] if sorted_by_oos else None,
            'best_oos_profit': round(sorted_by_oos[0]['oos_total_profit'], 2) if sorted_by_oos else 0,
            'worst_strategy': sorted_by_oos[-1]['strategy_id'] if sorted_by_oos else None,
            'worst_oos_profit': round(sorted_by_oos[-1]['oos_total_profit'], 2) if sorted_by_oos else 0,
        }

    # Group by symbol
    by_symbol = defaultdict(list)
    for c in comparisons:
        by_symbol[c['symbol']].append(c)

    symbol_stats = {}
    for symbol, symbol_comps in by_symbol.items():
        avg_deg = sum(c['profit_degradation_ratio'] for c in symbol_comps) / len(symbol_comps)
        symbol_stats[symbol] = {
            'avg_profit_degradation': round(avg_deg, 4),
            'total_strategies': len(symbol_comps),
        }

    # Group by timeframe
    by_timeframe = defaultdict(list)
    for c in comparisons:
        by_timeframe[c['timeframe']].append(c)

    timeframe_stats = {}
    for timeframe, tf_comps in by_timeframe.items():
        avg_deg = sum(c['profit_degradation_ratio'] for c in tf_comps) / len(tf_comps)
        timeframe_stats[timeframe] = {
            'avg_profit_degradation': round(avg_deg, 4),
            'total_strategies': len(tf_comps),
        }

    # Top strategies by OOS profit
    sorted_by_oos_profit = sorted(comparisons, key=lambda x: x['oos_total_profit'], reverse=True)[:10]
    top_strategies = [
        {
            'strategy_id': c['strategy_id'],
            'oos_total_profit': round(c['oos_total_profit'], 2),
            'degradation_ratio': round(c['profit_degradation_ratio'], 4),
            'robustness_score': round(c['robustness_score'], 4),
        }
        for c in sorted_by_oos_profit
    ]

    # Most robust strategies
    sorted_by_robustness = sorted(comparisons, key=lambda x: x['robustness_score'], reverse=True)[:10]
    most_robust = [
        {
            'strategy_id': c['strategy_id'],
            'degradation_ratio': round(c['profit_degradation_ratio'], 4),
            'robustness_score': round(c['robustness_score'], 4),
            'oos_total_profit': round(c['oos_total_profit'], 2),
        }
        for c in sorted_by_robustness
    ]

    # Warning flags
    warning_flags = []
    for c in comparisons:
        if c['profit_degradation_ratio'] < 0.5 and c['is_total_profit'] > 0:
            warning_flags.append({
                'strategy_id': c['strategy_id'],
                'issue': 'Profit degradation > 50%',
                'degradation_ratio': round(c['profit_degradation_ratio'], 4),
                'recommendation': 'Likely overfit, avoid using',
            })
        elif c['oos_total_profit'] < 0 and c['is_total_profit'] > 0:
            warning_flags.append({
                'strategy_id': c['strategy_id'],
                'issue': 'OOS losses when IS profitable',
                'oos_total_profit': round(c['oos_total_profit'], 2),
                'recommendation': 'Strategy failed OOS test',
            })

    # Construct final JSON
    analysis = {
        'summary': {
            'total_strategies': total_strategies,
            'total_windows': len(by_window),
            'avg_profit_degradation_ratio': round(avg_profit_degradation, 4),
            'avg_sharpe_degradation': round(avg_sharpe_degradation, 4),
            'avg_robustness_score': round(avg_robustness, 4),
            'strategies_with_positive_oos': positive_oos,
            'strategies_with_negative_oos': negative_oos,
            'high_robustness_strategies': high_robustness,
        },
        'by_window': window_stats,
        'by_symbol': symbol_stats,
        'by_timeframe': timeframe_stats,
        'top_strategies_by_oos_profit': top_strategies,
        'most_robust_strategies': most_robust,
        'warning_flags': warning_flags,
    }

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=4)

    _logger.info("Saved degradation analysis JSON to %s", output_path)


def generate_robustness_summary_csv(comparisons: List[dict], output_path: str):
    """
    Generate robustness summary CSV with aggregated metrics per strategy across all windows.

    Args:
        comparisons: List of comparison dictionaries
        output_path: Output CSV file path
    """
    if not comparisons:
        _logger.warning("No comparisons for robustness summary")
        return

    # Group by strategy (across all windows)
    by_strategy = defaultdict(list)
    for c in comparisons:
        strategy_id = c['strategy_id']
        by_strategy[strategy_id].append(c)

    # Calculate aggregated metrics
    summary_rows = []
    for strategy_id, strategy_comps in by_strategy.items():
        windows_tested = len(strategy_comps)
        avg_oos_profit = sum(c['oos_total_profit'] for c in strategy_comps) / windows_tested
        std_oos_profit = pd.Series([c['oos_total_profit'] for c in strategy_comps]).std()

        avg_degradation = sum(c['profit_degradation_ratio'] for c in strategy_comps) / windows_tested
        min_degradation = min(c['profit_degradation_ratio'] for c in strategy_comps)
        max_degradation = max(c['profit_degradation_ratio'] for c in strategy_comps)

        avg_robustness = sum(c['robustness_score'] for c in strategy_comps) / windows_tested

        # Consistency score (lower std_oos_profit relative to avg is better)
        consistency_score = 1.0 - min(std_oos_profit / (abs(avg_oos_profit) + 1.0), 1.0)

        # Overall robustness (combines degradation and consistency)
        overall_robustness = (avg_robustness + consistency_score) / 2.0

        # Recommendation
        if overall_robustness > 0.7 and avg_oos_profit > 0:
            recommendation = "Excellent"
        elif overall_robustness > 0.5 and avg_oos_profit > 0:
            recommendation = "Good"
        elif overall_robustness > 0.3 and avg_oos_profit > 0:
            recommendation = "Fair"
        elif avg_oos_profit < 0:
            recommendation = "Reject"
        else:
            recommendation = "Poor"

        # Extract metadata from first comparison
        first_comp = strategy_comps[0]

        summary_rows.append({
            'strategy_id': strategy_id,
            'symbol': first_comp['symbol'],
            'timeframe': first_comp['timeframe'],
            'entry_mixin': first_comp['entry_mixin'],
            'exit_mixin': first_comp['exit_mixin'],
            'windows_tested': windows_tested,
            'avg_oos_profit': round(avg_oos_profit, 2),
            'std_oos_profit': round(std_oos_profit, 2),
            'avg_degradation_ratio': round(avg_degradation, 4),
            'min_degradation_ratio': round(min_degradation, 4),
            'max_degradation_ratio': round(max_degradation, 4),
            'consistency_score': round(consistency_score, 4),
            'overall_robustness_score': round(overall_robustness, 4),
            'recommendation': recommendation,
        })

    # Sort by overall robustness (descending)
    summary_rows.sort(key=lambda x: x['overall_robustness_score'], reverse=True)

    # Write CSV
    fieldnames = [
        'strategy_id',
        'symbol',
        'timeframe',
        'entry_mixin',
        'exit_mixin',
        'windows_tested',
        'avg_oos_profit',
        'std_oos_profit',
        'avg_degradation_ratio',
        'min_degradation_ratio',
        'max_degradation_ratio',
        'consistency_score',
        'overall_robustness_score',
        'recommendation',
    ]

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    _logger.info("Saved robustness summary CSV to %s", output_path)
    _logger.info("  Total strategies: %d", len(summary_rows))


def main():
    """Main orchestrator for performance comparison."""
    start_time = dt.now()
    _logger.info("=" * 80)
    _logger.info("Performance Comparison Started")
    _logger.info("=" * 80)
    _logger.info("Start time: %s", start_time)

    # Load IS and OOS results
    _logger.info("Loading in-sample (IS) results...")
    is_results_by_year = load_all_results("results/optimization")

    _logger.info("Loading out-of-sample (OOS) results...")
    oos_results_by_year = load_all_results("results/validation")

    # Match IS and OOS results
    comparisons = []

    for oos_year, oos_results in oos_results_by_year.items():
        _logger.info("")
        _logger.info("Processing OOS results from %s", oos_year)

        for strategy_key, oos_result in oos_results.items():
            # Determine which IS year this OOS result corresponds to
            trained_on_year = oos_result.get('trained_on_year')

            if not trained_on_year:
                _logger.warning("OOS result for %s missing 'trained_on_year', skipping", strategy_key)
                continue

            # Find corresponding IS result
            if trained_on_year not in is_results_by_year:
                _logger.warning("IS results for year %s not found, skipping", trained_on_year)
                continue

            is_results = is_results_by_year[trained_on_year]

            if strategy_key not in is_results:
                _logger.warning("IS result for %s not found in year %s, skipping", strategy_key, trained_on_year)
                continue

            is_result = is_results[strategy_key]

            # Calculate degradation metrics
            metrics = calculate_degradation_metrics(is_result, oos_result)

            # Build comparison record
            comparison = {
                'strategy_id': strategy_key,
                'window_name': oos_result.get('window_name', ''),
                'symbol': oos_result.get('symbol', ''),
                'timeframe': oos_result.get('timeframe', ''),
                'entry_mixin': oos_result.get('best_params', {}).get('entry_logic', {}).get('name', ''),
                'exit_mixin': oos_result.get('best_params', {}).get('exit_logic', {}).get('name', ''),
                'is_period': trained_on_year,
                'oos_period': oos_year,
                **metrics
            }

            comparisons.append(comparison)

            _logger.debug(
                "  Compared: %s | IS Profit: %.2f | OOS Profit: %.2f | Degradation: %.2f%%",
                strategy_key,
                metrics['is_profit'],
                metrics['oos_profit'],
                metrics['profit_degradation_pct']
            )

    # Generate reports
    _logger.info("")
    _logger.info("=" * 80)
    _logger.info("Generating Reports")
    _logger.info("=" * 80)
    _logger.info("Total comparisons: %d", len(comparisons))

    if comparisons:
        # CSV report
        csv_output = "results/walk_forward_reports/performance_comparison.csv"
        generate_comparison_csv(comparisons, csv_output)

        # JSON analysis report
        json_output = "results/walk_forward_reports/degradation_analysis.json"
        generate_degradation_json(comparisons, json_output)

        # Robustness summary CSV
        robustness_output = "results/walk_forward_reports/robustness_summary.csv"
        generate_robustness_summary_csv(comparisons, robustness_output)

    else:
        _logger.warning("No comparisons generated. Check that IS and OOS results exist.")

    # Summary
    end_time = dt.now()
    duration = end_time - start_time

    _logger.info("")
    _logger.info("=" * 80)
    _logger.info("Performance Comparison Completed")
    _logger.info("=" * 80)
    _logger.info("End time: %s", end_time)
    _logger.info("Total duration: %s", duration)
    _logger.info("Reports saved to: results/walk_forward_reports/")
    _logger.info("=" * 80)


if __name__ == "__main__":
    main()
