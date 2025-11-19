"""
Plot Top Strategies from Performance Comparison

This script:
1. Reads performance_comparison.csv
2. Selects top 5 strategies by is_total_profit
3. Selects top 5 strategies by robustness_score
4. Creates plots for each selected strategy using the plotter
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from src.notification.logger import setup_logger
from src.backtester.plotter.run_plotter import ResultPlotter

_logger = setup_logger(__name__)


def load_performance_comparison(csv_path: str = "results/walk_forward_reports/performance_comparison.csv") -> pd.DataFrame:
    """Load performance comparison CSV"""
    try:
        df = pd.read_csv(csv_path)
        _logger.info("Loaded %d comparisons from %s", len(df), csv_path)
        return df
    except Exception:
        _logger.exception("Error loading performance comparison CSV:")
        return pd.DataFrame()


def select_top_strategies(df: pd.DataFrame, n: int = 5) -> dict:
    """
    Select top strategies based on different criteria.

    Args:
        df: Performance comparison dataframe
        n: Number of top strategies to select

    Returns:
        Dictionary with strategy selections
    """
    selections = {}

    # 1. Top 5 by IS total profit
    top_by_is_profit = df.nlargest(n, 'is_total_profit')
    selections['top_is_profit'] = top_by_is_profit[[
        'strategy_id', 'is_period', 'oos_period', 'symbol', 'timeframe',
        'is_total_profit', 'oos_total_profit', 'robustness_score'
    ]]

    _logger.info("\n" + "="*80)
    _logger.info("TOP %d STRATEGIES BY IN-SAMPLE PROFIT:", n)
    _logger.info("="*80)
    for idx, row in selections['top_is_profit'].iterrows():
        _logger.info("  %s (%s)", row['strategy_id'], row['is_period'])
        _logger.info("    IS Profit: $%.2f | OOS Profit: $%.2f | Robustness: %.3f",
                    row['is_total_profit'], row['oos_total_profit'], row['robustness_score'])

    # 2. Top 5 by robustness score
    top_by_robustness = df.nlargest(n, 'robustness_score')
    selections['top_robustness'] = top_by_robustness[[
        'strategy_id', 'is_period', 'oos_period', 'symbol', 'timeframe',
        'is_total_profit', 'oos_total_profit', 'robustness_score'
    ]]

    _logger.info("\n" + "="*80)
    _logger.info("TOP %d STRATEGIES BY ROBUSTNESS:", n)
    _logger.info("="*80)
    for idx, row in selections['top_robustness'].iterrows():
        _logger.info("  %s (%s)", row['strategy_id'], row['is_period'])
        _logger.info("    IS Profit: $%.2f | OOS Profit: $%.2f | Robustness: %.3f",
                    row['is_total_profit'], row['oos_total_profit'], row['robustness_score'])

    return selections


def get_json_path_for_strategy(strategy_id: str, is_period: str) -> str:
    """
    Get the JSON file path for a strategy.

    The actual filename includes timestamps and date ranges, so we need to find
    the matching file by pattern.

    Args:
        strategy_id: Strategy identifier (e.g., BTCUSDT_4h_RSI_ATR)
        is_period: In-sample period (year)

    Returns:
        Path to JSON file, or None if not found
    """
    import glob

    # The strategy_id format is: SYMBOL_TIMEFRAME_ENTRY_EXIT
    # But actual files are: SYMBOL_TIMEFRAME_STARTDATE_ENDDATE_ENTRY_EXIT_TIMESTAMP.json
    # Example: ETHUSDT_1h_20200101_20201231_RSIOrBBEntryMixin_AdvancedATRExitMixin_20251115_072257.json

    directory = f"results/walk_forward_reports/{is_period}"

    # Try to find files matching the pattern
    # Pattern: {directory}/*_{strategy_parts}.json
    parts = strategy_id.split('_')
    if len(parts) >= 4:
        # Extract symbol, timeframe, entry, exit
        symbol = parts[0]
        timeframe = parts[1]
        entry = '_'.join(parts[2:-1])  # Entry might have underscores
        exit_mixin = parts[-1]

        # Search pattern: SYMBOL_TIMEFRAME_*_ENTRY_EXIT_*.json
        pattern = f"{directory}/{symbol}_{timeframe}_*_{entry}_{exit_mixin}_*.json"
        matches = glob.glob(pattern)

        if matches:
            # Return the most recent file (by timestamp in filename)
            return sorted(matches)[-1]

    # Fallback: try direct match (might work if no timestamp)
    direct_path = f"{directory}/{strategy_id}.json"
    if os.path.exists(direct_path):
        return direct_path

    return None


def plot_selected_strategies(selections: dict, output_dir: str = "results/walk_forward_reports/"):
    """
    Create plots for selected strategies.

    Args:
        selections: Dictionary of strategy selections
        output_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize plotter
    plotter = ResultPlotter()

    # Track processed files and missing files
    processed_files = set()
    missing_files = []
    skipped_duplicates = 0

    # Process each selection category
    for category, df in selections.items():
        _logger.info("\n" + "="*80)
        _logger.info("Processing %s strategies...", category.replace('_', ' ').upper())
        _logger.info("="*80)

        category_processed = 0
        for idx, row in df.iterrows():
            strategy_id = row['strategy_id']
            is_period = row['is_period']

            # Get JSON file path
            json_file = get_json_path_for_strategy(strategy_id, is_period)

            # Check if file exists
            if json_file is None:
                _logger.warning("⚠ JSON file not found for: %s (%s)", strategy_id, is_period)
                missing_files.append({
                    'strategy_id': strategy_id,
                    'is_period': is_period,
                    'file': f"results/walk_forward_reports/{is_period}/{strategy_id}_*.json",
                    'category': category
                })
                continue

            # Skip if already processed
            if json_file in processed_files:
                _logger.debug("Skipping %s (already processed)", strategy_id)
                skipped_duplicates += 1
                continue

            _logger.info("✓ Plotting: %s (%s)", strategy_id, is_period)

            try:
                # Create output subdirectory for this category
                category_dir = os.path.join(output_dir, category)
                os.makedirs(category_dir, exist_ok=True)

                # Process the JSON file
                plotter.process_json_file(json_file)

                # Move the generated plot to the category directory
                plot_file = json_file.replace('.json', '.png')
                if os.path.exists(plot_file):
                    new_plot_path = os.path.join(
                        category_dir,
                        f"{strategy_id}_{is_period}.png"
                    )
                    os.rename(plot_file, new_plot_path)
                    _logger.info("  → Saved: %s", new_plot_path)
                    category_processed += 1

                processed_files.add(json_file)

            except Exception:
                _logger.exception("✗ Error plotting %s:", json_file)

        _logger.info("Category %s: %d plots created", category, category_processed)

    # Summary
    _logger.info("\n" + "="*80)
    _logger.info("PLOTTING COMPLETED!")
    _logger.info("="*80)
    _logger.info("✓ Total plots created: %d", len(processed_files))
    _logger.info("⊘ Skipped duplicates: %d", skipped_duplicates)
    _logger.info("⚠ Missing files: %d", len(missing_files))
    _logger.info("Output directory: %s", output_dir)

    # Report missing files
    if missing_files:
        _logger.warning("\n" + "="*80)
        _logger.warning("MISSING JSON FILES REPORT")
        _logger.warning("="*80)
        _logger.warning("The following strategies were selected but JSON files not found:")
        for item in missing_files:
            _logger.warning("  - %s (%s) [%s]", item['strategy_id'], item['is_period'], item['category'])
        _logger.warning("\nPossible reasons:")
        _logger.warning("  1. Optimization wasn't run for these years")
        _logger.warning("  2. JSON files were deleted or moved")
        _logger.warning("  3. Strategy naming mismatch")
        _logger.warning("\nTo fix: Re-run walk_forward_optimizer.py for the missing years")

    return processed_files, missing_files


def create_summary_report(selections: dict, output_dir: str = "results/walk_forward_reports"):
    """
    Create a summary report of selected strategies.

    Args:
        selections: Dictionary of strategy selections
        output_dir: Directory to save report
    """
    report_path = os.path.join(output_dir, "selection_summary.txt")

    with open(report_path, 'w') as f:
        f.write("TOP STRATEGIES SELECTION SUMMARY\n")
        f.write("="*80 + "\n\n")

        for category, df in selections.items():
            f.write(f"\n{category.replace('_', ' ').upper()}\n")
            f.write("-"*80 + "\n")

            for idx, row in df.iterrows():
                f.write(f"\n{idx+1}. {row['strategy_id']} ({row['is_period']})\n")
                f.write(f"   Symbol/Timeframe: {row['symbol']} / {row['timeframe']}\n")
                f.write(f"   IS Period: {row['is_period']} | OOS Period: {row['oos_period']}\n")
                f.write(f"   IS Profit: ${row['is_total_profit']:.2f}\n")
                f.write(f"   OOS Profit: ${row['oos_total_profit']:.2f}\n")
                f.write(f"   Robustness Score: {row['robustness_score']:.4f}\n")

                # Calculate profit percentage
                is_profit_pct = (row['is_total_profit'] / 1000) * 100
                oos_profit_pct = (row['oos_total_profit'] / 1000) * 100
                f.write(f"   IS Return: {is_profit_pct:.2f}%\n")
                f.write(f"   OOS Return: {oos_profit_pct:.2f}%\n")

            f.write("\n")

    _logger.info("Summary report saved: %s", report_path)


def main():
    """Main function to plot top strategies"""
    _logger.info("="*80)
    _logger.info("TOP STRATEGIES PLOTTER")
    _logger.info("="*80)

    # Load performance comparison
    df = load_performance_comparison()
    if df.empty:
        _logger.error("No data loaded. Exiting.")
        return

    # Select top strategies
    selections = select_top_strategies(df, n=5)

    # Create summary report
    create_summary_report(selections)

    # Plot selected strategies
    plot_selected_strategies(selections)

    _logger.info("\n" + "="*80)
    _logger.info("ALL DONE!")
    _logger.info("="*80)
    _logger.info("📊 Check results/walk_forward_reports/ for:")
    _logger.info("   - PNG plots in categorized folders")
    _logger.info("   - selection_summary.txt with detailed metrics")


if __name__ == "__main__":
    main()
