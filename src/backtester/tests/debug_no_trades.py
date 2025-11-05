#!/usr/bin/env python
"""
Quick Diagnostic Script for "No Trades" Issue
----------------------------------------------

Run this script to diagnose why your backtest generated 0 trades.

Usage:
    python debug_no_trades.py config/backtester/your_config.json
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.backtester.tests.backtest_debugger import BacktestDebugger


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_no_trades.py <config_file>")
        print("\nAvailable configs:")

        config_dir = Path("config/backtester")
        if config_dir.exists():
            for config in config_dir.glob("*.json"):
                print(f"  - {config}")

        sys.exit(1)

    config_path = sys.argv[1]

    print("\n" + "="*80)
    print("DIAGNOSING: WHY NO TRADES?")
    print("="*80)

    debugger = BacktestDebugger(config_path)

    # Load data
    print("\nLoading data...")
    df = debugger.load_data()
    print(f"✓ Loaded {len(df)} bars")

    # Analyze entry conditions
    print("\n" + "="*80)
    debugger.analyze_entry_conditions()

    # Suggest adjustments
    print("\n" + "="*80)
    print("\nWould you like parameter suggestions? (y/n): ", end='')
    try:
        response = input().strip().lower()
        if response == 'y':
            debugger.suggest_parameter_adjustments()
    except EOFError:
        print("y")
        debugger.suggest_parameter_adjustments()

    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    print()


if __name__ == "__main__":
    main()
