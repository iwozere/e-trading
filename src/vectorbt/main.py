"""
Main CLI Entry Point

src/vectorbt/main.py optimize --interval 1h --trials 2 --symbols BTC,XRP,LTC,ETH --strategy src/vectorbt/configs/default_strategy.json

Orchestrates the entire vectorbt optimization pipeline:
1. Data loading
2. Signal generation
3. Optuna optimization
4. Reporting
5. Strategy promotion
"""

import sys
import argparse
import os
from pathlib import Path
from typing import Optional, List, Union
from datetime import datetime

# Add project root to sys.path to allow absolute imports from 'src'
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vectorbt.pipeline.manager import StudyManager
from src.vectorbt.tools.promoter import StrategyPromoter
from src.notification.logger import setup_logger, setup_multiprocessing_logging

# -----------------------------------------------------------------------------
# DEBUG CONFIGURATION (VS CODE)
# -----------------------------------------------------------------------------
# This section allows running and debugging from VS Code (F5) without manually
# entering CLI arguments. These will be used if CLI arguments are omitted.
DEBUG_MODE = True # Set to False for production use
DEFAULT_INTERVALS = "4h,1h,30m,15m,5m"  # Comma-separated
DEFAULT_SYMBOLS = "BTC,XRP,LTC,ETH"   # Comma-separated
DEFAULT_TRIALS = 10
DEFAULT_STRATEGY = "src/vectorbt/configs/triple-filter-trend-rider.json"
DEFAULT_BATCH = True
DEFAULT_JOBS = 6

# -----------------------------------------------------------------------------
# CLI DEFAULTS
# -----------------------------------------------------------------------------
CLI_DEFAULT_TOP_N = 5
CLI_DEFAULT_USER_ID = 1
CLI_DEFAULT_MIN_CALMAR = 1.5
CLI_DEFAULT_MAX_DRAWDOWN = 0.4
CLI_DEFAULT_MIN_TRADES = 20

# Initialize component-level logger
_logger = setup_logger("vectorbt_pipeline")

def run_optimization(
    interval: str,
    n_trials: int = 100,
    n_jobs: Optional[int] = None,
    study_name: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    strategy_path: str = DEFAULT_STRATEGY,
    strategy_name: str = "triple-filter-trend-rider",
):
    """
    Run optimization study.
    """
    manager = StudyManager()
    study = manager.run_optimization(
        interval=interval,
        n_trials=n_trials,
        n_jobs=n_jobs,
        study_name=study_name,
        symbols=symbols,
        strategy_path=strategy_path,
        strategy_name=strategy_name,
    )

    if study is None:
        _logger.error(f"Optimization failed for {symbols} at {interval}")
        return False

    _logger.info(f"✅ Optimization complete for '{study.study_name}'")
    return True


def promote_strategies(
    study_name: str,
    top_n: int = 5,
    user_id: int = 1,
    min_calmar: float = 1.5,
    max_drawdown: float = 0.4,
    min_trades: int = 20,
):
    """
    Promote top strategies from SQLite to PostgreSQL.
    """
    promoter = StrategyPromoter(
        min_calmar=min_calmar, max_drawdown=max_drawdown, min_trades=min_trades
    )

    promoted_ids = promoter.promote_top_trials(
        study_name=study_name, top_n=top_n, user_id=user_id
    )

    if not promoted_ids:
        _logger.warning(f"No strategies were promoted from {study_name}")
        return False

    _logger.info(f"✅ Promoted {len(promoted_ids)} strategies to PostgreSQL")
    return True


def list_studies():
    """List all available optimization studies."""
    promoter = StrategyPromoter()
    studies = promoter.list_studies()

    if not studies:
        print("No studies found in the database.")
        return

    print("\n📚 Available optimization studies:")
    for study in studies:
        print(f"  - {study}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Vectorbt Trading Strategy Optimization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Run optimization study")
    optimize_parser.add_argument(
        "--interval", type=str, default=DEFAULT_INTERVALS, help=f"Data interval(s), comma-separated (default: {DEFAULT_INTERVALS})"
    )
    optimize_parser.add_argument(
        "--trials", type=int, default=DEFAULT_TRIALS, help=f"Number of trials (default: {DEFAULT_TRIALS})"
    )
    optimize_parser.add_argument(
        "--jobs", type=int, default=DEFAULT_JOBS, help=f"Number of parallel jobs (default: {DEFAULT_JOBS})"
    )
    optimize_parser.add_argument(
        "--symbols", type=str, default=DEFAULT_SYMBOLS, help=f"Comma-separated symbols to optimize (default: {DEFAULT_SYMBOLS})"
    )
    optimize_parser.add_argument(
        "--study-name", type=str, default=None, help="Custom study name (default: auto)"
    )
    optimize_parser.add_argument(
        "--auto-promote",
        action="store_true",
        help="Automatically promote top strategies after optimization",
    )
    optimize_parser.add_argument(
        "--batch",
        action="store_true",
        default=DEFAULT_BATCH,
        help=f"Run independent optimization for each symbol/interval sequentially (default: {DEFAULT_BATCH})",
    )
    optimize_parser.add_argument(
        "--strategy",
        type=str,
        default=DEFAULT_STRATEGY,
        help=f"Path to strategy JSON config (default: {DEFAULT_STRATEGY})"
    )

    # Promote command
    promote_parser = subparsers.add_parser(
        "promote", help="Promote strategies to production"
    )
    promote_parser.add_argument(
        "study_name", type=str, help="Name of the study to promote from"
    )
    promote_parser.add_argument(
        "--top-n", type=int, default=CLI_DEFAULT_TOP_N, help=f"Number of top trials to promote (default: {CLI_DEFAULT_TOP_N})"
    )
    promote_parser.add_argument(
        "--user-id", type=int, default=CLI_DEFAULT_USER_ID, help=f"User ID to associate (default: {CLI_DEFAULT_USER_ID})"
    )
    promote_parser.add_argument(
        "--min-calmar",
        type=float,
        default=CLI_DEFAULT_MIN_CALMAR,
        help=f"Minimum Calmar ratio (default: {CLI_DEFAULT_MIN_CALMAR})",
    )
    promote_parser.add_argument(
        "--max-drawdown",
        type=float,
        default=CLI_DEFAULT_MAX_DRAWDOWN,
        help=f"Max drawdown threshold (default: {CLI_DEFAULT_MAX_DRAWDOWN})",
    )
    promote_parser.add_argument(
        "--min-trades",
        type=int,
        default=CLI_DEFAULT_MIN_TRADES,
        help=f"Minimum trades required (default: {CLI_DEFAULT_MIN_TRADES})",
    )

    # List studies command
    subparsers.add_parser("list-studies", help="List all optimization studies")

    # Handle VS Code Debugging (no args provided)
    if len(sys.argv) == 1 and DEBUG_MODE:
        _logger.info("🐞 Debug mode: Running 'optimize' with DEFAULT_* constants")
        args = parser.parse_args(["optimize"])
    else:
        args = parser.parse_args()

    # Initialise multiprocessing-safe logging once in the main process
    setup_multiprocessing_logging()

    _logger.info("Initializing Vectorbt Trading Pipeline CLI")

    # 1. Extract clean strategy name from path
    strategy_name = os.path.basename(args.strategy).replace(".json", "") if hasattr(args, 'strategy') else "default_strategy"

    # Execute command
    if args.command == "optimize":
        symbols_list = [s.strip() for s in args.symbols.split(",")] if args.symbols else []
        intervals_list = [i.strip() for i in args.interval.split(",")] if args.interval else []

        _logger.info(f"📊 Targets: {len(symbols_list)} symbols, {len(intervals_list)} intervals")

        for interval in intervals_list:
            # Determine if we run as Batch or Portfolio
            if args.batch and symbols_list:
                _logger.info(f"🚀 Starting BATCH optimization for interval: {interval}")
                for symbol in symbols_list:
                    _logger.info(f"--- Optimizing {symbol} [{interval}] ---")
                    success = run_optimization(
                        interval=interval,
                        n_trials=args.trials,
                        n_jobs=args.jobs,
                        symbols=[symbol],
                        strategy_path=args.strategy,
                        strategy_name=strategy_name,
                    )
                    if success and args.auto_promote:
                        study_name = f"optimization_{symbol}_{interval}"
                        promote_strategies(study_name=study_name)
            else:
                # Standard Portfolio or Single Symbol run
                _logger.info(f"🚀 Starting PORTFOLIO optimization for interval: {interval}")
                success = run_optimization(
                    interval=interval,
                    n_trials=args.trials,
                    n_jobs=args.jobs,
                    study_name=args.study_name,
                    symbols=symbols_list,
                    strategy_path=args.strategy,
                    strategy_name=strategy_name,
                )
                if success and args.auto_promote:
                    symbol_slug = "-".join(sorted([s.upper() for s in symbols_list])) if symbols_list else "PORTFOLIO"
                    study_name = args.study_name or f"optimization_{symbol_slug}_{interval}"
                    promote_strategies(study_name=study_name)

    elif args.command == "promote":
        promote_strategies(
            study_name=args.study_name,
            top_n=args.top_n,
            user_id=args.user_id,
            min_calmar=args.min_calmar,
            max_drawdown=args.max_drawdown,
            min_trades=args.min_trades,
        )

    elif args.command == "list-studies":
        list_studies()

    else:
        parser.print_help()

    # Explicitly shutdown logging to ensure clean exit of background threads
    from src.notification.logger import shutdown_multiprocessing_logging
    shutdown_multiprocessing_logging()


if __name__ == "__main__":
    main()
