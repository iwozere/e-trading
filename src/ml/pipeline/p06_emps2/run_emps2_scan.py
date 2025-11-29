#!/usr/bin/env python3
"""
EMPS2 Universe Scanner - CLI Entry Point

Command-line interface for running the EMPS2 pre-screening pipeline.

Usage:
    python src/ml/pipeline/p06_emps2/run_emps2_scan.py
    python src/ml/pipeline/p06_emps2/run_emps2_scan.py --aggressive
    python src/ml/pipeline/p06_emps2/run_emps2_scan.py --min-cap 100000000 --max-cap 2000000000
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.ml.pipeline.p06_emps2.config import EMPS2PipelineConfig, EMPS2FilterConfig
from src.ml.pipeline.p06_emps2.emps2_pipeline import EMPS2Pipeline

_logger = setup_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="EMPS2 Universe Scanner - Enhanced Explosive Move Pre-Screener",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (with rolling memory enabled)
  python run_emps2_scan.py

  # Use aggressive filtering (stricter criteria)
  python run_emps2_scan.py --aggressive

  # Use conservative filtering (broader universe)
  python run_emps2_scan.py --conservative

  # Disable rolling memory (single-day scan only)
  python run_emps2_scan.py --no-rolling-memory

  # Custom rolling memory settings
  python run_emps2_scan.py --lookback-days-rolling 15 --phase1-threshold 7

  # Disable alerts (no Telegram/Email notifications)
  python run_emps2_scan.py --no-alerts

  # Custom market cap range
  python run_emps2_scan.py --min-cap 100000000 --max-cap 2000000000

  # Custom volatility threshold
  python run_emps2_scan.py --min-volatility 0.025 --min-range 0.07

  # Force refresh (bypass caches and fetch fresh data)
  python run_emps2_scan.py --force-refresh

  # Quiet mode (less logging)
  python run_emps2_scan.py --quiet
        """
    )

    # Preset configurations
    preset_group = parser.add_mutually_exclusive_group()
    preset_group.add_argument(
        '--aggressive',
        action='store_true',
        help='Use aggressive filtering (stricter criteria for highest volatility)'
    )
    preset_group.add_argument(
        '--conservative',
        action='store_true',
        help='Use conservative filtering (broader universe)'
    )

    # Fundamental filter parameters
    fundamental_group = parser.add_argument_group('Fundamental Filters')
    fundamental_group.add_argument(
        '--min-price',
        type=float,
        help='Minimum stock price (default: 1.0)'
    )
    fundamental_group.add_argument(
        '--min-volume',
        type=int,
        help='Minimum average volume (default: 400000)'
    )
    fundamental_group.add_argument(
        '--min-cap',
        type=int,
        help='Minimum market cap (default: 50000000 = $50M)'
    )
    fundamental_group.add_argument(
        '--max-cap',
        type=int,
        help='Maximum market cap (default: 5000000000 = $5B)'
    )
    fundamental_group.add_argument(
        '--max-float',
        type=int,
        help='Maximum float shares (default: 60000000 = 60M)'
    )

    # Volatility filter parameters
    volatility_group = parser.add_argument_group('Volatility Filters')
    volatility_group.add_argument(
        '--min-volatility',
        type=float,
        help='Minimum ATR/Price ratio (default: 0.02 = 2%%)'
    )
    volatility_group.add_argument(
        '--min-range',
        type=float,
        help='Minimum price range (default: 0.05 = 5%%)'
    )
    volatility_group.add_argument(
        '--lookback-days',
        type=int,
        help='Lookback period in days (default: 7)'
    )
    volatility_group.add_argument(
        '--interval',
        type=str,
        choices=['5m', '15m', '30m', '1h'],
        help='Data interval (default: 15m)'
    )
    volatility_group.add_argument(
        '--atr-period',
        type=int,
        help='ATR calculation period (default: 14)'
    )

    # Rolling memory parameters
    rolling_group = parser.add_argument_group('Rolling Memory & Phase Detection')
    rolling_group.add_argument(
        '--no-rolling-memory',
        action='store_true',
        help='Disable rolling memory (single-day scan only)'
    )
    rolling_group.add_argument(
        '--lookback-days-rolling',
        type=int,
        help='Rolling memory lookback period (default: 10 days)'
    )
    rolling_group.add_argument(
        '--phase1-threshold',
        type=int,
        help='Minimum appearances for Phase 1 (default: 5)'
    )
    rolling_group.add_argument(
        '--no-alerts',
        action='store_true',
        help='Disable Telegram/Email alerts for Phase 2 transitions'
    )

    # Execution parameters
    execution_group = parser.add_argument_group('Execution Options')
    execution_group.add_argument(
        '--force-refresh',
        action='store_true',
        help='Force refresh (bypass cache and fetch fresh data)'
    )
    execution_group.add_argument(
        '--quiet',
        action='store_true',
        help='Quiet mode (less verbose logging)'
    )
    execution_group.add_argument(
        '--no-summary',
        action='store_true',
        help='Skip summary generation'
    )

    return parser.parse_args()


def print_header(title: str):
    """Print formatted header."""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")


def print_results_summary(final_df, config: EMPS2PipelineConfig):
    """
    Print formatted results summary.

    Args:
        final_df: Final DataFrame
        config: Pipeline configuration
    """
    print_header("Final Results")

    if final_df.empty:
        print("[WARNING] No candidates found\n")
        return

    print(f"[OK] Found {len(final_df)} explosive move candidates\n")

    # Display top 20
    print("Top 20 candidates:")
    print("-" * 70)

    display_cols = ['ticker', 'market_cap', 'avg_volume', 'sector', 'current_price']
    available_cols = [col for col in display_cols if col in final_df.columns]

    if available_cols:
        for idx, row in final_df[available_cols].head(20).iterrows():
            ticker = row.get('ticker', 'N/A')
            market_cap = row.get('market_cap', 0)
            volume = row.get('avg_volume', 0)
            sector = row.get('sector', 'N/A')
            price = row.get('current_price', 0)

            market_cap_str = f"${market_cap/1e6:.0f}M" if market_cap > 0 else "N/A"
            volume_str = f"{volume/1e3:.0f}K" if volume > 0 else "N/A"

            print(f"{ticker:6s} | Cap: {market_cap_str:8s} | Vol: {volume_str:8s} | "
                  f"Price: ${price:6.2f} | {sector}")
    else:
        # Fallback if columns not available
        for ticker in final_df['ticker'].head(20):
            print(f"  {ticker}")

    if len(final_df) > 20:
        print(f"  ... and {len(final_df) - 20} more")

    print()


def main():
    """Main execution function."""
    args = parse_args()

    print_header("EMPS2 Universe Scanner")

    # Create configuration
    if args.aggressive:
        print("[INFO] Using AGGRESSIVE filtering preset\n")
        config = EMPS2PipelineConfig.create_aggressive()
    elif args.conservative:
        print("[INFO] Using CONSERVATIVE filtering preset\n")
        config = EMPS2PipelineConfig.create_conservative()
    else:
        print("[INFO] Using DEFAULT filtering settings\n")
        config = EMPS2PipelineConfig.create_default()

    # Apply custom parameters
    if args.min_price is not None:
        config.filter_config.min_price = args.min_price
    if args.min_volume is not None:
        config.filter_config.min_avg_volume = args.min_volume
    if args.min_cap is not None:
        config.filter_config.min_market_cap = args.min_cap
    if args.max_cap is not None:
        config.filter_config.max_market_cap = args.max_cap
    if args.max_float is not None:
        config.filter_config.max_float = args.max_float
    if args.min_volatility is not None:
        config.filter_config.min_volatility_threshold = args.min_volatility
    if args.min_range is not None:
        config.filter_config.min_price_range = args.min_range
    if args.lookback_days is not None:
        config.filter_config.lookback_days = args.lookback_days
    if args.interval is not None:
        config.filter_config.interval = args.interval
    if args.atr_period is not None:
        config.filter_config.atr_period = args.atr_period

    # Apply rolling memory parameters
    if args.no_rolling_memory:
        config.rolling_memory_config.enabled = False
    if args.lookback_days_rolling is not None:
        config.rolling_memory_config.lookback_days = args.lookback_days_rolling
    if args.phase1_threshold is not None:
        config.rolling_memory_config.phase1_min_appearances = args.phase1_threshold
    if args.no_alerts:
        config.rolling_memory_config.send_alerts = False

    # Update config flags
    if args.quiet:
        config.verbose_logging = False
    if args.no_summary:
        config.generate_summary = False

    # Determine force_refresh (default is False, use cache unless --force-refresh is specified)
    force_refresh = args.force_refresh

    # Display configuration
    print("Configuration:")
    print(f"  Min Price: ${config.filter_config.min_price}")
    print(f"  Min Volume: {config.filter_config.min_avg_volume:,}")
    print(f"  Market Cap: ${config.filter_config.min_market_cap/1e6:.0f}M - "
          f"${config.filter_config.max_market_cap/1e9:.1f}B")
    print(f"  Max Float: {config.filter_config.max_float/1e6:.0f}M shares")
    print(f"  Min Volatility (ATR/Price): {config.filter_config.min_volatility_threshold*100:.1f}%")
    print(f"  Min Price Range: {config.filter_config.min_price_range*100:.1f}%")
    print(f"  Lookback: {config.filter_config.lookback_days} days")
    print(f"  Interval: {config.filter_config.interval}")
    print(f"  Force Refresh: {force_refresh}")
    print(f"\nRolling Memory:")
    print(f"  Enabled: {config.rolling_memory_config.enabled}")
    if config.rolling_memory_config.enabled:
        print(f"  Lookback Days: {config.rolling_memory_config.lookback_days}")
        print(f"  Phase 1 Threshold: {config.rolling_memory_config.phase1_min_appearances} appearances")
        print(f"  Send Alerts: {config.rolling_memory_config.send_alerts}")
    print()

    # Create and run pipeline
    try:
        pipeline = EMPS2Pipeline(config)
        final_df = pipeline.run(force_refresh=force_refresh)

        # Print results
        print_results_summary(final_df, config)

        # Results location
        today = datetime.now().strftime('%Y-%m-%d')
        results_dir = PROJECT_ROOT / 'results' / 'emps2' / today

        print_header("Output Files")
        print(f"Results saved to: {results_dir}\n")
        print("Files:")
        print(f"  - pipeline.log                    (Full scan log)")
        print(f"  - 01_nasdaq_universe.csv          (Full NASDAQ universe)")
        print(f"  - 02_fundamental_raw_data.csv     (Raw fundamental data)")
        print(f"  - 03_fundamental_filtered.csv     (After fundamental filters)")
        print(f"  - 04_volatility_diagnostics.csv   (ALL tickers with metrics & failure reasons)")
        print(f"  - 05_volatility_filtered.csv      (After volatility filters)")
        print(f"  - 06_prefiltered_universe.csv     (Final results)")
        if config.rolling_memory_config.enabled:
            print(f"  - 07_rolling_candidates.csv       (10-day rolling memory)")
            print(f"  - 08_phase1_watchlist.csv         (Phase 1: Quiet Accumulation)")
            print(f"  - 09_phase2_alerts.csv            (Phase 2: Hot Candidates 🔥)")
        if config.generate_summary:
            print(f"  - summary.json                    (Pipeline summary)")
        print()

        print_header("Scan Complete")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        return 0

    except Exception:
        _logger.exception("Error running pipeline:")
        print("\n[ERROR] Pipeline failed. Check logs for details.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
