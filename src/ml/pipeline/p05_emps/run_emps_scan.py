#!/usr/bin/env python3
"""
EMPS Universe Scanner - Example Script

Demonstrates full integration of EMPS with:
1. FMP Data Downloader
2. EMPS Universe Selection (standalone P05)
3. Optional P04 Short Squeeze Integration

Usage:
    python run_emps_scan.py --limit 50 --min-score 0.5 --output results.csv
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.data.downloader.fmp_data_downloader import FMPDataDownloader
from src.data.downloader.yahoo_data_downloader import YahooDataDownloader
from src.ml.pipeline.p05_emps.universe_loader import EMPSUniverseConfig
from src.ml.pipeline.p05_emps.emps_integration import create_emps_scanner
from src.ml.pipeline.p05_emps.emps import DEFAULTS as EMPS_DEFAULTS

logger = setup_logger(__name__)

# Optional DB imports
try:
    from src.data.db.services.short_squeeze_service import ShortSqueezeService
    DB_AVAILABLE = True
except ImportError:
    logger.warning("Database modules not available. DB storage will be disabled.")
    DB_AVAILABLE = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="EMPS Universe Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan top 20 tickers with Yahoo Finance (free, default)
  python run_emps_scan.py --limit 20

  # Scan with custom threshold and save results (auto-saves to results/emps/YYYY-MM-DD/)
  python run_emps_scan.py --limit 50 --min-score 0.6 --output emps_results.csv

  # Save universe to CSV and database
  python run_emps_scan.py --limit 30 --universe-output universe.csv --save-universe-to-db

  # Use FMP data provider instead of Yahoo Finance
  python run_emps_scan.py --limit 20 --data-provider fmp

  # Custom universe configuration with P04 integration
  python run_emps_scan.py --min-cap 50000000 --max-cap 5000000000 --combine-p04
        """
    )

    # Scan parameters
    parser.add_argument(
        '--limit',
        type=int,
        default=20,
        help='Maximum number of tickers to scan (default: 20)'
    )
    parser.add_argument(
        '--min-score',
        type=float,
        default=0.5,
        help='Minimum EMPS score to include in results (default: 0.5)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file for results (filename or full path). If filename only, saved to results/emps/YYYY-MM-DD/'
    )
    parser.add_argument(
        '--combine-p04',
        action='store_true',
        help='Enable P04 short squeeze integration'
    )

    # Universe configuration
    parser.add_argument(
        '--min-cap',
        type=int,
        default=100_000_000,
        help='Minimum market cap (default: 100M)'
    )
    parser.add_argument(
        '--max-cap',
        type=int,
        default=10_000_000_000,
        help='Maximum market cap (default: 10B)'
    )
    parser.add_argument(
        '--min-volume',
        type=int,
        default=500_000,
        help='Minimum average volume (default: 500K)'
    )

    # EMPS parameters
    parser.add_argument(
        '--emps-threshold',
        type=float,
        default=0.6,
        help='EMPS explosion flag threshold (default: 0.6)'
    )
    parser.add_argument(
        '--interval',
        type=str,
        default='15m',
        choices=['5m', '15m', '30m', '1h'],
        help='Data interval for EMPS calculation (default: 15m - optimized for screening)'
    )
    parser.add_argument(
        '--days-back',
        type=int,
        default=7,
        help='Days of historical data to fetch (default: 2 - optimized for screening)'
    )
    parser.add_argument(
        '--save-universe-to-db',
        action='store_true',
        help='Save selected universe to ss_snapshot table in database'
    )
    parser.add_argument(
        '--universe-output',
        type=str,
        help='Output CSV file for universe (filename or full path). If filename only, saved to results/emps/YYYY-MM-DD/'
    )
    parser.add_argument(
        '--data-provider',
        type=str,
        default='yfinance',
        choices=['fmp', 'yfinance'],
        help='Data provider to use: fmp (requires API key, paid for intraday) or yfinance (free, no API key, last 60 days intraday). Default: yfinance'
    )

    return parser.parse_args()


def print_header(title: str):
    """Print formatted header."""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")


def print_universe_summary(universe: list, config: EMPSUniverseConfig):
    """Print formatted universe summary."""
    print_header("Selected Universe")
    print(f"Universe size: {len(universe)} tickers")
    print(f"Market cap range: ${config.min_market_cap/1e6:.0f}M - ${config.max_market_cap/1e9:.1f}B")
    print(f"Min avg volume: {config.min_avg_volume:,}")
    print(f"Exchanges: {', '.join(config.exchanges)}")
    print(f"\nFirst 50 tickers:")

    # Print tickers in rows of 10
    for i in range(0, min(50, len(universe)), 10):
        print("  " + ", ".join(universe[i:i+10]))

    if len(universe) > 50:
        print(f"  ... and {len(universe) - 50} more")
    print()


def get_results_path(filename: str) -> Path:
    """
    Get standardized results path under results/emps/yyyy-mm-dd/.

    Args:
        filename: Name of the output file

    Returns:
        Path object with full path including date-based folder
    """
    today = datetime.now().strftime('%Y-%m-%d')
    results_dir = PROJECT_ROOT / 'results' / 'emps' / today
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / filename


def save_universe_to_csv(universe: list, filepath: str, config: EMPSUniverseConfig):
    """Save universe to CSV file."""
    try:
        # If filepath is just a filename, use standardized results path
        if not Path(filepath).is_absolute() and '/' not in filepath and '\\' not in filepath:
            output_path = get_results_path(filepath)
        else:
            output_path = Path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create DataFrame with universe metadata
        df = pd.DataFrame({
            'ticker': universe,
            'scan_date': datetime.now().strftime('%Y-%m-%d'),
            'min_market_cap': config.min_market_cap,
            'max_market_cap': config.max_market_cap,
            'min_avg_volume': config.min_avg_volume,
            'exchanges': ','.join(config.exchanges)
        })

        df.to_csv(output_path, index=False)
        print(f"[OK] Universe saved to: {output_path}\n")
        logger.info("Universe saved to CSV: %s", output_path)

    except Exception as e:
        print(f"[ERROR] Failed to save universe to CSV: {e}\n")
        logger.error("Failed to save universe to CSV: %s", e)


def save_universe_to_db(universe: list, config: EMPSUniverseConfig):
    """Save universe tickers to ss_snapshot table."""
    if not DB_AVAILABLE:
        print("[WARNING] Database not available. Skipping DB save.\n")
        return

    try:
        run_date = datetime.now().date()

        # Use the service layer instead of direct repository access
        service = ShortSqueezeService()

        # Create snapshot records for each ticker in universe
        snapshots_data = []
        for ticker in universe:
            snapshot_data = {
                'ticker': ticker.upper(),
                'run_date': run_date,
                'market_cap': None,  # Will be populated during EMPS scan
                'avg_volume_14d': None,
                'screener_score': None,
                'short_interest_pct': None,
                'days_to_cover': None,
                'float_shares': None,
                'data_quality': None,
                'raw_payload': {'source': 'emps_universe_loader', 'config': {
                    'min_market_cap': config.min_market_cap,
                    'max_market_cap': config.max_market_cap,
                    'min_avg_volume': config.min_avg_volume
                }}
            }
            snapshots_data.append(snapshot_data)

        # Save via service (which handles UoW and transactions)
        count = service.save_screener_results(snapshots_data, run_date)

        print(f"[OK] Universe saved to database: {count} tickers in ss_snapshot table\n")
        logger.info("Universe saved to database: %d tickers for run_date=%s", count, run_date)

    except Exception as e:
        print(f"[ERROR] Failed to save universe to database: {e}\n")
        logger.error("Failed to save universe to database: %s", e)


def print_results_summary(df: pd.DataFrame):
    """Print formatted results summary."""
    if df.empty:
        print("[WARNING] No candidates found above threshold\n")
        return

    print(f"[OK] Found {len(df)} candidates:\n")

    # Format columns for display
    display_cols = ['ticker', 'emps_score', 'explosion_flag', 'vol_zscore', 'vwap_dev', 'rv_ratio']

    # Add P04 columns if present
    if 'short_interest_pct' in df.columns:
        display_cols.extend(['short_interest_pct', 'days_to_cover', 'combined_score'])

    # Filter to available columns
    display_cols = [col for col in display_cols if col in df.columns]

    # Format numeric columns
    format_dict = {
        'emps_score': '{:.3f}',
        'vol_zscore': '{:.2f}',
        'vwap_dev': '{:.3f}',
        'rv_ratio': '{:.2f}',
        'short_interest_pct': '{:.1f}',
        'days_to_cover': '{:.2f}',
        'combined_score': '{:.3f}',
    }

    # Print table
    print(df[display_cols].head(20).to_string(index=False, formatters=format_dict))

    # Print statistics
    print(f"\n{'─'*70}")
    print("Statistics:")
    print(f"{'─'*70}")
    print(f"  Total candidates: {len(df)}")
    print(f"  Explosion flags: {df['explosion_flag'].sum()}")
    if 'hard_flag' in df.columns:
        print(f"  Hard flags: {df['hard_flag'].sum()}")
    print(f"  Avg EMPS score: {df['emps_score'].mean():.3f}")
    print(f"  Max EMPS score: {df['emps_score'].max():.3f}")

    if 'combined_score' in df.columns:
        print(f"  Avg combined score: {df['combined_score'].mean():.3f}")
        print(f"  Max combined score: {df['combined_score'].max():.3f}")

    print()


def main():
    """Main execution function."""
    args = parse_args()

    print_header("EMPS Universe Scanner")

    # Display configuration
    print("Configuration:")
    print(f"  Data provider: {args.data_provider.upper()}")
    print(f"  Scan limit: {args.limit} tickers")
    print(f"  Min EMPS score: {args.min_score}")
    print(f"  Data interval: {args.interval}")
    print(f"  Days back: {args.days_back}")
    print(f"  Universe: ${args.min_cap/1e6:.0f}M - ${args.max_cap/1e9:.1f}B market cap")
    print(f"  Min volume: {args.min_volume:,}")
    print(f"  P04 integration: {'Enabled' if args.combine_p04 else 'Disabled'}")
    if args.output:
        print(f"  Output file: {args.output}")
    print()

    # Initialize data downloader based on provider choice
    logger.info("Initializing %s downloader...", args.data_provider)
    try:
        if args.data_provider == 'yfinance':
            downloader = YahooDataDownloader()
            print("[INFO] Using Yahoo Finance (free, no API key required)")
            print("[INFO] Intraday data available for last 60 days")
        else:  # fmp
            downloader = FMPDataDownloader()
            print("[INFO] Using FMP (requires API key)")

        if not downloader.test_connection():
            print(f"[ERROR] {args.data_provider.upper()} connection failed")
            return 1
        print(f"[OK] {args.data_provider.upper()} connection successful\n")
    except Exception as e:
        print(f"[ERROR] Failed to initialize {args.data_provider}: {e}")
        return 1

    # Configure universe
    universe_config = EMPSUniverseConfig(
        min_market_cap=args.min_cap,
        max_market_cap=args.max_cap,
        min_avg_volume=args.min_volume,
        exchanges=['NYSE', 'NASDAQ'],
        max_universe_size=1000
    )

    # Configure EMPS parameters
    emps_params = {
        **EMPS_DEFAULTS,
        'combined_score_thresh': args.emps_threshold
    }

    # Configure data fetch parameters
    fetch_params = {
        'interval': args.interval,
        'days_back': args.days_back
    }

    # Create scanner
    logger.info("Initializing EMPS scanner...")
    try:
        scanner = create_emps_scanner(downloader, universe_config, emps_params, fetch_params)
        print("[OK] EMPS scanner initialized\n")
    except Exception as e:
        print(f"[ERROR] Failed to initialize scanner: {e}")
        return 1

    # Load and display universe
    logger.info("Loading universe...")
    universe = scanner.universe_loader.load_universe()

    if not universe:
        print("[ERROR] Failed to load universe\n")
        return 1

    # Print universe summary
    print_universe_summary(universe, universe_config)

    # Save universe to CSV if requested
    if args.universe_output:
        save_universe_to_csv(universe, args.universe_output, universe_config)

    # Save universe to database if requested
    if args.save_universe_to_db:
        save_universe_to_db(universe, universe_config)

    # Run scan
    print_header("Scanning Universe")
    print(f"[INFO] Scanning {args.limit} tickers...\n")

    try:
        if args.combine_p04:
            results = scanner.scan_with_p04_integration(
                limit=args.limit,
                combine_scores=True
            )
        else:
            results = scanner.scan_universe(
                limit=args.limit,
                min_emps_score=args.min_score
            )

        print_header("Results")
        print_results_summary(results)

        # Save to file if requested
        if args.output and not results.empty:
            # If filepath is just a filename, use standardized results path
            if not Path(args.output).is_absolute() and '/' not in args.output and '\\' not in args.output:
                output_path = get_results_path(args.output)
            else:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)

            results.to_csv(output_path, index=False)
            print(f"[OK] Results saved to: {output_path}\n")

        # Print top candidates
        if not results.empty:
            print_header("Top 5 Candidates")

            for idx, row in results.head(5).iterrows():
                print(f"{idx+1}. {row['ticker']}")
                print(f"   EMPS Score: {row['emps_score']:.3f}")
                print(f"   Explosion Flag: {row['explosion_flag']}")
                if 'combined_score' in row:
                    print(f"   Combined Score: {row['combined_score']:.3f}")
                print(f"   Vol Z-Score: {row['vol_zscore']:.2f}")
                print(f"   VWAP Deviation: {row['vwap_dev']:.3f}")
                if 'short_interest_pct' in row and pd.notna(row['short_interest_pct']):
                    print(f"   Short Interest: {row['short_interest_pct']:.1f}%")
                print()

        print_header("Scan Complete")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        return 0

    except Exception as e:
        logger.exception("Error during scan:")
        print(f"\n[ERROR] Scan failed: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
