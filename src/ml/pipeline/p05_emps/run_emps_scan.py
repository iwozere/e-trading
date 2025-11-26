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
from src.ml.pipeline.p05_emps.universe_loader import EMPSUniverseConfig, EMPSUniverseLoader
from src.ml.pipeline.p05_emps.emps_integration import EMPSUniverseScanner
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
        default=1000,
        help='Maximum number of tickers to scan (default: 1000)'
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
    parser.add_argument(
        '--max-universe-size',
        type=int,
        default=1000,
        help='Maximum universe size before narrowing (default: 1000). Known explosive tickers are prioritized.'
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
        default=20,
        help='Days of historical data to fetch (default: 20 calendar days ~ 14 trading days for sufficient EMPS rolling windows)'
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
    parser.add_argument(
        '--force-refresh-universe',
        action='store_true',
        default=True,
        help='Force refresh universe from screener by default; set to False programmatically if needed (useful when FMP API becomes available after previous failure)'
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
        'emps_score': lambda x: f'{x:.3f}',
        'vol_zscore': lambda x: f'{x:.2f}',
        'vwap_dev': lambda x: f'{x:.3f}',
        'rv_ratio': lambda x: f'{x:.2f}',
        'short_interest_pct': lambda x: f'{x:.1f}',
        'days_to_cover': lambda x: f'{x:.2f}',
        'combined_score': lambda x: f'{x:.3f}',
    }

    # Print table (only use formatters for columns that exist)
    formatters_to_use = {k: v for k, v in format_dict.items() if k in display_cols}
    print(df[display_cols].head(20).to_string(index=False, formatters=formatters_to_use))

    # Print statistics
    print(f"\n{'-'*70}")
    print("Statistics:")
    print(f"{'-'*70}")
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

    # Initialize data downloaders
    # Strategy: Use FMP for universe screening (has screener API)
    #           Use Yahoo Finance for OHLCV data (free, no API key)
    logger.info("Initializing data downloaders...")

    try:
        # Always use FMP for universe screening (better screener API)
        fmp_downloader = FMPDataDownloader()
        print("[INFO] Universe provider: FMP (screener API)")

        if not fmp_downloader.test_connection():
            print("[ERROR] FMP connection failed - needed for universe screening")
            return 1
        print("[OK] FMP connection successful (universe screening)\n")

    except Exception as e:
        print(f"[ERROR] Failed to initialize FMP for universe: {e}")
        return 1

    try:
        # Use selected provider for OHLCV data
        if args.data_provider == 'yfinance':
            ohlcv_downloader = YahooDataDownloader()
            print("[INFO] OHLCV provider: Yahoo Finance (free, no API key required)")
            print("[INFO] Intraday data available for last 60 days")
        else:  # fmp
            ohlcv_downloader = FMPDataDownloader()
            print("[INFO] OHLCV provider: FMP (requires API key)")
            # Reuse FMP connection for both if user chose FMP
            ohlcv_downloader = fmp_downloader

        if ohlcv_downloader != fmp_downloader:
            if not ohlcv_downloader.test_connection():
                print(f"[ERROR] {args.data_provider.upper()} connection failed")
                return 1
            print(f"[OK] {args.data_provider.upper()} connection successful (OHLCV data)\n")

    except Exception as e:
        print(f"[ERROR] Failed to initialize {args.data_provider} for OHLCV: {e}")
        return 1

    # Configure universe (using FMP for screening)
    universe_config = EMPSUniverseConfig(
        min_market_cap=args.min_cap,
        max_market_cap=args.max_cap,
        min_avg_volume=args.min_volume,
        exchanges=['NYSE', 'NASDAQ'],
        max_universe_size=args.max_universe_size
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

    # Create scanner with hybrid approach:
    # - Universe loader uses FMP (for screener)
    # - Data adapter uses selected provider (for OHLCV)
    logger.info("Initializing EMPS scanner with hybrid providers...")
    try:
        # Universe loader uses FMP (has screener API)
        universe_loader = EMPSUniverseLoader(fmp_downloader, universe_config)

        # Create scanner (it will create its own data adapter internally)
        scanner = EMPSUniverseScanner(
            downloader=ohlcv_downloader,
            universe_loader=universe_loader,
            emps_params=emps_params,
            fetch_params=fetch_params
        )

        print("[OK] EMPS scanner initialized (FMP universe + Yahoo OHLCV)\n")
    except Exception as e:
        print(f"[ERROR] Failed to initialize scanner: {e}")
        return 1

    # Load and display universe
    logger.info("Loading universe...")
    universe = scanner.universe_loader.load_universe(force_refresh=args.force_refresh_universe)

    if not universe:
        print("[ERROR] Failed to load universe\n")
        return 1

    # Print universe summary
    print_universe_summary(universe, universe_config)

    # Save universe to CSV (always, with custom filename if provided)
    universe_filename = args.universe_output if args.universe_output else 'universe.csv'
    save_universe_to_csv(universe, universe_filename, universe_config)

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
