#!/usr/bin/env python3
"""
EMPS Universe Scanner - Example Script

Demonstrates full integration of EMPS with:
1. FMP Data Downloader
2. P04 Universe Selection
3. Combined EMPS + Short Squeeze Scoring

Usage:
    python run_emps_scan.py --limit 50 --min-score 0.5 --output results.csv
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.data.downloader.fmp_data_downloader import FMPDataDownloader
from src.ml.pipeline.p04_short_squeeze.config.data_classes import UniverseConfig
from src.ml.pipeline.p05_emps.emps_p04_integration import create_emps_scanner
from src.ml.pipeline.p05_emps.emps import DEFAULTS as EMPS_DEFAULTS

logger = setup_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="EMPS Universe Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan top 20 tickers with default settings
  python run_emps_scan.py --limit 20

  # Scan with custom threshold and save results
  python run_emps_scan.py --limit 50 --min-score 0.6 --output emps_results.csv

  # Enable P04 integration for combined scoring
  python run_emps_scan.py --limit 30 --combine-p04

  # Custom universe configuration
  python run_emps_scan.py --min-cap 50000000 --max-cap 5000000000 --min-volume 1000000
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
        help='Output CSV file path (optional)'
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

    return parser.parse_args()


def print_header(title: str):
    """Print formatted header."""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")


def print_results_summary(df: pd.DataFrame):
    """Print formatted results summary."""
    if df.empty:
        print("⚠️  No candidates found above threshold\n")
        return

    print(f"✅ Found {len(df)} candidates:\n")

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
    print(f"  Scan limit: {args.limit} tickers")
    print(f"  Min EMPS score: {args.min_score}")
    print(f"  Universe: ${args.min_cap/1e6:.0f}M - ${args.max_cap/1e9:.1f}B market cap")
    print(f"  Min volume: {args.min_volume:,}")
    print(f"  P04 integration: {'Enabled' if args.combine_p04 else 'Disabled'}")
    if args.output:
        print(f"  Output file: {args.output}")
    print()

    # Initialize FMP downloader
    logger.info("Initializing FMP downloader...")
    try:
        fmp = FMPDataDownloader()
        if not fmp.test_connection():
            print("❌ FMP connection failed")
            return 1
        print("✅ FMP connection successful\n")
    except Exception as e:
        print(f"❌ Failed to initialize FMP: {e}")
        return 1

    # Configure universe
    universe_config = UniverseConfig(
        min_market_cap=args.min_cap,
        max_market_cap=args.max_cap,
        min_avg_volume=args.min_volume,
        exchanges=['NYSE', 'NASDAQ']
    )

    # Configure EMPS parameters
    emps_params = {
        **EMPS_DEFAULTS,
        'combined_score_thresh': args.emps_threshold
    }

    # Create scanner
    logger.info("Initializing EMPS scanner...")
    try:
        scanner = create_emps_scanner(fmp, universe_config, emps_params)
        print("✅ EMPS scanner initialized\n")
    except Exception as e:
        print(f"❌ Failed to initialize scanner: {e}")
        return 1

    # Run scan
    print_header("Scanning Universe")
    print(f"⏳ Scanning {args.limit} tickers...\n")

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
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            results.to_csv(output_path, index=False)
            print(f"✅ Results saved to: {output_path}\n")

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
        print(f"\n❌ Scan failed: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
