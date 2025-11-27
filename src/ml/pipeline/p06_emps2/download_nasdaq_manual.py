#!/usr/bin/env python3
"""
Manual NASDAQ Universe Download Helper

Use this script if you have network/firewall issues with the automatic download.

Steps:
1. Download these two files manually in your browser:
   - ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt
   - ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt

2. Save them to: results/emps2/YYYY-MM-DD/ (today's date folder)

3. Run this script to process them into the cache format

Or just run this script and it will try FTP download automatically.
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import io
from ftplib import FTP

import pandas as pd

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def download_nasdaq_files():
    """Try downloading with FTP protocol using ftplib."""
    try:
        print("Connecting to NASDAQ Trader FTP server...")
        print("  Server: ftp.nasdaqtrader.com")
        print("  Directory: SymbolDirectory")
        print()

        # Connect to FTP server
        ftp = FTP('ftp.nasdaqtrader.com', timeout=60)
        ftp.login()  # Anonymous login

        print("Connected successfully!")
        print(f"Server response: {ftp.getwelcome()}\n")

        # Change to SymbolDirectory
        ftp.cwd('SymbolDirectory')

        # Download NASDAQ listed
        print("Downloading nasdaqlisted.txt...")
        nasdaq_data = []
        ftp.retrlines('RETR nasdaqlisted.txt', nasdaq_data.append)
        nasdaq_text = '\n'.join(nasdaq_data)
        print(f"  Downloaded {len(nasdaq_data)} lines")

        # Download other listed
        print("Downloading otherlisted.txt...")
        other_data = []
        ftp.retrlines('RETR otherlisted.txt', other_data.append)
        other_text = '\n'.join(other_data)
        print(f"  Downloaded {len(other_data)} lines")

        ftp.quit()
        print("\nDownloads successful!\n")
        return nasdaq_text, other_text

    except Exception as e:
        print(f"FTP download failed: {e}\n")
        return None, None


def process_manual_files(results_dir: Path):
    """Process manually downloaded files."""
    nasdaq_file = results_dir / "nasdaqlisted.txt"
    other_file = results_dir / "otherlisted.txt"

    if not nasdaq_file.exists() or not other_file.exists():
        print(f"Manual files not found in {results_dir}")
        print(f"  Looking for: nasdaqlisted.txt and otherlisted.txt")
        return None, None

    print(f"Found manual files in {results_dir}")

    with open(nasdaq_file, 'r') as f:
        nasdaq_text = f.read()

    with open(other_file, 'r') as f:
        other_text = f.read()

    return nasdaq_text, other_text


def process_nasdaq_data(nasdaq_text: str, other_text: str, results_dir: Path):
    """Process NASDAQ data into cache format."""
    try:
        # Parse CSV files
        df1 = pd.read_csv(io.StringIO(nasdaq_text), sep="|")
        df2 = pd.read_csv(io.StringIO(other_text), sep="|")

        # Combine
        df = pd.concat([df1, df2], ignore_index=True)
        print(f"Total symbols: {len(df)}")
        print(f"Columns: {df.columns.tolist()}\n")

        # Drop empty rows
        df = df.dropna(how='all')

        # Filter test issues
        if "Test Issue" in df.columns:
            df = df[df["Test Issue"] == "N"]
            print(f"After excluding test issues: {len(df)}")

        # Extract tickers
        if "Symbol" not in df.columns:
            print(f"ERROR: Symbol column not found. Available columns: {df.columns.tolist()}")
            return

        tickers = df["Symbol"].dropna().tolist()
        print(f"Extracted tickers: {len(tickers)}\n")

        # Filter tickers (alphabetic only, 1-5 chars)
        filtered_tickers = []
        for ticker in tickers:
            if not ticker or not isinstance(ticker, str):
                continue
            ticker = ticker.strip().upper()
            if ticker.isalpha() and 1 <= len(ticker) <= 5:
                filtered_tickers.append(ticker)

        print(f"After filtering (alphabetic, 1-5 chars): {len(filtered_tickers)}")
        print(f"Removed: {len(tickers) - len(filtered_tickers)}\n")

        # Save to cache
        cache_file = results_dir / "nasdaq_universe_cache.json"
        cache_data = {
            'tickers': filtered_tickers,
            'created_at': datetime.now().isoformat(),
            'config': {
                'exclude_test_issues': True,
                'alphabetic_only': True
            }
        }

        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)

        print(f"✓ Saved cache to: {cache_file}")

        # Save full universe CSV
        df_filtered = df[df["Symbol"].isin(filtered_tickers)]
        universe_file = results_dir / "nasdaq_universe.csv"
        df_filtered.to_csv(universe_file, index=False)

        print(f"✓ Saved universe to: {universe_file}")
        print(f"\nSuccess! {len(filtered_tickers)} tickers ready for pipeline.")

        # Show sample
        print(f"\nSample tickers: {', '.join(filtered_tickers[:20])}")
        if len(filtered_tickers) > 20:
            print(f"... and {len(filtered_tickers) - 20} more")

    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main execution."""
    print("="*70)
    print(" NASDAQ Universe Manual Download Helper")
    print("="*70)
    print()

    # Create results directory
    today = datetime.now().strftime('%Y-%m-%d')
    results_dir = PROJECT_ROOT / 'results' / 'emps2' / today
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results directory: {results_dir}\n")

    # Try automatic HTTP download first
    nasdaq_text, other_text = download_nasdaq_files()

    # If HTTP download failed, try manual files
    if not nasdaq_text or not other_text:
        print("Automatic download failed. Checking for manual files...")
        nasdaq_text, other_text = process_manual_files(results_dir)

    if not nasdaq_text or not other_text:
        print("\n" + "="*70)
        print(" Manual Download Required")
        print("="*70)
        print("\nPlease download these files manually in your browser:")
        print("  1. ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt")
        print("  2. ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt")
        print(f"\nSave them to: {results_dir}")
        print("  as: nasdaqlisted.txt and otherlisted.txt")
        print("\nThen run this script again.")
        return 1

    # Process the data
    print("Processing NASDAQ data...")
    process_nasdaq_data(nasdaq_text, other_text, results_dir)

    print("\n" + "="*70)
    print(" Done! You can now run the pipeline with --no-force-refresh")
    print("="*70)
    print("\nCommand:")
    print("  python src/ml/pipeline/p06_emps2/run_emps2_scan.py --no-force-refresh")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
