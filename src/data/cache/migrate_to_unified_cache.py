#!/usr/bin/env python3
"""
Migration Script: Provider-based Cache to Unified Cache

This script migrates data from the old structure:
    provider/symbol/timeframe/year/

To the new unified structure:
    symbol/timeframe/year/

Usage:
    python src/data/migrate_to_unified_cache.py --old-cache d:/data-cache --new-cache d:/data-cache-v2
    python src/data/migrate_to_unified_cache.py --migrate --dry-run
"""

import argparse
import sys
import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.cache.unified_cache import configure_unified_cache


class CacheMigrator:
    """Migrate from provider-based to unified cache structure."""

    def __init__(self, old_cache_dir: str, new_cache_dir: str):
        """
        Initialize migrator.

        Args:
            old_cache_dir: Path to old provider-based cache
            new_cache_dir: Path to new unified cache
        """
        self.old_cache_dir = Path(old_cache_dir)
        self.new_cache_dir = Path(new_cache_dir)
        self.new_cache = configure_unified_cache(cache_dir=new_cache_dir)

        # Migration statistics
        self.stats = {
            'symbols_migrated': 0,
            'timeframes_migrated': 0,
            'years_migrated': 0,
            'files_migrated': 0,
            'errors': 0,
            'skipped': 0
        }

    def analyze_old_structure(self) -> Dict[str, Any]:
        """Analyze the old cache structure."""
        print("🔍 Analyzing old cache structure...")

        structure = {}
        total_size = 0

        if not self.old_cache_dir.exists():
            print(f"❌ Old cache directory does not exist: {self.old_cache_dir}")
            return structure

        for provider_dir in self.old_cache_dir.iterdir():
            if not provider_dir.is_dir() or provider_dir.name.startswith('.'):
                continue

            provider = provider_dir.name
            structure[provider] = {}

            for symbol_dir in provider_dir.iterdir():
                if not symbol_dir.is_dir():
                    continue

                symbol = symbol_dir.name
                structure[provider][symbol] = {}

                for timeframe_dir in symbol_dir.iterdir():
                    if not timeframe_dir.is_dir():
                        continue

                    timeframe = timeframe_dir.name
                    structure[provider][symbol][timeframe] = {}

                    for year_dir in timeframe_dir.iterdir():
                        if not year_dir.is_dir() or not year_dir.name.isdigit():
                            continue

                        year = int(year_dir.name)
                        year_files = list(year_dir.glob('*'))

                        # Calculate size
                        year_size = sum(f.stat().st_size for f in year_files if f.is_file())
                        total_size += year_size

                        structure[provider][symbol][timeframe][year] = {
                            'files': len(year_files),
                            'size_bytes': year_size,
                            'path': str(year_dir)
                        }

        print(f"📊 Analysis complete:")
        print(f"   Providers: {len(structure)}")
        print(f"   Total symbols: {sum(len(symbols) for symbols in structure.values())}")
        print(f"   Total size: {total_size / (1024**3):.2f} GB")

        return structure

    def migrate_data(self, structure: Dict[str, Any], dry_run: bool = False) -> bool:
        """Migrate data from old to new structure."""
        print(f"\n🚀 Starting migration...")
        if dry_run:
            print("🧪 DRY RUN MODE - No files will be moved")

        try:
            for provider, symbols in structure.items():
                print(f"\n📁 Processing provider: {provider}")

                for symbol, timeframes in symbols.items():
                    print(f"  📊 Symbol: {symbol}")

                    for timeframe, years in timeframes.items():
                        print(f"    ⏱️  Timeframe: {timeframe}")

                        for year, year_info in years.items():
                            print(f"      📅 Year: {year}")

                            old_path = Path(year_info['path'])
                            if not old_path.exists():
                                print(f"        ⚠️  Path no longer exists, skipping")
                                self.stats['skipped'] += 1
                                continue

                            # Find CSV files
                            csv_files = list(old_path.glob('*.csv'))
                            if not csv_files:
                                print(f"        ⚠️  No CSV files found, skipping")
                                self.stats['skipped'] += 1
                                continue

                            for csv_file in csv_files:
                                if self._migrate_file(csv_file, symbol, timeframe, year, provider, dry_run):
                                    self.stats['files_migrated'] += 1
                                else:
                                    self.stats['errors'] += 1

                            self.stats['years_migrated'] += 1

                        self.stats['timeframes_migrated'] += 1

                    self.stats['symbols_migrated'] += 1

            return True

        except Exception as e:
            print(f"❌ Migration failed: {str(e)}")
            return False

    def _migrate_file(self, csv_file: Path, symbol: str, timeframe: str,
                      year: int, provider: str, dry_run: bool) -> bool:
        """Migrate a single CSV file."""
        try:
            # Create new path
            new_dir = self.new_cache_dir / symbol / timeframe / str(year)
            new_file = new_dir / f"{year}.csv.gz"
            metadata_file = new_dir / f"{year}.metadata.json"

            if dry_run:
                print(f"        🧪 Would migrate: {csv_file.name} -> {new_file}")
                return True

            # Create directory
            new_dir.mkdir(parents=True, exist_ok=True)

            # Read and compress CSV
            import pandas as pd
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

            # Save as compressed CSV
            with open(new_file, 'wb') as f:
                import gzip
                with gzip.open(f, 'wt', encoding='utf-8') as gz:
                    df.to_csv(gz, index=True)

            # Create metadata
            metadata = {
                'symbol': symbol,
                'timeframe': timeframe,
                'year': year,
                'data_source': provider,
                'migrated_at': datetime.now().isoformat(),
                'original_file': str(csv_file),
                'file_info': {
                    'format': 'csv.gz',
                    'size_bytes': new_file.stat().st_size,
                    'rows': len(df),
                    'columns': list(df.columns)
                }
            }

            # Save metadata
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"        ✅ Migrated: {csv_file.name} ({len(df)} rows)")
            return True

        except Exception as e:
            print(f"        ❌ Error migrating {csv_file.name}: {str(e)}")
            return False

    def print_migration_summary(self):
        """Print migration summary."""
        print(f"\n📊 MIGRATION SUMMARY")
        print("=" * 50)
        print(f"✅ Symbols migrated: {self.stats['symbols_migrated']}")
        print(f"✅ Timeframes migrated: {self.stats['timeframes_migrated']}")
        print(f"✅ Years migrated: {self.stats['years_migrated']}")
        print(f"✅ Files migrated: {self.stats['files_migrated']}")
        print(f"⚠️  Files skipped: {self.stats['skipped']}")
        print(f"❌ Errors: {self.stats['errors']}")

        if self.stats['errors'] == 0:
            print(f"\n🎉 Migration completed successfully!")
        else:
            print(f"\n⚠️  Migration completed with {self.stats['errors']} errors")


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(description="Migrate from provider-based to unified cache")
    parser.add_argument("--old-cache", type=str, default="./data-cache",
                       help="Path to old provider-based cache")
    parser.add_argument("--new-cache", type=str, default="./data-cache-v2",
                       help="Path to new unified cache")
    parser.add_argument("--migrate", action="store_true",
                       help="Actually perform migration (default is analyze only)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be migrated without actually doing it")

    args = parser.parse_args()

    print("🔄 Cache Migration Tool")
    print("=" * 50)
    print(f"📁 Old cache: {args.old_cache}")
    print(f"📁 New cache: {args.new_cache}")
    print()

    # Initialize migrator
    migrator = CacheMigrator(args.old_cache, args.new_cache)

    # Analyze old structure
    structure = migrator.analyze_old_structure()

    if not structure:
        print("❌ No data found to migrate")
        return

    # Ask for confirmation
    if args.migrate:
        if not args.dry_run:
            response = input("\n⚠️  This will move files from old to new cache. Continue? (y/N): ")
            if response.lower() != 'y':
                print("❌ Migration cancelled")
                return

        # Perform migration
        success = migrator.migrate_data(structure, dry_run=args.dry_run)
        if success:
            migrator.print_migration_summary()
        else:
            print("❌ Migration failed")
    else:
        print("\n💡 To perform migration, run with --migrate flag")
        print("💡 To see what would be migrated, run with --migrate --dry-run")


if __name__ == "__main__":
    main()
