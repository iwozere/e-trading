#!/usr/bin/env python3
"""
Cache Validation and Cleanup Script

This script:
1. Scans the cache directory for files that fail validation
2. Removes invalid files to free up space
3. Reports on cache health and cleanup results

Usage:
    python src/data/cleanup_failed_cache.py --validate-only
    python src/data/cleanup_failed_cache.py --cleanup
    python src/data/cleanup_failed_cache.py --validate-and-cleanup
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.utils.validation import validate_ohlcv_data, get_data_quality_score

# Import cache directory setting
try:
    from config.donotshare.donotshare import DATA_CACHE_DIR
except ImportError:
    DATA_CACHE_DIR = "c:/data-cache"


def validate_cache_file(file_path: Path) -> Dict[str, Any]:
    """Validate a single cache file and return validation results."""
    try:
        # Load the CSV file
        df = pd.read_csv(file_path, parse_dates=['timestamp'])

        # Set timestamp as index for validation
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')

        # Validate data quality
        is_valid, errors = validate_ohlcv_data(df)
        quality_score = get_data_quality_score(df)

        return {
            'file_path': str(file_path),
            'is_valid': is_valid,
            'errors': errors,
            'quality_score': quality_score['quality_score'],
            'rows': len(df),
            'columns': list(df.columns),
            'file_size_bytes': file_path.stat().st_size,
            'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime)
        }
    except Exception as e:
        return {
            'file_path': str(file_path),
            'is_valid': False,
            'errors': [f"File loading error: {str(e)}"],
            'quality_score': 0.0,
            'rows': 0,
            'columns': [],
            'file_size_bytes': file_path.stat().st_size if file_path.exists() else 0,
            'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime) if file_path.exists() else datetime.now()
        }


def scan_cache_directory(cache_dir: str) -> Dict[str, Any]:
    """Scan the cache directory and validate all files."""
    print(f"🔍 Scanning cache directory: {cache_dir}")
    print()

    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return {'error': f"Cache directory does not exist: {cache_dir}"}

    results = {
        'total_files': 0,
        'valid_files': 0,
        'invalid_files': 0,
        'failed_files': 0,
        'total_size_bytes': 0,
        'invalid_size_bytes': 0,
        'file_details': [],
        'invalid_files_list': []
    }

    # Find all CSV files
    csv_files = list(cache_path.rglob("*.csv"))
    csv_files.extend(list(cache_path.rglob("*.csv.gz")))  # Include compressed files

    print(f"📁 Found {len(csv_files)} CSV files to validate")
    print()

    for i, file_path in enumerate(csv_files, 1):
        print(f"🔄 Validating file {i}/{len(csv_files)}: {file_path.name}")

        # Validate the file
        validation_result = validate_cache_file(file_path)
        results['file_details'].append(validation_result)
        results['total_files'] += 1
        results['total_size_bytes'] += validation_result['file_size_bytes']

        if validation_result['is_valid']:
            results['valid_files'] += 1
            print(f"  ✅ Valid (Quality: {validation_result['quality_score']:.2f}, Rows: {validation_result['rows']})")
        else:
            results['invalid_files'] += 1
            results['invalid_size_bytes'] += validation_result['file_size_bytes']
            results['invalid_files_list'].append(validation_result)
            print(f"  ❌ Invalid: {', '.join(validation_result['errors'])}")

        # Progress indicator
        if i % 10 == 0:
            print(f"  📊 Progress: {i}/{len(csv_files)} ({i/len(csv_files)*100:.1f}%)")

    print()
    return results


def cleanup_invalid_files(cache_dir: str, dry_run: bool = True) -> Dict[str, Any]:
    """Remove invalid files from the cache."""
    print(f"🧹 {'DRY RUN: ' if dry_run else ''}Cleaning up invalid cache files")
    print()

    # First scan to find invalid files
    scan_results = scan_cache_directory(cache_dir)

    if 'error' in scan_results:
        return scan_results

    if scan_results['invalid_files'] == 0:
        print("✅ No invalid files found. Cache is clean!")
        return scan_results

    print("📊 Cleanup Summary:")
    print(f"  📁 Total files: {scan_results['total_files']}")
    print(f"  ✅ Valid files: {scan_results['valid_files']}")
    print(f"  ❌ Invalid files: {scan_results['invalid_files']}")
    print(f"  💾 Invalid files size: {scan_results['invalid_size_bytes'] / (1024*1024):.2f} MB")
    print()

    if dry_run:
        print("🔍 DRY RUN - No files will be deleted")
        print("Run with --cleanup to actually remove invalid files")
        print()

        # Show what would be deleted
        print("📋 Files that would be deleted:")
        for invalid_file in scan_results['invalid_files_list']:
            print(f"  ❌ {invalid_file['file_path']}")
            print(f"     Size: {invalid_file['file_size_bytes']} bytes")
            print(f"     Errors: {', '.join(invalid_file['errors'])}")
            print()
    else:
        print("🗑️  Removing invalid files...")
        removed_count = 0
        removed_size = 0

        for invalid_file in scan_results['invalid_files_list']:
            try:
                file_path = Path(invalid_file['file_path'])
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    removed_count += 1
                    removed_size += file_size
                    print(f"  ✅ Removed: {file_path.name} ({file_size} bytes)")
                else:
                    print(f"  ⚠️  File not found: {file_path.name}")
            except Exception as e:
                print(f"  ❌ Error removing {invalid_file['file_path']}: {str(e)}")

        print()
        print("🎉 Cleanup completed!")
        print(f"  📁 Files removed: {removed_count}")
        print(f"  💾 Space freed: {removed_size / (1024*1024):.2f} MB")

    return scan_results


def main():
    """Main function to run the cache validation and cleanup script."""
    parser = argparse.ArgumentParser(description="Cache Validation and Cleanup Script")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate cache files, don't remove anything")
    parser.add_argument("--cleanup", action="store_true",
                       help="Remove invalid files from cache")
    parser.add_argument("--validate-and-cleanup", action="store_true",
                       help="Validate and then cleanup invalid files")
    parser.add_argument("--cache-dir", type=str, default=DATA_CACHE_DIR,
                       help="Cache directory path")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be deleted without actually deleting")

    args = parser.parse_args()

    # If no arguments provided, default to validate-only
    if not any([args.validate_only, args.cleanup, args.validate_and_cleanup]):
        args.validate_only = True

    print("🧹 E-Trading Data Module - Cache Validation & Cleanup")
    print("=" * 60)
    print()

    try:
        if args.validate_only:
            print("🔍 VALIDATING CACHE FILES")
            print("-" * 30)
            results = scan_cache_directory(args.cache_dir)

            if 'error' not in results:
                print("\n📊 VALIDATION RESULTS")
                print("-" * 30)
                print(f"📁 Total files: {results['total_files']}")
                print(f"✅ Valid files: {results['valid_files']}")
                print(f"❌ Invalid files: {results['invalid_files']}")
                print(f"💾 Total cache size: {results['total_size_bytes'] / (1024*1024):.2f} MB")
                print(f"💾 Invalid files size: {results['invalid_size_bytes'] / (1024*1024):.2f} MB")

                if results['invalid_files'] > 0:
                    print(f"\n⚠️  Found {results['invalid_files']} invalid files!")
                    print("Run with --cleanup to remove them, or --validate-and-cleanup to do both")

        elif args.cleanup:
            print("🗑️  CLEANING UP INVALID FILES")
            print("-" * 30)
            cleanup_invalid_files(args.cache_dir, dry_run=args.dry_run)

        elif args.validate_and_cleanup:
            print("🔍 VALIDATING AND CLEANING UP")
            print("-" * 30)
            cleanup_invalid_files(args.cache_dir, dry_run=args.dry_run)

        print("\n🎉 OPERATION COMPLETED SUCCESSFULLY!")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
