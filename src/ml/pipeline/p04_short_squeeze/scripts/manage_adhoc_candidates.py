#!/usr/bin/env python3
"""
Ad-hoc Candidate Management Script for Short Squeeze Detection Pipeline

This script provides command-line utilities for managing manually added candidates,
including adding, removing, listing, and bulk operations.

Usage:
    python manage_adhoc_candidates.py <command> [options]

Commands:
    add         Add a new ad-hoc candidate
    remove      Remove (deactivate) an ad-hoc candidate
    list        List active ad-hoc candidates
    status      Show status of a specific candidate
    activate    Activate a previously deactivated candidate
    deactivate  Deactivate an active candidate
    expire      Run expiration process for TTL candidates
    extend      Extend TTL for a candidate
    bulk-add    Add multiple candidates from CSV file
    stats       Show ad-hoc candidate statistics
    cleanup     Clean up expired candidates

Examples:
    # Add a single candidate
    python manage_adhoc_candidates.py add AAPL "High volume spike observed"

    # Add with custom TTL
    python manage_adhoc_candidates.py add TSLA "Unusual options activity" --ttl 14

    # List all active candidates
    python manage_adhoc_candidates.py list

    # Show detailed status
    python manage_adhoc_candidates.py status GME

    # Remove a candidate
    python manage_adhoc_candidates.py remove AAPL

    # Bulk add from CSV
    python manage_adhoc_candidates.py bulk-add candidates.csv

    # Show statistics
    python manage_adhoc_candidates.py stats

    # Run expiration process
    python manage_adhoc_candidates.py expire
"""

import argparse
import sys
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.ml.pipeline.p04_short_squeeze.data.adhoc_manager import AdHocManager
from src.ml.pipeline.p04_short_squeeze.config.config_manager import ConfigManager

_logger = setup_logger(__name__)


class AdHocCandidateManager:
    """
    Command-line interface for ad-hoc candidate management.

    Provides comprehensive functionality for managing manually added candidates
    including CRUD operations, bulk operations, and status reporting.
    """

    def __init__(self):
        """Initialize the ad-hoc candidate manager."""
        self.adhoc_manager: Optional[AdHocManager] = None
        self.config_manager: Optional[ConfigManager] = None

    def setup_managers(self, config_path: Optional[str] = None) -> bool:
        """
        Setup the ad-hoc manager and configuration.

        Args:
            config_path: Optional path to configuration file

        Returns:
            True if setup successful, False otherwise
        """
        try:
            # Load configuration to get default TTL
            self.config_manager = ConfigManager(config_path)
            config = self.config_manager.load_config()
            default_ttl = config.adhoc.default_ttl_days

            # Initialize ad-hoc manager
            self.adhoc_manager = AdHocManager(default_ttl_days=default_ttl)

            return True

        except Exception:
            _logger.exception("Failed to setup managers:")
            return False

    def add_candidate(self, ticker: str, reason: str, ttl_days: Optional[int] = None) -> bool:
        """
        Add a new ad-hoc candidate.

        Args:
            ticker: Stock ticker symbol
            reason: Reason for adding the candidate
            ttl_days: Time-to-live in days

        Returns:
            True if candidate was added successfully, False otherwise
        """
        try:
            ticker = ticker.upper().strip()

            if not ticker:
                print("Error: Ticker cannot be empty")
                return False

            if not reason or not reason.strip():
                print("Error: Reason cannot be empty")
                return False

            print(f"Adding ad-hoc candidate: {ticker}")
            print(f"Reason: {reason}")
            if ttl_days:
                print(f"TTL: {ttl_days} days")

            success = self.adhoc_manager.add_candidate(ticker, reason, ttl_days)

            if success:
                print(f"✅ Successfully added ad-hoc candidate: {ticker}")
                return True
            else:
                print(f"❌ Failed to add ad-hoc candidate: {ticker}")
                return False

        except Exception as e:
            print(f"❌ Error adding candidate: {e}")
            return False

    def remove_candidate(self, ticker: str) -> bool:
        """
        Remove (deactivate) an ad-hoc candidate.

        Args:
            ticker: Stock ticker symbol

        Returns:
            True if candidate was removed successfully, False otherwise
        """
        try:
            ticker = ticker.upper().strip()

            if not ticker:
                print("Error: Ticker cannot be empty")
                return False

            print(f"Removing ad-hoc candidate: {ticker}")

            success = self.adhoc_manager.remove_candidate(ticker)

            if success:
                print(f"✅ Successfully removed ad-hoc candidate: {ticker}")
                return True
            else:
                print(f"❌ Failed to remove ad-hoc candidate: {ticker} (may not exist or already inactive)")
                return False

        except Exception as e:
            print(f"❌ Error removing candidate: {e}")
            return False

    def list_candidates(self, show_details: bool = False) -> bool:
        """
        List active ad-hoc candidates.

        Args:
            show_details: Show detailed information if True

        Returns:
            True if listing successful, False otherwise
        """
        try:
            candidates = self.adhoc_manager.get_active_candidates()

            if not candidates:
                print("No active ad-hoc candidates found.")
                return True

            print(f"\n📋 Active Ad-hoc Candidates ({len(candidates)}):")
            print("=" * 80)

            for i, candidate in enumerate(candidates, 1):
                print(f"{i:2d}. {candidate.ticker}")

                if show_details:
                    print(f"    Reason: {candidate.reason}")
                    print(f"    Added: {candidate.first_seen.strftime('%Y-%m-%d %H:%M:%S')}")

                    if candidate.expires_at:
                        days_left = (candidate.expires_at - datetime.now()).days
                        status = "⚠️ EXPIRING SOON" if days_left <= 3 else "✅ Active"
                        print(f"    Expires: {candidate.expires_at.strftime('%Y-%m-%d %H:%M:%S')} ({days_left} days left) {status}")

                    if candidate.promoted_by_screener:
                        print("    🎯 Promoted by screener")

                    print()

            return True

        except Exception as e:
            print(f"❌ Error listing candidates: {e}")
            return False

    def show_candidate_status(self, ticker: str) -> bool:
        """
        Show detailed status of a specific candidate.

        Args:
            ticker: Stock ticker symbol

        Returns:
            True if status shown successfully, False otherwise
        """
        try:
            ticker = ticker.upper().strip()
            candidate = self.adhoc_manager.get_candidate(ticker)

            if not candidate:
                print(f"❌ Ad-hoc candidate '{ticker}' not found")
                return False

            print(f"\n📊 Ad-hoc Candidate Status: {candidate.ticker}")
            print("=" * 50)
            print(f"Ticker: {candidate.ticker}")
            print(f"Reason: {candidate.reason}")
            print(f"Status: {'✅ Active' if candidate.active else '❌ Inactive'}")
            print(f"Added: {candidate.first_seen.strftime('%Y-%m-%d %H:%M:%S')}")

            if candidate.expires_at:
                days_left = (candidate.expires_at - datetime.now()).days
                status_emoji = "⚠️" if days_left <= 3 else "✅"
                print(f"Expires: {candidate.expires_at.strftime('%Y-%m-%d %H:%M:%S')} ({days_left} days left) {status_emoji}")

            if candidate.promoted_by_screener:
                print("🎯 Promoted by screener: Yes")
            else:
                print("🎯 Promoted by screener: No")

            return True

        except Exception as e:
            print(f"❌ Error showing candidate status: {e}")
            return False

    def activate_candidate(self, ticker: str) -> bool:
        """
        Activate a previously deactivated candidate.

        Args:
            ticker: Stock ticker symbol

        Returns:
            True if candidate was activated successfully, False otherwise
        """
        try:
            ticker = ticker.upper().strip()
            print(f"Activating ad-hoc candidate: {ticker}")

            success = self.adhoc_manager.activate_candidate(ticker)

            if success:
                print(f"✅ Successfully activated ad-hoc candidate: {ticker}")
                return True
            else:
                print(f"❌ Failed to activate ad-hoc candidate: {ticker}")
                return False

        except Exception as e:
            print(f"❌ Error activating candidate: {e}")
            return False

    def deactivate_candidate(self, ticker: str) -> bool:
        """
        Deactivate an active candidate.

        Args:
            ticker: Stock ticker symbol

        Returns:
            True if candidate was deactivated successfully, False otherwise
        """
        try:
            ticker = ticker.upper().strip()
            print(f"Deactivating ad-hoc candidate: {ticker}")

            success = self.adhoc_manager.deactivate_candidate(ticker)

            if success:
                print(f"✅ Successfully deactivated ad-hoc candidate: {ticker}")
                return True
            else:
                print(f"❌ Failed to deactivate ad-hoc candidate: {ticker}")
                return False

        except Exception as e:
            print(f"❌ Error deactivating candidate: {e}")
            return False

    def expire_candidates(self) -> bool:
        """
        Run expiration process for TTL candidates.

        Returns:
            True if expiration process completed successfully, False otherwise
        """
        try:
            print("Running ad-hoc candidate expiration process...")

            expired_tickers = self.adhoc_manager.expire_candidates()

            if expired_tickers:
                print(f"✅ Expired {len(expired_tickers)} candidates:")
                for ticker in expired_tickers:
                    print(f"  - {ticker}")
            else:
                print("✅ No candidates expired")

            return True

        except Exception as e:
            print(f"❌ Error running expiration process: {e}")
            return False

    def extend_ttl(self, ticker: str, additional_days: int) -> bool:
        """
        Extend TTL for a candidate.

        Args:
            ticker: Stock ticker symbol
            additional_days: Number of additional days

        Returns:
            True if TTL was extended successfully, False otherwise
        """
        try:
            ticker = ticker.upper().strip()

            if additional_days <= 0:
                print("Error: Additional days must be positive")
                return False

            print(f"Extending TTL for {ticker} by {additional_days} days...")

            success = self.adhoc_manager.extend_ttl(ticker, additional_days)

            if success:
                print(f"✅ Successfully extended TTL for {ticker}")
                return True
            else:
                print(f"❌ Failed to extend TTL for {ticker}")
                return False

        except Exception as e:
            print(f"❌ Error extending TTL: {e}")
            return False

    def bulk_add_candidates(self, csv_file: str) -> bool:
        """
        Add multiple candidates from CSV file.

        Args:
            csv_file: Path to CSV file with candidate data

        Returns:
            True if bulk add completed successfully, False otherwise
        """
        try:
            csv_path = Path(csv_file)
            if not csv_path.exists():
                print(f"❌ CSV file not found: {csv_file}")
                return False

            print(f"Loading candidates from CSV file: {csv_file}")

            candidates_data = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                # Validate required columns
                required_columns = {'ticker', 'reason'}
                if not required_columns.issubset(reader.fieldnames):
                    print(f"❌ CSV file must contain columns: {required_columns}")
                    print(f"Found columns: {reader.fieldnames}")
                    return False

                for row_num, row in enumerate(reader, 1):
                    try:
                        candidate_data = {
                            'ticker': row['ticker'].strip(),
                            'reason': row['reason'].strip(),
                            'ttl_days': int(row.get('ttl_days', 0)) or None
                        }

                        # Validate data
                        is_valid, errors = self.adhoc_manager.validate_candidate_data(candidate_data)
                        if not is_valid:
                            print(f"⚠️ Row {row_num}: Validation errors: {errors}")
                            continue

                        candidates_data.append(candidate_data)

                    except Exception as e:
                        print(f"⚠️ Row {row_num}: Error parsing data: {e}")
                        continue

            if not candidates_data:
                print("❌ No valid candidates found in CSV file")
                return False

            print(f"Found {len(candidates_data)} valid candidates in CSV")

            # Perform bulk add
            added_count, errors = self.adhoc_manager.bulk_add_candidates(candidates_data)

            print("\n📊 Bulk Add Results:")
            print(f"✅ Successfully added: {added_count}")
            print(f"❌ Errors: {len(errors)}")

            if errors:
                print("\nErrors:")
                for error in errors[:10]:  # Show first 10 errors
                    print(f"  - {error}")
                if len(errors) > 10:
                    print(f"  ... and {len(errors) - 10} more errors")

            return added_count > 0

        except Exception as e:
            print(f"❌ Error in bulk add: {e}")
            return False

    def show_statistics(self) -> bool:
        """
        Show ad-hoc candidate statistics.

        Returns:
            True if statistics shown successfully, False otherwise
        """
        try:
            stats = self.adhoc_manager.get_statistics()

            print("\n📊 Ad-hoc Candidate Statistics")
            print("=" * 40)
            print(f"Total Active: {stats['total_active']}")
            print(f"Promoted by Screener: {stats['promoted_by_screener']}")
            print(f"Expiring within 3 days: {stats['expiring_within_3_days']}")
            print(f"Average Age: {stats['average_age_days']} days")
            print(f"Default TTL: {stats['default_ttl_days']} days")
            print(f"Last Updated: {stats['last_updated'].strftime('%Y-%m-%d %H:%M:%S')}")

            # Show expiring candidates if any
            if stats['expiring_within_3_days'] > 0:
                print("\n⚠️ Candidates Expiring Soon:")
                expiring = self.adhoc_manager.get_expiring_candidates(3)
                for candidate in expiring:
                    days_left = (candidate.expires_at - datetime.now()).days
                    print(f"  - {candidate.ticker}: {days_left} days left")

            return True

        except Exception as e:
            print(f"❌ Error showing statistics: {e}")
            return False

    def cleanup_expired(self) -> bool:
        """
        Clean up expired candidates (same as expire).

        Returns:
            True if cleanup completed successfully, False otherwise
        """
        return self.expire_candidates()

    def create_sample_csv(self, output_file: str) -> bool:
        """
        Create a sample CSV file for bulk import.

        Args:
            output_file: Path to output CSV file

        Returns:
            True if sample file created successfully, False otherwise
        """
        try:
            sample_data = [
                {'ticker': 'AAPL', 'reason': 'High volume spike observed', 'ttl_days': 7},
                {'ticker': 'TSLA', 'reason': 'Unusual options activity', 'ttl_days': 14},
                {'ticker': 'GME', 'reason': 'Social media buzz increasing', 'ttl_days': 10},
            ]

            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['ticker', 'reason', 'ttl_days'])
                writer.writeheader()
                writer.writerows(sample_data)

            print(f"✅ Sample CSV file created: {output_file}")
            print("Edit this file and use 'bulk-add' command to import candidates")
            return True

        except Exception as e:
            print(f"❌ Error creating sample CSV: {e}")
            return False


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Manage ad-hoc candidates for short squeeze detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Usage:')[1] if 'Usage:' in __doc__ else ""
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Add command
    add_parser = subparsers.add_parser('add', help='Add a new ad-hoc candidate')
    add_parser.add_argument('ticker', help='Stock ticker symbol')
    add_parser.add_argument('reason', help='Reason for adding the candidate')
    add_parser.add_argument('--ttl', type=int, help='Time-to-live in days')

    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Remove an ad-hoc candidate')
    remove_parser.add_argument('ticker', help='Stock ticker symbol')

    # List command
    list_parser = subparsers.add_parser('list', help='List active ad-hoc candidates')
    list_parser.add_argument('--details', action='store_true', help='Show detailed information')

    # Status command
    status_parser = subparsers.add_parser('status', help='Show status of a specific candidate')
    status_parser.add_argument('ticker', help='Stock ticker symbol')

    # Activate command
    activate_parser = subparsers.add_parser('activate', help='Activate a candidate')
    activate_parser.add_argument('ticker', help='Stock ticker symbol')

    # Deactivate command
    deactivate_parser = subparsers.add_parser('deactivate', help='Deactivate a candidate')
    deactivate_parser.add_argument('ticker', help='Stock ticker symbol')

    # Expire command
    expire_parser = subparsers.add_parser('expire', help='Run expiration process')

    # Extend command
    extend_parser = subparsers.add_parser('extend', help='Extend TTL for a candidate')
    extend_parser.add_argument('ticker', help='Stock ticker symbol')
    extend_parser.add_argument('days', type=int, help='Additional days to extend')

    # Bulk-add command
    bulk_add_parser = subparsers.add_parser('bulk-add', help='Add candidates from CSV file')
    bulk_add_parser.add_argument('csv_file', help='Path to CSV file')

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show statistics')

    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up expired candidates')

    # Sample CSV command
    sample_parser = subparsers.add_parser('sample-csv', help='Create sample CSV file')
    sample_parser.add_argument('output_file', help='Output CSV file path')

    return parser


def main() -> int:
    """
    Main entry point for the ad-hoc candidate management script.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        parser = create_parser()
        args = parser.parse_args()

        if not args.command:
            parser.print_help()
            return 1

        # Setup logging
        if args.verbose:
            import logging
            logging.getLogger().setLevel(logging.DEBUG)

        # Initialize manager
        manager = AdHocCandidateManager()

        # Special case for sample-csv command (doesn't need database)
        if args.command == 'sample-csv':
            return 0 if manager.create_sample_csv(args.output_file) else 1

        # Setup managers for all other commands
        if not manager.setup_managers(args.config):
            return 1

        # Execute command
        success = False

        if args.command == 'add':
            success = manager.add_candidate(args.ticker, args.reason, args.ttl)

        elif args.command == 'remove':
            success = manager.remove_candidate(args.ticker)

        elif args.command == 'list':
            success = manager.list_candidates(args.details)

        elif args.command == 'status':
            success = manager.show_candidate_status(args.ticker)

        elif args.command == 'activate':
            success = manager.activate_candidate(args.ticker)

        elif args.command == 'deactivate':
            success = manager.deactivate_candidate(args.ticker)

        elif args.command == 'expire':
            success = manager.expire_candidates()

        elif args.command == 'extend':
            success = manager.extend_ttl(args.ticker, args.days)

        elif args.command == 'bulk-add':
            success = manager.bulk_add_candidates(args.csv_file)

        elif args.command == 'stats':
            success = manager.show_statistics()

        elif args.command == 'cleanup':
            success = manager.cleanup_expired()

        else:
            print(f"❌ Unknown command: {args.command}")
            parser.print_help()
            return 1

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        return 130
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)