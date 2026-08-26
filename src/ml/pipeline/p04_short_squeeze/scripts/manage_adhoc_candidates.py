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
import csv
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p04_short_squeeze.config.config_manager import ConfigManager
from src.ml.pipeline.p04_short_squeeze.data.adhoc_manager import AdHocManager
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class AdHocCandidateManager:
    """
    Command-line interface for ad-hoc candidate management.

    Provides comprehensive functionality for managing manually added candidates
    including CRUD operations, bulk operations, and status reporting.
    """

    def __init__(self):
        """Initialize the ad-hoc candidate manager."""
        self.adhoc_manager: AdHocManager | None = None
        self.config_manager: ConfigManager | None = None

    def _require_manager(self) -> AdHocManager:
        """Return the ad-hoc manager, failing fast if setup_managers() was not run."""
        if self.adhoc_manager is None:
            raise RuntimeError("setup_managers() must be called before using the CLI commands")
        return self.adhoc_manager

    def setup_managers(self, config_path: str | None = None) -> bool:
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

    def add_candidate(self, ticker: str, reason: str, ttl_days: int | None = None) -> bool:
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
                _logger.error("Cannot add candidate: ticker cannot be empty")
                return False

            if not reason or not reason.strip():
                _logger.error("Cannot add candidate: reason cannot be empty")
                return False

            _logger.info("Adding ad-hoc candidate: %s", ticker)
            _logger.info("Reason: %s", reason)
            if ttl_days:
                _logger.info("TTL: %d days", ttl_days)

            success = self._require_manager().add_candidate(ticker, reason, ttl_days)

            if success:
                _logger.info("Successfully added ad-hoc candidate: %s", ticker)
                return True
            else:
                _logger.error("Failed to add ad-hoc candidate: %s", ticker)
                return False

        except Exception:
            _logger.exception("Error adding candidate:")
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
                _logger.error("Cannot remove candidate: ticker cannot be empty")
                return False

            _logger.info("Removing ad-hoc candidate: %s", ticker)

            success = self._require_manager().remove_candidate(ticker)

            if success:
                _logger.info("Successfully removed ad-hoc candidate: %s", ticker)
                return True
            else:
                _logger.error("Failed to remove ad-hoc candidate: %s (may not exist or already inactive)", ticker)
                return False

        except Exception:
            _logger.exception("Error removing candidate:")
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
            candidates = self._require_manager().get_active_candidates()

            if not candidates:
                _logger.info("No active ad-hoc candidates found.")
                return True

            _logger.info("Active Ad-hoc Candidates (%d):", len(candidates))

            for i, candidate in enumerate(candidates, 1):
                _logger.info("%2d. %s", i, candidate.ticker)

                if show_details:
                    _logger.info("    Reason: %s", candidate.reason)
                    _logger.info("    Added: %s", candidate.first_seen.strftime("%Y-%m-%d %H:%M:%S"))

                    if candidate.expires_at:
                        days_left = (candidate.expires_at - datetime.now()).days
                        status = "EXPIRING SOON" if days_left <= 3 else "Active"
                        _logger.info(
                            "    Expires: %s (%d days left) %s",
                            candidate.expires_at.strftime("%Y-%m-%d %H:%M:%S"),
                            days_left,
                            status,
                        )

                    if candidate.promoted_by_screener:
                        _logger.info("    Promoted by screener")

            return True

        except Exception:
            _logger.exception("Error listing candidates:")
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
            candidate = self._require_manager().get_candidate(ticker)

            if not candidate:
                _logger.error("Ad-hoc candidate '%s' not found", ticker)
                return False

            _logger.info("Ad-hoc Candidate Status: %s", candidate.ticker)
            _logger.info("Ticker: %s", candidate.ticker)
            _logger.info("Reason: %s", candidate.reason)
            _logger.info("Status: %s", "Active" if candidate.active else "Inactive")
            _logger.info("Added: %s", candidate.first_seen.strftime("%Y-%m-%d %H:%M:%S"))

            if candidate.expires_at:
                days_left = (candidate.expires_at - datetime.now()).days
                _logger.info(
                    "Expires: %s (%d days left)", candidate.expires_at.strftime("%Y-%m-%d %H:%M:%S"), days_left
                )

            _logger.info("Promoted by screener: %s", "Yes" if candidate.promoted_by_screener else "No")

            return True

        except Exception:
            _logger.exception("Error showing candidate status:")
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
            _logger.info("Activating ad-hoc candidate: %s", ticker)

            success = self._require_manager().activate_candidate(ticker)

            if success:
                _logger.info("Successfully activated ad-hoc candidate: %s", ticker)
                return True
            else:
                _logger.error("Failed to activate ad-hoc candidate: %s", ticker)
                return False

        except Exception:
            _logger.exception("Error activating candidate:")
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
            _logger.info("Deactivating ad-hoc candidate: %s", ticker)

            success = self._require_manager().deactivate_candidate(ticker)

            if success:
                _logger.info("Successfully deactivated ad-hoc candidate: %s", ticker)
                return True
            else:
                _logger.error("Failed to deactivate ad-hoc candidate: %s", ticker)
                return False

        except Exception:
            _logger.exception("Error deactivating candidate:")
            return False

    def expire_candidates(self) -> bool:
        """
        Run expiration process for TTL candidates.

        Returns:
            True if expiration process completed successfully, False otherwise
        """
        try:
            _logger.info("Running ad-hoc candidate expiration process...")

            expired_tickers = self._require_manager().expire_candidates()

            if expired_tickers:
                _logger.info("Expired %d candidates: %s", len(expired_tickers), expired_tickers)
            else:
                _logger.info("No candidates expired")

            return True

        except Exception:
            _logger.exception("Error running expiration process:")
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
                _logger.error("Cannot extend TTL: additional days must be positive")
                return False

            _logger.info("Extending TTL for %s by %d days...", ticker, additional_days)

            success = self._require_manager().extend_ttl(ticker, additional_days)

            if success:
                _logger.info("Successfully extended TTL for %s", ticker)
                return True
            else:
                _logger.error("Failed to extend TTL for %s", ticker)
                return False

        except Exception:
            _logger.exception("Error extending TTL:")
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
                _logger.error("CSV file not found: %s", csv_file)
                return False

            _logger.info("Loading candidates from CSV file: %s", csv_file)

            candidates_data = []
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)

                # Validate required columns
                required_columns = {"ticker", "reason"}
                if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
                    _logger.error("CSV file must contain columns: %s", required_columns)
                    _logger.error("Found columns: %s", reader.fieldnames)
                    return False

                for row_num, row in enumerate(reader, 1):
                    try:
                        candidate_data = {
                            "ticker": row["ticker"].strip(),
                            "reason": row["reason"].strip(),
                            "ttl_days": int(row.get("ttl_days", 0)) or None,
                        }

                        # Validate data
                        is_valid, errors = self._require_manager().validate_candidate_data(candidate_data)
                        if not is_valid:
                            _logger.warning("Row %d: Validation errors: %s", row_num, errors)
                            continue

                        candidates_data.append(candidate_data)

                    except Exception as e:
                        _logger.warning("Row %d: Error parsing data: %s", row_num, e)
                        continue

            if not candidates_data:
                _logger.error("No valid candidates found in CSV file")
                return False

            _logger.info("Found %d valid candidates in CSV", len(candidates_data))

            # Perform bulk add
            added_count, errors = self._require_manager().bulk_add_candidates(candidates_data)

            _logger.info("Bulk Add Results: added=%d, errors=%d", added_count, len(errors))

            if errors:
                for error in errors[:10]:  # Show first 10 errors
                    _logger.warning("  - %s", error)
                if len(errors) > 10:
                    _logger.warning("  ... and %d more errors", len(errors) - 10)

            return added_count > 0

        except Exception:
            _logger.exception("Error in bulk add:")
            return False

    def show_statistics(self) -> bool:
        """
        Show ad-hoc candidate statistics.

        Returns:
            True if statistics shown successfully, False otherwise
        """
        try:
            stats = self._require_manager().get_statistics()

            _logger.info("Ad-hoc Candidate Statistics")
            _logger.info("Total Active: %s", stats["total_active"])
            _logger.info("Promoted by Screener: %s", stats["promoted_by_screener"])
            _logger.info("Expiring within 3 days: %s", stats["expiring_within_3_days"])
            _logger.info("Average Age: %s days", stats["average_age_days"])
            _logger.info("Default TTL: %s days", stats["default_ttl_days"])
            _logger.info("Last Updated: %s", stats["last_updated"].strftime("%Y-%m-%d %H:%M:%S"))

            # Show expiring candidates if any
            if stats["expiring_within_3_days"] > 0:
                _logger.info("Candidates Expiring Soon:")
                expiring = self._require_manager().get_expiring_candidates(3)
                for candidate in expiring:
                    if candidate.expires_at is None:
                        continue
                    days_left = (candidate.expires_at - datetime.now()).days
                    _logger.info("  - %s: %d days left", candidate.ticker, days_left)

            return True

        except Exception:
            _logger.exception("Error showing statistics:")
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
                {"ticker": "AAPL", "reason": "High volume spike observed", "ttl_days": 7},
                {"ticker": "TSLA", "reason": "Unusual options activity", "ttl_days": 14},
                {"ticker": "GME", "reason": "Social media buzz increasing", "ttl_days": 10},
            ]

            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["ticker", "reason", "ttl_days"])
                writer.writeheader()
                writer.writerows(sample_data)

            _logger.info("Sample CSV file created: %s", output_file)
            _logger.info("Edit this file and use 'bulk-add' command to import candidates")
            return True

        except Exception:
            _logger.exception("Error creating sample CSV:")
            return False


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Manage ad-hoc candidates for short squeeze detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if __doc__ and "Usage:" in __doc__ else "",
    )

    parser.add_argument("--config", "-c", type=str, help="Path to configuration file")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add a new ad-hoc candidate")
    add_parser.add_argument("ticker", help="Stock ticker symbol")
    add_parser.add_argument("reason", help="Reason for adding the candidate")
    add_parser.add_argument("--ttl", type=int, help="Time-to-live in days")

    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove an ad-hoc candidate")
    remove_parser.add_argument("ticker", help="Stock ticker symbol")

    # List command
    list_parser = subparsers.add_parser("list", help="List active ad-hoc candidates")
    list_parser.add_argument("--details", action="store_true", help="Show detailed information")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show status of a specific candidate")
    status_parser.add_argument("ticker", help="Stock ticker symbol")

    # Activate command
    activate_parser = subparsers.add_parser("activate", help="Activate a candidate")
    activate_parser.add_argument("ticker", help="Stock ticker symbol")

    # Deactivate command
    deactivate_parser = subparsers.add_parser("deactivate", help="Deactivate a candidate")
    deactivate_parser.add_argument("ticker", help="Stock ticker symbol")

    # Expire command
    subparsers.add_parser("expire", help="Run expiration process")

    # Extend command
    extend_parser = subparsers.add_parser("extend", help="Extend TTL for a candidate")
    extend_parser.add_argument("ticker", help="Stock ticker symbol")
    extend_parser.add_argument("days", type=int, help="Additional days to extend")

    # Bulk-add command
    bulk_add_parser = subparsers.add_parser("bulk-add", help="Add candidates from CSV file")
    bulk_add_parser.add_argument("csv_file", help="Path to CSV file")

    # Stats command
    subparsers.add_parser("stats", help="Show statistics")

    # Cleanup command
    subparsers.add_parser("cleanup", help="Clean up expired candidates")

    # Sample CSV command
    sample_parser = subparsers.add_parser("sample-csv", help="Create sample CSV file")
    sample_parser.add_argument("output_file", help="Output CSV file path")

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
        if args.command == "sample-csv":
            return 0 if manager.create_sample_csv(args.output_file) else 1

        # Setup managers for all other commands
        if not manager.setup_managers(args.config):
            return 1

        # Execute command
        success = False

        if args.command == "add":
            success = manager.add_candidate(args.ticker, args.reason, args.ttl)

        elif args.command == "remove":
            success = manager.remove_candidate(args.ticker)

        elif args.command == "list":
            success = manager.list_candidates(args.details)

        elif args.command == "status":
            success = manager.show_candidate_status(args.ticker)

        elif args.command == "activate":
            success = manager.activate_candidate(args.ticker)

        elif args.command == "deactivate":
            success = manager.deactivate_candidate(args.ticker)

        elif args.command == "expire":
            success = manager.expire_candidates()

        elif args.command == "extend":
            success = manager.extend_ttl(args.ticker, args.days)

        elif args.command == "bulk-add":
            success = manager.bulk_add_candidates(args.csv_file)

        elif args.command == "stats":
            success = manager.show_statistics()

        elif args.command == "cleanup":
            success = manager.cleanup_expired()

        else:
            _logger.error("Unknown command: %s", args.command)
            parser.print_help()
            return 1

        return 0 if success else 1

    except KeyboardInterrupt:
        _logger.warning("Operation cancelled by user")
        return 130
    except Exception:
        _logger.exception("Unexpected error:")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
