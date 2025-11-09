#!/usr/bin/env python3
"""
Weekly Universe Loader Script for Short Squeeze Detection Pipeline

This script runs weekly to load and filter the stock universe from FMP
based on market cap and other criteria. It does NOT perform individual
ticker analysis - that's handled by the daily volume detector.

Usage:
    python run_weekly_screener.py [options]

Examples:
    # Run with default configuration
    python run_weekly_screener.py

    # Run with custom configuration file
    python run_weekly_screener.py --config /path/to/config.yaml

    # Run with specific universe size limit
    python run_weekly_screener.py --max-universe 1000

    # Run in dry-run mode (no database writes)
    python run_weekly_screener.py --dry-run

    # Run with verbose logging
    python run_weekly_screener.py --verbose
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Dict, Any, List

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.data.downloader.fmp_data_downloader import FMPDataDownloader
from src.data.db.services.short_squeeze_service import ShortSqueezeService
from src.ml.pipeline.p04_short_squeeze.config.config_manager import ConfigManager
from src.ml.pipeline.p04_short_squeeze.core.universe_loader import create_universe_loader

_logger = setup_logger(__name__)


class WeeklyUniverseLoader:
    """
    Runner class for the weekly universe loader script.

    Handles command-line arguments, configuration loading, and orchestrates
    the weekly universe loading process with comprehensive error handling and metrics.
    This script ONLY loads the universe - no individual ticker analysis.
    """

    def __init__(self):
        """Initialize the weekly screener runner."""
        self.config_manager: Optional[ConfigManager] = None
        self.fmp_downloader: Optional[FMPDataDownloader] = None
        self.start_time: Optional[datetime] = None
        self.run_id: Optional[str] = None

    def parse_arguments(self) -> argparse.Namespace:
        """
        Parse command-line arguments.

        Returns:
            Parsed arguments namespace
        """
        parser = argparse.ArgumentParser(
            description="Load weekly stock universe for short squeeze detection",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=__doc__.split('Usage:')[1] if 'Usage:' in __doc__ else ""
        )

        parser.add_argument(
            '--config', '-c',
            type=str,
            help='Path to configuration file (default: uses default config location)'
        )

        parser.add_argument(
            '--max-universe',
            type=int,
            help='Maximum number of stocks in universe (for testing/debugging)'
        )

        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Run without writing results to database'
        )

        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose logging'
        )

        parser.add_argument(
            '--test-connection',
            action='store_true',
            help='Test API connections and exit'
        )

        parser.add_argument(
            '--run-id',
            type=str,
            help='Custom run ID (default: auto-generated timestamp)'
        )

        parser.add_argument(
            '--output-dir',
            type=str,
            help='Directory to save output reports (optional)'
        )

        return parser.parse_args()

    def setup_logging(self, verbose: bool) -> None:
        """
        Setup logging configuration.

        Args:
            verbose: Enable verbose logging if True
        """
        if verbose:
            import logging
            logging.getLogger().setLevel(logging.DEBUG)
            _logger.info("Verbose logging enabled")

    def load_configuration(self, config_path: Optional[str]) -> bool:
        """
        Load and validate configuration.

        Args:
            config_path: Optional path to configuration file

        Returns:
            True if configuration loaded successfully, False otherwise
        """
        try:
            _logger.info("Loading configuration...")
            self.config_manager = ConfigManager(config_path)
            config = self.config_manager.load_config()

            _logger.info("Configuration loaded successfully")
            _logger.info("Run ID: %s", config.run_id)

            # Get universe configuration
            universe_config = getattr(config.screener, 'universe', None)
            if universe_config:
                _logger.info("Universe filters: min_market_cap=$%.0fM, min_volume=%d, exchanges=%s",
                           universe_config.min_market_cap / 1_000_000,
                           universe_config.min_avg_volume,
                           universe_config.exchanges)

            return True

        except Exception:
            _logger.exception("Failed to load configuration:")
            return False

    def initialize_data_providers(self) -> bool:
        """
        Initialize and test data provider connections.

        Returns:
            True if all providers initialized successfully, False otherwise
        """
        try:
            _logger.info("Initializing data providers...")

            # Initialize FMP downloader
            self.fmp_downloader = FMPDataDownloader()

            # Test connection
            if not self.fmp_downloader.test_connection():
                _logger.error("FMP API connection test failed")
                return False

            _logger.info("FMP API connection successful")
            return True

        except Exception:
            _logger.exception("Failed to initialize data providers:")
            return False

    def load_universe(self, max_universe: Optional[int] = None) -> Optional[List[str]]:
        """
        Load the stock universe from FMP with market cap and volume filtering.

        Args:
            max_universe: Optional limit on universe size

        Returns:
            List of ticker symbols or None if failed
        """
        try:
            _logger.info("Loading stock universe from FMP...")

            # Get universe configuration
            config = self.config_manager.load_config()
            universe_config = getattr(config.screener, 'universe', None)

            if not universe_config:
                _logger.warning("No screener.universe config found, using default settings")
                from src.ml.pipeline.p04_short_squeeze.config.data_classes import UniverseConfig
                universe_config = UniverseConfig()

            universe_loader = create_universe_loader(self.fmp_downloader, universe_config)
            universe = universe_loader.load_universe()

            if not universe:
                _logger.error("Failed to load universe - no stocks returned")
                return None

            # Apply max universe limit if specified
            if max_universe and len(universe) > max_universe:
                _logger.info("Limiting universe from %d to %d stocks for testing",
                           len(universe), max_universe)
                universe = universe[:max_universe]

            _logger.info("Successfully loaded universe of %d stocks", len(universe))
            return universe

        except Exception:
            _logger.exception("Failed to load universe:")
            return None

    def store_universe(self, universe: List[str], dry_run: bool = False) -> Optional[Dict[str, Any]]:
        """
        Store the universe in the database for use by other pipeline components.

        Args:
            universe: List of ticker symbols
            dry_run: If True, don't write to database

        Returns:
            Dictionary with storage results or None if failed
        """
        try:
            _logger.info("Storing universe of %d stocks...", len(universe))

            current_date = date.today()
            _logger.info("Storing universe for run date: %s", current_date)

            if dry_run:
                _logger.info("DRY RUN MODE: Universe would be stored in database")
                stored_count = len(universe)
                strategy_breakdown = {'mid_cap': 500, 'small_cap': 300, 'known_candidates': len(universe) - 800}
            else:
                # Get universe loader to categorize tickers by strategy
                config = self.config_manager.load_config()
                universe_config = getattr(config.screener, 'universe', None)
                universe_loader = create_universe_loader(self.fmp_downloader, universe_config)

                # Get strategy breakdown from universe loader
                strategy_breakdown = self._get_strategy_breakdown(universe_loader, universe)

                # Store universe as a snapshot for other components to use
                universe_data = []

                for ticker in universe:
                    # Determine which strategy this ticker came from
                    strategy = self._determine_ticker_strategy(ticker, strategy_breakdown)

                    universe_data.append({
                        'ticker': ticker.upper(),
                        'run_date': current_date,
                        'screener_score': 0.0,  # No scoring in universe loading
                        'raw_payload': {
                            'universe_load': True,
                            'ticker': ticker,
                            'source': 'universe_loader',
                            'strategy': strategy,
                            'load_timestamp': datetime.now().isoformat()
                        },
                        'data_quality': 1.0  # High quality since it's from market cap filtering
                    })

                _logger.info("Preparing to store %d universe entries in ss_snapshot table", len(universe_data))

                service = ShortSqueezeService()
                stored_count = service.save_screener_results(universe_data, current_date)

                _logger.info("Database operation completed: %d entries stored for %s", stored_count, current_date)

            # Prepare results
            results_dict = {
                'run_id': self.run_id,
                'load_date': date.today().isoformat(),
                'universe_size': len(universe),
                'stored_count': stored_count,
                'universe_sample': universe[:10],  # First 10 tickers as sample
                'storage_success': stored_count > 0,
                'strategy_breakdown': strategy_breakdown
            }

            _logger.info("Universe storage completed: %d/%d stocks stored", stored_count, len(universe))
            _logger.info("Strategy breakdown: %s", strategy_breakdown)
            return results_dict

        except Exception:
            _logger.exception("Universe storage failed:")
            return None

    def _get_strategy_breakdown(self, universe_loader, universe: List[str]) -> Dict[str, int]:
        """
        Get breakdown of universe by loading strategy.

        Args:
            universe_loader: Universe loader instance
            universe: List of ticker symbols

        Returns:
            Dictionary with strategy breakdown
        """
        try:
            # Get known candidates
            known_candidates = universe_loader._get_known_short_interest_candidates()
            known_in_universe = [t for t in universe if t.upper() in [k.upper() for k in known_candidates]]

            # Estimate strategy breakdown (this is approximate since we don't track exact source)
            config = self.config_manager.load_config()
            universe_config = getattr(config.screener, 'universe', None)

            max_universe = getattr(universe_config, 'max_universe_size', 1000)
            strategy_1_limit = min(500, max_universe // 2)
            strategy_2_limit = min(300, max_universe // 3)

            # Approximate breakdown
            mid_cap_count = min(strategy_1_limit, len(universe))
            small_cap_count = min(strategy_2_limit, max(0, len(universe) - mid_cap_count))
            known_candidates_count = len(known_in_universe)
            other_count = max(0, len(universe) - mid_cap_count - small_cap_count)

            return {
                'mid_cap_strategy': mid_cap_count,
                'small_cap_strategy': small_cap_count,
                'known_candidates': known_candidates_count,
                'other': other_count,
                'total': len(universe)
            }

        except Exception as e:
            _logger.warning("Error getting strategy breakdown: %s", e)
            return {'total': len(universe), 'unknown': len(universe)}

    def _determine_ticker_strategy(self, ticker: str, strategy_breakdown: Dict[str, int]) -> str:
        """
        Determine which strategy a ticker likely came from.

        Args:
            ticker: Ticker symbol
            strategy_breakdown: Strategy breakdown dictionary

        Returns:
            Strategy name
        """
        # This is a simplified approach - in practice you'd need to track this during loading
        # For now, just return 'multi_strategy' since we combine all strategies
        return 'multi_strategy'

    def generate_performance_report(self, results: Dict[str, Any]) -> None:
        """
        Generate and log performance metrics report.

        Args:
            results: Universe loading results dictionary
        """
        try:
            _logger.info("=== WEEKLY UNIVERSE LOADER PERFORMANCE REPORT ===")
            _logger.info("Run ID: %s", results.get('run_id'))
            _logger.info("Load Date: %s", results.get('load_date'))

            # Universe loading results
            _logger.info("Universe Size: %d", results.get('universe_size', 0))
            _logger.info("Stored Count: %d", results.get('stored_count', 0))
            _logger.info("Storage Success: %s", "✅" if results.get('storage_success') else "❌")

            # Strategy breakdown
            strategy_breakdown = results.get('strategy_breakdown', {})
            if strategy_breakdown:
                _logger.info("Strategy Breakdown:")
                for strategy, count in strategy_breakdown.items():
                    _logger.info("  %s: %d tickers", strategy, count)

            # Sample of loaded universe
            universe_sample = results.get('universe_sample', [])
            if universe_sample:
                _logger.info("Universe Sample (first 10): %s", ', '.join(universe_sample))

            _logger.info("=== END PERFORMANCE REPORT ===")

        except Exception as e:
            _logger.warning("Failed to generate performance report: %s", e)

    def save_output_report(self, results: Dict[str, Any], output_dir: Optional[str]) -> None:
        """
        Save output report to file if output directory is specified.

        Args:
            results: Universe loading results dictionary
            output_dir: Directory to save output files
        """
        if not output_dir:
            return

        try:
            import json
            from pathlib import Path

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save JSON report
            run_id = results.get('run_id', 'unknown')
            json_file = output_path / f"weekly_universe_{run_id}.json"

            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)

            _logger.info("Output report saved to: %s", json_file)

        except Exception as e:
            _logger.warning("Failed to save output report: %s", e)

    def run(self) -> int:
        """
        Main execution method.

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            self.start_time = datetime.now()
            args = self.parse_arguments()

            # Set run ID
            self.run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

            _logger.info("Starting Weekly Universe Loader Script")
            _logger.info("Run ID: %s", self.run_id)
            _logger.info("Arguments: %s", vars(args))

            # Setup logging
            self.setup_logging(args.verbose)

            # Load configuration
            if not self.load_configuration(args.config):
                return 1

            # Initialize data providers
            if not self.initialize_data_providers():
                return 1

            # Test connection mode
            if args.test_connection:
                _logger.info("API connection test successful - exiting")
                return 0

            # Load universe
            universe = self.load_universe(args.max_universe)
            if not universe:
                return 1

            # Store universe
            results = self.store_universe(universe, args.dry_run)
            if not results:
                return 1

            # Generate performance report
            self.generate_performance_report(results)

            # Save output report if requested
            self.save_output_report(results, args.output_dir)

            # Calculate total runtime
            end_time = datetime.now()
            total_runtime = (end_time - self.start_time).total_seconds()

            _logger.info("Weekly universe loader script completed successfully in %.2f seconds", total_runtime)
            return 0

        except KeyboardInterrupt:
            _logger.warning("Script interrupted by user")
            return 130  # Standard exit code for SIGINT
        except Exception as e:
            _logger.error("Unexpected error in weekly screener script: %s", e, exc_info=True)
            return 1


def main() -> int:
    """
    Main entry point for the weekly universe loader script.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    runner = WeeklyUniverseLoader()
    return runner.run()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)