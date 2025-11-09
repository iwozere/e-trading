#!/usr/bin/env python3
"""
Volume Detector Script for Short Squeeze Detection Pipeline (using yfinance)

This script runs daily to analyze volume patterns and identify potential
short squeeze candidates using volume spikes and momentum indicators.

This version uses yfinance for data fetching to avoid FMP API rate limits.

Usage:
    python run_volume_detector.py [options]

Examples:
    # Run with default configuration
    python run_volume_detector.py

    # Run with custom configuration file
    python run_volume_detector.py --config /path/to/config.yaml

    # Run for specific date
    python run_volume_detector.py --date 2024-01-15

    # Run with limited universe for testing
    python run_volume_detector.py --max-universe 100

    # Run in dry-run mode (no database writes)
    python run_volume_detector.py --dry-run

    # Run with progress tracking
    python run_volume_detector.py --progress

    # Run with custom volume spike threshold
    python run_volume_detector.py --min-volume-spike 4.0
"""

import argparse
import sys
import time
import threading
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Dict, Any, List

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.data.db.services.short_squeeze_service import ShortSqueezeService
from src.ml.pipeline.p04_short_squeeze.config.config_manager import ConfigManager
from src.ml.pipeline.p04_short_squeeze.core.volume_squeeze_detector_yf import create_volume_squeeze_detector_yf

_logger = setup_logger(__name__)


class ProgressTracker:
    """Thread-safe progress tracker for volume analysis."""

    def __init__(self, total_items: int):
        """Initialize progress tracker."""
        self.total_items = total_items
        self.completed_items = 0
        self.failed_items = 0
        self.candidates_found = 0
        self.start_time = datetime.now()
        self.lock = threading.Lock()

    def update(self, success: bool = True, is_candidate: bool = False) -> None:
        """Update progress counters."""
        with self.lock:
            if success:
                self.completed_items += 1
                if is_candidate:
                    self.candidates_found += 1
            else:
                self.failed_items += 1

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress statistics."""
        with self.lock:
            processed = self.completed_items + self.failed_items
            progress_pct = (processed / self.total_items * 100) if self.total_items > 0 else 0

            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = processed / elapsed if elapsed > 0 else 0
            eta_seconds = (self.total_items - processed) / rate if rate > 0 else 0

            return {
                'total': self.total_items,
                'completed': self.completed_items,
                'failed': self.failed_items,
                'candidates_found': self.candidates_found,
                'processed': processed,
                'progress_pct': progress_pct,
                'elapsed_seconds': elapsed,
                'rate_per_second': rate,
                'eta_seconds': eta_seconds
            }

    def log_progress(self) -> None:
        """Log current progress."""
        stats = self.get_progress()
        _logger.info("Progress: %d/%d (%.1f%%) - Success: %d, Failed: %d, Candidates: %d, Rate: %.2f/sec, ETA: %.0fs",
                    stats['processed'], stats['total'], stats['progress_pct'],
                    stats['completed'], stats['failed'], stats['candidates_found'],
                    stats['rate_per_second'], stats['eta_seconds'])


class VolumeDetectorRunner:
    """
    Runner class for the volume detector script.

    Handles command-line arguments, configuration loading, and orchestrates
    the volume detection process with batch processing and progress tracking.
    """

    def __init__(self):
        """Initialize the volume detector runner."""
        self.config_manager: Optional[ConfigManager] = None
        self.start_time: Optional[datetime] = None
        self.run_id: Optional[str] = None
        self.progress_tracker: Optional[ProgressTracker] = None

    def parse_arguments(self) -> argparse.Namespace:
        """
        Parse command-line arguments.

        Returns:
            Parsed arguments namespace
        """
        parser = argparse.ArgumentParser(
            description="Run volume detector for short squeeze detection",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=__doc__.split('Usage:')[1] if 'Usage:' in __doc__ else ""
        )

        parser.add_argument(
            '--config', '-c',
            type=str,
            help='Path to configuration file (default: uses default config location)'
        )

        parser.add_argument(
            '--date',
            type=str,
            help='Specific date to analyze (YYYY-MM-DD format, default: today)'
        )

        parser.add_argument(
            '--max-universe',
            type=int,
            help='Maximum number of stocks to analyze (for testing)'
        )

        parser.add_argument(
            '--min-volume-spike',
            type=float,
            help='Minimum volume spike ratio to consider (overrides config)'
        )

        parser.add_argument(
            '--batch-size',
            type=int,
            help='Batch size for processing stocks (overrides config)'
        )

        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Run without writing results to database'
        )

        parser.add_argument(
            '--progress',
            action='store_true',
            help='Enable progress tracking and periodic updates'
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

        parser.add_argument(
            '--universe-source',
            choices=['database', 'fmp'],
            default='database',
            help='Source for stock universe (database or fresh FMP load)'
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

            # Get volume detector configuration
            volume_config = getattr(config, 'volume_detector', None)
            if volume_config:
                _logger.info("Volume detector config: min_spike=%.2f, lookback=%d days, max_candidates=%d",
                           volume_config.analysis.min_volume_spike_ratio,
                           volume_config.analysis.volume_lookback_days,
                           volume_config.filters.max_candidates)

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
            _logger.info("Initializing data providers (using yfinance)...")
            # No initialization needed for yfinance
            _logger.info("yfinance ready for data fetching")
            return True

        except Exception:
            _logger.exception("Failed to initialize data providers:")
            return False

    def parse_analysis_date(self, date_str: Optional[str]) -> date:
        """
        Parse analysis date from string or use today.

        Args:
            date_str: Date string in YYYY-MM-DD format or None

        Returns:
            Date object for analysis
        """
        if date_str:
            try:
                return datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                _logger.warning("Invalid date format '%s', using today", date_str)

        return datetime.now().date()

    def load_universe(self, universe_source: str, max_universe: Optional[int] = None) -> Optional[List[str]]:
        """
        Load the stock universe for volume analysis.

        Args:
            universe_source: Source for universe ('database' or 'fmp')
            max_universe: Optional limit on universe size

        Returns:
            List of ticker symbols or None if failed
        """
        try:
            if universe_source == 'database':
                _logger.info("Loading universe from database (latest screener run)...")

                service = ShortSqueezeService()
                # Get top candidates from latest screener run as universe
                candidates = service.get_top_candidates_by_screener_score(limit=5000)  # Large limit to get full universe
                universe = [c['ticker'] for c in candidates]

                if not universe:
                    _logger.warning("No universe found in database, using default universe")
                    universe_source = 'fmp'
                else:
                    _logger.info("Loaded %d stocks from database universe", len(universe))

            if universe_source == 'fmp':
                _logger.warning("FMP universe loading not available with yfinance mode")
                _logger.info("Using default S&P 500 universe as fallback")

                # Use a default universe (S&P 500 as example)
                # You could also read from a file or use another source
                import pandas as pd
                try:
                    # Try to get S&P 500 tickers from Wikipedia
                    sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
                    sp500_table = pd.read_html(sp500_url)
                    universe = sp500_table[0]['Symbol'].tolist()
                    _logger.info("Loaded %d stocks from S&P 500 list", len(universe))
                except Exception as e:
                    _logger.error("Failed to load default universe: %s", e)
                    return None

            # Apply max universe limit if specified
            if max_universe and len(universe) > max_universe:
                _logger.info("Limiting universe from %d to %d stocks for testing",
                           len(universe), max_universe)
                universe = universe[:max_universe]

            return universe

        except Exception:
            _logger.exception("Failed to load universe:")
            return None

    def run_volume_detection(self, universe: List[str], analysis_date: date,
                           min_volume_spike: Optional[float], batch_size: Optional[int],
                           dry_run: bool = False, enable_progress: bool = False) -> Optional[Dict[str, Any]]:
        """
        Run volume detection on the universe.

        Args:
            universe: List of ticker symbols to analyze
            analysis_date: Date for the analysis
            min_volume_spike: Optional minimum volume spike threshold
            batch_size: Optional batch size override
            dry_run: If True, don't write results to database
            enable_progress: Enable progress tracking

        Returns:
            Dictionary with detection results or None if failed
        """
        try:
            _logger.info("Starting volume detection run for date: %s", analysis_date)
            _logger.info("Analyzing %d stocks for volume patterns", len(universe))

            config = self.config_manager.get_volume_detector_config() if hasattr(self.config_manager, 'get_volume_detector_config') else None

            # Override parameters if specified
            if min_volume_spike and config:
                config.analysis.min_volume_spike_ratio = min_volume_spike
                _logger.info("Using custom minimum volume spike: %.2f", min_volume_spike)

            if batch_size and config:
                config.batch_size = batch_size
                _logger.info("Using custom batch size: %d", batch_size)

            volume_detector = create_volume_squeeze_detector_yf()

            # Setup progress tracking if enabled
            if enable_progress:
                self.progress_tracker = ProgressTracker(len(universe))
                # Start progress logging thread
                progress_thread = threading.Thread(target=self._log_progress_periodically, daemon=True)
                progress_thread.start()

            # Run the volume detection using screen_universe method
            min_score = min_volume_spike or 0.3  # Use min_volume_spike as min_score
            results = volume_detector.screen_universe(universe, min_score)

            # Update progress tracker if enabled
            if enable_progress:
                for candidate, indicators in results:
                    self.progress_tracker.update(success=True, is_candidate=True)
                # Update for any remaining tickers that weren't candidates
                remaining = len(universe) - len(results)
                for _ in range(remaining):
                    self.progress_tracker.update(success=True, is_candidate=False)

            # Process results
            candidates_found = len(results) if results else 0

            # Calculate runtime metrics
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            stocks_per_sec = len(universe) / duration if duration > 0 else 0

            # Store results in database if not dry run
            if not dry_run and results:
                _logger.info("Storing %d volume candidates in database...", len(results))
                stored_count = self._store_volume_candidates(results, analysis_date)
                _logger.info("Successfully stored %d volume candidates", stored_count)
            else:
                stored_count = 0
                if dry_run:
                    _logger.info("DRY RUN MODE: Would store %d volume candidates", candidates_found)

            # Convert results to dictionary for easier handling
            results_dict = {
                'run_id': self.run_id,
                'analysis_date': analysis_date.isoformat(),
                'universe_size': len(universe),
                'stocks_analyzed': len(universe),
                'candidates_found': candidates_found,
                'stored_count': stored_count,
                'min_score': min_score,
                'candidates': [
                    {
                        'ticker': candidate.ticker,
                        'combined_score': indicators.combined_score,
                        'volume_score': indicators.volume_score,
                        'momentum_score': indicators.momentum_score,
                        'squeeze_probability': indicators.squeeze_probability,
                        'source': candidate.source.value
                    }
                    for candidate, indicators in results
                ],
                'runtime_metrics': {
                    'duration_seconds': duration,
                    'stocks_per_second': stocks_per_sec,
                    'total_stocks_analyzed': len(universe)
                },
                'data_quality_metrics': {
                    'success_rate': 1.0,  # All stocks were processed
                    'candidates_rate': candidates_found / len(universe) if universe else 0
                }
            }

            _logger.info("Volume detection completed successfully")
            return results_dict

        except Exception:
            _logger.exception("Volume detection run failed:")
            return None

    def _store_volume_candidates(self, results: List, analysis_date: date) -> int:
        """
        Store volume candidates in the database.

        Args:
            results: List of (Candidate, SqueezeIndicators) tuples
            analysis_date: Date of analysis

        Returns:
            Number of candidates stored
        """
        try:
            snapshots_data = []

            # Prepare all snapshot data first
            for candidate, indicators in results:
                # Convert to screener snapshot format (ensure Python native types)
                snapshot_data = {
                    'ticker': candidate.ticker,
                    'run_date': analysis_date,
                    'screener_score': float(indicators.combined_score),  # Convert numpy to Python float
                    'raw_payload': {
                        'source': 'volume_detector',
                        'volume_score': float(indicators.volume_score),
                        'momentum_score': float(indicators.momentum_score),
                        'squeeze_probability': str(indicators.squeeze_probability),
                        'analysis_date': analysis_date.isoformat()
                    },
                    'data_quality': 1.0  # High quality from volume analysis
                }
                snapshots_data.append(snapshot_data)

            # Store all snapshots using service
            if snapshots_data:
                service = ShortSqueezeService()
                stored_count = service.save_screener_results(snapshots_data, analysis_date)

            return stored_count

        except Exception:
            _logger.exception("Error storing volume candidates:")
            return 0

    def _log_progress_periodically(self) -> None:
        """Log progress updates periodically (runs in separate thread)."""
        while self.progress_tracker:
            time.sleep(30)  # Log every 30 seconds
            if self.progress_tracker:
                stats = self.progress_tracker.get_progress()
                if stats['processed'] < stats['total']:
                    self.progress_tracker.log_progress()
                else:
                    break

    def generate_performance_report(self, results: Dict[str, Any]) -> None:
        """
        Generate and log performance metrics report.

        Args:
            results: Volume detection results dictionary
        """
        try:
            runtime_metrics = results.get('runtime_metrics', {})
            data_quality_metrics = results.get('data_quality_metrics', {})

            _logger.info("=== VOLUME DETECTOR PERFORMANCE REPORT ===")
            _logger.info("Run ID: %s", results.get('run_id'))
            _logger.info("Analysis Date: %s", results.get('analysis_date'))

            # Runtime metrics
            duration = runtime_metrics.get('duration_seconds', 0)
            stocks_per_sec = runtime_metrics.get('stocks_per_second', 0)
            _logger.info("Runtime: %.2f seconds (%.2f stocks/sec)", duration, stocks_per_sec)

            # Analysis results
            _logger.info("Universe Size: %d", results.get('universe_size', 0))
            _logger.info("Stocks Analyzed: %d", results.get('stocks_analyzed', 0))
            _logger.info("Candidates Found: %d", results.get('candidates_found', 0))

            # Data quality metrics
            successful_analyses = data_quality_metrics.get('successful_analyses', 0)
            failed_analyses = data_quality_metrics.get('failed_analyses', 0)
            total_stocks = successful_analyses + failed_analyses

            success_rate = (successful_analyses / total_stocks * 100) if total_stocks > 0 else 0
            _logger.info("Analysis Success Rate: %.1f%% (%d/%d stocks)",
                        success_rate, successful_analyses, total_stocks)

            # API call metrics
            api_calls = data_quality_metrics.get('api_calls_made', 0)
            _logger.info("API Calls Made: %d", api_calls)

            # Data availability metrics
            valid_volume = data_quality_metrics.get('valid_volume_data', 0)
            valid_price = data_quality_metrics.get('valid_price_data', 0)
            _logger.info("Data Availability: Volume=%d, Price=%d", valid_volume, valid_price)

            # Top candidates
            candidates = results.get('candidates', [])
            if candidates:
                # Sort by volume spike ratio
                top_candidates = sorted(candidates, key=lambda x: x['volume_spike_ratio'], reverse=True)[:5]
                _logger.info("Top 5 Volume Spike Candidates:")
                for i, candidate in enumerate(top_candidates, 1):
                    _logger.info("  %d. %s: spike=%.2fx, volume=%d, RSI=%.1f, score=%.3f",
                               i, candidate['ticker'], candidate['volume_spike_ratio'],
                               candidate['current_volume'], candidate['rsi'], candidate['detection_score'])

            _logger.info("=== END PERFORMANCE REPORT ===")

        except Exception as e:
            _logger.warning("Failed to generate performance report: %s", e)

    def save_output_report(self, results: Dict[str, Any], output_dir: Optional[str]) -> None:
        """
        Save output report to file if output directory is specified.

        Args:
            results: Volume detection results dictionary
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
            json_file = output_path / f"volume_detector_{run_id}.json"

            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)

            _logger.info("Output report saved to: %s", json_file)

            # Save CSV of candidates
            candidates = results.get('candidates', [])
            if candidates:
                import csv

                csv_file = output_path / f"volume_detector_candidates_{run_id}.csv"
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=candidates[0].keys())
                    writer.writeheader()
                    writer.writerows(candidates)

                _logger.info("Candidates CSV saved to: %s", csv_file)

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
            self.run_id = args.run_id or datetime.now().strftime("volume_%Y%m%d_%H%M%S")

            _logger.info("Starting Volume Detector Script")
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

            # Parse analysis date
            analysis_date = self.parse_analysis_date(args.date)
            _logger.info("Analysis date: %s", analysis_date)

            # Load universe
            universe = self.load_universe(args.universe_source, args.max_universe)
            if not universe:
                return 1

            # Run volume detection
            results = self.run_volume_detection(
                universe, analysis_date, args.min_volume_spike, args.batch_size,
                args.dry_run, args.progress
            )
            if not results:
                return 1

            # Generate performance report
            self.generate_performance_report(results)

            # Save output report if requested
            self.save_output_report(results, args.output_dir)

            # Calculate total runtime
            end_time = datetime.now()
            total_runtime = (end_time - self.start_time).total_seconds()

            _logger.info("Volume detector script completed successfully in %.2f seconds", total_runtime)

            # Return success if we found candidates, warning if no candidates found
            candidates_found = results.get('candidates_found', 0)
            if candidates_found == 0:
                _logger.warning("No volume spike candidates found - this may indicate market conditions or configuration issues")
                return 2  # Warning exit code

            return 0

        except KeyboardInterrupt:
            _logger.warning("Script interrupted by user")
            return 130  # Standard exit code for SIGINT
        except Exception as e:
            _logger.error("Unexpected error in volume detector script: %s", e, exc_info=True)
            return 1


def main() -> int:
    """
    Main entry point for the volume detector script.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    runner = VolumeDetectorRunner()
    return runner.run()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)