#!/usr/bin/env python3
"""
Daily Deep Scan Script for Short Squeeze Detection Pipeline

This script runs the daily deep scan to analyze identified candidates with
real-time transient metrics like volume spikes, sentiment, and options data.

Usage:
    python run_daily_deep_scan.py [options]

Examples:
    # Run with default configuration
    python run_daily_deep_scan.py

    # Run with custom configuration file
    python run_daily_deep_scan.py --config /path/to/config.yaml

    # Run with specific candidates only
    python run_daily_deep_scan.py --tickers AAPL,TSLA,GME

    # Run in dry-run mode (no database writes)
    python run_daily_deep_scan.py --dry-run

    # Run with progress tracking
    python run_daily_deep_scan.py --progress

    # Run with custom batch size
    python run_daily_deep_scan.py --batch-size 5
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Dict, Any, List
import threading

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.data.downloader.fmp_data_downloader import FMPDataDownloader
from src.data.downloader.finnhub_data_downloader import FinnhubDataDownloader
from src.ml.pipeline.p04_short_squeeze.config.config_manager import ConfigManager
from src.ml.pipeline.p04_short_squeeze.core.daily_deep_scan import create_daily_deep_scan
from src.ml.pipeline.p04_short_squeeze.core.models import Candidate, StructuralMetrics, CandidateSource

_logger = setup_logger(__name__)


class ProgressTracker:
    """Thread-safe progress tracker for batch processing."""

    def __init__(self, total_items: int):
        """Initialize progress tracker."""
        self.total_items = total_items
        self.completed_items = 0
        self.failed_items = 0
        self.start_time = datetime.now()
        self.lock = threading.Lock()

    def update(self, success: bool = True) -> None:
        """Update progress counters."""
        with self.lock:
            if success:
                self.completed_items += 1
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
                'processed': processed,
                'progress_pct': progress_pct,
                'elapsed_seconds': elapsed,
                'rate_per_second': rate,
                'eta_seconds': eta_seconds
            }

    def log_progress(self) -> None:
        """Log current progress."""
        stats = self.get_progress()
        _logger.info("Progress: %d/%d (%.1f%%) - Success: %d, Failed: %d, Rate: %.2f/sec, ETA: %.0fs",
                    stats['processed'], stats['total'], stats['progress_pct'],
                    stats['completed'], stats['failed'], stats['rate_per_second'], stats['eta_seconds'])


class DailyDeepScanRunner:
    """
    Runner class for the daily deep scan script.

    Handles command-line arguments, configuration loading, and orchestrates
    the daily deep scan process with batch processing and progress tracking.
    """

    def __init__(self):
        """Initialize the daily deep scan runner."""
        self.config_manager: Optional[ConfigManager] = None
        self.fmp_downloader: Optional[FMPDataDownloader] = None
        self.finnhub_downloader: Optional[FinnhubDataDownloader] = None
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
            description="Run daily deep scan for short squeeze detection",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=__doc__.split('Usage:')[1] if 'Usage:' in __doc__ else ""
        )

        parser.add_argument(
            '--config', '-c',
            type=str,
            help='Path to configuration file (default: uses default config location)'
        )

        parser.add_argument(
            '--tickers',
            type=str,
            help='Comma-separated list of specific tickers to scan (overrides database candidates)'
        )

        parser.add_argument(
            '--batch-size',
            type=int,
            help='Batch size for processing candidates (overrides config)'
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
            '--scan-date',
            type=str,
            help='Specific date to scan (YYYY-MM-DD format, default: today)'
        )

        parser.add_argument(
            '--max-candidates',
            type=int,
            help='Maximum number of candidates to process (for testing)'
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
            _logger.info("Deep scan config: batch_size=%d, api_delay=%.2fs",
                        config.deep_scan.batch_size, config.deep_scan.api_delay_seconds)

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

            # Test FMP connection
            if not self.fmp_downloader.test_connection():
                _logger.error("FMP API connection test failed")
                return False

            _logger.info("FMP API connection successful")

            # Initialize Finnhub downloader (optional)
            try:
                from config.donotshare.donotshare import FINNHUB_KEY
                if FINNHUB_KEY:
                    self.finnhub_downloader = FinnhubDataDownloader(FINNHUB_KEY)
                    _logger.info("Finnhub API initialized with key from config")
                else:
                    self.finnhub_downloader = None
                    _logger.warning("Finnhub API key not found in config - sentiment and options data will be unavailable")
            except ImportError:
                self.finnhub_downloader = None
                _logger.warning("Could not import Finnhub API key from config - sentiment and options data will be unavailable")
            except Exception as e:
                self.finnhub_downloader = None
                _logger.warning("Failed to initialize Finnhub downloader: %s", e)
            return True

        except Exception:
            _logger.exception("Failed to initialize data providers:")
            return False

    def parse_scan_date(self, date_str: Optional[str]) -> date:
        """
        Parse scan date from string or use today.

        Args:
            date_str: Date string in YYYY-MM-DD format or None

        Returns:
            Date object for scanning
        """
        if date_str:
            try:
                return datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                _logger.warning("Invalid date format '%s', using today", date_str)

        return datetime.now().date()

    def create_manual_candidates(self, ticker_list: List[str]) -> List[Candidate]:
        """
        Create candidate objects from manual ticker list.

        Args:
            ticker_list: List of ticker symbols

        Returns:
            List of Candidate objects with placeholder data
        """
        candidates = []

        for ticker in ticker_list:
            ticker = ticker.strip().upper()
            if not ticker:
                continue

            # Create placeholder structural metrics for manual candidates
            structural_metrics = StructuralMetrics(
                short_interest_pct=0.0,  # Will be updated during scan if available
                days_to_cover=0.0,
                float_shares=1,
                avg_volume_14d=1,
                market_cap=1
            )

            candidate = Candidate(
                ticker=ticker,
                screener_score=0.0,  # Manual candidates start with 0 score
                structural_metrics=structural_metrics,
                last_updated=datetime.now(),
                source=CandidateSource.ADHOC
            )

            candidates.append(candidate)

        _logger.info("Created %d manual candidates from ticker list", len(candidates))
        return candidates

    def load_candidates(self, manual_tickers: Optional[str], max_candidates: Optional[int]) -> Optional[List[Candidate]]:
        """
        Load candidates for deep scan.

        Args:
            manual_tickers: Comma-separated ticker list or None
            max_candidates: Maximum number of candidates to load

        Returns:
            List of candidates or None if failed
        """
        try:
            if manual_tickers:
                # Use manual ticker list
                ticker_list = [t.strip().upper() for t in manual_tickers.split(',') if t.strip()]
                if not ticker_list:
                    _logger.error("No valid tickers found in manual list")
                    return None

                candidates = self.create_manual_candidates(ticker_list)
            else:
                # Load from database (this will be handled by the deep scan module)
                _logger.info("Loading candidates from database...")
                candidates = None  # Let deep scan module load from database

            # Apply max candidates limit if specified
            if candidates and max_candidates and len(candidates) > max_candidates:
                _logger.info("Limiting candidates from %d to %d for testing",
                           len(candidates), max_candidates)
                candidates = candidates[:max_candidates]

            if candidates:
                _logger.info("Loaded %d candidates for deep scan", len(candidates))
            else:
                _logger.info("Will load candidates from database during deep scan")

            return candidates

        except Exception:
            _logger.exception("Failed to load candidates:")
            return None

    def run_deep_scan(self, candidates: Optional[List[Candidate]], scan_date: date,
                     batch_size: Optional[int], dry_run: bool = False,
                     enable_progress: bool = False) -> Optional[Dict[str, Any]]:
        """
        Run the daily deep scan on candidates.

        Args:
            candidates: List of candidates to scan or None to load from database
            scan_date: Date for the scan
            batch_size: Optional batch size override
            dry_run: If True, don't write results to database
            enable_progress: Enable progress tracking

        Returns:
            Dictionary with scan results or None if failed
        """
        try:
            _logger.info("Starting daily deep scan run for date: %s", scan_date)

            config = self.config_manager.get_deep_scan_config()

            # Override batch size if specified
            if batch_size:
                config.batch_size = batch_size
                _logger.info("Using custom batch size: %d", batch_size)

            deep_scan = create_daily_deep_scan(
                self.fmp_downloader,
                self.finnhub_downloader,
                config
            )

            # Override database writes for dry run
            if dry_run:
                _logger.info("DRY RUN MODE: Results will not be saved to database")
                original_store = deep_scan._store_results
                deep_scan._store_results = lambda *args, **kwargs: _logger.info("DRY RUN: Skipping database write")

            # Setup progress tracking if enabled
            if enable_progress and candidates:
                self.progress_tracker = ProgressTracker(len(candidates))

                # Monkey patch the scan method to update progress
                original_scan = deep_scan._scan_candidate

                def progress_scan_candidate(candidate, metrics):
                    try:
                        result = original_scan(candidate, metrics)
                        self.progress_tracker.update(success=result is not None)
                        return result
                    except Exception:
                        self.progress_tracker.update(success=False)
                        raise

                deep_scan._scan_candidate = progress_scan_candidate

                # Start progress logging thread
                progress_thread = threading.Thread(target=self._log_progress_periodically, daemon=True)
                progress_thread.start()

            # Run the deep scan
            results = deep_scan.run_deep_scan(candidates)


            # --- SENTIMENT INTEGRATION BLOCK START ---
            # try to import sync wrapper of the async collector (adjust path if needed)
            try:
                # if your collect_sentiment_async lives at src/common/sentiments/collect_sentiment_async.py
                from src.common.sentiments.collect_sentiment_async import collect_sentiment_batch_sync
                sentiment_available = True
            except Exception as _e:
                _logger.warning("Sentiment module not available: %s. Skipping sentiment collection.", _e)
                sentiment_available = False

            # Only run sentiment collection if we have a candidate list and the module exists
            if sentiment_available and candidates:
                try:
                    # Extract tickers for this batch
                    batch_tickers = [c.ticker for c in candidates]
                    _logger.info("Collecting sentiment for %d tickers: %s", len(batch_tickers), ", ".join(batch_tickers[:10]))
                    # You can pass config options here (or None to use defaults)
                    try:
                        sentiment_cfg = config.sentiment._asdict() if hasattr(config, "sentiment") else None
                    except Exception:
                        sentiment_cfg = None

                    # Run sync wrapper (it will internally run async loop and return dict)
                    sentiment_map = collect_sentiment_batch_sync(batch_tickers, lookback_hours=24, config=sentiment_cfg, history_lookup=None)

                    # Attach sentiment to candidate transient_metrics (or fallback to candidate._sentiment)
                    for c in candidates:
                        feats = sentiment_map.get(c.ticker)
                        if not feats:
                            # mark missing or neutral
                            try:
                                setattr(c.transient_metrics, "sentiment_24h", 0.5)
                                setattr(c.transient_metrics, "sentiment_score_raw", 0.0)
                                setattr(c.transient_metrics, "sentiment_payload", None)
                            except Exception:
                                setattr(c, "_sentiment", None)
                            continue

                        # Map dataclass-like object to expected fields
                        try:
                            # if feats is dataclass, attr names as in SentimentFeatures dataclass
                            sent_norm = getattr(feats, "sentiment_normalized", None) or feats.get("sentiment_normalized") if isinstance(feats, dict) else None
                            sent_raw = getattr(feats, "raw_payload", None) or (feats.get("raw_payload") if isinstance(feats, dict) else None)
                            sent_score = getattr(feats, "sentiment_score_24h", None) or (feats.get("sentiment_score_24h") if isinstance(feats, dict) else None)

                            if hasattr(c, "transient_metrics") and c.transient_metrics is not None:
                                setattr(c.transient_metrics, "sentiment_24h", float(sent_norm) if sent_norm is not None else 0.5)
                                setattr(c.transient_metrics, "sentiment_score_raw", float(sent_score) if sent_score is not None else 0.0)
                                setattr(c.transient_metrics, "sentiment_payload", sent_raw)
                            else:
                                # fallback: attach to candidate
                                setattr(c, "_sentiment", feats)
                        except Exception as e:
                            _logger.debug("Failed attaching sentiment to candidate %s: %s", c.ticker, e)

                    _logger.info("Sentiment collection done; attached to transient_metrics for %d candidates", len(candidates))

                except Exception:
                    _logger.exception("Sentiment collection failed, continuing without sentiments:")
            # --- SENTIMENT INTEGRATION BLOCK END ---


            # Restore original methods if they were patched
            if dry_run:
                deep_scan._store_results = original_store

            # Convert results to dictionary for easier handling
            results_dict = {
                'run_id': results.run_id,
                'run_date': results.run_date.isoformat(),
                'candidates_processed': results.candidates_processed,
                'scored_candidates_count': len(results.scored_candidates),
                'scored_candidates': [
                    {
                        'ticker': sc.candidate.ticker,
                        'squeeze_score': sc.squeeze_score,
                        'screener_score': sc.candidate.screener_score,
                        'volume_spike': sc.transient_metrics.volume_spike,
                        'sentiment_24h': sc.transient_metrics.sentiment_24h,
                        'call_put_ratio': sc.transient_metrics.call_put_ratio,
                        'borrow_fee_pct': sc.transient_metrics.borrow_fee_pct,
                        'source': sc.candidate.source.value,
                        'alert_level': sc.alert_level
                    }
                    for sc in results.scored_candidates
                ],
                'data_quality_metrics': results.data_quality_metrics,
                'runtime_metrics': results.runtime_metrics
            }

            _logger.info("Daily deep scan completed successfully")
            return results_dict

        except Exception:
            _logger.exception("Daily deep scan run failed:")
            return None

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
            results: Deep scan results dictionary
        """
        try:
            runtime_metrics = results.get('runtime_metrics', {})
            data_quality_metrics = results.get('data_quality_metrics', {})

            _logger.info("=== DAILY DEEP SCAN PERFORMANCE REPORT ===")
            _logger.info("Run ID: %s", results.get('run_id'))
            _logger.info("Scan Date: %s", results.get('run_date'))

            # Runtime metrics
            duration = runtime_metrics.get('duration_seconds', 0)
            candidates_per_sec = runtime_metrics.get('candidates_per_second', 0)
            _logger.info("Runtime: %.2f seconds (%.2f candidates/sec)", duration, candidates_per_sec)

            # Scanning results
            _logger.info("Candidates Processed: %d", results.get('candidates_processed', 0))
            _logger.info("Scored Candidates: %d", results.get('scored_candidates_count', 0))

            # Data quality metrics
            successful_scans = data_quality_metrics.get('successful_scans', 0)
            failed_scans = data_quality_metrics.get('failed_scans', 0)
            total_candidates = successful_scans + failed_scans

            success_rate = (successful_scans / total_candidates * 100) if total_candidates > 0 else 0
            _logger.info("Scan Success Rate: %.1f%% (%d/%d candidates)",
                        success_rate, successful_scans, total_candidates)

            # API call metrics
            fmp_calls = data_quality_metrics.get('api_calls_fmp', 0)
            finnhub_calls = data_quality_metrics.get('api_calls_finnhub', 0)
            _logger.info("API Calls: FMP=%d, Finnhub=%d", fmp_calls, finnhub_calls)

            # Data availability metrics
            valid_volume = data_quality_metrics.get('valid_volume_data', 0)
            valid_sentiment = data_quality_metrics.get('valid_sentiment_data', 0)
            valid_options = data_quality_metrics.get('valid_options_data', 0)
            valid_borrow = data_quality_metrics.get('valid_borrow_rates', 0)
            valid_finra = data_quality_metrics.get('finra_data_available', 0)

            _logger.info("Data Availability: Volume=%d, Sentiment=%d, Options=%d, Borrow=%d, FINRA=%d",
                        valid_volume, valid_sentiment, valid_options, valid_borrow, valid_finra)

            # Top scored candidates
            scored_candidates = results.get('scored_candidates', [])
            if scored_candidates:
                # Sort by squeeze score
                top_candidates = sorted(scored_candidates, key=lambda x: x['squeeze_score'], reverse=True)[:5]
                _logger.info("Top 5 Scored Candidates:")
                for i, candidate in enumerate(top_candidates, 1):
                    _logger.info("  %d. %s: squeeze=%.3f, volume_spike=%.2f, sentiment=%.2f",
                               i, candidate['ticker'], candidate['squeeze_score'],
                               candidate['volume_spike'], candidate['sentiment_24h'])

            _logger.info("=== END PERFORMANCE REPORT ===")

        except Exception as e:
            _logger.warning("Failed to generate performance report: %s", e)

    def save_output_report(self, results: Dict[str, Any], output_dir: Optional[str]) -> None:
        """
        Save output report to file if output directory is specified.

        Args:
            results: Deep scan results dictionary
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
            json_file = output_path / f"daily_deep_scan_{run_id}.json"

            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)

            _logger.info("Output report saved to: %s", json_file)

            # Save CSV of scored candidates
            scored_candidates = results.get('scored_candidates', [])
            if scored_candidates:
                import csv

                csv_file = output_path / f"daily_deep_scan_candidates_{run_id}.csv"
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=scored_candidates[0].keys())
                    writer.writeheader()
                    writer.writerows(scored_candidates)

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
            self.run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

            _logger.info("Starting Daily Deep Scan Script")
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

            # Parse scan date
            scan_date = self.parse_scan_date(args.scan_date)
            _logger.info("Scan date: %s", scan_date)

            # Load candidates
            candidates = self.load_candidates(args.tickers, args.max_candidates)
            if candidates is not None and len(candidates) == 0:
                _logger.warning("No candidates to process")
                return 0

            # Run deep scan
            results = self.run_deep_scan(
                candidates, scan_date, args.batch_size,
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

            _logger.info("Daily deep scan script completed successfully in %.2f seconds", total_runtime)
            return 0

        except KeyboardInterrupt:
            _logger.warning("Script interrupted by user")
            return 130  # Standard exit code for SIGINT
        except Exception as e:
            _logger.error("Unexpected error in daily deep scan script: %s", e, exc_info=True)
            return 1


def main() -> int:
    """
    Main entry point for the daily deep scan script.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    runner = DailyDeepScanRunner()
    return runner.run()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)