#!/usr/bin/env python3
"""
Data Pipeline Runner

This script runs the complete data pipeline for downloading and processing market data.

Pipeline Steps:
1. Download 1-minute data from Alpaca (step01_download_alpaca_1m.py)
2. Calculate higher timeframes (step02_calculate_timeframes.py)

Features:
- Run individual steps or complete pipeline
- Comprehensive logging and error handling
- Pipeline statistics and reporting
- Configurable parameters for each step

Usage:
    # Run complete pipeline
    python src/data/cache/pipeline/run_pipeline.py

    # Run specific steps
    python src/data/cache/pipeline/run_pipeline.py --steps 1
    python src/data/cache/pipeline/run_pipeline.py --steps 1,2

    # Run with specific parameters
    python src/data/cache/pipeline/run_pipeline.py --tickers AAPL,MSFT --timeframes 5m,15m,1h

    # Force refresh all data
    python src/data/cache/pipeline/run_pipeline.py --force-refresh
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import argparse
import subprocess
import time
from datetime import datetime
from typing import List, Dict, Any

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class DataPipelineRunner:
    """
    Runs the complete data pipeline with comprehensive logging and error handling.
    """

    def __init__(self):
        """Initialize the pipeline runner."""
        self.pipeline_dir = Path(__file__).parent
        self.stats = {
            'pipeline_start_time': None,
            'pipeline_end_time': None,
            'total_duration': 0,
            'steps_executed': [],
            'steps_failed': [],
            'step_results': {}
        }

    def run_step(self, step_num: int, step_name: str, script_path: Path,
                 args: List[str] = None) -> Dict[str, Any]:
        """
        Run a single pipeline step.

        Args:
            step_num: Step number
            step_name: Step description
            script_path: Path to step script
            args: Additional arguments for the script

        Returns:
            Dictionary with step results
        """
        step_start_time = time.time()

        _logger.info("=" * 80)
        _logger.info("PIPELINE STEP %d: %s", step_num, step_name)
        _logger.info("=" * 80)

        try:
            # Build command
            cmd = [sys.executable, str(script_path)]
            if args:
                cmd.extend(args)

            _logger.info("Executing: %s", ' '.join(cmd))

            # Run the step
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT
            )

            step_duration = time.time() - step_start_time

            # Log output
            if result.stdout:
                _logger.info("Step %d output:\n%s", step_num, result.stdout)

            if result.stderr:
                _logger.warning("Step %d stderr:\n%s", step_num, result.stderr)

            # Check result
            if result.returncode == 0:
                _logger.info("✅ Step %d completed successfully (%.1f seconds)", step_num, step_duration)
                self.stats['steps_executed'].append(step_num)
                return {
                    'success': True,
                    'duration': step_duration,
                    'return_code': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                _logger.error("❌ Step %d failed with return code %d (%.1f seconds)",
                             step_num, result.returncode, step_duration)
                self.stats['steps_failed'].append(step_num)
                return {
                    'success': False,
                    'duration': step_duration,
                    'return_code': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'error': f"Process failed with return code {result.returncode}"
                }

        except Exception as e:
            step_duration = time.time() - step_start_time
            _logger.error("❌ Step %d failed with exception: %s (%.1f seconds)", step_num, e, step_duration)
            self.stats['steps_failed'].append(step_num)
            return {
                'success': False,
                'duration': step_duration,
                'error': str(e)
            }

    def run_pipeline(self, steps: List[int], tickers: str = None, timeframes: str = None,
                    start_date: str = None, end_date: str = None, force_refresh: bool = False,
                    cache_dir: str = None) -> Dict[str, Any]:
        """
        Run the complete pipeline or specific steps.

        Args:
            steps: List of step numbers to run
            tickers: Comma-separated list of tickers
            timeframes: Comma-separated list of timeframes
            start_date: Start date for downloads
            end_date: End date for downloads
            force_refresh: Force refresh all data
            cache_dir: Cache directory path

        Returns:
            Dictionary with pipeline results
        """
        self.stats['pipeline_start_time'] = datetime.now()

        _logger.info("🚀 STARTING DATA PIPELINE")
        _logger.info("Steps to run: %s", steps)
        _logger.info("Parameters:")
        if tickers:
            _logger.info("  Tickers: %s", tickers)
        if timeframes:
            _logger.info("  Timeframes: %s", timeframes)
        if start_date:
            _logger.info("  Start date: %s", start_date)
        if end_date:
            _logger.info("  End date: %s", end_date)
        _logger.info("  Force refresh: %s", force_refresh)
        if cache_dir:
            _logger.info("  Cache directory: %s", cache_dir)

        # Step 1: Download 1-minute data from Alpaca
        if 1 in steps:
            step1_args = []
            if tickers:
                step1_args.extend(['--tickers', tickers])
            if start_date:
                step1_args.extend(['--start-date', start_date])
            if end_date:
                step1_args.extend(['--end-date', end_date])
            if force_refresh:
                step1_args.append('--force-refresh')
            if cache_dir:
                step1_args.extend(['--cache-dir', cache_dir])

            step1_result = self.run_step(
                1,
                "Download 1-Minute Data from Alpaca",
                self.pipeline_dir / "step01_download_alpaca_1m.py",
                step1_args
            )
            self.stats['step_results']['step1'] = step1_result

            # Stop if step 1 failed and we need its output for step 2
            if not step1_result['success'] and 2 in steps:
                _logger.error("Step 1 failed, cannot proceed to step 2")
                return self._finalize_pipeline_stats()

        # Step 2: Calculate higher timeframes
        if 2 in steps:
            step2_args = []
            if tickers:
                step2_args.extend(['--tickers', tickers])
            if timeframes:
                step2_args.extend(['--timeframes', timeframes])
            if force_refresh:
                step2_args.append('--force-refresh')
            if cache_dir:
                step2_args.extend(['--cache-dir', cache_dir])

            step2_result = self.run_step(
                2,
                "Calculate Higher Timeframes",
                self.pipeline_dir / "step02_calculate_timeframes.py",
                step2_args
            )
            self.stats['step_results']['step2'] = step2_result

        return self._finalize_pipeline_stats()

    def _finalize_pipeline_stats(self) -> Dict[str, Any]:
        """Finalize pipeline statistics and print summary."""
        self.stats['pipeline_end_time'] = datetime.now()
        self.stats['total_duration'] = (
            self.stats['pipeline_end_time'] - self.stats['pipeline_start_time']
        ).total_seconds()

        self.print_pipeline_summary()
        return self.stats

    def print_pipeline_summary(self):
        """Print comprehensive pipeline summary."""
        _logger.info("=" * 80)
        _logger.info("PIPELINE EXECUTION SUMMARY")
        _logger.info("=" * 80)

        # Overall statistics
        total_steps = len(self.stats['steps_executed']) + len(self.stats['steps_failed'])
        successful_steps = len(self.stats['steps_executed'])
        failed_steps = len(self.stats['steps_failed'])

        _logger.info("📊 EXECUTION STATISTICS:")
        _logger.info("   Pipeline duration: %.1f seconds", self.stats['total_duration'])
        _logger.info("   Steps executed: %d", total_steps)
        _logger.info("   ✅ Successful: %d", successful_steps)
        _logger.info("   ❌ Failed: %d", failed_steps)

        # Step details
        if self.stats['steps_executed']:
            _logger.info("\n✅ SUCCESSFUL STEPS:")
            for step_num in self.stats['steps_executed']:
                step_key = f'step{step_num}'
                if step_key in self.stats['step_results']:
                    duration = self.stats['step_results'][step_key]['duration']
                    _logger.info("   Step %d: %.1f seconds", step_num, duration)

        if self.stats['steps_failed']:
            _logger.info("\n❌ FAILED STEPS:")
            for step_num in self.stats['steps_failed']:
                step_key = f'step{step_num}'
                if step_key in self.stats['step_results']:
                    error = self.stats['step_results'][step_key].get('error', 'Unknown error')
                    _logger.info("   Step %d: %s", step_num, error)

        # Overall result
        if failed_steps == 0:
            _logger.info("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        elif successful_steps > 0:
            _logger.info("\n⚠️ PIPELINE COMPLETED WITH SOME FAILURES")
        else:
            _logger.info("\n💥 PIPELINE FAILED")

        _logger.info("=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run the complete data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python src/data/cache/pipeline/run_pipeline.py

  # Run specific steps
  python src/data/cache/pipeline/run_pipeline.py --steps 1
  python src/data/cache/pipeline/run_pipeline.py --steps 2
  python src/data/cache/pipeline/run_pipeline.py --steps 1,2

  # Run with specific parameters
  python src/data/cache/pipeline/run_pipeline.py --tickers AAPL,MSFT --timeframes 5m,15m,1h

  # Force refresh all data
  python src/data/cache/pipeline/run_pipeline.py --force-refresh

Pipeline Steps:
  1. Download 1-minute data from Alpaca
  2. Calculate higher timeframes (5m, 15m, 1h, 4h, 1d)
        """
    )

    parser.add_argument(
        "--steps",
        type=str,
        default="1,2",
        help="Comma-separated list of steps to run (default: 1,2)"
    )

    parser.add_argument(
        "--tickers",
        type=str,
        help="Comma-separated list of tickers to process (default: discover from cache)"
    )

    parser.add_argument(
        "--timeframes",
        type=str,
        help="Comma-separated list of timeframes to calculate (default: 5m,15m,1h,4h,1d)"
    )

    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for data download (YYYY-MM-DD, default: 2020-01-01)"
    )

    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for data download (YYYY-MM-DD, default: yesterday)"
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Cache directory path"
    )

    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh: re-download and recalculate all data"
    )

    args = parser.parse_args()

    try:
        # Parse steps
        steps = [int(s.strip()) for s in args.steps.split(",")]
        valid_steps = [1, 2]
        invalid_steps = [s for s in steps if s not in valid_steps]
        if invalid_steps:
            _logger.error("Invalid steps: %s. Valid steps: %s", invalid_steps, valid_steps)
            sys.exit(1)

        # Initialize and run pipeline
        runner = DataPipelineRunner()
        results = runner.run_pipeline(
            steps=steps,
            tickers=args.tickers,
            timeframes=args.timeframes,
            start_date=args.start_date,
            end_date=args.end_date,
            force_refresh=args.force_refresh,
            cache_dir=args.cache_dir
        )

        # Exit with appropriate code
        if len(results['steps_failed']) == 0:
            sys.exit(0)  # Success
        elif len(results['steps_executed']) > 0:
            sys.exit(0)  # Partial success
        else:
            sys.exit(1)  # Complete failure

    except KeyboardInterrupt:
        _logger.info("Pipeline cancelled by user")
        sys.exit(1)
    except Exception:
        _logger.exception("Pipeline fatal error:")
        sys.exit(1)


if __name__ == "__main__":
    main()