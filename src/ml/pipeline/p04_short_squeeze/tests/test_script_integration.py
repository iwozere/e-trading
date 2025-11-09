"""
Integration tests for Short Squeeze Detection Pipeline scripts.

Tests the executable scripts for weekly screener, daily deep scan, and ad-hoc
candidate management with focus on command-line argument parsing, error handling,
and end-to-end execution with sample data.
"""

import unittest
import subprocess
import sys
import tempfile
import csv
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

# Simple logging setup for tests
import logging
_logger = logging.getLogger(__name__)


class ScriptIntegrationTestBase(unittest.TestCase):
    """Base class for script integration tests."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.scripts_dir = PROJECT_ROOT / "src" / "ml" / "pipeline" / "p04_short_squeeze" / "scripts"

    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def run_script(self, script_name: str, args: list = None, expect_success: bool = True) -> subprocess.CompletedProcess:
        """
        Run a script with given arguments.

        Args:
            script_name: Name of the script file
            args: List of command-line arguments
            expect_success: Whether to expect successful execution

        Returns:
            CompletedProcess result
        """
        script_path = self.scripts_dir / script_name
        cmd = [sys.executable, str(script_path)]
        if args:
            cmd.extend(args)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout for tests
        )

        if expect_success and result.returncode != 0:
            self.fail(f"Script {script_name} failed with return code {result.returncode}. "
                     f"STDOUT: {result.stdout}, STDERR: {result.stderr}")

        return result

    def create_test_config(self) -> str:
        """Create a test configuration file."""
        config_content = """run_id: "test_run_001"

screener:
  universe:
    market_cap_min: 100000000
    avg_volume_min: 100000
    exchanges: ["NASDAQ", "NYSE"]
    max_universe_size: 100

  filters:
    si_percent_min: 0.15
    days_to_cover_min: 2.0
    top_k_candidates: 10

  batch_size: 5
  api_delay_seconds: 0.1

deep_scan:
  batch_size: 3
  api_delay_seconds: 0.1
  volume_lookback_days: 14
  sentiment_hours: 24

adhoc:
  default_ttl_days: 7
  max_active_candidates: 50

scoring:
  weights:
    structural: 0.4
    transient: 0.6

  alert_thresholds:
    high: 0.8
    medium: 0.6
    low: 0.4

database:
  connection_string: "sqlite:///:memory:"

logging:
  level: "INFO"
"""
        config_path = os.path.join(self.test_dir, "test_config.yaml")
        with open(config_path, 'w') as f:
            f.write(config_content)
        return config_path


class TestWeeklyScreenerScript(ScriptIntegrationTestBase):
    """Test cases for the weekly screener script."""

    def test_help_argument(self):
        """Test that help argument works."""
        result = self.run_script("run_weekly_screener.py", ["--help"], expect_success=False)
        self.assertEqual(result.returncode, 0)
        self.assertIn("weekly screener", result.stdout.lower())

    def test_invalid_arguments(self):
        """Test handling of invalid arguments."""
        result = self.run_script("run_weekly_screener.py", ["--invalid-arg"], expect_success=False)
        self.assertNotEqual(result.returncode, 0)

    @patch('src.ml.pipeline.p04_short_squeeze.scripts.run_weekly_screener.FMPDataDownloader')
    @patch('src.ml.pipeline.p04_short_squeeze.scripts.run_weekly_screener.ConfigManager')
    def test_test_connection_mode(self, mock_config_manager, mock_fmp_downloader):
        """Test the test connection mode."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config_manager.return_value.load_config.return_value = mock_config

        mock_downloader = MagicMock()
        mock_downloader.test_connection.return_value = True
        mock_fmp_downloader.return_value = mock_downloader

        config_path = self.create_test_config()
        result = self.run_script("run_weekly_screener.py", ["--config", config_path, "--test-connection"])

        self.assertEqual(result.returncode, 0)
        self.assertIn("connection test successful", result.stdout.lower())

    @patch('src.ml.pipeline.p04_short_squeeze.scripts.run_weekly_screener.FMPDataDownloader')
    @patch('src.ml.pipeline.p04_short_squeeze.scripts.run_weekly_screener.ConfigManager')
    @patch('src.ml.pipeline.p04_short_squeeze.scripts.run_weekly_screener.create_universe_loader')
    @patch('src.ml.pipeline.p04_short_squeeze.scripts.run_weekly_screener.create_weekly_screener')
    def test_dry_run_mode(self, mock_create_screener, mock_create_universe, mock_config_manager, mock_fmp_downloader):
        """Test dry run mode execution."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config_manager.return_value.load_config.return_value = mock_config
        mock_config_manager.return_value.get_screener_config.return_value = mock_config

        mock_downloader = MagicMock()
        mock_downloader.test_connection.return_value = True
        mock_fmp_downloader.return_value = mock_downloader

        mock_universe_loader = MagicMock()
        mock_universe_loader.load_universe.return_value = ["AAPL", "TSLA", "GME"]
        mock_create_universe.return_value = mock_universe_loader

        mock_screener = MagicMock()
        mock_results = MagicMock()
        mock_results.run_id = "test_run"
        mock_results.run_date = datetime.now().date()
        mock_results.total_universe = 3
        mock_results.candidates_found = 2
        mock_results.top_candidates = []
        mock_results.data_quality_metrics = {}
        mock_results.runtime_metrics = {}
        mock_screener.run_screener.return_value = mock_results
        mock_create_screener.return_value = mock_screener

        config_path = self.create_test_config()
        result = self.run_script("run_weekly_screener.py", [
            "--config", config_path,
            "--dry-run",
            "--max-universe", "3"
        ])

        self.assertEqual(result.returncode, 0)
        self.assertIn("dry run mode", result.stdout.lower())

    def test_verbose_logging(self):
        """Test verbose logging option."""
        config_path = self.create_test_config()

        with patch('src.ml.pipeline.p04_short_squeeze.scripts.run_weekly_screener.FMPDataDownloader') as mock_fmp:
            mock_downloader = MagicMock()
            mock_downloader.test_connection.return_value = True
            mock_fmp.return_value = mock_downloader

            result = self.run_script("run_weekly_screener.py", [
                "--config", config_path,
                "--test-connection",
                "--verbose"
            ])

            self.assertEqual(result.returncode, 0)

    def test_output_directory_creation(self):
        """Test output directory creation and file saving."""
        output_dir = os.path.join(self.test_dir, "output")
        config_path = self.create_test_config()

        with patch('src.ml.pipeline.p04_short_squeeze.scripts.run_weekly_screener.FMPDataDownloader') as mock_fmp:
            mock_downloader = MagicMock()
            mock_downloader.test_connection.return_value = True
            mock_fmp.return_value = mock_downloader

            result = self.run_script("run_weekly_screener.py", [
                "--config", config_path,
                "--test-connection",
                "--output-dir", output_dir
            ])

            self.assertEqual(result.returncode, 0)


class TestDailyDeepScanScript(ScriptIntegrationTestBase):
    """Test cases for the daily deep scan script."""

    def test_help_argument(self):
        """Test that help argument works."""
        result = self.run_script("run_daily_deep_scan.py", ["--help"], expect_success=False)
        self.assertEqual(result.returncode, 0)
        self.assertIn("daily deep scan", result.stdout.lower())

    def test_invalid_arguments(self):
        """Test handling of invalid arguments."""
        result = self.run_script("run_daily_deep_scan.py", ["--invalid-arg"], expect_success=False)
        self.assertNotEqual(result.returncode, 0)

    @patch('src.ml.pipeline.p04_short_squeeze.scripts.run_daily_deep_scan.FMPDataDownloader')
    @patch('src.ml.pipeline.p04_short_squeeze.scripts.run_daily_deep_scan.FinnhubDataDownloader')
    @patch('src.ml.pipeline.p04_short_squeeze.scripts.run_daily_deep_scan.ConfigManager')
    def test_test_connection_mode(self, mock_config_manager, mock_finnhub, mock_fmp):
        """Test the test connection mode."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config_manager.return_value.load_config.return_value = mock_config

        mock_fmp_downloader = MagicMock()
        mock_fmp_downloader.test_connection.return_value = True
        mock_fmp.return_value = mock_fmp_downloader

        mock_finnhub_downloader = MagicMock()
        mock_finnhub.return_value = mock_finnhub_downloader

        config_path = self.create_test_config()
        result = self.run_script("run_daily_deep_scan.py", ["--config", config_path, "--test-connection"])

        self.assertEqual(result.returncode, 0)
        self.assertIn("connection test successful", result.stdout.lower())

    @patch('src.ml.pipeline.p04_short_squeeze.scripts.run_daily_deep_scan.FMPDataDownloader')
    @patch('src.ml.pipeline.p04_short_squeeze.scripts.run_daily_deep_scan.FinnhubDataDownloader')
    @patch('src.ml.pipeline.p04_short_squeeze.scripts.run_daily_deep_scan.ConfigManager')
    @patch('src.ml.pipeline.p04_short_squeeze.scripts.run_daily_deep_scan.create_daily_deep_scan')
    def test_manual_tickers_mode(self, mock_create_deep_scan, mock_config_manager, mock_finnhub, mock_fmp):
        """Test manual tickers mode execution."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config_manager.return_value.load_config.return_value = mock_config
        mock_config_manager.return_value.get_deep_scan_config.return_value = mock_config

        mock_fmp_downloader = MagicMock()
        mock_fmp_downloader.test_connection.return_value = True
        mock_fmp.return_value = mock_fmp_downloader

        mock_finnhub_downloader = MagicMock()
        mock_finnhub.return_value = mock_finnhub_downloader

        mock_deep_scan = MagicMock()
        mock_results = MagicMock()
        mock_results.run_id = "test_run"
        mock_results.run_date = datetime.now().date()
        mock_results.candidates_processed = 2
        mock_results.scored_candidates = []
        mock_results.data_quality_metrics = {}
        mock_results.runtime_metrics = {}
        mock_deep_scan.run_deep_scan.return_value = mock_results
        mock_create_deep_scan.return_value = mock_deep_scan

        config_path = self.create_test_config()
        result = self.run_script("run_daily_deep_scan.py", [
            "--config", config_path,
            "--tickers", "AAPL,TSLA",
            "--dry-run"
        ])

        self.assertEqual(result.returncode, 0)
        self.assertIn("dry run mode", result.stdout.lower())

    def test_scan_date_parsing(self):
        """Test scan date parsing functionality."""
        config_path = self.create_test_config()

        with patch('src.ml.pipeline.p04_short_squeeze.scripts.run_daily_deep_scan.FMPDataDownloader') as mock_fmp:
            with patch('src.ml.pipeline.p04_short_squeeze.scripts.run_daily_deep_scan.FinnhubDataDownloader') as mock_finnhub:
                mock_fmp_downloader = MagicMock()
                mock_fmp_downloader.test_connection.return_value = True
                mock_fmp.return_value = mock_fmp_downloader

                mock_finnhub_downloader = MagicMock()
                mock_finnhub.return_value = mock_finnhub_downloader

                result = self.run_script("run_daily_deep_scan.py", [
                    "--config", config_path,
                    "--test-connection",
                    "--scan-date", "2024-01-15"
                ])

                self.assertEqual(result.returncode, 0)

    def test_progress_tracking(self):
        """Test progress tracking option."""
        config_path = self.create_test_config()

        with patch('src.ml.pipeline.p04_short_squeeze.scripts.run_daily_deep_scan.FMPDataDownloader') as mock_fmp:
            with patch('src.ml.pipeline.p04_short_squeeze.scripts.run_daily_deep_scan.FinnhubDataDownloader') as mock_finnhub:
                mock_fmp_downloader = MagicMock()
                mock_fmp_downloader.test_connection.return_value = True
                mock_fmp.return_value = mock_fmp_downloader

                mock_finnhub_downloader = MagicMock()
                mock_finnhub.return_value = mock_finnhub_downloader

                result = self.run_script("run_daily_deep_scan.py", [
                    "--config", config_path,
                    "--test-connection",
                    "--progress"
                ])

                self.assertEqual(result.returncode, 0)


class TestAdHocCandidateManagementScript(ScriptIntegrationTestBase):
    """Test cases for the ad-hoc candidate management script."""

    def test_help_argument(self):
        """Test that help argument works."""
        result = self.run_script("manage_adhoc_candidates.py", ["--help"], expect_success=False)
        self.assertEqual(result.returncode, 0)
        self.assertIn("manage ad-hoc candidates", result.stdout.lower())

    def test_no_command_shows_help(self):
        """Test that running without command shows help."""
        result = self.run_script("manage_adhoc_candidates.py", [], expect_success=False)
        self.assertNotEqual(result.returncode, 0)

    def test_invalid_command(self):
        """Test handling of invalid commands."""
        result = self.run_script("manage_adhoc_candidates.py", ["invalid-command"], expect_success=False)
        self.assertNotEqual(result.returncode, 0)

    def test_sample_csv_creation(self):
        """Test sample CSV file creation."""
        output_file = os.path.join(self.test_dir, "sample.csv")
        result = self.run_script("manage_adhoc_candidates.py", ["sample-csv", output_file])

        self.assertEqual(result.returncode, 0)
        self.assertTrue(os.path.exists(output_file))

        # Verify CSV content
        with open(output_file, 'r') as f:
            content = f.read()
            self.assertIn("ticker", content)
            self.assertIn("reason", content)
            self.assertIn("ttl_days", content)

    @patch('src.ml.pipeline.p04_short_squeeze.scripts.manage_adhoc_candidates.AdHocManager')
    @patch('src.ml.pipeline.p04_short_squeeze.scripts.manage_adhoc_candidates.ConfigManager')
    def test_add_candidate_command(self, mock_config_manager, mock_adhoc_manager):
        """Test add candidate command."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.adhoc.default_ttl_days = 7
        mock_config_manager.return_value.load_config.return_value = mock_config

        mock_manager = MagicMock()
        mock_manager.add_candidate.return_value = True
        mock_adhoc_manager.return_value = mock_manager

        config_path = self.create_test_config()
        result = self.run_script("manage_adhoc_candidates.py", [
            "--config", config_path,
            "add", "AAPL", "Test reason"
        ])

        self.assertEqual(result.returncode, 0)
        mock_manager.add_candidate.assert_called_once_with("AAPL", "Test reason", None)

    @patch('src.ml.pipeline.p04_short_squeeze.scripts.manage_adhoc_candidates.AdHocManager')
    @patch('src.ml.pipeline.p04_short_squeeze.scripts.manage_adhoc_candidates.ConfigManager')
    def test_list_candidates_command(self, mock_config_manager, mock_adhoc_manager):
        """Test list candidates command."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.adhoc.default_ttl_days = 7
        mock_config_manager.return_value.load_config.return_value = mock_config

        mock_manager = MagicMock()
        mock_manager.get_active_candidates.return_value = []
        mock_adhoc_manager.return_value = mock_manager

        config_path = self.create_test_config()
        result = self.run_script("manage_adhoc_candidates.py", [
            "--config", config_path,
            "list"
        ])

        self.assertEqual(result.returncode, 0)
        mock_manager.get_active_candidates.assert_called_once()

    @patch('src.ml.pipeline.p04_short_squeeze.scripts.manage_adhoc_candidates.AdHocManager')
    @patch('src.ml.pipeline.p04_short_squeeze.scripts.manage_adhoc_candidates.ConfigManager')
    def test_stats_command(self, mock_config_manager, mock_adhoc_manager):
        """Test statistics command."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.adhoc.default_ttl_days = 7
        mock_config_manager.return_value.load_config.return_value = mock_config

        mock_manager = MagicMock()
        mock_stats = {
            'total_active': 5,
            'promoted_by_screener': 2,
            'expiring_within_3_days': 1,
            'average_age_days': 3.5,
            'default_ttl_days': 7,
            'last_updated': datetime.now()
        }
        mock_manager.get_statistics.return_value = mock_stats
        mock_adhoc_manager.return_value = mock_manager

        config_path = self.create_test_config()
        result = self.run_script("manage_adhoc_candidates.py", [
            "--config", config_path,
            "stats"
        ])

        self.assertEqual(result.returncode, 0)
        mock_manager.get_statistics.assert_called_once()

    def test_bulk_add_with_invalid_csv(self):
        """Test bulk add with invalid CSV file."""
        invalid_csv = os.path.join(self.test_dir, "nonexistent.csv")

        result = self.run_script("manage_adhoc_candidates.py", [
            "bulk-add", invalid_csv
        ], expect_success=False)

        self.assertNotEqual(result.returncode, 0)

    @patch('src.ml.pipeline.p04_short_squeeze.scripts.manage_adhoc_candidates.AdHocManager')
    @patch('src.ml.pipeline.p04_short_squeeze.scripts.manage_adhoc_candidates.ConfigManager')
    def test_bulk_add_with_valid_csv(self, mock_config_manager, mock_adhoc_manager):
        """Test bulk add with valid CSV file."""
        # Create test CSV
        csv_file = os.path.join(self.test_dir, "test_candidates.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ticker', 'reason', 'ttl_days'])
            writer.writerow(['AAPL', 'Test reason 1', '7'])
            writer.writerow(['TSLA', 'Test reason 2', '14'])

        # Setup mocks
        mock_config = MagicMock()
        mock_config.adhoc.default_ttl_days = 7
        mock_config_manager.return_value.load_config.return_value = mock_config

        mock_manager = MagicMock()
        mock_manager.validate_candidate_data.return_value = (True, [])
        mock_manager.bulk_add_candidates.return_value = (2, [])
        mock_adhoc_manager.return_value = mock_manager

        config_path = self.create_test_config()
        result = self.run_script("manage_adhoc_candidates.py", [
            "--config", config_path,
            "bulk-add", csv_file
        ])

        self.assertEqual(result.returncode, 0)
        mock_manager.bulk_add_candidates.assert_called_once()

    def test_verbose_logging(self):
        """Test verbose logging option."""
        result = self.run_script("manage_adhoc_candidates.py", [
            "--verbose",
            "sample-csv", os.path.join(self.test_dir, "test.csv")
        ])

        self.assertEqual(result.returncode, 0)


class TestScriptPerformanceAndResourceUsage(ScriptIntegrationTestBase):
    """Test cases for script performance and resource usage."""

    def test_script_timeout_handling(self):
        """Test that scripts handle timeouts gracefully."""
        # This test ensures scripts don't hang indefinitely
        config_path = self.create_test_config()

        try:
            result = subprocess.run([
                sys.executable,
                str(self.scripts_dir / "run_weekly_screener.py"),
                "--config", config_path,
                "--test-connection"
            ], capture_output=True, text=True, timeout=5)

            # Should complete within timeout or fail gracefully
            self.assertIsNotNone(result.returncode)

        except subprocess.TimeoutExpired:
            self.fail("Script did not complete within reasonable time")

    def test_memory_usage_reasonable(self):
        """Test that scripts don't consume excessive memory."""
        # This is a basic test - in production you might use memory profiling tools
        config_path = self.create_test_config()

        result = self.run_script("manage_adhoc_candidates.py", [
            "sample-csv", os.path.join(self.test_dir, "test.csv")
        ])

        self.assertEqual(result.returncode, 0)
        # Script should complete without memory errors

    def test_error_handling_robustness(self):
        """Test error handling in various failure scenarios."""
        # Test with invalid config path
        result = self.run_script("run_weekly_screener.py", [
            "--config", "/nonexistent/config.yaml"
        ], expect_success=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("error", result.stderr.lower() + result.stdout.lower())

    def test_keyboard_interrupt_handling(self):
        """Test that scripts handle keyboard interrupts gracefully."""
        # This test verifies that scripts can be interrupted cleanly
        # In a real scenario, you might send SIGINT to test this
        pass  # Placeholder for interrupt testing


class TestEndToEndScriptExecution(ScriptIntegrationTestBase):
    """End-to-end integration tests with sample data."""

    @patch('src.ml.pipeline.p04_short_squeeze.scripts.run_weekly_screener.FMPDataDownloader')
    @patch('src.ml.pipeline.p04_short_squeeze.scripts.run_weekly_screener.ConfigManager')
    @patch('src.ml.pipeline.p04_short_squeeze.scripts.run_weekly_screener.create_universe_loader')
    @patch('src.ml.pipeline.p04_short_squeeze.scripts.run_weekly_screener.create_weekly_screener')
    def test_weekly_screener_end_to_end(self, mock_create_screener, mock_create_universe,
                                       mock_config_manager, mock_fmp_downloader):
        """Test complete weekly screener execution with mocked data."""
        # Setup comprehensive mocks for end-to-end test
        mock_config = MagicMock()
        mock_config.run_id = "test_e2e_001"
        mock_config_manager.return_value.load_config.return_value = mock_config
        mock_config_manager.return_value.get_screener_config.return_value = mock_config

        mock_downloader = MagicMock()
        mock_downloader.test_connection.return_value = True
        mock_fmp_downloader.return_value = mock_downloader

        mock_universe_loader = MagicMock()
        mock_universe_loader.load_universe.return_value = ["AAPL", "TSLA", "GME", "AMC", "NVDA"]
        mock_create_universe.return_value = mock_universe_loader

        # Create realistic mock results

        mock_candidate = MagicMock()
        mock_candidate.ticker = "GME"
        mock_candidate.screener_score = 0.85
        mock_candidate.structural_metrics.short_interest_pct = 0.25
        mock_candidate.structural_metrics.days_to_cover = 5.2
        mock_candidate.structural_metrics.market_cap = 1000000000
        mock_candidate.source.value = "SCREENER"

        mock_screener = MagicMock()
        mock_results = MagicMock()
        mock_results.run_id = "test_e2e_001"
        mock_results.run_date = datetime.now().date()
        mock_results.total_universe = 5
        mock_results.candidates_found = 3
        mock_results.top_candidates = [mock_candidate]
        mock_results.data_quality_metrics = {
            'total_tickers': 5,
            'successful_fetches': 4,
            'failed_fetches': 1,
            'api_calls_made': 15
        }
        mock_results.runtime_metrics = {
            'duration_seconds': 45.2,
            'tickers_per_second': 0.11
        }
        mock_screener.run_screener.return_value = mock_results
        mock_create_screener.return_value = mock_screener

        # Run the test
        config_path = self.create_test_config()
        output_dir = os.path.join(self.test_dir, "output")

        result = self.run_script("run_weekly_screener.py", [
            "--config", config_path,
            "--max-universe", "5",
            "--output-dir", output_dir,
            "--verbose"
        ])

        self.assertEqual(result.returncode, 0)
        self.assertIn("completed successfully", result.stdout.lower())

        # Verify output files were created
        self.assertTrue(os.path.exists(output_dir))

    @patch('src.ml.pipeline.p04_short_squeeze.scripts.run_daily_deep_scan.FMPDataDownloader')
    @patch('src.ml.pipeline.p04_short_squeeze.scripts.run_daily_deep_scan.FinnhubDataDownloader')
    @patch('src.ml.pipeline.p04_short_squeeze.scripts.run_daily_deep_scan.ConfigManager')
    @patch('src.ml.pipeline.p04_short_squeeze.scripts.run_daily_deep_scan.create_daily_deep_scan')
    def test_daily_deep_scan_end_to_end(self, mock_create_deep_scan, mock_config_manager,
                                       mock_finnhub, mock_fmp):
        """Test complete daily deep scan execution with mocked data."""
        # Setup comprehensive mocks
        mock_config = MagicMock()
        mock_config.run_id = "test_e2e_002"
        mock_config_manager.return_value.load_config.return_value = mock_config
        mock_config_manager.return_value.get_deep_scan_config.return_value = mock_config

        mock_fmp_downloader = MagicMock()
        mock_fmp_downloader.test_connection.return_value = True
        mock_fmp.return_value = mock_fmp_downloader

        mock_finnhub_downloader = MagicMock()
        mock_finnhub.return_value = mock_finnhub_downloader

        # Create realistic mock results
        mock_scored_candidate = MagicMock()
        mock_scored_candidate.candidate.ticker = "GME"
        mock_scored_candidate.squeeze_score = 0.78
        mock_scored_candidate.candidate.screener_score = 0.85
        mock_scored_candidate.transient_metrics.volume_spike = 2.5
        mock_scored_candidate.transient_metrics.sentiment_24h = 0.65
        mock_scored_candidate.transient_metrics.call_put_ratio = 1.8
        mock_scored_candidate.transient_metrics.borrow_fee_pct = 0.15
        mock_scored_candidate.candidate.source.value = "SCREENER"
        mock_scored_candidate.alert_level = "HIGH"

        mock_deep_scan = MagicMock()
        mock_results = MagicMock()
        mock_results.run_id = "test_e2e_002"
        mock_results.run_date = datetime.now().date()
        mock_results.candidates_processed = 3
        mock_results.scored_candidates = [mock_scored_candidate]
        mock_results.data_quality_metrics = {
            'successful_scans': 2,
            'failed_scans': 1,
            'api_calls_fmp': 8,
            'api_calls_finnhub': 6,
            'valid_volume_data': 2,
            'valid_sentiment_data': 1,
            'valid_options_data': 2,
            'valid_borrow_rates': 1
        }
        mock_results.runtime_metrics = {
            'duration_seconds': 25.8,
            'candidates_per_second': 0.12
        }
        mock_deep_scan.run_deep_scan.return_value = mock_results
        mock_create_deep_scan.return_value = mock_deep_scan

        # Run the test
        config_path = self.create_test_config()
        output_dir = os.path.join(self.test_dir, "output")

        result = self.run_script("run_daily_deep_scan.py", [
            "--config", config_path,
            "--tickers", "GME,AMC,AAPL",
            "--output-dir", output_dir,
            "--progress"
        ])

        self.assertEqual(result.returncode, 0)
        self.assertIn("completed successfully", result.stdout.lower())


if __name__ == '__main__':
    unittest.main()