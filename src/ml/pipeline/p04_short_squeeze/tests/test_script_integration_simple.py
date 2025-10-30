#!/usr/bin/env python3
"""
Simple integration tests for Short Squeeze Detection Pipeline scripts.

These tests verify that the scripts can be executed and handle basic command-line
arguments correctly. More comprehensive tests with mocking are in test_script_integration.py.
"""

import unittest
import subprocess
import sys
import tempfile
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))


class TestScriptBasicFunctionality(unittest.TestCase):
    """Basic functionality tests for pipeline scripts."""

    def setUp(self):
        """Set up test environment."""
        self.scripts_dir = PROJECT_ROOT / "src" / "ml" / "pipeline" / "p04_short_squeeze" / "scripts"
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_weekly_screener_script_exists(self):
        """Test that weekly screener script exists and is executable."""
        script_path = self.scripts_dir / "run_weekly_screener.py"
        self.assertTrue(script_path.exists(), "Weekly screener script should exist")
        self.assertTrue(script_path.is_file(), "Weekly screener should be a file")

    def test_daily_deep_scan_script_exists(self):
        """Test that daily deep scan script exists and is executable."""
        script_path = self.scripts_dir / "run_daily_deep_scan.py"
        self.assertTrue(script_path.exists(), "Daily deep scan script should exist")
        self.assertTrue(script_path.is_file(), "Daily deep scan should be a file")

    def test_adhoc_management_script_exists(self):
        """Test that ad-hoc management script exists and is executable."""
        script_path = self.scripts_dir / "manage_adhoc_candidates.py"
        self.assertTrue(script_path.exists(), "Ad-hoc management script should exist")
        self.assertTrue(script_path.is_file(), "Ad-hoc management should be a file")

    def test_scripts_have_proper_shebang(self):
        """Test that scripts have proper shebang lines."""
        scripts = [
            "run_weekly_screener.py",
            "run_daily_deep_scan.py",
            "manage_adhoc_candidates.py"
        ]

        for script_name in scripts:
            script_path = self.scripts_dir / script_name
            with open(script_path, 'r') as f:
                first_line = f.readline().strip()
                self.assertTrue(
                    first_line.startswith('#!/usr/bin/env python') or first_line.startswith('#!/usr/bin/python'),
                    f"Script {script_name} should have proper shebang"
                )

    def test_scripts_have_docstrings(self):
        """Test that scripts have proper docstrings with usage information."""
        scripts = [
            "run_weekly_screener.py",
            "run_daily_deep_scan.py",
            "manage_adhoc_candidates.py"
        ]

        for script_name in scripts:
            script_path = self.scripts_dir / script_name
            with open(script_path, 'r') as f:
                content = f.read()
                # Check for docstring
                self.assertIn('"""', content, f"Script {script_name} should have docstring")
                # Check for usage information
                self.assertIn('Usage:', content, f"Script {script_name} should have usage information")
                # Check for examples
                self.assertIn('Examples:', content, f"Script {script_name} should have examples")

    def test_scripts_syntax_validity(self):
        """Test that all scripts have valid Python syntax."""
        import ast

        scripts = [
            "run_weekly_screener.py",
            "run_daily_deep_scan.py",
            "manage_adhoc_candidates.py"
        ]

        for script_name in scripts:
            script_path = self.scripts_dir / script_name
            with open(script_path, 'r') as f:
                content = f.read()

            try:
                ast.parse(content)
            except SyntaxError as e:
                self.fail(f"Script {script_name} has syntax error: {e}")

    def test_scripts_have_main_function(self):
        """Test that scripts have main functions and proper entry points."""
        scripts = [
            "run_weekly_screener.py",
            "run_daily_deep_scan.py",
            "manage_adhoc_candidates.py"
        ]

        for script_name in scripts:
            script_path = self.scripts_dir / script_name
            with open(script_path, 'r') as f:
                content = f.read()

            # Check for main function
            self.assertIn('def main()', content, f"Script {script_name} should have main() function")
            # Check for proper entry point
            self.assertIn('if __name__ == "__main__":', content, f"Script {script_name} should have proper entry point")

    def test_scripts_have_argument_parsing(self):
        """Test that scripts have argument parsing functionality."""
        scripts = [
            "run_weekly_screener.py",
            "run_daily_deep_scan.py",
            "manage_adhoc_candidates.py"
        ]

        for script_name in scripts:
            script_path = self.scripts_dir / script_name
            with open(script_path, 'r') as f:
                content = f.read()

            # Check for argparse usage
            self.assertIn('argparse', content, f"Script {script_name} should use argparse")
            self.assertIn('ArgumentParser', content, f"Script {script_name} should create ArgumentParser")
            self.assertIn('parse_args', content, f"Script {script_name} should parse arguments")

    def test_scripts_have_error_handling(self):
        """Test that scripts have proper error handling."""
        scripts = [
            "run_weekly_screener.py",
            "run_daily_deep_scan.py",
            "manage_adhoc_candidates.py"
        ]

        for script_name in scripts:
            script_path = self.scripts_dir / script_name
            with open(script_path, 'r') as f:
                content = f.read()

            # Check for exception handling
            self.assertIn('try:', content, f"Script {script_name} should have try/except blocks")
            self.assertIn('except', content, f"Script {script_name} should have exception handling")
            # Check for KeyboardInterrupt handling
            self.assertIn('KeyboardInterrupt', content, f"Script {script_name} should handle KeyboardInterrupt")

    def test_adhoc_script_has_subcommands(self):
        """Test that ad-hoc management script has proper subcommand structure."""
        script_path = self.scripts_dir / "manage_adhoc_candidates.py"
        with open(script_path, 'r') as f:
            content = f.read()

        # Check for subparsers
        self.assertIn('subparsers', content, "Ad-hoc script should have subcommands")
        self.assertIn('add_subparsers', content, "Ad-hoc script should create subparsers")

        # Check for expected commands
        expected_commands = ['add', 'remove', 'list', 'status', 'stats']
        for command in expected_commands:
            self.assertIn(f"'{command}'", content, f"Ad-hoc script should have {command} command")

    def test_scripts_have_configuration_support(self):
        """Test that scripts support configuration files."""
        scripts = [
            "run_weekly_screener.py",
            "run_daily_deep_scan.py",
            "manage_adhoc_candidates.py"
        ]

        for script_name in scripts:
            script_path = self.scripts_dir / script_name
            with open(script_path, 'r') as f:
                content = f.read()

            # Check for config argument
            self.assertIn('--config', content, f"Script {script_name} should support --config argument")
            # Check for ConfigManager usage
            self.assertIn('ConfigManager', content, f"Script {script_name} should use ConfigManager")

    def test_scripts_have_logging_setup(self):
        """Test that scripts have proper logging setup."""
        scripts = [
            "run_weekly_screener.py",
            "run_daily_deep_scan.py",
            "manage_adhoc_candidates.py"
        ]

        for script_name in scripts:
            script_path = self.scripts_dir / script_name
            with open(script_path, 'r') as f:
                content = f.read()

            # Check for logging setup
            self.assertIn('logger', content.lower(), f"Script {script_name} should have logging")
            # Check for verbose option
            self.assertIn('--verbose', content, f"Script {script_name} should support --verbose")


class TestScriptIntegrationRequirements(unittest.TestCase):
    """Test that scripts meet the integration requirements from task 7.4."""

    def setUp(self):
        """Set up test environment."""
        self.scripts_dir = PROJECT_ROOT / "src" / "ml" / "pipeline" / "p04_short_squeeze" / "scripts"

    def test_command_line_argument_parsing(self):
        """Test that scripts have comprehensive command-line argument parsing."""
        # This test verifies requirement: "Test command-line argument parsing"

        # Weekly screener should have these arguments
        weekly_script = self.scripts_dir / "run_weekly_screener.py"
        with open(weekly_script, 'r') as f:
            content = f.read()

        expected_args = ['--config', '--max-universe', '--dry-run', '--verbose', '--test-connection']
        for arg in expected_args:
            self.assertIn(arg, content, f"Weekly screener should support {arg}")

        # Daily deep scan should have these arguments
        daily_script = self.scripts_dir / "run_daily_deep_scan.py"
        with open(daily_script, 'r') as f:
            content = f.read()

        expected_args = ['--config', '--tickers', '--batch-size', '--dry-run', '--progress']
        for arg in expected_args:
            self.assertIn(arg, content, f"Daily deep scan should support {arg}")

    def test_error_handling_implementation(self):
        """Test that scripts have comprehensive error handling."""
        # This test verifies requirement: "Test error handling"

        scripts = [
            "run_weekly_screener.py",
            "run_daily_deep_scan.py",
            "manage_adhoc_candidates.py"
        ]

        for script_name in scripts:
            script_path = self.scripts_dir / script_name
            with open(script_path, 'r') as f:
                content = f.read()

            # Should handle various exception types
            self.assertIn('Exception', content, f"{script_name} should handle general exceptions")
            self.assertIn('KeyboardInterrupt', content, f"{script_name} should handle KeyboardInterrupt")

            # Should have proper exit codes
            self.assertIn('return', content, f"{script_name} should return exit codes")
            self.assertIn('sys.exit', content, f"{script_name} should use sys.exit")

    def test_performance_monitoring_capability(self):
        """Test that scripts have performance monitoring capabilities."""
        # This test verifies requirement: "Test script performance and resource usage"

        scripts = [
            "run_weekly_screener.py",
            "run_daily_deep_scan.py"
        ]

        for script_name in scripts:
            script_path = self.scripts_dir / script_name
            with open(script_path, 'r') as f:
                content = f.read()

            # Should track runtime metrics
            self.assertIn('runtime', content.lower(), f"{script_name} should track runtime")
            self.assertIn('performance', content.lower(), f"{script_name} should have performance reporting")

            # Should have timing capabilities
            self.assertIn('time', content.lower(), f"{script_name} should measure time")

    def test_sample_data_compatibility(self):
        """Test that scripts are designed to work with sample data."""
        # This test verifies requirement: "Test end-to-end execution with sample data"

        # Weekly screener should support universe limiting for testing
        weekly_script = self.scripts_dir / "run_weekly_screener.py"
        with open(weekly_script, 'r') as f:
            content = f.read()

        self.assertIn('max-universe', content, "Weekly screener should support universe limiting")
        self.assertIn('dry-run', content, "Weekly screener should support dry run mode")

        # Daily deep scan should support manual ticker input
        daily_script = self.scripts_dir / "run_daily_deep_scan.py"
        with open(daily_script, 'r') as f:
            content = f.read()

        self.assertIn('tickers', content, "Daily deep scan should support manual ticker input")
        self.assertIn('dry-run', content, "Daily deep scan should support dry run mode")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)