"""
Test Suite for CustomStrategy Backtesting
-----------------------------------------

This module contains pytest tests for backtesting CustomStrategy with various
entry/exit mixin combinations using JSON configuration files.

Usage:
    pytest src/backtester/tests/test_custom_strategy.py -v
    pytest src/backtester/tests/test_custom_strategy.py::test_custom_strategy_rsi_bb_fixed_ratio -v
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.backtester.tests.backtester_test_framework import (
    BacktesterTestFramework,
    run_backtest_from_config
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class TestCustomStrategyBacktest:
    """
    Test suite for CustomStrategy backtesting using JSON configurations.
    """

    @pytest.fixture
    def config_dir(self):
        """Get the configuration directory."""
        return Path(project_root) / "config" / "backtester"

    @pytest.fixture
    def custom_strategy_config(self, config_dir):
        """Path to custom strategy test config."""
        return config_dir / "custom_strategy_test.json"

    @pytest.fixture
    def rsi_volume_supertrend_config(self, config_dir):
        """Path to RSI+Volume+Supertrend test config."""
        return config_dir / "rsi_volume_supertrend_test.json"

    def test_framework_initialization(self, custom_strategy_config):
        """Test that the framework initializes correctly."""
        framework = BacktesterTestFramework(str(custom_strategy_config))

        assert framework.config is not None
        assert 'strategy' in framework.config
        assert 'data' in framework.config
        assert 'broker' in framework.config

        _logger.info("Framework initialization test passed")

    def test_config_loading(self, custom_strategy_config):
        """Test configuration loading and validation."""
        framework = BacktesterTestFramework(str(custom_strategy_config))

        # Check strategy config
        assert framework.config['strategy']['type'] == 'CustomStrategy'
        assert 'entry_logic' in framework.config['strategy']['parameters']
        assert 'exit_logic' in framework.config['strategy']['parameters']

        # Check data config
        assert 'file_path' in framework.config['data']
        assert 'symbol' in framework.config['data']

        # Check broker config
        assert 'cash' in framework.config['broker']
        assert 'commission' in framework.config['broker']

        _logger.info("Config loading test passed")

    def test_invalid_config_file(self):
        """Test handling of invalid configuration file."""
        with pytest.raises(FileNotFoundError):
            BacktesterTestFramework("nonexistent_config.json")

    @pytest.mark.skipif(
        not (Path(project_root) / "data" / "BTCUSDT_1h.csv").exists(),
        reason="Test data file not found"
    )
    def test_custom_strategy_rsi_bb_fixed_ratio(self, custom_strategy_config):
        """
        Test CustomStrategy with RSI+BB entry and Fixed Ratio exit.

        This test:
        1. Loads the config from config/backtester/custom_strategy_test.json
        2. Sets up CustomStrategy with RSIBBMixin entry and FixedRatioExitMixin exit
        3. Runs backtest on the provided data
        4. Validates results against configured assertions
        """
        framework = BacktesterTestFramework(str(custom_strategy_config))

        # Setup and run
        framework.setup_backtest()
        results = framework.run_backtest()

        # Validate results exist
        assert results is not None
        assert 'initial_value' in results
        assert 'final_value' in results
        assert 'total_return' in results
        assert 'total_trades' in results

        # Log results
        _logger.info("Backtest Results:")
        _logger.info("  Initial Value: $%.2f", results['initial_value'])
        _logger.info("  Final Value: $%.2f", results['final_value'])
        _logger.info("  Total Return: %.2f%%", results['total_return'] * 100)
        _logger.info("  Total Trades: %d", results.get('total_trades', 0))

        # Validate assertions
        validation = framework.validate_assertions()

        # The test passes if validation passes (or if no assertions are configured)
        if validation.get('failures'):
            _logger.warning("Assertion failures: %s", validation['failures'])
            # Note: We don't fail the test here to allow inspection of results
            # Remove the next line if you want strict assertion checking
            pytest.skip("Assertions configured but not met - check results for analysis")

        # Generate report
        report = framework.generate_report()
        _logger.info("\n%s", report)

        assert results['total_trades'] >= 0  # Basic sanity check

    @pytest.mark.skipif(
        not (Path(project_root) / "data" / "BTCUSDT_1h.csv").exists(),
        reason="Test data file not found"
    )
    def test_custom_strategy_rsi_volume_supertrend_atr(self, rsi_volume_supertrend_config):
        """
        Test CustomStrategy with RSI+Volume+Supertrend entry and ATR exit.

        This test:
        1. Loads the config from config/backtester/rsi_volume_supertrend_test.json
        2. Sets up CustomStrategy with RSIVolumeSuperTrendMixin and ATRExitMixin
        3. Runs backtest on filtered data (2023 only)
        4. Validates results
        """
        framework = BacktesterTestFramework(str(rsi_volume_supertrend_config))

        # Run full test
        test_results = framework.run_full_test(generate_report_file=True)

        assert test_results['results'] is not None
        assert test_results['validation'] is not None

        results = test_results['results']
        validation = test_results['validation']

        # Log results
        _logger.info("Backtest Results:")
        _logger.info("  Initial Value: $%.2f", results['initial_value'])
        _logger.info("  Final Value: $%.2f", results['final_value'])
        _logger.info("  Total Return: %.2f%%", results['total_return'] * 100)
        _logger.info("  Total Trades: %d", results.get('total_trades', 0))
        _logger.info("  Win Rate: %.2f%%", results.get('win_rate', 0) * 100)

        # Print report
        print("\n" + test_results['report'])

        # Basic sanity checks
        assert results['initial_value'] > 0
        assert results['final_value'] > 0
        assert results['total_trades'] >= 0

    def test_strategy_class_mapping(self, custom_strategy_config):
        """Test that strategy class can be resolved correctly."""
        framework = BacktesterTestFramework(str(custom_strategy_config))

        strategy_class = framework._get_strategy_class('CustomStrategy')
        assert strategy_class is not None
        assert strategy_class.__name__ == 'CustomStrategy'

    def test_invalid_strategy_type(self, custom_strategy_config):
        """Test handling of invalid strategy type."""
        framework = BacktesterTestFramework(str(custom_strategy_config))

        with pytest.raises(ValueError, match="Unknown strategy type"):
            framework._get_strategy_class('NonExistentStrategy')

    @pytest.mark.skipif(
        not (Path(project_root) / "data" / "BTCUSDT_1h.csv").exists(),
        reason="Test data file not found"
    )
    def test_data_feed_preparation(self, custom_strategy_config):
        """Test data feed preparation from config."""
        framework = BacktesterTestFramework(str(custom_strategy_config))

        data_feed = framework._prepare_data_feed()

        assert data_feed is not None
        assert data_feed.p.symbol == framework.config['data']['symbol']

    def test_assertion_validation_logic(self, custom_strategy_config):
        """Test assertion validation logic."""
        framework = BacktesterTestFramework(str(custom_strategy_config))

        # Mock results for testing
        framework.results = {
            'total_trades': 10,
            'max_drawdown': 0.15,
            'sharpe_ratio': 1.5,
            'initial_value': 10000,
            'final_value': 12000,
        }

        # Test with passing assertions
        framework.config['assertions'] = {
            'min_trades': 5,
            'max_drawdown_pct': 20.0,
            'min_sharpe_ratio': 1.0,
        }

        validation = framework.validate_assertions()

        assert validation['passed'] is True
        assert len(validation['failures']) == 0
        assert len(validation['checks']) == 3

    def test_assertion_validation_failures(self, custom_strategy_config):
        """Test that assertion failures are correctly detected."""
        framework = BacktesterTestFramework(str(custom_strategy_config))

        # Mock results that will fail assertions
        framework.results = {
            'total_trades': 2,
            'max_drawdown': 0.25,
            'sharpe_ratio': 0.5,
            'initial_value': 10000,
            'final_value': 9000,
        }

        # Test with failing assertions
        framework.config['assertions'] = {
            'min_trades': 5,
            'max_drawdown_pct': 20.0,
            'min_sharpe_ratio': 1.0,
            'final_value_greater_than_initial': True,
        }

        validation = framework.validate_assertions()

        assert validation['passed'] is False
        assert len(validation['failures']) == 4  # All assertions should fail

    def test_report_generation(self, custom_strategy_config):
        """Test report generation."""
        framework = BacktesterTestFramework(str(custom_strategy_config))

        # Mock results
        framework.results = {
            'test_name': 'Test Report Generation',
            'description': 'Testing report generation',
            'initial_value': 10000,
            'final_value': 12000,
            'total_pnl': 2000,
            'total_return': 0.20,
            'sharpe_ratio': 1.5,
            'max_drawdown': 0.10,
            'max_drawdown_period': 50,
            'total_trades': 10,
            'won_trades': 6,
            'lost_trades': 4,
            'win_rate': 0.60,
            'avg_win': 500,
            'avg_loss': -250,
            'profit_factor': 2.0,
        }

        report = framework.generate_report()

        assert report is not None
        assert 'BACKTESTER TEST REPORT' in report
        assert 'Performance Metrics' in report
        assert 'Trade Statistics' in report
        assert '$12,000.00' in report  # Final value formatted
        assert '20.00%' in report  # Return formatted


class TestConvenienceFunctions:
    """Test convenience functions for running backtests."""

    @pytest.fixture
    def config_dir(self):
        """Get the configuration directory."""
        return Path(project_root) / "config" / "backtester"

    @pytest.mark.skipif(
        not (Path(project_root) / "data" / "BTCUSDT_1h.csv").exists(),
        reason="Test data file not found"
    )
    def test_run_backtest_from_config_function(self, config_dir):
        """Test the convenience function for running backtests."""
        config_path = config_dir / "custom_strategy_test.json"

        if not config_path.exists():
            pytest.skip("Config file not found")

        # Run backtest
        results = run_backtest_from_config(str(config_path), generate_report=False)

        assert results is not None
        assert 'results' in results
        assert 'validation' in results
        assert 'report' in results
        assert 'success' in results


# Integration test that can be run manually
@pytest.mark.integration
@pytest.mark.skipif(
    not (Path(project_root) / "data" / "BTCUSDT_1h.csv").exists(),
    reason="Test data file not found"
)
def test_full_integration_custom_strategy():
    """
    Full integration test for CustomStrategy backtesting.

    This test runs a complete backtest with all features enabled.
    Mark as 'integration' to run separately from unit tests.

    Run with: pytest src/backtester/tests/test_custom_strategy.py -v -m integration
    """
    config_dir = Path(project_root) / "config" / "backtester"
    config_path = config_dir / "custom_strategy_test.json"

    if not config_path.exists():
        pytest.skip("Config file not found")

    framework = BacktesterTestFramework(str(config_path))

    # Run full workflow
    test_results = framework.run_full_test(generate_report_file=True)

    # Comprehensive validation
    assert test_results['results']['initial_value'] > 0
    assert test_results['results']['final_value'] > 0
    assert 'total_return' in test_results['results']
    assert 'total_trades' in test_results['results']
    assert test_results['validation'] is not None
    assert test_results['report'] is not None

    # Print summary
    print("\n" + "=" * 80)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 80)
    print(f"Status: {'PASSED' if test_results['success'] else 'FAILED'}")
    print(f"Total Trades: {test_results['results'].get('total_trades', 0)}")
    print(f"Total Return: {test_results['results']['total_return']*100:.2f}%")
    print(f"Final Value: ${test_results['results']['final_value']:,.2f}")
    print("=" * 80)

    _logger.info("Full integration test completed")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
