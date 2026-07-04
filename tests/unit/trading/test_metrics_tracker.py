import json
from unittest.mock import patch

import pytest

from src.trading.metrics_tracker import MetricsRegistry, PerformanceMetrics


class TestPerformanceMetrics:
    def test_initial_metrics(self):
        metrics = PerformanceMetrics(bot_id="bot_1", symbol="BTCUSDT")
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.total_pnl == 0.0

    def test_update_metrics_win(self):
        metrics = PerformanceMetrics(bot_id="bot_1", symbol="BTCUSDT", current_balance=1000.0, peak_balance=1000.0)
        metrics.update(100.0, 10.0, 1100.0)
        assert metrics.total_trades == 1
        assert metrics.winning_trades == 1
        assert metrics.win_rate == 100.0
        assert metrics.total_pnl == 100.0
        assert metrics.current_balance == 1100.0

    def test_update_metrics_loss(self):
        metrics = PerformanceMetrics(bot_id="bot_1", symbol="BTCUSDT", current_balance=1000.0, peak_balance=1000.0)
        metrics.update(-50.0, -5.0, 950.0)
        assert metrics.total_trades == 1
        assert metrics.winning_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.total_pnl == -50.0

    def test_update_zero_peak_balance_no_division_error(self):
        """peak_balance 0 (e.g. initial_balance 0) must not raise in update()."""
        metrics = PerformanceMetrics(bot_id="bot_z", symbol="BTCUSDT", current_balance=0.0, peak_balance=0.0)
        metrics.update(-10.0, 0.0, 0.0)
        assert metrics.total_trades == 1
        assert metrics.current_drawdown == 0.0

    def test_drawdown_calculation(self):
        metrics = PerformanceMetrics(bot_id="bot_1", symbol="BTCUSDT", current_balance=10000.0, peak_balance=10000.0)

        # Drawdown to 9000
        metrics.update(-1000.0, -10.0, 9000.0)
        assert metrics.max_drawdown == 10.0

        # Recovery to 9500 (peak still 10000)
        metrics.update(500.0, 5.0, 9500.0)
        assert metrics.max_drawdown == 10.0

        # New Peak 11000
        metrics.update(1500.0, 15.0, 11000.0)
        assert metrics.peak_balance == 11000.0
        assert metrics.max_drawdown == 10.0


class TestMetricsRegistry:
    @pytest.fixture
    def temp_registry(self, tmp_path):
        """Fixture to create a MetricsRegistry with a temporary file."""
        metrics_file = tmp_path / "metrics.json"

        # We need to bypass the singleton for testing OR reset it
        # Since MetricsRegistry might have been initialized already, we patch its file path
        with patch("src.trading.metrics_tracker.DATA_DIR", tmp_path):
            registry = MetricsRegistry()
            # Force re-initialization of the file path for the test
            registry.metrics_file = metrics_file
            registry.bot_metrics = {}
            yield registry

    def test_get_metrics_creates_new(self, temp_registry):
        metrics = temp_registry.get_metrics("bot_123", "ETHUSDT", 5000.0)
        assert isinstance(metrics, PerformanceMetrics)
        assert "bot_123" in temp_registry.bot_metrics
        assert metrics.symbol == "ETHUSDT"
        assert metrics.current_balance == 5000.0

    def test_record_trade_updates_and_saves(self, temp_registry):
        temp_registry.record_trade("bot_123", "BTCUSDT", 200.0, 2.0, 10200.0)

        metrics = temp_registry.get_metrics("bot_123", "BTCUSDT", 10200.0)
        assert metrics.total_trades == 1
        assert metrics.total_pnl == 200.0

        # Verify file was created/updated
        assert temp_registry.metrics_file.exists()
        with open(temp_registry.metrics_file) as f:
            data = json.load(f)
            assert "bot_123" in data
            assert data["bot_123"]["total_pnl"] == 200.0
