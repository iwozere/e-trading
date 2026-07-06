import os
import tempfile
from datetime import datetime, timedelta
from typing import Any, Dict, List
import pytest
from src.analytics.advanced_analytics import AdvancedAnalytics
from src.model.analytics import PerformanceMetrics, Trade


def test_advanced_analytics_initialization():
    """Test initializing AdvancedAnalytics."""
    analytics = AdvancedAnalytics(risk_free_rate=0.02)
    assert analytics.risk_free_rate == 0.02
    assert len(analytics.trades) == 0
    assert analytics.metrics is None


def test_calculate_metrics_with_no_trades():
    """Test calculating metrics when no trades have been added."""
    analytics = AdvancedAnalytics()
    metrics = analytics.calculate_metrics()
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.total_trades == 0
    assert metrics.win_rate == 0.0


def test_calculate_metrics_with_trades():
    """Test calculating metrics with winning and losing trades."""
    analytics = AdvancedAnalytics()
    
    # Add one winning trade and one losing trade
    now = datetime.now()
    trades_data = [
        {
            "entry_time": (now - timedelta(hours=2)).isoformat(),
            "exit_time": (now - timedelta(hours=1)).isoformat(),
            "symbol": "BTCUSDT",
            "side": "BUY",
            "entry_price": 50000.0,
            "exit_price": 51000.0,
            "quantity": 1.0,
            "pnl": 1000.0,
            "commission": 10.0,
            "net_pnl": 990.0,
            "exit_reason": "take_profit",
        },
        {
            "entry_time": (now - timedelta(hours=4)).isoformat(),
            "exit_time": (now - timedelta(hours=3)).isoformat(),
            "symbol": "BTCUSDT",
            "side": "SELL",
            "entry_price": 50000.0,
            "exit_price": 50500.0,
            "quantity": 1.0,
            "pnl": -500.0,
            "commission": 10.0,
            "net_pnl": -510.0,
            "exit_reason": "stop_loss",
        }
    ]
    
    analytics.add_trades(trades_data)
    assert len(analytics.trades) == 2
    
    metrics = analytics.calculate_metrics()
    assert metrics.total_trades == 2
    assert metrics.winning_trades == 1
    assert metrics.losing_trades == 1
    assert metrics.win_rate == 50.0
    assert metrics.avg_win == 990.0
    assert metrics.avg_loss == -510.0
    assert metrics.payoff_ratio == 990.0 / 510.0
    assert metrics.avg_trade_duration == timedelta(hours=1)


def test_generate_report_json():
    """Test generating JSON performance report."""
    analytics = AdvancedAnalytics()
    now = datetime.now()
    trades_data = [
        {
            "entry_time": (now - timedelta(hours=2)).isoformat(),
            "exit_time": (now - timedelta(hours=1)).isoformat(),
            "symbol": "BTCUSDT",
            "side": "BUY",
            "entry_price": 50000.0,
            "exit_price": 51000.0,
            "quantity": 1.0,
            "pnl": 1000.0,
            "commission": 10.0,
            "net_pnl": 990.0,
            "exit_reason": "take_profit",
        }
    ]
    analytics.add_trades(trades_data)
    analytics.calculate_metrics()

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = analytics.generate_performance_report(output_dir=tmpdir)
        assert os.path.exists(report_path)
        _, ext = os.path.splitext(report_path)
        assert ext.lower() in [".pdf", ".xlsx", ".json"]
