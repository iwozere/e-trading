#!/usr/bin/env python3
"""
Unit Tests for Main API Endpoints
--------------------------------

Tests for the core FastAPI application endpoints including:
- Health check and root endpoints
- Strategy management CRUD operations
- Strategy lifecycle management (start/stop/restart)
- System monitoring endpoints
- Configuration management
"""

import pytest
from unittest.mock import patch, AsyncMock
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.api.services import StrategyValidationError, StrategyOperationError


class TestRootAndHealthEndpoints:
    """Test cases for root and health check endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns correct information."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Trading Web UI API"
        assert data["version"] == "1.0.0"

    def test_health_check_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "trading_system_available" in data
        assert isinstance(data["trading_system_available"], bool)

    def test_test_auth_endpoint_authenticated(self, authenticated_client_admin, mock_admin_user):
        """Test authentication test endpoint with valid user."""
        response = authenticated_client_admin.get("/api/test-auth")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Authentication successful"
        assert "user" in data
        assert data["user"]["email"] == mock_admin_user.email

    def test_test_auth_endpoint_unauthenticated(self, client):
        """Test authentication test endpoint without authentication."""
        response = client.get("/api/test-auth")

        assert response.status_code == 403  # No authorization header


class TestStrategyManagementEndpoints:
    """Test cases for strategy management CRUD operations."""

    @patch('src.api.main.strategy_service')
    def test_list_strategies_success(self, mock_service, authenticated_client_admin):
        """Test successful strategy listing."""
        mock_strategies = [
            {
                "instance_id": "strategy-1",
                "name": "Test Strategy 1",
                "status": "running",
                "uptime_seconds": 3600.0,
                "error_count": 0,
                "last_error": None,
                "broker_type": "paper",
                "trading_mode": "paper",
                "symbol": "BTCUSDT",
                "strategy_type": "sma_crossover"
            }
        ]
        mock_service.get_all_strategies_status.return_value = mock_strategies

        response = authenticated_client_admin.get("/api/strategies")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["instance_id"] == "strategy-1"
        assert data[0]["name"] == "Test Strategy 1"
        assert data[0]["status"] == "running"

    @patch('src.api.main.strategy_service')
    def test_list_strategies_error(self, mock_service, authenticated_client_admin):
        """Test strategy listing with service error."""
        mock_service.get_all_strategies_status.side_effect = Exception("Service error")

        response = authenticated_client_admin.get("/api/strategies")

        assert response.status_code == 500
        data = response.json()
        assert "Service error" in data["detail"]

    @patch('src.api.main.strategy_service')
    def test_create_strategy_success(self, mock_service, authenticated_client_trader, sample_strategy_config):
        """Test successful strategy creation."""
        from unittest.mock import AsyncMock
        mock_service.create_strategy = AsyncMock(return_value={
            "message": "Strategy created successfully",
            "strategy_id": "test-strategy"
        })

        response = authenticated_client_trader.post("/api/strategies", json=sample_strategy_config)

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Strategy created successfully"
        assert data["strategy_id"] == "test-strategy"
        mock_service.create_strategy.assert_called_once()

    @patch('src.api.main.strategy_service')
    def test_create_strategy_validation_error(self, mock_service, authenticated_client_trader, sample_strategy_config):
        """Test strategy creation with validation error."""
        mock_service.create_strategy.side_effect = StrategyValidationError("Invalid configuration")

        response = authenticated_client_trader.post("/api/strategies", json=sample_strategy_config)

        assert response.status_code == 400
        data = response.json()
        assert "Invalid configuration" in data["detail"]

    @patch('src.api.main.strategy_service')
    def test_create_strategy_operation_error(self, mock_service, authenticated_client_trader, sample_strategy_config):
        """Test strategy creation with operation error."""
        mock_service.create_strategy.side_effect = StrategyOperationError("Service unavailable")

        response = authenticated_client_trader.post("/api/strategies", json=sample_strategy_config)

        assert response.status_code == 503
        data = response.json()
        assert "Service unavailable" in data["detail"]

    def test_create_strategy_unauthorized(self, authenticated_client_viewer, sample_strategy_config):
        """Test strategy creation with insufficient permissions."""
        response = authenticated_client_viewer.post("/api/strategies", json=sample_strategy_config)

        assert response.status_code == 403
        data = response.json()
        assert "Access denied" in data["detail"]

    @patch('src.api.main.strategy_service')
    def test_get_strategy_success(self, mock_service, authenticated_client_admin, sample_strategy_status):
        """Test successful strategy retrieval."""
        mock_service.get_strategy_status.return_value = sample_strategy_status

        response = authenticated_client_admin.get("/api/strategies/test-strategy")

        assert response.status_code == 200
        data = response.json()
        assert data["instance_id"] == sample_strategy_status["instance_id"]
        assert data["name"] == sample_strategy_status["name"]

    @patch('src.api.main.strategy_service')
    def test_get_strategy_not_found(self, mock_service, authenticated_client_admin):
        """Test strategy retrieval when strategy not found."""
        mock_service.get_strategy_status.return_value = None

        response = authenticated_client_admin.get("/api/strategies/nonexistent")

        assert response.status_code == 404
        data = response.json()
        assert "Strategy not found" in data["detail"]

    @patch('src.api.main.strategy_service')
    def test_update_strategy_success(self, mock_service, authenticated_client_trader, sample_strategy_config):
        """Test successful strategy update."""
        mock_service.update_strategy = AsyncMock(return_value={
            "message": "Strategy updated successfully",
            "strategy_id": "test-strategy"
        })

        response = authenticated_client_trader.put("/api/strategies/test-strategy", json=sample_strategy_config)

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Strategy updated successfully"
        assert data["strategy_id"] == "test-strategy"

    @patch('src.api.main.strategy_service')
    def test_update_strategy_not_found(self, mock_service, authenticated_client_trader, sample_strategy_config):
        """Test strategy update when strategy not found."""
        mock_service.update_strategy.side_effect = StrategyOperationError("Strategy not found")

        response = authenticated_client_trader.put("/api/strategies/nonexistent", json=sample_strategy_config)

        assert response.status_code == 404
        data = response.json()
        assert "Strategy not found" in data["detail"]

    @patch('src.api.main.strategy_service')
    def test_delete_strategy_success(self, mock_service, authenticated_client_trader):
        """Test successful strategy deletion."""
        mock_service.delete_strategy = AsyncMock(return_value={
            "message": "Strategy deleted successfully",
            "strategy_id": "test-strategy"
        })

        response = authenticated_client_trader.delete("/api/strategies/test-strategy")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Strategy deleted successfully"
        assert data["strategy_id"] == "test-strategy"

    @patch('src.api.main.strategy_service')
    def test_delete_strategy_not_found(self, mock_service, authenticated_client_trader):
        """Test strategy deletion when strategy not found."""
        mock_service.delete_strategy.side_effect = StrategyOperationError("Strategy not found")

        response = authenticated_client_trader.delete("/api/strategies/nonexistent")

        assert response.status_code == 404
        data = response.json()
        assert "Strategy not found" in data["detail"]


class TestStrategyLifecycleEndpoints:
    """Test cases for strategy lifecycle management."""

    @patch('src.api.main.strategy_service')
    def test_start_strategy_success(self, mock_service, authenticated_client_trader):
        """Test successful strategy start."""
        mock_service.start_strategy = AsyncMock(return_value={
            "message": "Strategy started successfully",
            "strategy_id": "test-strategy"
        })

        action_data = {"action": "start", "confirm_live_trading": False}
        response = authenticated_client_trader.post("/api/strategies/test-strategy/start", json=action_data)

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Strategy started successfully"
        mock_service.start_strategy.assert_called_once_with("test-strategy", confirm_live_trading=False)

    @patch('src.api.main.strategy_service')
    def test_start_strategy_confirmation_required(self, mock_service, authenticated_client_trader):
        """Test strategy start requiring live trading confirmation."""
        mock_service.start_strategy.side_effect = StrategyOperationError("Live trading confirmation required")

        action_data = {"action": "start", "confirm_live_trading": False}
        response = authenticated_client_trader.post("/api/strategies/test-strategy/start", json=action_data)

        assert response.status_code == 400
        data = response.json()
        assert "confirmation" in data["detail"].lower()

    @patch('src.api.main.strategy_service')
    def test_start_strategy_not_found(self, mock_service, authenticated_client_trader):
        """Test strategy start when strategy not found."""
        mock_service.start_strategy.side_effect = StrategyOperationError("Strategy not found")

        action_data = {"action": "start", "confirm_live_trading": False}
        response = authenticated_client_trader.post("/api/strategies/nonexistent/start", json=action_data)

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    @patch('src.api.main.strategy_service')
    def test_stop_strategy_success(self, mock_service, authenticated_client_trader):
        """Test successful strategy stop."""
        mock_service.stop_strategy = AsyncMock(return_value={
            "message": "Strategy stopped successfully",
            "strategy_id": "test-strategy"
        })

        response = authenticated_client_trader.post("/api/strategies/test-strategy/stop")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Strategy stopped successfully"
        mock_service.stop_strategy.assert_called_once_with("test-strategy")

    @patch('src.api.main.strategy_service')
    def test_restart_strategy_success(self, mock_service, authenticated_client_trader):
        """Test successful strategy restart."""
        mock_service.restart_strategy = AsyncMock(return_value={
            "message": "Strategy restarted successfully",
            "strategy_id": "test-strategy"
        })

        action_data = {"action": "restart", "confirm_live_trading": True}
        response = authenticated_client_trader.post("/api/strategies/test-strategy/restart", json=action_data)

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Strategy restarted successfully"
        mock_service.restart_strategy.assert_called_once_with("test-strategy", confirm_live_trading=True)


class TestSystemMonitoringEndpoints:
    """Test cases for system monitoring endpoints."""

    @patch('src.api.main.monitoring_service')
    @patch('src.api.main.strategy_service')
    def test_get_system_status_success(self, mock_strategy_service, mock_monitoring_service, authenticated_client_admin):
        """Test successful system status retrieval."""
        # Mock strategy service status
        mock_strategy_service.get_service_status.return_value = {
            "available": True,
            "active_strategies": 2,
            "total_strategies": 5
        }

        # Mock monitoring service metrics
        mock_monitoring_service.get_comprehensive_metrics.return_value = {
            'cpu': {'usage_percent': 25.5},
            'memory': {'usage_percent': 60.2},
            'temperature': {'average_celsius': 45.0},
            'disk': {
                'partitions': {
                    '/': {'usage_percent': 75.0},
                    '/home': {'usage_percent': 50.0}
                }
            }
        }

        response = authenticated_client_admin.get("/api/system/status")

        assert response.status_code == 200
        data = response.json()
        assert data["service_name"] == "Enhanced Multi-Strategy Trading System"
        assert data["version"] == "2.0.0"
        assert data["status"] == "running"
        assert data["active_strategies"] == 2
        assert data["total_strategies"] == 5
        assert data["system_metrics"]["cpu_percent"] == 25.5
        assert data["system_metrics"]["memory_percent"] == 60.2
        assert data["system_metrics"]["temperature_c"] == 45.0
        assert data["system_metrics"]["disk_usage_percent"] == 75.0  # Max of partitions

    @patch('src.api.main.monitoring_service')
    def test_get_system_metrics_success(self, mock_service, authenticated_client_admin):
        """Test successful system metrics retrieval."""
        mock_metrics = {
            'cpu': {'usage_percent': 30.0, 'cores': 4},
            'memory': {'usage_percent': 65.0, 'total_gb': 8.0},
            'disk': {'usage_percent': 80.0, 'free_gb': 50.0}
        }
        mock_service.get_comprehensive_metrics.return_value = mock_metrics

        response = authenticated_client_admin.get("/api/monitoring/metrics")

        assert response.status_code == 200
        data = response.json()
        assert data == mock_metrics

    @patch('src.api.main.monitoring_service')
    def test_get_system_alerts_success(self, mock_service, authenticated_client_admin):
        """Test successful system alerts retrieval."""
        mock_alerts = [
            {"id": 1, "message": "High CPU usage", "severity": "warning"},
            {"id": 2, "message": "Low disk space", "severity": "error"}
        ]
        mock_service.get_alerts.return_value = mock_alerts

        response = authenticated_client_admin.get("/api/monitoring/alerts")

        assert response.status_code == 200
        data = response.json()
        assert data["alerts"] == mock_alerts
        mock_service.get_alerts.assert_called_once_with(False)  # unacknowledged_only=False

    @patch('src.api.main.monitoring_service')
    def test_get_system_alerts_unacknowledged_only(self, mock_service, authenticated_client_admin):
        """Test system alerts retrieval with unacknowledged filter."""
        mock_service.get_alerts.return_value = []

        response = authenticated_client_admin.get("/api/monitoring/alerts?unacknowledged_only=true")

        assert response.status_code == 200
        mock_service.get_alerts.assert_called_once_with(True)

    @patch('src.api.main.monitoring_service')
    def test_acknowledge_alert_success(self, mock_service, authenticated_client_trader):
        """Test successful alert acknowledgment."""
        mock_service.acknowledge_alert.return_value = True

        response = authenticated_client_trader.post("/api/monitoring/alerts/1/acknowledge")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Alert acknowledged successfully"
        mock_service.acknowledge_alert.assert_called_once_with(1)

    @patch('src.api.main.monitoring_service')
    def test_acknowledge_alert_not_found(self, mock_service, authenticated_client_trader):
        """Test alert acknowledgment when alert not found."""
        mock_service.acknowledge_alert.return_value = False

        response = authenticated_client_trader.post("/api/monitoring/alerts/999/acknowledge")

        assert response.status_code == 404
        data = response.json()
        assert "Alert not found" in data["detail"]

    @patch('src.api.main.monitoring_service')
    def test_get_performance_history_success(self, mock_service, authenticated_client_admin):
        """Test successful performance history retrieval."""
        mock_history = [
            {"timestamp": "2024-01-01T00:00:00Z", "cpu": 25.0, "memory": 60.0},
            {"timestamp": "2024-01-01T01:00:00Z", "cpu": 30.0, "memory": 65.0}
        ]
        mock_service.get_performance_history.return_value = mock_history

        response = authenticated_client_admin.get("/api/monitoring/history?hours=2")

        assert response.status_code == 200
        data = response.json()
        assert data["history"] == mock_history
        mock_service.get_performance_history.assert_called_once_with(2)

    def test_get_performance_history_invalid_hours(self, authenticated_client_admin):
        """Test performance history with invalid hours parameter."""
        response = authenticated_client_admin.get("/api/monitoring/history?hours=25")

        assert response.status_code == 400
        data = response.json()
        assert "Hours must be between 1 and 24" in data["detail"]


class TestConfigurationEndpoints:
    """Test cases for configuration management endpoints."""

    @patch('src.api.main.strategy_service')
    def test_update_strategy_parameters_success(self, mock_service, authenticated_client_trader):
        """Test successful strategy parameter update."""
        mock_result = {"message": "Parameters updated", "updated_params": {"fast_period": 15}}
        mock_service.update_strategy_parameters = AsyncMock(return_value=mock_result)

        parameters = {"fast_period": 15, "slow_period": 25}
        response = authenticated_client_trader.put("/api/strategies/test-strategy/parameters", json=parameters)

        assert response.status_code == 200
        data = response.json()
        assert data == mock_result
        mock_service.update_strategy_parameters.assert_called_once_with("test-strategy", parameters)

    @patch('src.api.main.strategy_service')
    def test_get_strategy_templates_success(self, mock_service, authenticated_client_admin):
        """Test successful strategy templates retrieval."""
        mock_templates = [
            {"name": "SMA Crossover", "config": {"type": "sma_crossover"}},
            {"name": "RSI Strategy", "config": {"type": "rsi"}}
        ]
        mock_service.get_strategy_templates.return_value = mock_templates

        response = authenticated_client_admin.get("/api/config/templates")

        assert response.status_code == 200
        data = response.json()
        assert data["templates"] == mock_templates

    @patch('src.api.main.strategy_service')
    def test_validate_configuration_valid(self, mock_service, authenticated_client_admin, sample_strategy_config):
        """Test configuration validation with valid config."""
        mock_service.validate_strategy_config.return_value = None  # No exception means valid

        response = authenticated_client_admin.post("/api/config/validate", json=sample_strategy_config)

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["errors"] == []

    @patch('src.api.main.strategy_service')
    def test_validate_configuration_invalid(self, mock_service, authenticated_client_admin, sample_strategy_config):
        """Test configuration validation with invalid config."""
        mock_service.validate_strategy_config.side_effect = StrategyValidationError("Invalid symbol")

        response = authenticated_client_admin.post("/api/config/validate", json=sample_strategy_config)

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert "Invalid symbol" in data["errors"]


if __name__ == "__main__":
    pytest.main([__file__])