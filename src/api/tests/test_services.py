#!/usr/bin/env python3
"""
Unit Tests for Application Services
----------------------------------

Tests for the application service layer including:
- WebUI App Service (database operations, user management)
- Telegram App Service (Telegram bot management)
- Strategy Management Service (strategy operations)
- System Monitoring Service (system metrics and alerts)
"""

import pytest
from unittest.mock import patch, Mock, MagicMock
from datetime import datetime, timezone
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.api.services.webui_app_service import WebUIAppService
from src.api.services.telegram_app_service import TelegramAppService
from src.api.services.monitoring_service import SystemMonitoringService
from src.api.services.strategy_service import StrategyManagementService, StrategyValidationError, StrategyOperationError


class TestWebUIAppService:
    """Test cases for WebUI Application Service."""

    def setup_method(self):
        """Set up test dependencies."""
        self.service = WebUIAppService()

    @patch('src.api.services.webui_app_service.get_database_service')
    def test_get_db_session(self, mock_get_db_service):
        """Test database session generator."""
        mock_db_service = Mock()
        mock_uow = Mock()
        mock_session = Mock()

        mock_get_db_service.return_value = mock_db_service
        # Mock the context manager properly
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = mock_uow
        mock_context_manager.__exit__.return_value = None
        mock_db_service.uow.return_value = mock_context_manager
        mock_uow.session = mock_session

        # Test the generator
        session_gen = self.service.get_db_session()
        session = next(session_gen)

        assert session == mock_session

    @patch('src.api.services.webui_app_service.get_database_service')
    @patch('src.api.services.webui_app_service.PROJECT_ROOT')
    def test_init_database_success(self, mock_project_root, mock_get_db_service):
        """Test successful database initialization."""
        # Mock project root and database directory
        mock_project_root.__truediv__.return_value.mkdir.return_value = None

        # Mock database service
        mock_db_service = Mock()
        mock_get_db_service.return_value = mock_db_service

        # Mock users service to return existing users
        with patch('src.api.services.webui_app_service.users_service') as mock_users_service:
            mock_users_service.list_telegram_users_dto.return_value = [{"id": 1}]  # Existing users

            self.service.init_database()

            mock_db_service.init_databases.assert_called_once()

    @patch('src.api.services.webui_app_service.get_database_service')
    @patch('src.api.services.webui_app_service.users_service')
    def test_create_default_users(self, mock_users_service, mock_get_db_service):
        """Test creation of default users."""
        # Mock no existing users
        mock_users_service.list_telegram_users_dto.return_value = []

        # Mock database service
        mock_db_service = Mock()
        mock_uow = Mock()
        mock_session = Mock()

        mock_get_db_service.return_value = mock_db_service
        # Mock the context manager properly
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = mock_uow
        mock_context_manager.__exit__.return_value = None
        mock_db_service.uow.return_value = mock_context_manager
        mock_uow.s = mock_session

        self.service.create_default_users()

        # Verify users were added to session
        assert mock_session.add.call_count == 3  # admin, trader, viewer
        mock_session.commit.assert_called_once()

    @patch('src.api.services.webui_app_service.get_database_service')
    def test_authenticate_user_success(self, mock_get_db_service):
        """Test successful user authentication."""
        # Mock database service and user
        mock_db_service = Mock()
        mock_uow = Mock()
        mock_session = Mock()
        mock_user = Mock()

        mock_user.is_active = True
        mock_user.verify_password.return_value = True
        mock_user.last_login = None
        mock_user.to_dict.return_value = {"id": 1, "email": "test@example.com"}

        mock_get_db_service.return_value = mock_db_service
        # Mock the context manager properly
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = mock_uow
        mock_context_manager.__exit__.return_value = None
        mock_db_service.uow.return_value = mock_context_manager
        mock_uow.s = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = mock_user

        result = self.service.authenticate_user("test@example.com", "password")

        assert result is not None
        assert result["email"] == "test@example.com"
        mock_user.verify_password.assert_called_once_with("password")
        mock_session.commit.assert_called_once()

    @patch('src.api.services.webui_app_service.get_database_service')
    def test_authenticate_user_not_found(self, mock_get_db_service):
        """Test user authentication when user not found."""
        mock_db_service = Mock()
        mock_uow = Mock()
        mock_session = Mock()

        mock_get_db_service.return_value = mock_db_service
        # Mock the context manager properly
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = mock_uow
        mock_context_manager.__exit__.return_value = None
        mock_db_service.uow.return_value = mock_context_manager
        mock_uow.s = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None

        result = self.service.authenticate_user("nonexistent@example.com", "password")

        assert result is None

    @patch('src.api.services.webui_app_service.webui_service')
    def test_log_user_action_success(self, mock_webui_service):
        """Test successful user action logging."""
        mock_webui_service.audit_log.return_value = 123

        result = self.service.log_user_action(
            user_id=1,
            action="login",
            resource_type="authentication",
            resource_id="session_123",
            details={"method": "jwt"}
        )

        assert result == 123
        mock_webui_service.audit_log.assert_called_once_with(
            user_id=1,
            action="login",
            resource_type="authentication",
            resource_id="session_123",
            details={"method": "jwt"},
            ip_address=None,
            user_agent=None
        )

    @patch('src.api.services.webui_app_service.webui_service')
    def test_get_system_config(self, mock_webui_service):
        """Test system configuration retrieval."""
        mock_config = {"key": "value", "setting": "enabled"}
        mock_webui_service.get_config.return_value = mock_config

        result = self.service.get_system_config("test_key")

        assert result == mock_config
        mock_webui_service.get_config.assert_called_once_with("test_key")

    @patch('src.api.services.webui_app_service.webui_service')
    def test_set_system_config(self, mock_webui_service):
        """Test system configuration setting."""
        mock_result = {"id": 1, "key": "test_key", "value": {"setting": "value"}}
        mock_webui_service.set_config.return_value = mock_result

        config_value = {"setting": "value"}
        result = self.service.set_system_config("test_key", config_value, "Test configuration")

        assert result == mock_result
        mock_webui_service.set_config.assert_called_once_with("test_key", config_value, "Test configuration")


class TestTelegramAppService:
    """Test cases for Telegram Application Service."""

    def setup_method(self):
        """Set up test dependencies."""
        self.service = TelegramAppService()

    @patch('src.api.services.telegram_app_service.users_service')
    def test_get_user_stats(self, mock_users_service):
        """Test getting Telegram user statistics."""
        mock_users = [
            {'verified': True, 'approved': True, 'is_admin': False},
            {'verified': True, 'approved': False, 'is_admin': False},
            {'verified': False, 'approved': False, 'is_admin': False},
            {'verified': True, 'approved': True, 'is_admin': True}
        ]
        mock_users_service.list_telegram_users_dto.return_value = mock_users

        result = self.service.get_user_stats()

        assert result['total_users'] == 4
        assert result['verified_users'] == 3
        assert result['approved_users'] == 2
        assert result['pending_approvals'] == 1  # verified but not approved
        assert result['admin_users'] == 1

    @patch('src.api.services.telegram_app_service.users_service')
    def test_get_users_list_all(self, mock_users_service):
        """Test getting all Telegram users."""
        mock_users = [
            {
                'telegram_user_id': '123456789',
                'email': 'test@example.com',
                'verified': True,
                'approved': True,
                'language': 'en',
                'is_admin': False,
                'max_alerts': 5,
                'max_schedules': 5,
                'created_at': datetime.now(timezone.utc),
                'updated_at': datetime.now(timezone.utc)
            }
        ]
        mock_users_service.list_telegram_users_dto.return_value = mock_users

        result = self.service.get_users_list()

        assert len(result) == 1
        assert result[0]['telegram_user_id'] == '123456789'
        assert result[0]['verified'] is True
        assert result[0]['approved'] is True

    @patch('src.api.services.telegram_app_service.users_service')
    def test_get_users_list_filtered(self, mock_users_service):
        """Test getting filtered Telegram users."""
        mock_users = [
            {'verified': True, 'approved': True},
            {'verified': True, 'approved': False},
            {'verified': False, 'approved': False}
        ]
        mock_users_service.list_telegram_users_dto.return_value = mock_users

        # Test verified filter
        result = self.service.get_users_list(filter_type="verified")
        assert len(result) == 2  # Only verified users

        # Test approved filter
        result = self.service.get_users_list(filter_type="approved")
        assert len(result) == 1  # Only approved users

        # Test pending filter
        result = self.service.get_users_list(filter_type="pending")
        assert len(result) == 1  # Verified but not approved

    @patch('src.api.services.telegram_app_service.users_service')
    def test_verify_user_success(self, mock_users_service):
        """Test successful user verification."""
        mock_users_service.update_telegram_profile.return_value = None

        result = self.service.verify_user("123456789")

        assert "verified successfully" in result["message"]
        mock_users_service.update_telegram_profile.assert_called_once_with("123456789", verified=True)

    @patch('src.api.services.telegram_app_service.users_service')
    def test_approve_user_success(self, mock_users_service):
        """Test successful user approval."""
        # Mock user profile with verified status
        mock_users_service.get_telegram_profile.return_value = {'verified': True}
        mock_users_service.update_telegram_profile.return_value = None

        result = self.service.approve_user("123456789")

        assert "approved successfully" in result["message"]
        mock_users_service.update_telegram_profile.assert_called_once_with("123456789", approved=True)

    @patch('src.api.services.telegram_app_service.users_service')
    def test_approve_user_not_verified(self, mock_users_service):
        """Test user approval when user is not verified."""
        # Mock user profile without verified status
        mock_users_service.get_telegram_profile.return_value = {'verified': False}

        with pytest.raises(ValueError, match="User must be verified before approval"):
            self.service.approve_user("123456789")

    @patch('src.api.services.telegram_app_service.users_service')
    def test_approve_user_not_found(self, mock_users_service):
        """Test user approval when user not found."""
        mock_users_service.get_telegram_profile.return_value = None

        with pytest.raises(ValueError, match="User not found"):
            self.service.approve_user("123456789")

    @patch('src.api.services.telegram_app_service.telegram_service')
    def test_get_alert_stats(self, mock_telegram_service):
        """Test getting alert statistics."""
        mock_alerts = [
            {'id': 1, 'active': True},
            {'id': 2, 'active': True},
            {'id': 3, 'active': False}
        ]
        mock_telegram_service.list_active_alerts.return_value = mock_alerts

        result = self.service.get_alert_stats()

        assert result['total_alerts'] == 3
        assert result['active_alerts'] == 3  # Only active alerts returned by service

    @patch('src.api.services.telegram_app_service.telegram_service')
    def test_send_broadcast_success(self, mock_telegram_service):
        """Test successful broadcast sending."""
        mock_users = [
            {'approved': True},
            {'approved': True},
            {'approved': False}  # This user won't receive broadcast
        ]
        mock_telegram_service.list_users.return_value = mock_users
        mock_telegram_service.log_broadcast.return_value = 123

        result = self.service.send_broadcast("Test message")

        assert result['total_recipients'] == 2  # Only approved users
        assert result['successful_deliveries'] == 2
        assert result['failed_deliveries'] == 0
        assert result['broadcast_id'] == '123'
        mock_telegram_service.log_broadcast.assert_called_once()


class TestStrategyManagementService:
    """Test cases for Strategy Management Service."""

    def setup_method(self):
        """Set up test dependencies."""
        self.mock_strategy_manager = Mock()
        self.service = StrategyManagementService(self.mock_strategy_manager)
        # Force the service to be available for testing
        self.service.is_available = True

    def test_init_with_none_manager(self):
        """Test initialization with None strategy manager."""
        service = StrategyManagementService(None)
        assert service.strategy_manager is None

    def test_get_all_strategies_status_with_manager(self):
        """Test getting all strategies status with manager available."""
        mock_strategies = [
            {"id": "strategy-1", "status": "running"},
            {"id": "strategy-2", "status": "stopped"}
        ]
        self.mock_strategy_manager.get_all_status.return_value = mock_strategies

        result = self.service.get_all_strategies_status()

        assert result == mock_strategies
        self.mock_strategy_manager.get_all_status.assert_called_once()

    def test_get_all_strategies_status_without_manager(self):
        """Test getting all strategies status without manager."""
        service = StrategyManagementService(None)

        result = service.get_all_strategies_status()

        assert result == []

    @pytest.mark.asyncio
    @patch('src.api.services.strategy_service.StrategyInstance')
    async def test_create_strategy_success(self, mock_strategy_instance):
        """Test successful strategy creation."""
        config = {
            "id": "test-strategy",
            "name": "Test Strategy",
            "symbol": "BTCUSDT",
            "broker": {"type": "paper", "trading_mode": "paper"},
            "strategy": {"type": "sma"}
        }
        expected_result = {
            "strategy_id": "test-strategy",
            "name": "Test Strategy",
            "status": "created",
            "message": "Strategy created successfully"
        }

        # Mock strategy_instances as an empty dict to simulate no existing strategies
        self.mock_strategy_manager.strategy_instances = {}

        # Mock StrategyInstance creation
        mock_instance = Mock()
        mock_strategy_instance.return_value = mock_instance

        result = await self.service.create_strategy(config)

        assert result == expected_result
        # Verify that the strategy was added to the manager's instances
        assert "test-strategy" in self.mock_strategy_manager.strategy_instances

    @pytest.mark.asyncio
    async def test_create_strategy_without_manager(self):
        """Test strategy creation without manager."""
        service = StrategyManagementService(None)
        config = {"id": "test-strategy"}

        with pytest.raises(StrategyOperationError, match="Strategy manager not available"):
            await service.create_strategy(config)

    def test_validate_strategy_config_success(self):
        """Test successful strategy configuration validation."""
        config = {"id": "test-strategy", "symbol": "BTCUSDT"}

        # Should not raise exception for valid config
        self.service.validate_strategy_config(config)

    def test_validate_strategy_config_missing_id(self):
        """Test strategy configuration validation with missing ID."""
        config = {"symbol": "BTCUSDT"}  # Missing required 'id' field

        with pytest.raises(StrategyValidationError, match="Strategy ID is required"):
            self.service.validate_strategy_config(config)

    def test_validate_strategy_config_missing_symbol(self):
        """Test strategy configuration validation with missing symbol."""
        config = {"id": "test-strategy"}  # Missing required 'symbol' field

        with pytest.raises(StrategyValidationError, match="Trading symbol is required"):
            self.service.validate_strategy_config(config)

    def test_get_service_status_with_manager(self):
        """Test getting service status with manager available."""
        mock_strategies = [
            {"status": "running"},
            {"status": "running"},
            {"status": "stopped"}
        ]
        self.mock_strategy_manager.get_all_strategies_status.return_value = mock_strategies

        result = self.service.get_service_status()

        assert result["available"] is True
        assert result["total_strategies"] == 3
        assert result["active_strategies"] == 2  # Only running strategies

    def test_get_service_status_without_manager(self):
        """Test getting service status without manager."""
        service = StrategyManagementService(None)

        result = service.get_service_status()

        assert result["available"] is False
        assert result["total_strategies"] == 0
        assert result["active_strategies"] == 0

    def test_get_strategy_templates(self):
        """Test getting strategy templates."""
        expected_templates = [
            {"name": "SMA Crossover", "type": "sma_crossover"},
            {"name": "RSI Strategy", "type": "rsi"}
        ]

        result = self.service.get_strategy_templates()

        # Should return default templates
        assert isinstance(result, list)
        assert len(result) > 0


class TestSystemMonitoringService:
    """Test cases for System Monitoring Service."""

    def setup_method(self):
        """Set up test dependencies."""
        self.service = SystemMonitoringService()

    @patch('src.api.services.monitoring_service.psutil')
    def test_get_comprehensive_metrics_success(self, mock_psutil):
        """Test successful comprehensive metrics retrieval."""
        # Mock psutil responses
        mock_psutil.cpu_percent.return_value = 25.5
        mock_psutil.virtual_memory.return_value.percent = 60.2
        mock_psutil.disk_usage.return_value.percent = 75.0

        # Mock temperature (may not be available on all systems)
        mock_psutil.sensors_temperatures.return_value = {
            'cpu_thermal': [Mock(current=45.0)]
        }

        result = self.service.get_comprehensive_metrics()

        assert result['cpu']['usage_percent'] == 25.5
        assert result['memory']['usage_percent'] == 60.2
        assert 'disk' in result
        assert 'temperature' in result

    def test_get_alerts_empty(self):
        """Test getting alerts when none exist."""
        result = self.service.get_alerts()

        assert result == []

    def test_get_alerts_unacknowledged_only(self):
        """Test getting only unacknowledged alerts."""
        result = self.service.get_alerts(unacknowledged_only=True)

        assert result == []

    def test_acknowledge_alert_invalid_index(self):
        """Test acknowledging alert with invalid index."""
        result = self.service.acknowledge_alert(999)

        assert result is False

    def test_get_performance_history_empty(self):
        """Test getting performance history when none exists."""
        result = self.service.get_performance_history(1)

        assert result == []

    def test_get_performance_history_invalid_hours(self):
        """Test getting performance history with invalid hours."""
        # Should handle invalid input gracefully
        result = self.service.get_performance_history(0)
        assert result == []

        result = self.service.get_performance_history(-1)
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__])