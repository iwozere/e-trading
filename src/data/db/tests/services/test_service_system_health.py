"""
Comprehensive tests for SystemHealthService.

Tests cover:
- System health status updates
- Health retrieval operations
- System overview and statistics
- Channel health monitoring (backward compatibility)
- Utility methods
"""
import pytest
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from src.data.db.services.system_health_service import SystemHealthService
from src.data.db.models.model_system_health import SystemHealthStatus


class TestSystemHealthServiceBasicOperations:
    """Tests for basic system health operations."""

    def test_update_system_health(self, mock_database_service, db_session):
        """Test updating system health status."""
        service = SystemHealthService(db_service=mock_database_service)

        health = service.update_system_health(
            system="telegram_bot",
            status=SystemHealthStatus.HEALTHY,
            response_time_ms=150
        )

        assert health is not None
        assert health.system == "telegram_bot"
        assert health.status == SystemHealthStatus.HEALTHY.value

    def test_update_system_health_with_component(self, mock_database_service, db_session):
        """Test updating system health with component."""
        service = SystemHealthService(db_service=mock_database_service)

        health = service.update_system_health(
            system="notification",
            component="email",
            status=SystemHealthStatus.HEALTHY,
            response_time_ms=200
        )

        assert health is not None
        assert health.system == "notification"
        assert health.component == "email"

    def test_update_system_health_degraded(self, mock_database_service, db_session):
        """Test updating system health to degraded."""
        service = SystemHealthService(db_service=mock_database_service)

        health = service.update_system_health(
            system="api_service",
            status=SystemHealthStatus.DEGRADED,
            response_time_ms=2000,
            error_message="High latency detected"
        )

        assert health is not None
        assert health.status == SystemHealthStatus.DEGRADED.value
        assert health.error_message == "High latency detected"

    def test_update_system_health_with_metadata(self, mock_database_service, db_session):
        """Test updating system health with metadata."""
        service = SystemHealthService(db_service=mock_database_service)

        metadata = {"version": "1.0.0", "uptime_seconds": 3600}
        health = service.update_system_health(
            system="trading_bot",
            status=SystemHealthStatus.HEALTHY,
            metadata=metadata
        )

        assert health is not None
        assert health.metadata is not None


class TestSystemHealthServiceRetrieval:
    """Tests for health retrieval operations."""

    def test_get_system_health(self, mock_database_service, db_session):
        """Test getting system health."""
        service = SystemHealthService(db_service=mock_database_service)

        # Create health record first
        service.update_system_health(
            system="database",
            status=SystemHealthStatus.HEALTHY
        )

        # Retrieve it
        health_data = service.get_system_health(system="database")

        assert health_data is not None
        assert health_data["system"] == "database"
        assert health_data["status"] == SystemHealthStatus.HEALTHY.value

    def test_get_system_health_with_component(self, mock_database_service, db_session):
        """Test getting system health with component."""
        service = SystemHealthService(db_service=mock_database_service)

        # Create health record
        service.update_system_health(
            system="notification",
            component="telegram",
            status=SystemHealthStatus.HEALTHY
        )

        # Retrieve it
        health_data = service.get_system_health(
            system="notification",
            component="telegram"
        )

        assert health_data is not None
        assert health_data["system"] == "notification"
        assert health_data["component"] == "telegram"

    def test_get_system_health_not_found(self, mock_database_service, db_session):
        """Test getting non-existent system health returns None."""
        service = SystemHealthService(db_service=mock_database_service)

        health_data = service.get_system_health(system="nonexistent_system")

        assert health_data is None

    def test_get_all_systems_health(self, mock_database_service, db_session):
        """Test getting health for all systems."""
        service = SystemHealthService(db_service=mock_database_service)

        # Create multiple health records
        service.update_system_health(system="telegram_bot", status=SystemHealthStatus.HEALTHY)
        service.update_system_health(system="api_service", status=SystemHealthStatus.DEGRADED)
        service.update_system_health(system="web_ui", status=SystemHealthStatus.DOWN)

        # Get all
        all_health = service.get_all_systems_health()

        assert isinstance(all_health, list)
        assert len(all_health) >= 3

    def test_get_unhealthy_systems(self, mock_database_service, db_session):
        """Test getting only unhealthy systems."""
        service = SystemHealthService(db_service=mock_database_service)

        # Create mix of healthy and unhealthy
        service.update_system_health(system="healthy_sys", status=SystemHealthStatus.HEALTHY)
        service.update_system_health(system="degraded_sys", status=SystemHealthStatus.DEGRADED)
        service.update_system_health(system="down_sys", status=SystemHealthStatus.DOWN)

        # Get unhealthy
        unhealthy = service.get_unhealthy_systems()

        assert isinstance(unhealthy, list)
        # Should include degraded and down systems
        unhealthy_names = [s["system"] for s in unhealthy]
        assert "degraded_sys" in unhealthy_names or "down_sys" in unhealthy_names


class TestSystemHealthServiceOverview:
    """Tests for system overview and statistics."""

    def test_get_systems_overview(self, mock_database_service, db_session):
        """Test getting systems overview."""
        service = SystemHealthService(db_service=mock_database_service)

        # Create some health records
        service.update_system_health(system="sys1", status=SystemHealthStatus.HEALTHY)
        service.update_system_health(system="sys2", status=SystemHealthStatus.DEGRADED)

        # Get overview
        overview = service.get_systems_overview()

        assert isinstance(overview, dict)
        assert "overall_status" in overview
        assert "timestamp" in overview
        assert "systems_overview" in overview
        assert "statistics" in overview


class TestSystemHealthServiceChannelCompatibility:
    """Tests for notification channel health (backward compatibility)."""

    def test_update_notification_channel_health(self, mock_database_service, db_session):
        """Test updating notification channel health."""
        service = SystemHealthService(db_service=mock_database_service)

        health = service.update_notification_channel_health(
            channel="email",
            status=SystemHealthStatus.HEALTHY,
            response_time_ms=100
        )

        assert health is not None
        assert health.system == "notification"
        assert health.component == "email"

    def test_get_notification_channels_health(self, mock_database_service, db_session):
        """Test getting all notification channel health."""
        service = SystemHealthService(db_service=mock_database_service)

        # Create channel health records
        service.update_notification_channel_health(channel="email", status=SystemHealthStatus.HEALTHY)
        service.update_notification_channel_health(channel="telegram", status=SystemHealthStatus.HEALTHY)

        # Get all channels
        channels = service.get_notification_channels_health()

        assert isinstance(channels, list)
        assert len(channels) >= 2

    def test_get_notification_channel_health(self, mock_database_service, db_session):
        """Test getting specific notification channel health."""
        service = SystemHealthService(db_service=mock_database_service)

        # Create channel health
        service.update_notification_channel_health(
            channel="slack",
            status=SystemHealthStatus.HEALTHY
        )

        # Get it
        channel_health = service.get_notification_channel_health(channel="slack")

        assert channel_health is not None
        assert channel_health["channel"] == "slack"


class TestSystemHealthServiceSpecificSystems:
    """Tests for system-specific health update methods."""

    def test_update_telegram_bot_health(self, mock_database_service, db_session):
        """Test updating Telegram bot health."""
        service = SystemHealthService(db_service=mock_database_service)

        health = service.update_telegram_bot_health(
            status=SystemHealthStatus.HEALTHY,
            response_time_ms=100
        )

        assert health is not None
        assert health.system == "telegram_bot"

    def test_update_api_service_health(self, mock_database_service, db_session):
        """Test updating API service health."""
        service = SystemHealthService(db_service=mock_database_service)

        health = service.update_api_service_health(
            status=SystemHealthStatus.HEALTHY
        )

        assert health is not None
        assert health.system == "api_service"

    def test_update_web_ui_health(self, mock_database_service, db_session):
        """Test updating Web UI health."""
        service = SystemHealthService(db_service=mock_database_service)

        health = service.update_web_ui_health(
            status=SystemHealthStatus.HEALTHY
        )

        assert health is not None
        assert health.system == "web_ui"

    def test_update_trading_bot_health(self, mock_database_service, db_session):
        """Test updating trading bot health."""
        service = SystemHealthService(db_service=mock_database_service)

        health = service.update_trading_bot_health(
            status=SystemHealthStatus.DEGRADED,
            error_message="Connection issues"
        )

        assert health is not None
        assert health.system == "trading_bot"
        assert health.status == SystemHealthStatus.DEGRADED.value

    def test_update_database_health(self, mock_database_service, db_session):
        """Test updating database health."""
        service = SystemHealthService(db_service=mock_database_service)

        health = service.update_database_health(
            status=SystemHealthStatus.HEALTHY,
            response_time_ms=50
        )

        assert health is not None
        assert health.system == "database"


class TestSystemHealthServiceUtilities:
    """Tests for utility methods."""

    def test_cleanup_stale_records(self, mock_database_service, db_session):
        """Test cleaning up stale records."""
        service = SystemHealthService(db_service=mock_database_service)

        # Create some records
        service.update_system_health(system="test_sys", status=SystemHealthStatus.HEALTHY)

        # Clean up
        deleted_count = service.cleanup_stale_records(stale_threshold_hours=24)

        assert isinstance(deleted_count, int)
        assert deleted_count >= 0

    def test_delete_system_health(self, mock_database_service, db_session):
        """Test deleting system health record."""
        service = SystemHealthService(db_service=mock_database_service)

        # Create a record
        service.update_system_health(system="temp_sys", status=SystemHealthStatus.HEALTHY)

        # Delete it
        success = service.delete_system_health(system="temp_sys")

        assert isinstance(success, bool)


class TestSystemHealthServiceIntegration:
    """Integration tests for system health workflows."""

    def test_full_health_monitoring_lifecycle(self, mock_database_service, db_session):
        """Test complete health monitoring lifecycle."""
        service = SystemHealthService(db_service=mock_database_service)

        # 1. Update multiple system healths
        service.update_telegram_bot_health(status=SystemHealthStatus.HEALTHY)
        service.update_api_service_health(status=SystemHealthStatus.HEALTHY)
        service.update_trading_bot_health(
            status=SystemHealthStatus.DEGRADED,
            error_message="Minor issues"
        )

        # 2. Get overview
        overview = service.get_systems_overview()
        assert overview["overall_status"] in ["HEALTHY", "DEGRADED"]

        # 3. Get unhealthy systems
        unhealthy = service.get_unhealthy_systems()
        assert any(s["system"] == "trading_bot" for s in unhealthy)

        # 4. Get all systems
        all_systems = service.get_all_systems_health()
        assert len(all_systems) >= 3

    def test_notification_channel_monitoring(self, mock_database_service, db_session):
        """Test notification channel monitoring workflow."""
        service = SystemHealthService(db_service=mock_database_service)

        # Update multiple channels
        service.update_notification_channel_health("email", SystemHealthStatus.HEALTHY)
        service.update_notification_channel_health("telegram", SystemHealthStatus.HEALTHY)
        service.update_notification_channel_health(
            "slack",
            SystemHealthStatus.DOWN,
            error_message="API unreachable"
        )

        # Get all channels
        channels = service.get_notification_channels_health()
        assert len(channels) >= 3

        # Get specific channel
        slack_health = service.get_notification_channel_health("slack")
        assert slack_health["status"] == SystemHealthStatus.DOWN.value
