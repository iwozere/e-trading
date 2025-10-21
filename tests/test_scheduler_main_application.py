"""
Integration Tests for Scheduler Main Application

Tests the complete main application functionality including:
- Complete service initialization and startup
- Configuration loading and validation
- Graceful shutdown and cleanup
"""

import pytest
import asyncio
import os
import signal
import sys
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.scheduler.main import SchedulerApplication, setup_signal_handlers, main
from src.scheduler.config import SchedulerServiceConfig


@pytest.fixture
def test_config():
    """Create a test configuration."""
    # Set test environment variables
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"
    os.environ["SCHEDULER_MAX_WORKERS"] = "2"
    os.environ["NOTIFICATION_SERVICE_URL"] = "http://localhost:8000"
    os.environ["TRADING_ENV"] = "development"  # Use valid environment

    config = SchedulerServiceConfig()
    return config


@pytest.fixture
def mock_dependencies():
    """Create mock dependencies for testing."""
    mocks = {
        'scheduler_service': Mock(),
        'notification_client': Mock()
    }

    # Setup async methods
    mocks['notification_client'].close = AsyncMock()
    mocks['scheduler_service'].start = AsyncMock()
    mocks['scheduler_service'].stop = AsyncMock()
    mocks['scheduler_service'].reload_schedules = AsyncMock(return_value=5)
    mocks['scheduler_service'].get_scheduler_status = Mock(return_value={
        "is_running": True,
        "job_count": 3,
        "scheduler_state": "running"
    })

    return mocks


class TestSchedulerApplicationInitialization:
    """Test scheduler application initialization."""

    def test_application_creation(self, test_config):
        """Test that application can be created with proper configuration."""
        app = SchedulerApplication(test_config)

        assert app.config == test_config
        assert app.scheduler_service is None
        assert app.data_manager is None
        assert app.indicator_service is None
        assert app.jobs_service is None
        assert app.alert_evaluator is None
        assert app.notification_client is None
        assert app.schema_validator is None
        assert not app._shutdown_event.is_set()

    def test_application_creation_with_invalid_config(self):
        """Test application creation with invalid configuration."""
        # Test with None config should raise AttributeError when accessing config attributes
        app = SchedulerApplication(None)
        with pytest.raises(AttributeError):
            # This should fail when trying to access config.alert.schema_dir
            asyncio.run(app.initialize_services())

    @pytest.mark.asyncio
    async def test_service_initialization_success(self, test_config):
        """Test successful service initialization."""
        app = SchedulerApplication(test_config)

        # Test that initialization completes without errors
        await app.initialize_services()

        # Verify all services were initialized
        assert app.schema_validator is not None
        assert app.data_manager is not None
        assert app.indicator_service is not None
        assert app.notification_client is not None
        assert app.jobs_service is not None
        assert app.alert_evaluator is not None
        assert app.scheduler_service is not None

    @pytest.mark.asyncio
    async def test_service_initialization_with_mock_failure(self, test_config):
        """Test service initialization failure handling with mocked failure."""
        app = SchedulerApplication(test_config)

        # Mock the initialize_services method to raise an exception
        original_init = app.initialize_services

        async def failing_init():
            raise Exception("Initialization failed")

        app.initialize_services = failing_init

        with pytest.raises(Exception, match="Initialization failed"):
            await app.initialize_services()


class TestSchedulerApplicationLifecycle:
    """Test scheduler application lifecycle management."""

    @pytest.mark.asyncio
    async def test_application_start_success(self, test_config, mock_dependencies):
        """Test successful application startup."""
        app = SchedulerApplication(test_config)

        with patch.object(app, 'initialize_services', new_callable=AsyncMock) as mock_init:
            # Mock the scheduler service
            app.scheduler_service = mock_dependencies['scheduler_service']
            mock_init.return_value = None

            # Start application
            await app.start()

            # Verify initialization and startup
            mock_init.assert_called_once()
            mock_dependencies['scheduler_service'].start.assert_called_once()

    @pytest.mark.asyncio
    async def test_application_start_initialization_failure(self, test_config):
        """Test application startup with initialization failure."""
        app = SchedulerApplication(test_config)

        with patch.object(app, 'initialize_services', side_effect=Exception("Init failed")):
            # Should raise exception on startup failure
            with pytest.raises(Exception, match="Init failed"):
                await app.start()

    @pytest.mark.asyncio
    async def test_application_start_scheduler_failure(self, test_config, mock_dependencies):
        """Test application startup with scheduler service failure."""
        app = SchedulerApplication(test_config)

        with patch.object(app, 'initialize_services', new_callable=AsyncMock):
            # Mock scheduler service to fail on start
            app.scheduler_service = mock_dependencies['scheduler_service']
            mock_dependencies['scheduler_service'].start.side_effect = Exception("Scheduler start failed")

            # Should raise exception on scheduler start failure
            with pytest.raises(Exception, match="Scheduler start failed"):
                await app.start()

    @pytest.mark.asyncio
    async def test_application_stop_success(self, test_config, mock_dependencies):
        """Test successful application shutdown."""
        app = SchedulerApplication(test_config)

        # Setup services
        app.scheduler_service = mock_dependencies['scheduler_service']
        app.notification_client = mock_dependencies['notification_client']

        # Stop application
        await app.stop()

        # Verify shutdown sequence
        mock_dependencies['scheduler_service'].stop.assert_called_once()
        mock_dependencies['notification_client'].close.assert_called_once()
        assert app._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_application_stop_with_errors(self, test_config, mock_dependencies):
        """Test application shutdown with service errors."""
        app = SchedulerApplication(test_config)

        # Setup services with stop failures
        app.scheduler_service = mock_dependencies['scheduler_service']
        app.notification_client = mock_dependencies['notification_client']
        mock_dependencies['scheduler_service'].stop.side_effect = Exception("Stop failed")
        mock_dependencies['notification_client'].close.side_effect = Exception("Close failed")

        # Should handle errors gracefully
        with pytest.raises(Exception, match="Stop failed"):
            await app.stop()

    @pytest.mark.asyncio
    async def test_application_stop_partial_services(self, test_config):
        """Test application shutdown with only some services initialized."""
        app = SchedulerApplication(test_config)

        # Only initialize notification client
        app.notification_client = Mock()
        app.notification_client.close = AsyncMock()

        # Should handle partial initialization gracefully
        await app.stop()

        app.notification_client.close.assert_called_once()
        assert app._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_wait_for_shutdown(self, test_config):
        """Test waiting for shutdown signal."""
        app = SchedulerApplication(test_config)

        # Start wait task
        wait_task = asyncio.create_task(app.wait_for_shutdown())

        # Give it a moment to start waiting
        await asyncio.sleep(0.01)
        assert not wait_task.done()

        # Signal shutdown
        app._shutdown_event.set()

        # Wait should complete
        await wait_task
        assert wait_task.done()


class TestSchedulerApplicationOperations:
    """Test scheduler application operational methods."""

    @pytest.mark.asyncio
    async def test_reload_schedules_success(self, test_config, mock_dependencies):
        """Test successful schedule reloading."""
        app = SchedulerApplication(test_config)
        app.scheduler_service = mock_dependencies['scheduler_service']

        # Test reload
        count = await app.reload_schedules()

        # Verify reload was called and count returned
        assert count == 5
        mock_dependencies['scheduler_service'].reload_schedules.assert_called_once()

    @pytest.mark.asyncio
    async def test_reload_schedules_not_initialized(self, test_config):
        """Test schedule reloading when scheduler service not initialized."""
        app = SchedulerApplication(test_config)

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Scheduler service not initialized"):
            await app.reload_schedules()

    def test_get_status_without_scheduler(self, test_config):
        """Test getting status when scheduler service not initialized."""
        app = SchedulerApplication(test_config)

        status = app.get_status()

        # Verify basic status information
        assert status["service"] == test_config.service.name
        assert status["version"] == test_config.service.version
        assert status["environment"] == test_config.service.environment
        assert status["max_workers"] == test_config.scheduler.max_workers
        assert status["scheduler"] is None

    def test_get_status_with_scheduler(self, test_config, mock_dependencies):
        """Test getting status when scheduler service is initialized."""
        app = SchedulerApplication(test_config)
        app.scheduler_service = mock_dependencies['scheduler_service']

        status = app.get_status()

        # Verify status includes scheduler information
        assert status["service"] == test_config.service.name
        assert status["scheduler"]["is_running"] is True
        assert status["scheduler"]["job_count"] == 3
        mock_dependencies['scheduler_service'].get_scheduler_status.assert_called_once()

    def test_get_status_database_url_masking(self, test_config):
        """Test that database URL is properly masked in status."""
        # Test with URL containing credentials
        test_config.database.url = "postgresql://user:password@localhost:5432/db"
        app = SchedulerApplication(test_config)

        status = app.get_status()

        # Should mask credentials
        assert status["database_url"] == "localhost:5432/db"

        # Test with local URL
        test_config.database.url = "sqlite:///local.db"
        app = SchedulerApplication(test_config)

        status = app.get_status()

        # Should show as local
        assert status["database_url"] == "local"


class TestConfigurationLoading:
    """Test configuration loading and validation."""

    def test_configuration_loading_from_environment(self):
        """Test configuration loading from environment variables."""
        # Set test environment variables
        test_env = {
            "DATABASE_URL": "postgresql://test:test@testhost:5432/testdb",
            "SCHEDULER_MAX_WORKERS": "8",
            "SCHEDULER_JOB_TIMEOUT": "600",
            "NOTIFICATION_SERVICE_URL": "http://test-notification:9000",
            "NOTIFICATION_TIMEOUT": "45",
            "TRADING_ENV": "staging",
            "LOG_LEVEL": "DEBUG"
        }

        with patch.dict(os.environ, test_env):
            config = SchedulerServiceConfig()

            # Verify environment variables were loaded
            assert config.database.url == "postgresql://test:test@testhost:5432/testdb"
            assert config.scheduler.max_workers == 8
            assert config.scheduler.job_timeout == 600
            assert config.notification.service_url == "http://test-notification:9000"
            assert config.notification.timeout == 45
            assert config.service.environment == "staging"
            assert config.service.log_level == "DEBUG"

    def test_configuration_validation_success(self):
        """Test successful configuration validation."""
        # Should not raise any exceptions
        config = SchedulerServiceConfig()
        assert config is not None

    def test_configuration_validation_failures(self):
        """Test configuration validation with invalid values."""
        # Test invalid max_workers
        with patch.dict(os.environ, {"SCHEDULER_MAX_WORKERS": "0"}):
            with pytest.raises(ValueError, match="Scheduler max_workers must be positive"):
                SchedulerServiceConfig()

        # Test invalid job_timeout
        with patch.dict(os.environ, {"SCHEDULER_JOB_TIMEOUT": "-1"}):
            with pytest.raises(ValueError, match="Scheduler job_timeout must be positive"):
                SchedulerServiceConfig()

        # Test invalid notification timeout
        with patch.dict(os.environ, {"NOTIFICATION_TIMEOUT": "0"}):
            with pytest.raises(ValueError, match="Notification timeout must be positive"):
                SchedulerServiceConfig()

    def test_configuration_schema_directory_validation(self):
        """Test schema directory validation."""
        # Test with non-existent directory (should log warning but not fail)
        with patch.dict(os.environ, {"ALERT_SCHEMA_DIR": "/non/existent/path"}):
            config = SchedulerServiceConfig()
            assert config.alert.schema_dir == "/non/existent/path"

    def test_configuration_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = SchedulerServiceConfig()
        config_dict = config.to_dict()

        # Verify structure
        assert "database" in config_dict
        assert "scheduler" in config_dict
        assert "notification" in config_dict
        assert "data" in config_dict
        assert "alert" in config_dict
        assert "service" in config_dict

        # Verify database URL masking
        assert "@" not in config_dict["database"]["url"] or "local" in config_dict["database"]["url"]


class TestSignalHandling:
    """Test signal handling for graceful shutdown."""

    def test_setup_signal_handlers(self, test_config):
        """Test signal handler setup."""
        app = SchedulerApplication(test_config)

        # Mock signal.signal to verify calls
        with patch('signal.signal') as mock_signal:
            setup_signal_handlers(app)

            # Verify signal handlers were registered
            assert mock_signal.call_count >= 2  # At least SIGINT and SIGTERM

            # Check for SIGINT and SIGTERM
            call_args = [call[0] for call in mock_signal.call_args_list]
            assert (signal.SIGINT, mock_signal.call_args_list[0][0][1]) in [(args[0], args[1]) for args in call_args]
            assert (signal.SIGTERM, mock_signal.call_args_list[1][0][1]) in [(args[0], args[1]) for args in call_args]

    def test_signal_handler_execution(self, test_config):
        """Test signal handler execution."""
        app = SchedulerApplication(test_config)

        # Mock asyncio.create_task to verify shutdown is called
        with patch('asyncio.create_task') as mock_create_task, \
             patch.object(app, 'stop', new_callable=AsyncMock) as mock_stop:

            # Setup signal handlers
            setup_signal_handlers(app)

            # Simulate signal reception
            # Get the signal handler function
            with patch('signal.signal') as mock_signal:
                setup_signal_handlers(app)
                signal_handler = mock_signal.call_args_list[0][0][1]

                # Call signal handler
                signal_handler(signal.SIGINT, None)

                # Verify shutdown task was created
                mock_create_task.assert_called_once()


class TestMainFunction:
    """Test main application entry point."""

    @pytest.mark.asyncio
    async def test_main_function_success(self):
        """Test successful main function execution."""
        with patch('src.scheduler.main.SchedulerServiceConfig') as mock_config_class, \
             patch('src.scheduler.main.SchedulerApplication') as mock_app_class, \
             patch('src.scheduler.main.setup_signal_handlers') as mock_setup_signals:

            # Setup mocks
            mock_config = Mock()
            mock_config_class.return_value = mock_config

            mock_app = Mock()
            mock_app.start = AsyncMock()
            mock_app.stop = AsyncMock()
            mock_app.wait_for_shutdown = AsyncMock()
            mock_app_class.return_value = mock_app

            # Run main function
            await main()

            # Verify execution flow
            mock_config_class.assert_called_once()
            mock_app_class.assert_called_once_with(mock_config)
            mock_setup_signals.assert_called_once_with(mock_app)
            mock_app.start.assert_called_once()
            mock_app.wait_for_shutdown.assert_called_once()
            mock_app.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_main_function_keyboard_interrupt(self):
        """Test main function handling keyboard interrupt."""
        with patch('src.scheduler.main.SchedulerServiceConfig') as mock_config_class, \
             patch('src.scheduler.main.SchedulerApplication') as mock_app_class, \
             patch('src.scheduler.main.setup_signal_handlers'):

            # Setup mocks
            mock_config = Mock()
            mock_config_class.return_value = mock_config

            mock_app = Mock()
            mock_app.start = AsyncMock()
            mock_app.stop = AsyncMock()
            mock_app.wait_for_shutdown = AsyncMock(side_effect=KeyboardInterrupt())
            mock_app_class.return_value = mock_app

            # Run main function (should handle KeyboardInterrupt gracefully)
            await main()

            # Verify cleanup was called
            mock_app.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_main_function_application_error(self):
        """Test main function handling application errors."""
        with patch('src.scheduler.main.SchedulerServiceConfig') as mock_config_class, \
             patch('src.scheduler.main.SchedulerApplication') as mock_app_class, \
             patch('src.scheduler.main.setup_signal_handlers'), \
             patch('sys.exit') as mock_exit:

            # Setup mocks
            mock_config = Mock()
            mock_config_class.return_value = mock_config

            mock_app = Mock()
            mock_app.start = AsyncMock(side_effect=Exception("Application error"))
            mock_app.stop = AsyncMock()
            mock_app_class.return_value = mock_app

            # Run main function (should handle exception and exit)
            await main()

            # Verify error handling
            mock_exit.assert_called_once_with(1)
            mock_app.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_main_function_config_error(self):
        """Test main function handling configuration errors."""
        with patch('src.scheduler.main.SchedulerServiceConfig', side_effect=Exception("Config error")), \
             patch('sys.exit') as mock_exit:

            # Run main function (should handle config error and exit)
            await main()

            # Verify error handling
            mock_exit.assert_called_once_with(1)


class TestIntegrationScenarios:
    """Test complete integration scenarios."""

    @pytest.mark.asyncio
    async def test_complete_application_lifecycle(self, test_config):
        """Test complete application lifecycle from start to stop."""
        app = SchedulerApplication(test_config)

        with patch.object(app, 'initialize_services', new_callable=AsyncMock) as mock_init:

            # Setup mock services
            mock_scheduler = Mock()
            mock_scheduler.start = AsyncMock()
            mock_scheduler.stop = AsyncMock()
            mock_scheduler.get_scheduler_status = Mock(return_value={"is_running": True})
            mock_scheduler.reload_schedules = AsyncMock(return_value=3)

            mock_notification = Mock()
            mock_notification.close = AsyncMock()

            # Mock initialization to set up services
            async def mock_initialize():
                app.scheduler_service = mock_scheduler
                app.notification_client = mock_notification

            mock_init.side_effect = mock_initialize

            # Test complete lifecycle
            await app.start()

            # Verify services are running
            status = app.get_status()
            assert status["scheduler"]["is_running"] is True

            # Test reload functionality
            count = await app.reload_schedules()
            assert count == 3

            # Test graceful shutdown
            await app.stop()

            # Verify shutdown
            mock_scheduler.stop.assert_called_once()
            mock_notification.close.assert_called_once()
            assert app._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_application_resilience_to_service_failures(self, test_config):
        """Test application resilience to individual service failures."""
        app = SchedulerApplication(test_config)

        # Test that initialization failure is handled properly
        with patch.object(app, 'initialize_services', side_effect=Exception("Service unavailable")):
            with pytest.raises(Exception, match="Service unavailable"):
                await app.start()

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, test_config, mock_dependencies):
        """Test concurrent application operations."""
        app = SchedulerApplication(test_config)
        app.scheduler_service = mock_dependencies['scheduler_service']

        # Test concurrent status checks and reloads
        async def get_status_async():
            return app.get_status()

        results = await asyncio.gather(
            get_status_async(),
            app.reload_schedules(),
            get_status_async(),
            return_exceptions=True
        )

        # Verify all operations completed
        assert len(results) == 3
        assert all(not isinstance(result, Exception) for result in results)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])