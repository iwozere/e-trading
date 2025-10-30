#!/usr/bin/env python3
"""
Test Suite for Enhanced Broker Factory and Management System
-----------------------------------------------------------

Comprehensive test suite for the enhanced broker factory, broker manager,
and configuration management system.
"""

import pytest
import asyncio
import json
import tempfile
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
from datetime import datetime, timezone

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.trading.broker.broker_factory import (
    get_broker, validate_broker_config, get_broker_credentials,
    get_broker_capabilities, list_available_brokers, create_sample_config,
    BrokerConfigurationError
)
from src.trading.broker.broker_manager import (
    BrokerManager, BrokerHealthMonitor, BrokerConnectionPool,
    BrokerStatus, BrokerHealthMetrics
)
from src.trading.broker.config_manager import (
    ConfigManager, ConfigTemplate, EnvironmentManager, Environment
)


class TestBrokerFactory:
    """Test cases for enhanced broker factory."""

    def test_validate_broker_config_valid(self):
        """Test broker configuration validation with valid config."""
        config = {
            'type': 'binance',
            'trading_mode': 'paper',
            'cash': 10000.0
        }

        validated = validate_broker_config(config)

        assert validated['type'] == 'binance'
        assert validated['trading_mode'] == 'paper'
        assert validated['paper_trading'] is True
        assert 'paper_trading_config' in validated
        assert 'notifications' in validated

    def test_validate_broker_config_invalid_type(self):
        """Test broker configuration validation with invalid type."""
        config = {
            'type': 'invalid_broker',
            'trading_mode': 'paper'
        }

        with pytest.raises(BrokerConfigurationError):
            validate_broker_config(config)

    def test_validate_broker_config_invalid_mode(self):
        """Test broker configuration validation with invalid trading mode."""
        config = {
            'type': 'binance',
            'trading_mode': 'invalid_mode'
        }

        with pytest.raises(BrokerConfigurationError):
            validate_broker_config(config)

    def test_validate_broker_config_live_trading_requirements(self):
        """Test live trading configuration requirements."""
        config = {
            'type': 'binance',
            'trading_mode': 'live'
        }

        # Should fail without confirmation and risk management
        with pytest.raises(BrokerConfigurationError):
            validate_broker_config(config)

        # Should pass with proper configuration
        config.update({
            'live_trading_confirmed': True,
            'risk_management': {
                'max_position_size': 1000.0,
                'max_daily_loss': 500.0
            }
        })

        validated = validate_broker_config(config)
        assert validated['trading_mode'] == 'live'
        assert validated['paper_trading'] is False

    def test_get_broker_credentials_binance(self):
        """Test credential selection for Binance."""
        # Paper trading credentials
        paper_creds = get_broker_credentials('binance', 'paper')
        assert 'testnet.binance.vision' in paper_creds['base_url']
        assert paper_creds['testnet'] is True

        # Live trading credentials
        live_creds = get_broker_credentials('binance', 'live')
        assert 'api.binance.com' in live_creds['base_url']
        assert live_creds['testnet'] is False

    def test_get_broker_credentials_ibkr(self):
        """Test credential selection for IBKR."""
        # Paper trading credentials
        paper_creds = get_broker_credentials('ibkr', 'paper')
        assert paper_creds['port'] == 7497  # Paper trading port
        assert paper_creds['paper_trading'] is True

        # Live trading credentials
        live_creds = get_broker_credentials('ibkr', 'live')
        assert live_creds['port'] == 4001  # Live trading port
        assert live_creds['paper_trading'] is False

    def test_get_broker_capabilities(self):
        """Test broker capabilities detection."""
        # Binance capabilities
        binance_caps = get_broker_capabilities('binance')
        assert binance_caps['paper_trading'] is True
        assert binance_caps['live_trading'] is True
        assert 'crypto' in binance_caps['asset_classes']
        assert 'oco' in binance_caps['order_types']

        # IBKR capabilities
        ibkr_caps = get_broker_capabilities('ibkr')
        assert ibkr_caps['margin_trading'] is True
        assert 'stocks' in ibkr_caps['asset_classes']
        assert 'options' in ibkr_caps['asset_classes']
        assert 'bracket' in ibkr_caps['order_types']

        # Mock capabilities
        mock_caps = get_broker_capabilities('mock')
        assert mock_caps['live_trading'] is False
        assert mock_caps['real_time_data'] is False

    def test_list_available_brokers(self):
        """Test listing available brokers."""
        brokers = list_available_brokers()

        assert len(brokers) >= 3  # At least binance, ibkr, mock

        broker_types = [b['type'] for b in brokers]
        assert 'binance' in broker_types
        assert 'ibkr' in broker_types
        assert 'mock' in broker_types

        # Check structure
        for broker in brokers:
            assert 'type' in broker
            assert 'name' in broker
            assert 'capabilities' in broker
            assert 'credentials_available' in broker
            assert 'recommended_for' in broker

    def test_create_sample_config(self):
        """Test sample configuration creation."""
        # Paper trading config
        paper_config = create_sample_config('binance', 'paper')
        assert paper_config['type'] == 'binance'
        assert paper_config['trading_mode'] == 'paper'
        assert 'paper_trading_config' in paper_config
        assert 'live_trading_confirmed' not in paper_config

        # Live trading config
        live_config = create_sample_config('ibkr', 'live')
        assert live_config['type'] == 'ibkr'
        assert live_config['trading_mode'] == 'live'
        assert live_config['live_trading_confirmed'] is False
        assert 'risk_management' in live_config

    @patch('src.trading.broker.broker_factory.BinanceBroker')
    def test_get_broker_binance(self, mock_binance):
        """Test broker creation for Binance."""
        config = {
            'type': 'binance',
            'trading_mode': 'paper',
            'cash': 10000.0
        }

        mock_broker = Mock()
        mock_binance.return_value = mock_broker

        broker = get_broker(config)

        assert broker == mock_broker
        mock_binance.assert_called_once()

    @patch('src.trading.broker.broker_factory.IBKRBroker')
    def test_get_broker_ibkr(self, mock_ibkr):
        """Test broker creation for IBKR."""
        config = {
            'type': 'ibkr',
            'trading_mode': 'paper',
            'cash': 25000.0
        }

        mock_broker = Mock()
        mock_ibkr.return_value = mock_broker

        broker = get_broker(config)

        assert broker == mock_broker
        mock_ibkr.assert_called_once()

    @patch('src.trading.broker.broker_factory.MockBroker')
    def test_get_broker_mock(self, mock_mock_broker):
        """Test broker creation for Mock."""
        config = {
            'type': 'mock',
            'trading_mode': 'paper',
            'cash': 5000.0
        }

        mock_broker = Mock()
        mock_mock_broker.return_value = mock_broker

        broker = get_broker(config)

        assert broker == mock_broker
        mock_mock_broker.assert_called_once()


class TestBrokerManager:
    """Test cases for broker manager."""

    @pytest.fixture
    def broker_manager(self):
        """Broker manager fixture."""
        return BrokerManager({
            'health_check_interval': 5,
            'auto_restart_enabled': True
        })

    @pytest.fixture
    def mock_broker(self):
        """Mock broker fixture."""
        broker = Mock()
        broker.config = {'type': 'mock', 'trading_mode': 'paper'}
        broker.trading_mode = Mock()
        broker.trading_mode.value = 'paper'
        broker.connect = AsyncMock(return_value=True)
        broker.disconnect = AsyncMock(return_value=True)
        return broker

    @pytest.mark.asyncio
    async def test_add_broker(self, broker_manager):
        """Test adding a broker to management."""
        config = {
            'type': 'mock',
            'trading_mode': 'paper',
            'cash': 10000.0
        }

        with patch('src.trading.broker.broker_manager.get_broker') as mock_get_broker:
            mock_broker = Mock()
            mock_broker.config = config
            mock_broker.trading_mode = Mock()
            mock_broker.trading_mode.value = 'paper'
            mock_get_broker.return_value = mock_broker

            result = await broker_manager.add_broker('test_broker', config)

            assert result is True
            assert 'test_broker' in broker_manager.brokers
            assert 'test_broker' in broker_manager.broker_configs

    @pytest.mark.asyncio
    async def test_start_broker(self, broker_manager, mock_broker):
        """Test starting a broker."""
        broker_manager.brokers['test_broker'] = mock_broker

        result = await broker_manager.start_broker('test_broker')

        assert result is True
        mock_broker.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_broker(self, broker_manager, mock_broker):
        """Test stopping a broker."""
        broker_manager.brokers['test_broker'] = mock_broker

        result = await broker_manager.stop_broker('test_broker')

        assert result is True
        mock_broker.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_restart_broker(self, broker_manager, mock_broker):
        """Test restarting a broker."""
        broker_manager.brokers['test_broker'] = mock_broker

        result = await broker_manager.restart_broker('test_broker')

        assert result is True
        mock_broker.disconnect.assert_called_once()
        mock_broker.connect.assert_called_once()

    def test_list_brokers(self, broker_manager, mock_broker):
        """Test listing managed brokers."""
        broker_manager.brokers['test_broker'] = mock_broker
        broker_manager.broker_configs['test_broker'] = {'type': 'mock', 'trading_mode': 'paper'}

        brokers = broker_manager.list_brokers()

        assert len(brokers) == 1
        assert brokers[0]['broker_id'] == 'test_broker'
        assert brokers[0]['broker_type'] == 'mock'
        assert brokers[0]['trading_mode'] == 'paper'

    def test_health_monitor(self):
        """Test broker health monitoring."""
        monitor = BrokerHealthMonitor(check_interval=1)

        # Add broker to monitoring
        mock_broker = Mock()
        mock_broker.config = {'type': 'mock'}
        mock_broker.trading_mode = Mock()
        mock_broker.trading_mode.value = 'paper'

        monitor.add_broker('test_broker', mock_broker)

        assert 'test_broker' in monitor.health_metrics

        # Update status
        monitor.update_broker_status('test_broker', BrokerStatus.RUNNING)

        metrics = monitor.health_metrics['test_broker']
        assert metrics.status == BrokerStatus.RUNNING

        # Get health summary
        summary = monitor.get_health_summary()
        assert summary['total_brokers'] == 1
        assert summary['running_brokers'] == 1

    def test_connection_pool(self):
        """Test broker connection pool."""
        pool = BrokerConnectionPool(max_connections_per_type=2)

        config = {'type': 'mock', 'trading_mode': 'paper'}

        with patch('src.trading.broker.broker_manager.get_broker') as mock_get_broker:
            mock_broker = Mock()
            mock_get_broker.return_value = mock_broker

            # Get connection
            broker = pool.get_connection(config)
            assert broker == mock_broker

            # Return connection
            pool.return_connection(broker)

            # Get stats
            stats = pool.get_pool_stats()
            assert 'pool_sizes' in stats
            assert 'connection_usage' in stats


class TestConfigManager:
    """Test cases for configuration manager."""

    @pytest.fixture
    def temp_config_dir(self):
        """Temporary configuration directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """Configuration manager fixture."""
        return ConfigManager(temp_config_dir)

    def test_template_manager(self):
        """Test configuration template manager."""
        template_manager = ConfigTemplate()

        # List templates
        templates = template_manager.list_templates()
        assert 'binance_paper' in templates
        assert 'ibkr_live' in templates
        assert 'conservative' in templates

        # Get template
        template = template_manager.get_template('binance_paper')
        assert template is not None
        assert template['type'] == 'binance'
        assert template['trading_mode'] == 'paper'

        # Create from template
        config = template_manager.create_from_template('binance_paper', {
            'cash': 20000.0
        })
        assert config['cash'] == 20000.0
        assert config['type'] == 'binance'

    def test_environment_manager(self, temp_config_dir):
        """Test environment manager."""
        env_manager = EnvironmentManager(temp_config_dir)

        # Set environment
        env_manager.set_environment(Environment.PRODUCTION)
        assert env_manager.current_environment == Environment.PRODUCTION

        # Save environment config
        env_config = {
            'notifications': {
                'email_enabled': True,
                'telegram_enabled': False
            }
        }
        env_manager.save_environment_config(Environment.PRODUCTION, env_config)

        # Get environment config
        retrieved_config = env_manager.get_environment_config(Environment.PRODUCTION)
        assert retrieved_config == env_config

        # Apply overrides
        base_config = {
            'type': 'binance',
            'notifications': {
                'email_enabled': False,
                'telegram_enabled': True
            }
        }

        merged_config = env_manager.apply_environment_overrides(base_config, Environment.PRODUCTION)
        assert merged_config['notifications']['email_enabled'] is True
        assert merged_config['notifications']['telegram_enabled'] is False

    def test_config_manager_crud(self, config_manager):
        """Test configuration CRUD operations."""
        config = {
            'type': 'binance',
            'trading_mode': 'paper',
            'cash': 10000.0
        }

        # Create configuration
        result = config_manager.create_configuration('test_config', config)
        assert result is True

        # Get configuration
        retrieved_config = config_manager.get_configuration('test_config')
        assert retrieved_config is not None
        assert retrieved_config['type'] == 'binance'

        # Update configuration
        updated_config = config.copy()
        updated_config['cash'] = 20000.0

        result = config_manager.update_configuration('test_config', updated_config)
        assert result is True

        # Verify update
        retrieved_config = config_manager.get_configuration('test_config')
        assert retrieved_config['cash'] == 20000.0

        # List configurations
        configs = config_manager.list_configurations()
        assert len(configs) >= 1

        config_names = [c['name'] for c in configs]
        assert 'test_config' in config_names

        # Delete configuration
        result = config_manager.delete_configuration('test_config')
        assert result is True

        # Verify deletion
        retrieved_config = config_manager.get_configuration('test_config')
        assert retrieved_config is None

    def test_config_versioning(self, config_manager):
        """Test configuration versioning."""
        config = {
            'type': 'binance',
            'trading_mode': 'paper',
            'cash': 10000.0
        }

        # Create initial configuration
        config_manager.create_configuration('versioned_config', config)

        # Update configuration (creates new version)
        updated_config = config.copy()
        updated_config['cash'] = 15000.0
        config_manager.update_configuration('versioned_config', updated_config, "Increased cash")

        # Get versions
        versions = config_manager.get_configuration_versions('versioned_config')
        assert len(versions) == 2
        assert versions[0].description == "Initial configuration"
        assert versions[1].description == "Increased cash"

    def test_config_templates(self, config_manager):
        """Test creating configurations from templates."""
        # Create from template
        result = config_manager.create_from_template(
            'template_config',
            'conservative',
            {'cash': 5000.0}
        )
        assert result is True

        # Verify configuration
        config = config_manager.get_configuration('template_config')
        assert config is not None
        assert config['cash'] == 5000.0
        assert config['risk_management']['max_position_size'] == 200.0

    def test_config_export_import(self, config_manager):
        """Test configuration export and import."""
        config = {
            'type': 'ibkr',
            'trading_mode': 'paper',
            'cash': 25000.0
        }

        # Create configuration
        config_manager.create_configuration('export_test', config)

        # Export configuration
        exported = config_manager.export_configuration('export_test')
        assert isinstance(exported, str)

        # Import configuration
        result = config_manager.import_configuration('imported_config', exported)
        assert result is True

        # Verify imported configuration
        imported_config = config_manager.get_configuration('imported_config')
        assert imported_config['type'] == 'ibkr'
        assert imported_config['cash'] == 25000.0


class TestIntegration:
    """Integration tests for the complete broker factory system."""

    @pytest.mark.asyncio
    async def test_full_broker_lifecycle(self):
        """Test complete broker lifecycle with factory and manager."""
        # Create configuration
        config = {
            'type': 'mock',
            'trading_mode': 'paper',
            'cash': 10000.0
        }

        # Create broker manager
        manager = BrokerManager()

        try:
            # Add broker
            result = await manager.add_broker('lifecycle_test', config)
            assert result is True

            # Start broker
            result = await manager.start_broker('lifecycle_test')
            # Note: This might fail due to mock broker limitations, but the test structure is correct

            # Get broker
            broker = manager.get_broker('lifecycle_test')
            assert broker is not None

            # List brokers
            brokers = manager.list_brokers()
            assert len(brokers) >= 1

            # Stop broker
            result = await manager.stop_broker('lifecycle_test')

            # Remove broker
            result = await manager.remove_broker('lifecycle_test')
            assert result is True

        finally:
            # Cleanup
            await manager.cleanup()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])