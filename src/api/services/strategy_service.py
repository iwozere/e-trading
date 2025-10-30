"""
Strategy Management Service
--------------------------

Service layer for managing trading strategies through the web UI.
Provides a clean interface between the web API and the enhanced trading system.
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json
import asyncio
from datetime import datetime
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# Import trading system components with error handling
try:
    from src.trading.strategy_manager import StrategyManager, StrategyInstance
    from src.model.config_models import StrategyConfig as TradingStrategyConfig
    TRADING_SYSTEM_AVAILABLE = True
except ImportError as e:
    _logger.warning("Trading system not available: %s", e)
    StrategyManager = None
    StrategyInstance = None
    TradingStrategyConfig = None
    TRADING_SYSTEM_AVAILABLE = False


class StrategyValidationError(Exception):
    """Raised when strategy configuration validation fails."""
    pass


class StrategyOperationError(Exception):
    """Raised when strategy operations fail."""
    pass


class StrategyManagementService:
    """
    Service for managing trading strategies through the web UI.

    This service provides:
    - Strategy CRUD operations
    - Configuration validation
    - Status monitoring
    - Parameter updates
    - Error handling and logging
    """

    def __init__(self, strategy_manager: Optional[StrategyManager] = None):
        """
        Initialize the strategy management service.

        Args:
            strategy_manager: Enhanced strategy manager instance
        """
        self.strategy_manager = strategy_manager
        self.is_available = TRADING_SYSTEM_AVAILABLE and strategy_manager is not None

        if not self.is_available:
            _logger.warning("Strategy management service initialized without trading system")

    def validate_strategy_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate strategy configuration.

        Args:
            config: Strategy configuration dictionary

        Returns:
            Dict: Validated and normalized configuration

        Raises:
            StrategyValidationError: If validation fails
        """
        try:
            # Required fields validation
            required_fields = ['id', 'name', 'symbol', 'broker', 'strategy']
            missing_fields = [field for field in required_fields if field not in config]

            if missing_fields:
                raise StrategyValidationError(f"Missing required fields: {', '.join(missing_fields)}")

            # Validate strategy ID format
            strategy_id = config['id']
            if not isinstance(strategy_id, str) or not strategy_id.strip():
                raise StrategyValidationError("Strategy ID must be a non-empty string")

            # Validate broker configuration
            broker_config = config['broker']
            if not isinstance(broker_config, dict):
                raise StrategyValidationError("Broker configuration must be a dictionary")

            required_broker_fields = ['type', 'trading_mode']
            missing_broker_fields = [field for field in required_broker_fields if field not in broker_config]

            if missing_broker_fields:
                raise StrategyValidationError(f"Missing broker fields: {', '.join(missing_broker_fields)}")

            # Validate trading mode
            valid_trading_modes = ['paper', 'live']
            if broker_config['trading_mode'] not in valid_trading_modes:
                raise StrategyValidationError(f"Trading mode must be one of: {', '.join(valid_trading_modes)}")

            # Validate strategy configuration
            strategy_config = config['strategy']
            if not isinstance(strategy_config, dict):
                raise StrategyValidationError("Strategy configuration must be a dictionary")

            # Validate symbol format
            symbol = config['symbol']
            if not isinstance(symbol, str) or not symbol.strip():
                raise StrategyValidationError("Symbol must be a non-empty string")

            # Set default values for optional fields
            validated_config = {
                'id': strategy_id.strip(),
                'name': config['name'].strip() if isinstance(config['name'], str) else strategy_id,
                'enabled': config.get('enabled', True),
                'symbol': symbol.strip().upper(),
                'broker': broker_config,
                'strategy': strategy_config,
                'data': config.get('data', {}),
                'trading': config.get('trading', {}),
                'risk_management': config.get('risk_management', {}),
                'notifications': config.get('notifications', {})
            }

            _logger.info("Strategy configuration validated successfully: %s", strategy_id)
            return validated_config

        except Exception as e:
            _logger.exception("Strategy configuration validation failed:")
            if isinstance(e, StrategyValidationError):
                raise
            raise StrategyValidationError(f"Configuration validation error: {str(e)}")

    async def create_strategy(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new trading strategy.

        Args:
            config: Strategy configuration

        Returns:
            Dict: Created strategy information

        Raises:
            StrategyValidationError: If configuration is invalid
            StrategyOperationError: If creation fails
        """
        if not self.is_available:
            raise StrategyOperationError("Trading system not available")

        try:
            # Validate configuration
            validated_config = self.validate_strategy_config(config)
            strategy_id = validated_config['id']

            # Check if strategy already exists
            if strategy_id in self.strategy_manager.strategy_instances:
                raise StrategyOperationError(f"Strategy '{strategy_id}' already exists")

            # Create strategy instance
            instance = StrategyInstance(strategy_id, validated_config)
            self.strategy_manager.strategy_instances[strategy_id] = instance

            _logger.info("Strategy created successfully: %s", strategy_id)

            return {
                'strategy_id': strategy_id,
                'name': validated_config['name'],
                'status': 'created',
                'message': 'Strategy created successfully'
            }

        except (StrategyValidationError, StrategyOperationError):
            raise
        except Exception as e:
            _logger.exception("Failed to create strategy:")
            raise StrategyOperationError(f"Strategy creation failed: {str(e)}")

    async def update_strategy(self, strategy_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing strategy configuration.

        Args:
            strategy_id: Strategy identifier
            config: Updated configuration

        Returns:
            Dict: Update result information

        Raises:
            StrategyOperationError: If update fails
        """
        if not self.is_available:
            raise StrategyOperationError("Trading system not available")

        try:
            # Check if strategy exists
            if strategy_id not in self.strategy_manager.strategy_instances:
                raise StrategyOperationError(f"Strategy '{strategy_id}' not found")

            # Validate new configuration
            validated_config = self.validate_strategy_config(config)

            # Get existing instance
            instance = self.strategy_manager.strategy_instances[strategy_id]

            # Check if strategy is running
            if instance.status == 'running':
                _logger.warning("Updating configuration of running strategy: %s", strategy_id)

            # Update configuration
            instance.config = validated_config

            _logger.info("Strategy updated successfully: %s", strategy_id)

            return {
                'strategy_id': strategy_id,
                'name': validated_config['name'],
                'status': 'updated',
                'message': 'Strategy updated successfully'
            }

        except (StrategyValidationError, StrategyOperationError):
            raise
        except Exception as e:
            _logger.exception("Failed to update strategy %s:", strategy_id)
            raise StrategyOperationError(f"Strategy update failed: {str(e)}")

    async def delete_strategy(self, strategy_id: str, force: bool = False) -> Dict[str, Any]:
        """
        Delete a trading strategy.

        Args:
            strategy_id: Strategy identifier
            force: Force deletion even if running

        Returns:
            Dict: Deletion result information

        Raises:
            StrategyOperationError: If deletion fails
        """
        if not self.is_available:
            raise StrategyOperationError("Trading system not available")

        try:
            # Check if strategy exists
            if strategy_id not in self.strategy_manager.strategy_instances:
                raise StrategyOperationError(f"Strategy '{strategy_id}' not found")

            instance = self.strategy_manager.strategy_instances[strategy_id]

            # Check if strategy is running
            if instance.status == 'running' and not force:
                raise StrategyOperationError(f"Cannot delete running strategy '{strategy_id}'. Stop it first or use force=True")

            # Stop strategy if running
            if instance.status == 'running':
                _logger.warning("Force stopping strategy before deletion: %s", strategy_id)
                await self.stop_strategy(strategy_id)

            # Remove from manager
            del self.strategy_manager.strategy_instances[strategy_id]

            _logger.info("Strategy deleted successfully: %s", strategy_id)

            return {
                'strategy_id': strategy_id,
                'status': 'deleted',
                'message': 'Strategy deleted successfully'
            }

        except StrategyOperationError:
            raise
        except Exception as e:
            _logger.exception("Failed to delete strategy %s:", strategy_id)
            raise StrategyOperationError(f"Strategy deletion failed: {str(e)}")

    async def start_strategy(self, strategy_id: str, confirm_live_trading: bool = False) -> Dict[str, Any]:
        """
        Start a trading strategy.

        Args:
            strategy_id: Strategy identifier
            confirm_live_trading: Confirmation for live trading

        Returns:
            Dict: Start operation result

        Raises:
            StrategyOperationError: If start fails
        """
        if not self.is_available:
            raise StrategyOperationError("Trading system not available")

        try:
            # Check if strategy exists
            if strategy_id not in self.strategy_manager.strategy_instances:
                raise StrategyOperationError(f"Strategy '{strategy_id}' not found")

            instance = self.strategy_manager.strategy_instances[strategy_id]

            # Check live trading confirmation
            if (instance.config.get('broker', {}).get('trading_mode') == 'live' and
                not confirm_live_trading):
                raise StrategyOperationError("Live trading requires explicit confirmation")

            # Start strategy
            success = await self.strategy_manager.start_strategy(strategy_id)

            if not success:
                raise StrategyOperationError(f"Failed to start strategy '{strategy_id}'")

            _logger.info("Strategy started successfully: %s", strategy_id)

            return {
                'strategy_id': strategy_id,
                'status': 'started',
                'message': 'Strategy started successfully'
            }

        except StrategyOperationError:
            raise
        except Exception as e:
            _logger.exception("Failed to start strategy %s:", strategy_id)
            raise StrategyOperationError(f"Strategy start failed: {str(e)}")

    async def stop_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """
        Stop a trading strategy.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Dict: Stop operation result

        Raises:
            StrategyOperationError: If stop fails
        """
        if not self.is_available:
            raise StrategyOperationError("Trading system not available")

        try:
            # Check if strategy exists
            if strategy_id not in self.strategy_manager.strategy_instances:
                raise StrategyOperationError(f"Strategy '{strategy_id}' not found")

            # Stop strategy
            success = await self.strategy_manager.stop_strategy(strategy_id)

            if not success:
                raise StrategyOperationError(f"Failed to stop strategy '{strategy_id}'")

            _logger.info("Strategy stopped successfully: %s", strategy_id)

            return {
                'strategy_id': strategy_id,
                'status': 'stopped',
                'message': 'Strategy stopped successfully'
            }

        except StrategyOperationError:
            raise
        except Exception as e:
            _logger.exception("Failed to stop strategy %s:", strategy_id)
            raise StrategyOperationError(f"Strategy stop failed: {str(e)}")

    async def restart_strategy(self, strategy_id: str, confirm_live_trading: bool = False) -> Dict[str, Any]:
        """
        Restart a trading strategy.

        Args:
            strategy_id: Strategy identifier
            confirm_live_trading: Confirmation for live trading

        Returns:
            Dict: Restart operation result

        Raises:
            StrategyOperationError: If restart fails
        """
        if not self.is_available:
            raise StrategyOperationError("Trading system not available")

        try:
            # Check if strategy exists
            if strategy_id not in self.strategy_manager.strategy_instances:
                raise StrategyOperationError(f"Strategy '{strategy_id}' not found")

            instance = self.strategy_manager.strategy_instances[strategy_id]

            # Check live trading confirmation
            if (instance.config.get('broker', {}).get('trading_mode') == 'live' and
                not confirm_live_trading):
                raise StrategyOperationError("Live trading requires explicit confirmation")

            # Restart strategy
            success = await self.strategy_manager.restart_strategy(strategy_id)

            if not success:
                raise StrategyOperationError(f"Failed to restart strategy '{strategy_id}'")

            _logger.info("Strategy restarted successfully: %s", strategy_id)

            return {
                'strategy_id': strategy_id,
                'status': 'restarted',
                'message': 'Strategy restarted successfully'
            }

        except StrategyOperationError:
            raise
        except Exception as e:
            _logger.exception("Failed to restart strategy %s:", strategy_id)
            raise StrategyOperationError(f"Strategy restart failed: {str(e)}")

    def get_strategy_status(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific strategy.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Dict: Strategy status information or None if not found
        """
        if not self.is_available:
            return None

        try:
            return self.strategy_manager.get_strategy_status(strategy_id)
        except Exception as e:
            _logger.exception("Failed to get strategy status %s:", strategy_id)
            return None

    def get_all_strategies_status(self) -> List[Dict[str, Any]]:
        """
        Get status of all strategies.

        Returns:
            List: List of strategy status information
        """
        if not self.is_available:
            return []

        try:
            return self.strategy_manager.get_all_status()
        except Exception as e:
            _logger.exception("Failed to get all strategies status:")
            return []

    async def update_strategy_parameters(self, strategy_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update strategy parameters while running.

        Args:
            strategy_id: Strategy identifier
            parameters: New parameter values

        Returns:
            Dict: Update result information

        Raises:
            StrategyOperationError: If parameter update fails
        """
        if not self.is_available:
            raise StrategyOperationError("Trading system not available")

        try:
            # Check if strategy exists
            if strategy_id not in self.strategy_manager.strategy_instances:
                raise StrategyOperationError(f"Strategy '{strategy_id}' not found")

            instance = self.strategy_manager.strategy_instances[strategy_id]

            # Update strategy parameters in configuration
            if 'strategy' not in instance.config:
                instance.config['strategy'] = {}

            if 'parameters' not in instance.config['strategy']:
                instance.config['strategy']['parameters'] = {}

            # Merge new parameters
            instance.config['strategy']['parameters'].update(parameters)

            # If strategy is running, apply parameters dynamically
            if instance.status == 'running' and hasattr(self.strategy_manager, 'update_strategy_parameters'):
                await self.strategy_manager.update_strategy_parameters(strategy_id, parameters)

            _logger.info("Strategy parameters updated successfully: %s", strategy_id)

            return {
                'strategy_id': strategy_id,
                'status': 'parameters_updated',
                'message': 'Strategy parameters updated successfully',
                'updated_parameters': parameters
            }

        except StrategyOperationError:
            raise
        except Exception as e:
            _logger.exception("Failed to update strategy parameters %s:", strategy_id)
            raise StrategyOperationError(f"Parameter update failed: {str(e)}")

    def get_strategy_templates(self) -> Dict[str, Any]:
        """
        Get available strategy templates.

        Returns:
            Dict: Available strategy templates
        """
        try:
            # Load templates from config examples
            templates_file = PROJECT_ROOT / "config/examples/enhanced_trading_config_examples.json"

            if templates_file.exists():
                with open(templates_file, 'r') as f:
                    templates = json.load(f)
                return templates
            else:
                _logger.warning("Strategy templates file not found: %s", templates_file)
                return {}

        except Exception as e:
            _logger.exception("Failed to load strategy templates:")
            return {}

    def get_service_status(self) -> Dict[str, Any]:
        """
        Get strategy management service status.

        Returns:
            Dict: Service status information
        """
        return {
            'service_name': 'Strategy Management Service',
            'available': self.is_available,
            'trading_system_available': TRADING_SYSTEM_AVAILABLE,
            'strategy_manager_connected': self.strategy_manager is not None,
            'total_strategies': len(self.strategy_manager.strategy_instances) if self.is_available else 0,
            'active_strategies': len([s for s in self.get_all_strategies_status() if s.get('status') == 'running'])
        }