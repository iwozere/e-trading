"""
Instance Service
----------------
Manages the lifecycle of multiple strategy instances.
Provides a central points for creating, starting, and stopping bot instances.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from src.trading.strategy_instance import StrategyInstance
from src.notification.logger import setup_logger
from src.notification.service.client import NotificationServiceClient

_logger = setup_logger(__name__)

class InstanceService:
    """
    Service for managing multiple trading strategy instances.
    Responsible for creating, starting, stopping, and monitoring instances.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(InstanceService, cls).__new__(cls)
        return cls._instance

    def __init__(self, notification_client: Optional[NotificationServiceClient] = None, 
                 trade_repository: Any = None):
        """Initialize the instance service."""
        # Only initialize once (singleton)
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.instances: Dict[str, StrategyInstance] = {}
        self.notification_client = notification_client
        self.trade_repository = trade_repository
        self._initialized = True
        _logger.info("InstanceService initialized")

    def create_instance(self, instance_id: str, config: Dict[str, Any]) -> StrategyInstance:
        """
        Create a new strategy instance.
        
        Args:
            instance_id: Unique identifier for the instance
            config: Configuration dictionaryfor the strategy
            
        Returns:
            The created StrategyInstance
        """
        if instance_id in self.instances:
            _logger.warning("Instance %s already exists, replacing it.", instance_id)
            
        instance = StrategyInstance(
            instance_id=instance_id,
            config=config,
            notification_client=self.notification_client,
            trade_repository=self.trade_repository
        )
        self.instances[instance_id] = instance
        _logger.info("Created strategy instance: %s (%s)", instance.name, instance_id)
        return instance

    async def start_instance(self, instance_id: str) -> bool:
        """Start a specific strategy instance."""
        if instance_id not in self.instances:
            _logger.error("Cannot start instance %s: not found", instance_id)
            return False
            
        return await self.instances[instance_id].start()

    async def stop_instance(self, instance_id: str) -> bool:
        """Stop a specific strategy instance."""
        if instance_id not in self.instances:
            _logger.error("Cannot stop instance %s: not found", instance_id)
            return False
            
        return await self.instances[instance_id].stop()

    async def stop_all_instances(self) -> None:
        """Stop all managed strategy instances gracefully."""
        _logger.info("Stopping all %d strategy instances...", len(self.instances))
        stop_tasks = []
        for instance_id in list(self.instances.keys()):
            stop_tasks.append(self.stop_instance(instance_id))
            
        if stop_tasks:
            await asyncio.gather(*stop_tasks)
        _logger.info("All instances stopped.")

    def get_instance_status(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific instance."""
        if instance_id in self.instances:
            return self.instances[instance_id].get_status()
        return None

    def get_all_statuses(self) -> List[Dict[str, Any]]:
        """Get status of all managed strategy instances."""
        return [instance.get_status() for instance in self.instances.values()]

    def remove_instance(self, instance_id: str) -> bool:
        """Remove an instance from the service (must be stopped first)."""
        if instance_id in self.instances:
            if self.instances[instance_id].status == 'running':
                _logger.error("Cannot remove running instance %s", instance_id)
                return False
            del self.instances[instance_id]
            return True
        return False
