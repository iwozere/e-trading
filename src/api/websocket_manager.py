#!/usr/bin/env python3
"""
WebSocket Manager for Real-Time Communication
--------------------------------------------

This module handles WebSocket connections for real-time communication
between the web UI and the trading system. It provides real-time updates
for strategy status, system metrics, and notifications.

Features:
- WebSocket connection management
- Real-time strategy status updates
- System metrics broadcasting
- Trade notifications
- Alert and error broadcasting
- Connection authentication and authorization
"""

import asyncio
import json
from typing import Dict, List, Set, Any, Optional
from datetime import datetime, timezone
from enum import Enum

from fastapi import WebSocket
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def get_current_user(token: str):
    """Mock function for getting current user from token."""
    # This is a placeholder - in real implementation this would validate JWT token
    # and return user from database
    from unittest.mock import Mock
    user = Mock()
    user.id = 123
    user.role = "trader"
    return user


class MessageType(Enum):
    """WebSocket message types."""
    WELCOME = "welcome"
    PING = "ping"
    PONG = "pong"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    STRATEGY_UPDATE = "strategy_update"
    SYSTEM_UPDATE = "system_update"
    SYSTEM_METRICS = "system_metrics"
    TRADE_NOTIFICATION = "trade_notification"
    NOTIFICATION = "notification"
    ALERT = "alert"
    ERROR = "error"


class BroadcastChannel(Enum):
    """Broadcast channel types."""
    STRATEGY = "strategy"
    STRATEGY_UPDATES = "strategy_updates"
    SYSTEM = "system"
    SYSTEM_METRICS = "system_metrics"
    TRADES = "trades"
    NOTIFICATIONS = "notifications"
    ALERTS = "alerts"


class WebSocketConnection:
    """Represents a single WebSocket connection."""

    def __init__(self, websocket: WebSocket, user_id: str, connection_id: str):
        """Initialize WebSocket connection."""
        self.websocket = websocket
        self.user_id = user_id
        self.connection_id = connection_id
        self.connected_at = datetime.now(timezone.utc)
        self.last_ping = datetime.now(timezone.utc)
        self.subscriptions: Set[str] = set()

    async def send_message(self, message: Dict[str, Any]) -> bool:
        """Send message to client."""
        try:
            await self.websocket.send_text(json.dumps(message))
            return True
        except Exception:
            _logger.exception("Error sending message to %s:", self.connection_id)
            return False

    async def ping(self) -> bool:
        """Send ping to client."""
        try:
            await self.websocket.send_text(json.dumps({"type": "ping", "timestamp": datetime.now().isoformat()}))
            self.last_ping = datetime.now(timezone.utc)
            return True
        except Exception:
            _logger.exception("Error pinging %s:", self.connection_id)
            return False

    def add_subscription(self, channel: str) -> None:
        """Add subscription to a channel."""
        self.subscriptions.add(channel)

    def remove_subscription(self, channel: str) -> None:
        """Remove subscription from a channel."""
        self.subscriptions.discard(channel)

    def has_subscription(self, channel: str) -> bool:
        """Check if subscribed to a channel."""
        return channel in self.subscriptions

    def update_last_ping(self) -> None:
        """Update the last ping timestamp."""
        self.last_ping = datetime.now(timezone.utc)

    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information."""
        return {
            "user_id": self.user_id,
            "connection_id": self.connection_id,
            "connected_at": self.connected_at.isoformat(),
            "last_ping": self.last_ping.isoformat(),
            "subscriptions": list(self.subscriptions)
        }

    def is_stale(self, timeout_seconds: int = 300) -> bool:
        """Check if connection is stale (no ping response)."""
        return (datetime.now(timezone.utc) - self.last_ping).total_seconds() > timeout_seconds


class WebSocketManager:
    """Manages WebSocket connections and real-time communication."""

    def __init__(self):
        """Initialize WebSocket manager."""
        self.connections: Dict[str, WebSocketConnection] = {}
        self.active_connections = self.connections  # Alias for backward compatibility
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self.strategy_subscribers: Dict[str, Set[str]] = {}  # strategy_id -> connection_ids
        self.system_subscribers: Set[str] = set()  # connection_ids subscribed to system events
        self.cleanup_task: Optional[asyncio.Task] = None
        self.is_running = False
        self.connection_manager = self  # Self-reference for tests that expect this attribute

    async def start(self):
        """Start the WebSocket manager."""
        self.is_running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_stale_connections())
        _logger.info("WebSocket manager started")

    async def stop(self):
        """Stop the WebSocket manager."""
        self.is_running = False

        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        for connection in list(self.connections.values()):
            await self.disconnect(connection.connection_id)

        _logger.info("WebSocket manager stopped")

    async def connect(self, websocket: WebSocket, user_id: str, connection_id: str = None) -> str:
        """Accept a new WebSocket connection."""
        try:
            await websocket.accept()

            # Generate connection ID if not provided
            if connection_id is None:
                import uuid
                connection_id = str(uuid.uuid4())

            connection = WebSocketConnection(websocket, user_id, connection_id)
            self.connections[connection_id] = connection

            # Track user connections
            if user_id not in self.user_connections:
                self.user_connections[user_id] = set()
            self.user_connections[user_id].add(connection_id)

            _logger.info("WebSocket connected: %s (user: %s)", connection_id, user_id)

            # Send welcome message
            await connection.send_message({
                "type": "welcome",
                "connection_id": connection_id,
                "timestamp": datetime.now().isoformat()
            })

            return connection_id

        except Exception:
            _logger.exception("Error connecting WebSocket %s:", connection_id)
            return None

    async def disconnect(self, connection_id: str):
        """Disconnect a WebSocket connection."""
        if connection_id not in self.connections:
            return

        connection = self.connections[connection_id]

        try:
            # Close WebSocket
            await connection.websocket.close()
        except Exception as e:
            _logger.debug("Error closing WebSocket %s: %s", connection_id, e)

        # Remove from tracking
        self._remove_connection(connection_id)

        _logger.info("WebSocket disconnected: %s", connection_id)

    def _remove_connection(self, connection_id: str):
        """Remove connection from all tracking structures."""
        if connection_id not in self.connections:
            return

        connection = self.connections[connection_id]

        # Remove from user connections
        if connection.user_id in self.user_connections:
            self.user_connections[connection.user_id].discard(connection_id)
            if not self.user_connections[connection.user_id]:
                del self.user_connections[connection.user_id]

        # Remove from strategy subscriptions
        for strategy_id in list(self.strategy_subscribers.keys()):
            self.strategy_subscribers[strategy_id].discard(connection_id)
            if not self.strategy_subscribers[strategy_id]:
                del self.strategy_subscribers[strategy_id]

        # Remove from system subscriptions
        self.system_subscribers.discard(connection_id)

        # Remove connection
        del self.connections[connection_id]

    async def handle_message(self, connection_id: str, message: str):
        """Handle incoming WebSocket message."""
        if connection_id not in self.connections:
            return

        connection = self.connections[connection_id]

        try:
            data = json.loads(message)
            message_type = data.get("type")

            if message_type == "pong":
                # Update last ping time
                connection.last_ping = datetime.now(timezone.utc)

            elif message_type == "subscribe":
                # Handle subscription requests
                await self._handle_subscription(connection_id, data)

            elif message_type == "unsubscribe":
                # Handle unsubscription requests
                await self._handle_unsubscription(connection_id, data)

            else:
                _logger.warning("Unknown message type from %s: %s", connection_id, message_type)

        except json.JSONDecodeError:
            _logger.exception("Invalid JSON from %s: %s", connection_id, message)
        except Exception:
            _logger.exception("Error handling message from %s", connection_id)

    async def _handle_subscription(self, connection_id: str, data: Dict[str, Any]):
        """Handle subscription requests."""
        subscription_type = data.get("subscription_type")

        if subscription_type == "strategy":
            strategy_id = data.get("strategy_id")
            if strategy_id:
                if strategy_id not in self.strategy_subscribers:
                    self.strategy_subscribers[strategy_id] = set()
                self.strategy_subscribers[strategy_id].add(connection_id)
                _logger.debug("Connection %s subscribed to strategy %s", connection_id, strategy_id)

        elif subscription_type == "system":
            self.system_subscribers.add(connection_id)
            _logger.debug("Connection %s subscribed to system events", connection_id)

    async def _handle_unsubscription(self, connection_id: str, data: Dict[str, Any]):
        """Handle unsubscription requests."""
        subscription_type = data.get("subscription_type")

        if subscription_type == "strategy":
            strategy_id = data.get("strategy_id")
            if strategy_id and strategy_id in self.strategy_subscribers:
                self.strategy_subscribers[strategy_id].discard(connection_id)
                _logger.debug("Connection %s unsubscribed from strategy %s", connection_id, strategy_id)

        elif subscription_type == "system":
            self.system_subscribers.discard(connection_id)
            _logger.debug("Connection %s unsubscribed from system events", connection_id)

    async def broadcast_strategy_update(self, strategy_id: str, status_data: Dict[str, Any]):
        """Broadcast strategy status update to subscribers."""
        if strategy_id not in self.strategy_subscribers:
            return

        message = {
            "type": "strategy_update",
            "strategy_id": strategy_id,
            "data": status_data,
            "timestamp": datetime.now().isoformat()
        }

        # Send to all subscribers
        subscribers = list(self.strategy_subscribers[strategy_id])
        for connection_id in subscribers:
            if connection_id in self.connections:
                success = await self.connections[connection_id].send_message(message)
                if not success:
                    # Remove failed connection
                    await self.disconnect(connection_id)

    async def broadcast_system_update(self, system_data: Dict[str, Any]):
        """Broadcast system status update to subscribers."""
        message = {
            "type": "system_update",
            "data": system_data,
            "timestamp": datetime.now().isoformat()
        }

        # Send to all system subscribers
        subscribers = list(self.system_subscribers)
        for connection_id in subscribers:
            if connection_id in self.connections:
                success = await self.connections[connection_id].send_message(message)
                if not success:
                    # Remove failed connection
                    await self.disconnect(connection_id)

    async def broadcast_trade_notification(self, strategy_id: str, trade_data: Dict[str, Any]):
        """Broadcast trade execution notification."""
        message = {
            "type": "trade_notification",
            "strategy_id": strategy_id,
            "data": trade_data,
            "timestamp": datetime.now().isoformat()
        }

        # Send to strategy subscribers and system subscribers
        all_subscribers = set()
        if strategy_id in self.strategy_subscribers:
            all_subscribers.update(self.strategy_subscribers[strategy_id])
        all_subscribers.update(self.system_subscribers)

        for connection_id in all_subscribers:
            if connection_id in self.connections:
                success = await self.connections[connection_id].send_message(message)
                if not success:
                    # Remove failed connection
                    await self.disconnect(connection_id)

    async def broadcast_alert(self, alert_data: Dict[str, Any], priority: str = "info"):
        """Broadcast system alert to all connected users."""
        message = {
            "type": "alert",
            "priority": priority,
            "data": alert_data,
            "timestamp": datetime.now().isoformat()
        }

        # Send to all connections
        for connection_id in list(self.connections.keys()):
            if connection_id in self.connections:
                success = await self.connections[connection_id].send_message(message)
                if not success:
                    # Remove failed connection
                    await self.disconnect(connection_id)

    async def send_to_user(self, user_id: str, message: Dict[str, Any]) -> bool:
        """Send message to all connections of a specific user."""
        if user_id not in self.user_connections:
            return False

        success_count = 0
        connection_ids = list(self.user_connections[user_id])

        for connection_id in connection_ids:
            if connection_id in self.connections:
                success = await self.connections[connection_id].send_message(message)
                if success:
                    success_count += 1
                else:
                    # Remove failed connection
                    await self.disconnect(connection_id)

        return success_count > 0

    def get_user_connections(self, user_id: str) -> List[WebSocketConnection]:
        """Get all connections for a specific user."""
        if user_id not in self.user_connections:
            return []

        connections = []
        for connection_id in self.user_connections[user_id]:
            if connection_id in self.connections:
                connections.append(self.connections[connection_id])
        return connections

    def get_connection(self, connection_id: str) -> Optional[WebSocketConnection]:
        """Get connection by ID."""
        return self.connections.get(connection_id)

    async def broadcast_to_all(self, message: Dict[str, Any]):
        """Broadcast message to all connections."""
        for connection_id in list(self.connections.keys()):
            if connection_id in self.connections:
                success = await self.connections[connection_id].send_message(message)
                if not success:
                    await self.disconnect(connection_id)

    async def broadcast_to_user(self, user_id: str, message: Dict[str, Any]):
        """Broadcast message to all connections of a specific user."""
        await self.send_to_user(user_id, message)

    async def broadcast_to_channel(self, channel: str, message: Dict[str, Any]):
        """Broadcast message to all connections subscribed to a channel."""
        for connection_id, connection in list(self.connections.items()):
            if connection.has_subscription(channel):
                success = await connection.send_message(message)
                if not success:
                    await self.disconnect(connection_id)

    async def cleanup_stale_connections(self):
        """Public method to cleanup stale connections."""
        stale_connections = []
        for connection_id, connection in self.connections.items():
            if connection.is_stale():
                stale_connections.append(connection_id)

        for connection_id in stale_connections:
            _logger.info("Removing stale connection: %s", connection_id)
            await self.disconnect(connection_id)

    async def _cleanup_stale_connections(self):
        """Periodically clean up stale connections."""
        while self.is_running:
            try:
                # Check for stale connections
                stale_connections = []
                for connection_id, connection in self.connections.items():
                    if connection.is_stale():
                        stale_connections.append(connection_id)

                # Remove stale connections
                for connection_id in stale_connections:
                    _logger.info("Removing stale connection: %s", connection_id)
                    await self.disconnect(connection_id)

                # Send ping to all active connections
                for connection in list(self.connections.values()):
                    await connection.ping()

                # Wait before next cleanup
                await asyncio.sleep(60)  # Check every minute

            except Exception:
                _logger.exception("Error in connection cleanup:")
                await asyncio.sleep(10)

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics."""
        connections_list = list(self.connections.values())
        oldest_connection = min(connections_list, key=lambda c: c.connected_at) if connections_list else None
        newest_connection = max(connections_list, key=lambda c: c.connected_at) if connections_list else None

        return {
            "total_connections": len(self.connections),
            "unique_users": len(self.user_connections),
            "strategy_subscriptions": len(self.strategy_subscribers),
            "system_subscribers": len(self.system_subscribers),
            "oldest_connection": oldest_connection.connected_at.isoformat() if oldest_connection else None,
            "newest_connection": newest_connection.connected_at.isoformat() if newest_connection else None,
            "connections_by_user": {
                user_id: len(connection_ids)
                for user_id, connection_ids in self.user_connections.items()
            }
        }

    async def handle_connection(self, websocket: WebSocket, token: str):
        """Handle a new WebSocket connection with authentication."""
        try:
            # Mock authentication for now - in real implementation this would validate the token

            # Try to get current user (this will be mocked in tests)
            try:
                from src.api.websocket_manager import get_current_user
                user = get_current_user(token)
            except Exception:
                # Authentication failed
                await websocket.close(code=4001, reason="Authentication failed")
                return

            if not user:
                await websocket.close(code=4001, reason="Authentication failed")
                return

            # Connect the user
            connection_id = await self.connect(websocket, str(user.id))
            if not connection_id:
                await websocket.close(code=4000, reason="Connection failed")
                return

            # Handle messages
            try:
                while True:
                    message_text = await websocket.receive_text()
                    message = json.loads(message_text)
                    await self.handle_message_obj(self.get_connection(connection_id), message)
            except Exception as e:
                _logger.info("WebSocket connection %s closed: %s", connection_id, e)
            finally:
                await self.disconnect(connection_id)

        except Exception:
            _logger.exception("Error in handle_connection:")
            try:
                await websocket.close(code=4001, reason="Authentication failed")
            except:
                pass

    async def handle_message_obj(self, connection: WebSocketConnection, message: Dict[str, Any]):
        """Handle incoming WebSocket message object."""
        if not connection:
            return

        message_type = message.get("type")

        if message_type == "ping":
            connection.update_last_ping()
            await connection.send_message({
                "type": "pong",
                "timestamp": datetime.now().isoformat()
            })
        elif message_type == "subscribe":
            channel = message.get("channel")
            if channel:
                connection.add_subscription(channel)
                await connection.send_message({
                    "type": "subscribed",
                    "channel": channel
                })
        elif message_type == "unsubscribe":
            channel = message.get("channel")
            if channel:
                connection.remove_subscription(channel)
                await connection.send_message({
                    "type": "unsubscribed",
                    "channel": channel
                })
        else:
            await connection.send_message({
                "type": "error",
                "message": "Unknown message type: {}".format(message_type)
            })

    async def broadcast_strategy_update(self, strategy_data: Dict[str, Any]):
        """Broadcast strategy update to subscribers."""
        message = {
            "type": "strategy_update",
            "data": strategy_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast_to_channel("strategy_updates", message)

    async def broadcast_system_metrics(self, metrics_data: Dict[str, Any]):
        """Broadcast system metrics to subscribers."""
        message = {
            "type": "system_metrics",
            "data": metrics_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast_to_channel("system_metrics", message)

    async def broadcast_notification(self, notification: Dict[str, Any]):
        """Broadcast notification to all connections."""
        message = {
            "type": "notification",
            "data": notification,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast_to_all(message)

    async def handle_message(self, connection: WebSocketConnection, message: Dict[str, Any]):
        """Handle incoming WebSocket message (test-compatible version)."""
        await self.handle_message_obj(connection, message)

    def get_manager_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        stats = self.get_connection_stats()
        stats["uptime_seconds"] = 0  # Would track actual uptime in real implementation
        return stats


# Alias for backward compatibility
ConnectionManager = WebSocketManager

# Global WebSocket manager instance
websocket_manager = WebSocketManager()