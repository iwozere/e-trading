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
import logging
from typing import Dict, List, Set, Any, Optional
from datetime import datetime, timezone
import weakref

from fastapi import WebSocket, WebSocketDisconnect
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


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
        except Exception as e:
            _logger.error(f"Error sending message to {self.connection_id}: {e}")
            return False

    async def ping(self) -> bool:
        """Send ping to client."""
        try:
            await self.websocket.send_text(json.dumps({"type": "ping", "timestamp": datetime.now().isoformat()}))
            self.last_ping = datetime.now(timezone.utc)
            return True
        except Exception as e:
            _logger.error(f"Error pinging {self.connection_id}: {e}")
            return False

    def is_stale(self, timeout_seconds: int = 300) -> bool:
        """Check if connection is stale (no ping response)."""
        return (datetime.now(timezone.utc) - self.last_ping).total_seconds() > timeout_seconds


class WebSocketManager:
    """Manages WebSocket connections and real-time communication."""

    def __init__(self):
        """Initialize WebSocket manager."""
        self.connections: Dict[str, WebSocketConnection] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self.strategy_subscribers: Dict[str, Set[str]] = {}  # strategy_id -> connection_ids
        self.system_subscribers: Set[str] = set()  # connection_ids subscribed to system events
        self.cleanup_task: Optional[asyncio.Task] = None
        self.is_running = False

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

    async def connect(self, websocket: WebSocket, user_id: str, connection_id: str) -> bool:
        """Accept a new WebSocket connection."""
        try:
            await websocket.accept()

            connection = WebSocketConnection(websocket, user_id, connection_id)
            self.connections[connection_id] = connection

            # Track user connections
            if user_id not in self.user_connections:
                self.user_connections[user_id] = set()
            self.user_connections[user_id].add(connection_id)

            _logger.info(f"WebSocket connected: {connection_id} (user: {user_id})")

            # Send welcome message
            await connection.send_message({
                "type": "welcome",
                "connection_id": connection_id,
                "timestamp": datetime.now().isoformat()
            })

            return True

        except Exception as e:
            _logger.error(f"Error connecting WebSocket {connection_id}: {e}")
            return False

    async def disconnect(self, connection_id: str):
        """Disconnect a WebSocket connection."""
        if connection_id not in self.connections:
            return

        connection = self.connections[connection_id]

        try:
            # Close WebSocket
            await connection.websocket.close()
        except Exception as e:
            _logger.debug(f"Error closing WebSocket {connection_id}: {e}")

        # Remove from tracking
        self._remove_connection(connection_id)

        _logger.info(f"WebSocket disconnected: {connection_id}")

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
                _logger.warning(f"Unknown message type from {connection_id}: {message_type}")

        except json.JSONDecodeError:
            _logger.error(f"Invalid JSON from {connection_id}: {message}")
        except Exception as e:
            _logger.error(f"Error handling message from {connection_id}: {e}")

    async def _handle_subscription(self, connection_id: str, data: Dict[str, Any]):
        """Handle subscription requests."""
        subscription_type = data.get("subscription_type")

        if subscription_type == "strategy":
            strategy_id = data.get("strategy_id")
            if strategy_id:
                if strategy_id not in self.strategy_subscribers:
                    self.strategy_subscribers[strategy_id] = set()
                self.strategy_subscribers[strategy_id].add(connection_id)
                _logger.debug(f"Connection {connection_id} subscribed to strategy {strategy_id}")

        elif subscription_type == "system":
            self.system_subscribers.add(connection_id)
            _logger.debug(f"Connection {connection_id} subscribed to system events")

    async def _handle_unsubscription(self, connection_id: str, data: Dict[str, Any]):
        """Handle unsubscription requests."""
        subscription_type = data.get("subscription_type")

        if subscription_type == "strategy":
            strategy_id = data.get("strategy_id")
            if strategy_id and strategy_id in self.strategy_subscribers:
                self.strategy_subscribers[strategy_id].discard(connection_id)
                _logger.debug(f"Connection {connection_id} unsubscribed from strategy {strategy_id}")

        elif subscription_type == "system":
            self.system_subscribers.discard(connection_id)
            _logger.debug(f"Connection {connection_id} unsubscribed from system events")

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
                    _logger.info(f"Removing stale connection: {connection_id}")
                    await self.disconnect(connection_id)

                # Send ping to all active connections
                for connection in list(self.connections.values()):
                    await connection.ping()

                # Wait before next cleanup
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                _logger.error(f"Error in connection cleanup: {e}")
                await asyncio.sleep(10)

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics."""
        return {
            "total_connections": len(self.connections),
            "unique_users": len(self.user_connections),
            "strategy_subscriptions": len(self.strategy_subscribers),
            "system_subscribers": len(self.system_subscribers),
            "connections_by_user": {
                user_id: len(connection_ids)
                for user_id, connection_ids in self.user_connections.items()
            }
        }


# Global WebSocket manager instance
websocket_manager = WebSocketManager()