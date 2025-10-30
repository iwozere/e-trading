#!/usr/bin/env python3
"""
Unit Tests for WebSocket Manager
------------------------------

Tests for WebSocket connection management, real-time communication,
and message broadcasting functionality.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.api.websocket_manager import (
    WebSocketConnection,
    WebSocketManager,
    ConnectionManager,
    MessageType,
    BroadcastChannel
)


class TestWebSocketConnection:
    """Test cases for WebSocket connection handling."""

    def setup_method(self):
        """Set up test dependencies."""
        self.mock_websocket = AsyncMock()
        self.connection = WebSocketConnection(
            websocket=self.mock_websocket,
            user_id="test_user_123",
            connection_id="conn_456"
        )

    def test_connection_initialization(self):
        """Test WebSocket connection initialization."""
        assert self.connection.websocket == self.mock_websocket
        assert self.connection.user_id == "test_user_123"
        assert self.connection.connection_id == "conn_456"
        assert isinstance(self.connection.connected_at, datetime)
        assert isinstance(self.connection.last_ping, datetime)
        assert isinstance(self.connection.subscriptions, set)
        assert len(self.connection.subscriptions) == 0

    @pytest.mark.asyncio
    async def test_send_message_success(self):
        """Test successful message sending."""
        message = {"type": "test", "data": {"key": "value"}}

        result = await self.connection.send_message(message)

        assert result is True
        self.mock_websocket.send_text.assert_called_once_with(json.dumps(message))

    @pytest.mark.asyncio
    async def test_send_message_failure(self):
        """Test message sending failure handling."""
        self.mock_websocket.send_text.side_effect = Exception("Connection lost")
        message = {"type": "test", "data": {}}

        result = await self.connection.send_message(message)

        assert result is False

    def test_add_subscription(self):
        """Test adding subscription to connection."""
        self.connection.add_subscription("strategy_updates")

        assert "strategy_updates" in self.connection.subscriptions
        assert len(self.connection.subscriptions) == 1

    def test_remove_subscription(self):
        """Test removing subscription from connection."""
        self.connection.add_subscription("strategy_updates")
        self.connection.add_subscription("system_metrics")

        self.connection.remove_subscription("strategy_updates")

        assert "strategy_updates" not in self.connection.subscriptions
        assert "system_metrics" in self.connection.subscriptions
        assert len(self.connection.subscriptions) == 1

    def test_has_subscription(self):
        """Test checking subscription existence."""
        assert not self.connection.has_subscription("strategy_updates")

        self.connection.add_subscription("strategy_updates")

        assert self.connection.has_subscription("strategy_updates")

    def test_update_last_ping(self):
        """Test updating last ping timestamp."""
        original_ping = self.connection.last_ping

        self.connection.update_last_ping()

        assert self.connection.last_ping > original_ping

    def test_get_connection_info(self):
        """Test getting connection information."""
        self.connection.add_subscription("test_channel")

        info = self.connection.get_connection_info()

        assert info["user_id"] == "test_user_123"
        assert info["connection_id"] == "conn_456"
        assert "connected_at" in info
        assert "last_ping" in info
        assert info["subscriptions"] == ["test_channel"]


class TestConnectionManager:
    """Test cases for connection management."""

    def setup_method(self):
        """Set up test dependencies."""
        self.manager = ConnectionManager()
        self.mock_websocket1 = AsyncMock()
        self.mock_websocket2 = AsyncMock()

    @pytest.mark.asyncio
    async def test_connect_user(self):
        """Test connecting a user."""
        connection_id = await self.manager.connect(
            websocket=self.mock_websocket1,
            user_id="user_123"
        )

        assert connection_id is not None
        assert len(self.manager.active_connections) == 1
        assert "user_123" in self.manager.user_connections

    @pytest.mark.asyncio
    async def test_connect_multiple_connections_same_user(self):
        """Test connecting multiple connections for same user."""
        conn_id1 = await self.manager.connect(self.mock_websocket1, "user_123")
        conn_id2 = await self.manager.connect(self.mock_websocket2, "user_123")

        assert conn_id1 != conn_id2
        assert len(self.manager.active_connections) == 2
        assert len(self.manager.user_connections["user_123"]) == 2

    @pytest.mark.asyncio
    async def test_disconnect_user(self):
        """Test disconnecting a user."""
        # First connect
        await self.manager.connect(self.mock_websocket1, "user_123")
        connection_id = list(self.manager.active_connections.keys())[0]

        # Then disconnect
        await self.manager.disconnect(connection_id)

        assert len(self.manager.active_connections) == 0
        assert "user_123" not in self.manager.user_connections

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent_connection(self):
        """Test disconnecting non-existent connection."""
        # Should not raise exception
        await self.manager.disconnect("nonexistent_id")

        assert len(self.manager.active_connections) == 0

    def test_get_user_connections(self):
        """Test getting connections for a user."""
        # Connect user
        asyncio.run(self.manager.connect(self.mock_websocket1, "user_123"))
        asyncio.run(self.manager.connect(self.mock_websocket2, "user_123"))

        connections = self.manager.get_user_connections("user_123")

        assert len(connections) == 2
        assert all(conn.user_id == "user_123" for conn in connections)

    def test_get_connection_by_id(self):
        """Test getting connection by ID."""
        conn_id = asyncio.run(self.manager.connect(self.mock_websocket1, "user_123"))

        connection = self.manager.get_connection(conn_id)

        assert connection is not None
        assert connection.connection_id == conn_id
        assert connection.user_id == "user_123"

    def test_get_nonexistent_connection(self):
        """Test getting non-existent connection."""
        connection = self.manager.get_connection("nonexistent_id")

        assert connection is None

    def test_get_connection_stats(self):
        """Test getting connection statistics."""
        asyncio.run(self.manager.connect(self.mock_websocket1, "user_123"))
        asyncio.run(self.manager.connect(self.mock_websocket2, "user_456"))

        stats = self.manager.get_connection_stats()

        assert stats["total_connections"] == 2
        assert stats["unique_users"] == 2
        assert "oldest_connection" in stats
        assert "newest_connection" in stats

    @pytest.mark.asyncio
    async def test_broadcast_to_all(self):
        """Test broadcasting message to all connections."""
        await self.manager.connect(self.mock_websocket1, "user_123")
        await self.manager.connect(self.mock_websocket2, "user_456")

        message = {"type": "broadcast", "data": "test message"}

        await self.manager.broadcast_to_all(message)

        # Each websocket should be called twice: once for welcome, once for broadcast
        assert self.mock_websocket1.send_text.call_count == 2
        assert self.mock_websocket2.send_text.call_count == 2

        # Check that the broadcast message was sent
        broadcast_call = json.dumps(message)
        assert any(call[0][0] == broadcast_call for call in self.mock_websocket1.send_text.call_args_list)
        assert any(call[0][0] == broadcast_call for call in self.mock_websocket2.send_text.call_args_list)

    @pytest.mark.asyncio
    async def test_broadcast_to_user(self):
        """Test broadcasting message to specific user."""
        await self.manager.connect(self.mock_websocket1, "user_123")
        await self.manager.connect(self.mock_websocket2, "user_456")

        message = {"type": "user_message", "data": "test"}

        await self.manager.broadcast_to_user("user_123", message)

        # websocket1 should be called twice: welcome + user message
        assert self.mock_websocket1.send_text.call_count == 2
        # websocket2 should be called once: only welcome message
        assert self.mock_websocket2.send_text.call_count == 1

        # Check that the user message was sent to websocket1
        user_message_call = json.dumps(message)
        assert any(call[0][0] == user_message_call for call in self.mock_websocket1.send_text.call_args_list)

    @pytest.mark.asyncio
    async def test_broadcast_to_channel(self):
        """Test broadcasting message to channel subscribers."""
        conn_id1 = await self.manager.connect(self.mock_websocket1, "user_123")
        conn_id2 = await self.manager.connect(self.mock_websocket2, "user_456")

        # Subscribe to channel
        connection1 = self.manager.get_connection(conn_id1)
        connection1.add_subscription("strategy_updates")

        message = {"type": "strategy_update", "data": {}}

        await self.manager.broadcast_to_channel("strategy_updates", message)

        # websocket1 should be called twice: welcome + channel message
        assert self.mock_websocket1.send_text.call_count == 2
        # websocket2 should be called once: only welcome message
        assert self.mock_websocket2.send_text.call_count == 1

        # Check that the channel message was sent to websocket1
        channel_message_call = json.dumps(message)
        assert any(call[0][0] == channel_message_call for call in self.mock_websocket1.send_text.call_args_list)


class TestWebSocketManager:
    """Test cases for WebSocket manager."""

    def setup_method(self):
        """Set up test dependencies."""
        self.manager = WebSocketManager()

    @pytest.mark.asyncio
    async def test_handle_connection_success(self):
        """Test successful WebSocket connection handling."""
        mock_websocket = AsyncMock()
        mock_user = Mock()
        mock_user.id = 123
        mock_user.role = "trader"

        # Mock authentication
        with patch('src.api.websocket_manager.get_current_user') as mock_auth:
            mock_auth.return_value = mock_user

            # Mock WebSocket accept and receive
            mock_websocket.accept.return_value = None
            mock_websocket.receive_text.side_effect = [
                json.dumps({"type": "ping"}),
                json.dumps({"type": "subscribe", "channel": "strategy_updates"}),
                # Simulate disconnect
                Exception("WebSocket disconnected")
            ]

            # Should handle connection without raising exception
            await self.manager.handle_connection(mock_websocket, "fake_token")

            mock_websocket.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_connection_authentication_failure(self):
        """Test WebSocket connection with authentication failure."""
        mock_websocket = AsyncMock()

        # Mock authentication failure
        with patch('src.api.websocket_manager.get_current_user') as mock_auth:
            mock_auth.side_effect = Exception("Invalid token")

            await self.manager.handle_connection(mock_websocket, "invalid_token")

            mock_websocket.close.assert_called_once_with(code=4001, reason="Authentication failed")

    @pytest.mark.asyncio
    async def test_handle_ping_message(self):
        """Test handling ping message."""
        mock_connection = Mock()
        mock_connection.update_last_ping = Mock()
        mock_connection.send_message = AsyncMock(return_value=True)

        message = {"type": "ping", "timestamp": "2024-01-01T00:00:00Z"}

        await self.manager.handle_message(mock_connection, message)

        mock_connection.update_last_ping.assert_called_once()
        mock_connection.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_subscribe_message(self):
        """Test handling subscribe message."""
        mock_connection = Mock()
        mock_connection.add_subscription = Mock()
        mock_connection.send_message = AsyncMock(return_value=True)

        message = {"type": "subscribe", "channel": "strategy_updates"}

        await self.manager.handle_message(mock_connection, message)

        mock_connection.add_subscription.assert_called_once_with("strategy_updates")
        mock_connection.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_unsubscribe_message(self):
        """Test handling unsubscribe message."""
        mock_connection = Mock()
        mock_connection.remove_subscription = Mock()
        mock_connection.send_message = AsyncMock(return_value=True)

        message = {"type": "unsubscribe", "channel": "strategy_updates"}

        await self.manager.handle_message(mock_connection, message)

        mock_connection.remove_subscription.assert_called_once_with("strategy_updates")
        mock_connection.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_unknown_message(self):
        """Test handling unknown message type."""
        mock_connection = Mock()
        mock_connection.send_message = AsyncMock(return_value=True)

        message = {"type": "unknown_type", "data": {}}

        await self.manager.handle_message(mock_connection, message)

        # Should send error response
        mock_connection.send_message.assert_called_once()
        call_args = mock_connection.send_message.call_args[0][0]
        assert call_args["type"] == "error"

    @pytest.mark.asyncio
    async def test_broadcast_strategy_update(self):
        """Test broadcasting strategy update."""
        with patch.object(self.manager.connection_manager, 'broadcast_to_channel') as mock_broadcast:
            strategy_data = {
                "strategy_id": "test_strategy",
                "status": "running",
                "uptime": 3600
            }

            await self.manager.broadcast_strategy_update(strategy_data)

            mock_broadcast.assert_called_once()
            call_args = mock_broadcast.call_args[0]
            assert call_args[0] == "strategy_updates"
            assert call_args[1]["type"] == "strategy_update"
            assert call_args[1]["data"] == strategy_data

    @pytest.mark.asyncio
    async def test_broadcast_system_metrics(self):
        """Test broadcasting system metrics."""
        with patch.object(self.manager.connection_manager, 'broadcast_to_channel') as mock_broadcast:
            metrics_data = {
                "cpu_percent": 25.5,
                "memory_percent": 60.2,
                "timestamp": "2024-01-01T00:00:00Z"
            }

            await self.manager.broadcast_system_metrics(metrics_data)

            mock_broadcast.assert_called_once()
            call_args = mock_broadcast.call_args[0]
            assert call_args[0] == "system_metrics"
            assert call_args[1]["type"] == "system_metrics"
            assert call_args[1]["data"] == metrics_data

    @pytest.mark.asyncio
    async def test_broadcast_notification(self):
        """Test broadcasting notification."""
        with patch.object(self.manager.connection_manager, 'broadcast_to_all') as mock_broadcast:
            notification = {
                "title": "Test Notification",
                "message": "This is a test",
                "severity": "info"
            }

            await self.manager.broadcast_notification(notification)

            mock_broadcast.assert_called_once()
            call_args = mock_broadcast.call_args[0][0]
            assert call_args["type"] == "notification"
            assert call_args["data"] == notification

    def test_get_manager_stats(self):
        """Test getting manager statistics."""
        with patch.object(self.manager.connection_manager, 'get_connection_stats') as mock_stats:
            mock_stats.return_value = {
                "total_connections": 5,
                "unique_users": 3
            }

            stats = self.manager.get_manager_stats()

            assert stats["total_connections"] == 5
            assert stats["unique_users"] == 3
            assert "uptime_seconds" in stats

    @pytest.mark.asyncio
    async def test_cleanup_stale_connections(self):
        """Test cleaning up stale connections."""
        # This would test the periodic cleanup task
        # For now, just verify the method exists and can be called
        await self.manager.cleanup_stale_connections()

        # No assertions needed as this is a cleanup method
        # In a real implementation, we'd mock the connection manager
        # and verify stale connections are removed


class TestMessageTypes:
    """Test cases for message type constants and validation."""

    def test_message_type_constants(self):
        """Test message type constants are defined."""
        assert hasattr(MessageType, 'PING')
        assert hasattr(MessageType, 'PONG')
        assert hasattr(MessageType, 'SUBSCRIBE')
        assert hasattr(MessageType, 'UNSUBSCRIBE')
        assert hasattr(MessageType, 'STRATEGY_UPDATE')
        assert hasattr(MessageType, 'SYSTEM_METRICS')
        assert hasattr(MessageType, 'NOTIFICATION')
        assert hasattr(MessageType, 'ERROR')

    def test_broadcast_channel_constants(self):
        """Test broadcast channel constants are defined."""
        assert hasattr(BroadcastChannel, 'STRATEGY_UPDATES')
        assert hasattr(BroadcastChannel, 'SYSTEM_METRICS')
        assert hasattr(BroadcastChannel, 'NOTIFICATIONS')
        assert hasattr(BroadcastChannel, 'ALERTS')


if __name__ == "__main__":
    pytest.main([__file__])