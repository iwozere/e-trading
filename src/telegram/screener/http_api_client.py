#!/usr/bin/env python3
"""
HTTP API Client for Bot Microservice
Provides a client interface for background services to send notifications via the bot's HTTP API.
"""

import asyncio
import aiohttp
from typing import Dict, Any, Optional
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class BotHttpApiClient:
    """
    HTTP API client for communicating with the bot microservice.
    Provides methods to send notifications without direct notification manager dependency.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def send_message(self, user_id: str, message: str, title: str = "Alkotrader Notification") -> bool:
        """
        Send a message to a specific user via the bot API.

        Args:
            user_id: Telegram user ID
            message: Message content
            title: Message title

        Returns:
            bool: True if message was queued successfully
        """
        try:
            if not self.session:
                raise RuntimeError("Client not initialized. Use async context manager.")

            async with self.session.post(f"{self.base_url}/api/send_message", json={
                "user_id": user_id,
                "message": message,
                "title": title
            }) as response:
                if response.status == 200:
                    result = await response.json()
                    success = result.get('success', False)
                    if success:
                        _logger.debug("Message queued successfully for user %s", user_id)
                    else:
                        _logger.warning("Failed to queue message for user %s: %s", user_id, result.get('message', 'Unknown error'))
                    return success
                else:
                    error_text = await response.text()
                    _logger.error("HTTP API error for user %s: %d - %s", user_id, response.status, error_text)
                    return False

        except Exception:
            _logger.exception("Error sending message to user %s: ", user_id)
            return False

    async def send_broadcast(self, message: str, title: str = "Alkotrader Announcement") -> Dict[str, Any]:
        """
        Send a broadcast message to all users via the bot API.

        Args:
            message: Message content
            title: Message title

        Returns:
            Dict with success status and counts
        """
        try:
            if not self.session:
                raise RuntimeError("Client not initialized. Use async context manager.")

            async with self.session.post(f"{self.base_url}/api/broadcast", json={
                "message": message,
                "title": title
            }) as response:
                if response.status == 200:
                    result = await response.json()
                    success = result.get('success', False)
                    if success:
                        success_count = result.get('success_count', 0)
                        total_count = result.get('total_count', 0)
                        _logger.info("Broadcast queued successfully: %d/%d users", success_count, total_count)
                    else:
                        _logger.warning("Failed to queue broadcast: %s", result.get('message', 'Unknown error'))
                    return result
                else:
                    error_text = await response.text()
                    _logger.error("HTTP API broadcast error: %d - %s", response.status, error_text)
                    return {"success": False, "error": error_text}

        except Exception as e:
            _logger.exception("Error sending broadcast: ")
            return {"success": False, "error": str(e)}

    async def get_status(self) -> Dict[str, Any]:
        """
        Get the bot API status and statistics.

        Returns:
            Dict with status information
        """
        try:
            if not self.session:
                raise RuntimeError("Client not initialized. Use async context manager.")

            async with self.session.get(f"{self.base_url}/api/status") as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    error_text = await response.text()
                    _logger.error("HTTP API status error: %d - %s", response.status, error_text)
                    return {"success": False, "error": error_text}

        except Exception as e:
            _logger.exception("Error getting status: ")
            return {"success": False, "error": str(e)}

    async def test_connection(self) -> bool:
        """
        Test the connection to the bot API.

        Returns:
            bool: True if connection is successful
        """
        try:
            if not self.session:
                raise RuntimeError("Client not initialized. Use async context manager.")

            async with self.session.get(f"{self.base_url}/api/test") as response:
                if response.status == 200:
                    result = await response.json()
                    _logger.debug("Bot API connection test successful: %s", result)
                    return True
                else:
                    _logger.error("Bot API connection test failed: HTTP %d", response.status)
                    return False

        except Exception:
            _logger.exception("Error testing bot API connection: ")
            return False


# Convenience function for one-off API calls
async def send_notification_via_api(user_id: str, message: str, title: str = "Alkotrader Notification") -> bool:
    """
    Convenience function to send a single notification via the bot API.

    Args:
        user_id: Telegram user ID
        message: Message content
        title: Message title

    Returns:
        bool: True if message was queued successfully
    """
    async with BotHttpApiClient() as client:
        return await client.send_message(user_id, message, title)


async def send_broadcast_via_api(message: str, title: str = "Alkotrader Announcement") -> Dict[str, Any]:
    """
    Convenience function to send a broadcast via the bot API.

    Args:
        message: Message content
        title: Message title

    Returns:
        Dict with success status and counts
    """
    async with BotHttpApiClient() as client:
        return await client.send_broadcast(message, title)


async def send_notification_to_admins(message: str, title: str = "System Notification") -> Dict[str, Any]:
    """
    Send a notification to all admin users via the bot API.

    Args:
        message: Message content
        title: Message title

    Returns:
        Dict with success status and counts
    """
    try:
        from src.data.db.services import telegram_service as db

        # Get all admin user IDs
        admin_ids = db.get_admin_user_ids()

        if not admin_ids:
            return {
                "success": False,
                "error": "No admin users found",
                "success_count": 0,
                "total_count": 0
            }

        success_count = 0
        total_count = len(admin_ids)

        # Send message to each admin
        for admin_id in admin_ids:
            try:
                success = await send_notification_via_api(
                    user_id=admin_id,
                    message=message,
                    title=title
                )
                if success:
                    success_count += 1
            except Exception as e:
                _logger.error("Failed to send notification to admin %s: %s", admin_id, e)

        return {
            "success": success_count > 0,
            "success_count": success_count,
            "total_count": total_count,
            "message": f"Notification sent to {success_count}/{total_count} admin users"
        }

    except Exception as e:
        _logger.exception("Error sending notification to admins:")
        return {
            "success": False,
            "error": str(e),
            "success_count": 0,
            "total_count": 0
        }


# Test function
async def test_api_client():
    """Test the HTTP API client."""
    print("Testing Bot HTTP API Client...")

    async with BotHttpApiClient() as client:
        # Test connection
        print("\n1. Testing connection...")
        if await client.test_connection():
            print("✅ Connection test passed")
        else:
            print("❌ Connection test failed")
            return

        # Test status
        print("\n2. Testing status endpoint...")
        status = await client.get_status()
        if status.get('success'):
            print(f"✅ Status test passed: {status}")
        else:
            print(f"❌ Status test failed: {status}")

        # Test broadcast (should work even without bot running)
        print("\n3. Testing broadcast endpoint...")
        broadcast_result = await client.send_broadcast("Test broadcast message", "Test Announcement")
        if broadcast_result.get('success'):
            print(f"✅ Broadcast test passed: {broadcast_result}")
        else:
            print(f"❌ Broadcast test failed: {broadcast_result}")


if __name__ == "__main__":
    asyncio.run(test_api_client())
