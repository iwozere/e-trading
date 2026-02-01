"""
Notification Service Client

Client library for interacting with the notification service API.
Provides a simple interface for sending notifications and checking delivery status.
"""

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class NotificationServiceError(Exception):
    """Base exception for notification service errors."""
    pass


class NotificationServiceUnavailableError(NotificationServiceError):
    """Raised when the notification service is unavailable."""
    pass


class NotificationRequest:
    """Notification request data structure."""

    def __init__(self, notification_type: str, title: str, message: str, **kwargs):
        self.notification_type = notification_type
        self.title = title
        self.message = message
        self.kwargs = kwargs


class NotificationResponse:
    """Notification response data structure."""

    def __init__(self, message_id: int, status: str, **kwargs):
        self.message_id = message_id
        self.status = status
        self.kwargs = kwargs


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Simple circuit breaker implementation."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED

    def call(self, func, *args, **kwargs):
        """Call function through circuit breaker."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise NotificationServiceUnavailableError("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        if self.last_failure_time is None:
            return True
        return (datetime.now(timezone.utc) - self.last_failure_time).seconds > self.recovery_timeout

    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN


class MessagePriority(str, Enum):
    """Message priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class MessageType(str, Enum):
    """Message types for categorization."""
    TRADE_ENTRY = "trade_entry"
    TRADE_EXIT = "trade_exit"
    ERROR = "error"
    ALERT = "alert"
    REPORT = "report"
    SYSTEM = "system"
    INFO = "info"


class NotificationServiceClient:
    """
    Client for interacting with the notification service API.

    Provides a simple interface for sending notifications and checking delivery status.
    Compatible with AsyncNotificationManager interface for easy migration.
    """

    def __init__(self,
                 service_url: Optional[str] = None,
                 base_url: str = None,  # For backward compatibility
                 timeout: int = 30,
                 max_retries: int = 3):
        """
        Initialize the notification service client.

        NOTE: This client now connects to the Main API Service instead of
        the notification service directly, as part of the database-centric architecture.

        Args:
            service_url: Base URL of the Main API service. If None, checks NOTIFICATION_SERVICE_URL env var or defaults to localhost.
            base_url: Deprecated parameter for backward compatibility
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        import os
        if service_url is None:
            service_url = os.environ.get("NOTIFICATION_SERVICE_URL", "http://localhost:8000")
        # Handle backward compatibility
        if base_url is not None:
            service_url = base_url

        # Check for database-only mode
        self.database_only_mode = service_url.startswith("database://")

        if self.database_only_mode:
            _logger.info("NotificationServiceClient configured for database-only mode (no HTTP API)")
            self.service_url = service_url  # Keep the database:// URL for identification
        else:
            # Ensure we're using the Main API service, not the notification service
            if ":8080" in service_url or ":5003" in service_url:
                _logger.warning(
                    "Redirecting notification service client from %s to Main API at http://localhost:8000 "
                    "(database-centric architecture)", service_url
                )
                service_url = "http://localhost:8000"

            # Always ensure we're pointing to the Main API service for database-centric architecture
            if not service_url.endswith(":8000") and "localhost" in service_url:
                service_url = "http://localhost:8000"

            self.service_url = service_url.rstrip('/')

        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None

        if self.database_only_mode:
            _logger.info("NotificationServiceClient initialized for database-only mode")
        else:
            _logger.info("NotificationServiceClient initialized for Main API at %s", self.service_url)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic.

        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional request parameters

        Returns:
            Response data

        Raises:
            Exception: If request fails after all retries
        """
        session = await self._get_session()
        url = f"{self.service_url}{endpoint}"

        for attempt in range(self.max_retries + 1):
            try:
                async with session.request(method, url, **kwargs) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 404:
                        raise ValueError(f"Resource not found: {endpoint}")
                    elif response.status >= 400:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")

            except aiohttp.ClientError as e:
                if attempt == self.max_retries:
                    raise Exception(f"Request failed after {self.max_retries + 1} attempts: {e}")

                # Wait before retry (exponential backoff)
                wait_time = 2 ** attempt
                _logger.warning("Request failed (attempt %d/%d), retrying in %ds: %s",
                              attempt + 1, self.max_retries + 1, wait_time, e)
                await asyncio.sleep(wait_time)

    async def send_notification(self,
                              notification_type: Union[str, MessageType],
                              title: str,
                              message: str,
                              priority: Union[str, MessagePriority] = MessagePriority.NORMAL,
                              data: Optional[Dict[str, Any]] = None,
                              source: str = "trading_bot",
                              channels: Optional[List[str]] = None,
                              attachments: Optional[dict] = None,
                              email_receiver: Optional[str] = None,
                              reply_to_message_id: Optional[int] = None,
                              telegram_chat_id: Optional[int] = None,
                              recipient_id: Optional[str] = None) -> bool:
        """
        Send a notification through the service.

        Compatible with AsyncNotificationManager.send_notification interface.

        Args:
            notification_type: Type of notification
            title: Notification title
            message: Notification message
            priority: Notification priority
            data: Additional data for the notification
            source: Source of the notification
            channels: Specific channels to use (None for all enabled channels)
            attachments: List of attachments to include with the notification
            email_receiver: Receiver email address for email notifications
            reply_to_message_id: Reply to message ID for Telegram messages
            telegram_chat_id: Telegram chat ID for Telegram messages
            recipient_id: Recipient user ID

        Returns:
            True if notification was queued successfully
        """
        try:
            # Convert enums to strings
            if isinstance(notification_type, MessageType):
                notification_type = notification_type.value
            if isinstance(priority, MessagePriority):
                priority = priority.value

            # Build message data
            message_data = {
                "message_type": str(notification_type),
                "priority": str(priority),
                "channels": channels or ["telegram", "email"],
                "recipient_id": recipient_id or email_receiver,
                "template_name": None,
                "content": {
                    "title": title,
                    "message": message,
                    "source": source
                },
                "message_metadata": data or {}
            }

            # Add compatibility data for legacy parameters
            if attachments:
                message_data["message_metadata"]["attachments"] = attachments

            if reply_to_message_id is not None:
                message_data["message_metadata"]["reply_to_message_id"] = reply_to_message_id

            if telegram_chat_id is not None:
                message_data["message_metadata"]["telegram_chat_id"] = telegram_chat_id

            if email_receiver:
                message_data["message_metadata"]["email_receiver"] = email_receiver

            # Skip HTTP API if in database-only mode
            if self.database_only_mode:
                _logger.debug("Database-only mode: skipping HTTP API, using direct database insertion")
            else:
                # Try HTTP API first
                try:
                    response = await self._make_request(
                        "POST",
                        "/api/notifications",
                        json=message_data,
                        headers={"Content-Type": "application/json"}
                    )

                    _logger.info("Notification sent successfully via HTTP API: %s", response.get("message_id"))
                    return True

                except Exception as api_error:
                    _logger.warning("HTTP API failed, attempting direct database fallback: %s", api_error)

            # Use direct database insertion (either as fallback or primary method)
            try:
                await self._send_notification_direct_to_db(message_data)
                if self.database_only_mode:
                    _logger.info("Notification queued successfully via direct database insertion")
                else:
                    _logger.info("Notification queued successfully via direct database fallback")
                return True

            except Exception as db_error:
                if self.database_only_mode:
                    _logger.error("Database insertion failed: %s", db_error)
                else:
                    _logger.error("Database fallback failed: %s", db_error)
                return False

        except Exception:
            _logger.exception("Failed to send notification:")
            return False

    async def _send_notification_direct_to_db(self, message_data: Dict[str, Any]) -> None:
        """
        Directly insert notification into database when HTTP API is unavailable.

        This fallback method ensures notifications are always queued even when
        the notification service HTTP API is not available.

        Args:
            message_data: Message data dictionary prepared for the API

        Raises:
            Exception: If database insertion fails
        """
        try:
            # Import here to avoid circular dependencies and reduce startup time
            from src.data.db.services.database_service import get_database_service
            from src.data.db.repos.repo_notification import MessageRepository
            from datetime import datetime, timezone
            import base64

            # Use the database service directly
            db_service = get_database_service()

            with db_service.uow() as uow:
                # Create message repository
                message_repo = MessageRepository(uow.s)

                # Process attachments if present
                processed_attachments = None
                if "attachments" in message_data.get("message_metadata", {}):
                    attachments = message_data["message_metadata"]["attachments"]
                    processed_attachments = {}

                    for filename, file_data in attachments.items():
                        if isinstance(file_data, bytes):
                            # Convert bytes to base64 for JSON storage
                            processed_attachments[filename] = {
                                "data": base64.b64encode(file_data).decode('utf-8'),
                                "type": "base64",
                                "size": len(file_data)
                            }
                        elif isinstance(file_data, str):
                            # File path
                            processed_attachments[filename] = {
                                "path": file_data,
                                "type": "file_path"
                            }
                        else:
                            # Already processed attachment data
                            processed_attachments[filename] = file_data

                # Prepare database message data
                # Convert priority to uppercase to match database constraints
                priority = message_data["priority"].upper() if message_data["priority"] else "NORMAL"

                # Prepare metadata without attachments (they're moved to content)
                metadata_copy = {**message_data.get("message_metadata", {})}
                metadata_copy.pop("attachments", None)  # Remove attachments from metadata
                metadata_copy["fallback_method"] = "direct_db"
                metadata_copy["fallback_timestamp"] = datetime.now(timezone.utc).isoformat()

                db_message_data = {
                    "message_type": message_data["message_type"],
                    "priority": priority,
                    "channels": message_data["channels"],
                    "recipient_id": message_data["recipient_id"],
                    "template_name": message_data.get("template_name"),
                    "content": {
                        **message_data["content"],
                        "attachments": processed_attachments
                    } if processed_attachments else message_data["content"],
                    "message_metadata": metadata_copy,
                    "scheduled_for": datetime.now(timezone.utc),
                    "max_retries": 3,
                    "retry_count": 0
                }

                # Create the message in database
                message = message_repo.create_message(db_message_data)
                _logger.info("Created notification message %s directly in database via fallback", message.id)
                # Commit is handled automatically by the UOW context manager

        except Exception:
            _logger.exception("Failed to insert notification directly to database:")
            raise

    async def send_trade_notification(self,
                                    symbol: str,
                                    side: str,
                                    price: float,
                                    quantity: float,
                                    entry_price: Optional[float] = None,
                                    pnl: Optional[float] = None,
                                    exit_type: Optional[str] = None,
                                    recipient_id: Optional[str] = None) -> bool:
        """
        Send a trade notification.

        Compatible with AsyncNotificationManager.send_trade_notification interface.

        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            price: Trade price
            quantity: Trade quantity
            entry_price: Entry price (for exits)
            pnl: Profit/loss percentage
            exit_type: Exit type (TP/SL)
            recipient_id: Recipient user ID

        Returns:
            True if notification was queued successfully
        """
        if side.upper() == "BUY":
            notification_type = MessageType.TRADE_ENTRY
            title = f"Buy Order: {symbol}"
            message = f"Buy {quantity} {symbol} at {price}"
        else:
            notification_type = MessageType.TRADE_EXIT
            title = f"Sell Order: {symbol}"
            message = f"Sell {quantity} {symbol} at {price}"
            if pnl is not None:
                message += f" (PnL: {pnl:.2f}%)"
            if exit_type:
                message += f" ({exit_type})"

        data = {
            "symbol": symbol,
            "side": side.upper(),
            "price": price,
            "quantity": quantity,
            "entry_price": entry_price,
            "pnl": pnl,
            "exit_type": exit_type,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        return await self.send_notification(
            notification_type=notification_type,
            title=title,
            message=message,
            priority=MessagePriority.HIGH,
            data=data,
            source="trading_bot",
            recipient_id=recipient_id
        )

    async def send_error_notification(self,
                                    error_message: str,
                                    source: str = "trading_bot",
                                    recipient_id: Optional[str] = None) -> bool:
        """
        Send an error notification.

        Compatible with AsyncNotificationManager.send_error_notification interface.

        Args:
            error_message: Error message
            source: Source of the error
            recipient_id: Recipient user ID

        Returns:
            True if notification was queued successfully
        """
        return await self.send_notification(
            notification_type=MessageType.ERROR,
            title="Error Alert",
            message=error_message,
            priority=MessagePriority.CRITICAL,
            source=source,
            recipient_id=recipient_id
        )

    async def get_message_status(self, message_id: int) -> Optional[Dict[str, Any]]:
        """
        Get message status and details.

        Args:
            message_id: Message ID

        Returns:
            Message details and status, or None if not found
        """
        try:
            response = await self._make_request("GET", f"/api/notifications/{message_id}")
            return response
        except ValueError:
            return None
        except Exception:
            _logger.exception("Failed to get message status:")
            return None

    async def get_delivery_status(self, message_id: int) -> List[Dict[str, Any]]:
        """
        Get delivery status for all channels of a message.

        Args:
            message_id: Message ID

        Returns:
            List of delivery statuses
        """
        try:
            response = await self._make_request("GET", f"/api/notifications/{message_id}/delivery")
            return response
        except Exception:
            _logger.exception("Failed to get delivery status:")
            return []

    async def list_messages(self,
                          status: Optional[str] = None,
                          priority: Optional[str] = None,
                          recipient_id: Optional[str] = None,
                          message_type: Optional[str] = None,
                          limit: int = 100,
                          offset: int = 0) -> List[Dict[str, Any]]:
        """
        List messages with optional filtering.

        Args:
            status: Filter by message status
            priority: Filter by message priority
            recipient_id: Filter by recipient ID
            message_type: Filter by message type
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of messages
        """
        try:
            params = {}
            if status:
                params["status"] = status
            if priority:
                params["priority"] = priority
            if recipient_id:
                params["recipient_id"] = recipient_id
            if message_type:
                params["message_type"] = message_type
            if limit:
                params["limit"] = limit
            if offset:
                params["offset"] = offset

            response = await self._make_request("GET", "/api/notifications", params=params)
            return response
        except Exception:
            _logger.exception("Failed to list messages:")
            return []

    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get service health status.

        Returns:
            Health status information
        """
        try:
            response = await self._make_request("GET", "/api/notifications/health")
            return response
        except Exception as e:
            _logger.exception("Failed to get health status:")
            return {"status": "unhealthy", "error": str(e)}

    async def get_channels_health(self) -> List[Dict[str, Any]]:
        """
        Get health status for all channels.

        Returns:
            List of channel health statuses
        """
        try:
            response = await self._make_request("GET", "/api/notifications/channels/health")
            return response
        except Exception:
            _logger.exception("Failed to get channels health:")
            return []

    async def send_to_admins(self,
                           title: str,
                           message: str,
                           notification_type: Union[str, MessageType] = MessageType.SYSTEM,
                           priority: Union[str, MessagePriority] = MessagePriority.HIGH,
                           data: Optional[Dict[str, Any]] = None,
                           channels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Send notification to all admin users.

        Args:
            title: Notification title
            message: Notification message
            notification_type: Type of notification
            priority: Notification priority (default: HIGH)
            data: Additional data for the notification
            channels: Specific channels to use (default: ["telegram"])

        Returns:
            Dict with success status and counts:
            {
                "success": bool,
                "success_count": int,
                "total_count": int,
                "message": str
            }
        """
        try:
            from src.data.db.services import telegram_service as db

            # Get all admin user IDs
            admin_ids = db.get_admin_user_ids()

            if not admin_ids:
                _logger.warning("No admin users found for notification")
                return {
                    "success": False,
                    "error": "No admin users found",
                    "success_count": 0,
                    "total_count": 0
                }

            success_count = 0
            total_count = len(admin_ids)
            failed_admins = []

            # Use telegram channel by default for admin notifications
            notification_channels = channels or ["telegram"]

            # Send message to each admin
            for admin_id in admin_ids:
                try:
                    # Prepare notification data
                    notification_data = data.copy() if data else {}
                    notification_data["admin_notification"] = True
                    notification_data["telegram_chat_id"] = admin_id

                    success = await self.send_notification(
                        notification_type=notification_type,
                        title=title,
                        message=message,
                        priority=priority,
                        data=notification_data,
                        channels=notification_channels,
                        telegram_chat_id=admin_id,
                        recipient_id=str(admin_id)
                    )

                    if success:
                        success_count += 1
                    else:
                        failed_admins.append(admin_id)

                except Exception as e:
                    _logger.error("Failed to send notification to admin %s: %s", admin_id, e)
                    failed_admins.append(admin_id)

            result = {
                "success": success_count > 0,
                "success_count": success_count,
                "total_count": total_count,
                "message": f"Notification sent to {success_count}/{total_count} admin users"
            }

            if failed_admins:
                result["failed_admins"] = failed_admins

            _logger.info("Admin notification sent: %d/%d successful", success_count, total_count)
            return result

        except Exception as e:
            _logger.exception("Error sending notification to admins:")
            return {
                "success": False,
                "error": str(e),
                "success_count": 0,
                "total_count": 0
            }

    # Context manager support
    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    # Compatibility methods for AsyncNotificationManager interface
    async def start(self):
        """Start the client (compatibility method)."""
        # No-op for client, service handles processing
        pass

    async def stop(self):
        """Stop the client (compatibility method)."""
        await self.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics (compatibility method)."""
        return {
            "service_url": self.service_url,
            "timeout": self.timeout.total,
            "max_retries": self.max_retries,
            "session_active": self._session is not None and not self._session.closed
        }


# Global client instance for easy access
_notification_client: Optional[NotificationServiceClient] = None


def get_notification_client() -> Optional[NotificationServiceClient]:
    """Get the global notification client instance."""
    return _notification_client


async def initialize_notification_client(service_url: str = "http://localhost:8000", **kwargs) -> NotificationServiceClient:
    """
    Initialize the global notification client.

    Args:
        service_url: Base URL of the notification service
        **kwargs: Additional client configuration

    Returns:
        Initialized notification client
    """
    global _notification_client

    if _notification_client is not None:
        await _notification_client.close()

    _notification_client = NotificationServiceClient(service_url=service_url, **kwargs)
    _logger.info("Notification service client initialized with URL: %s", service_url)

    return _notification_client