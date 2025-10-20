"""
Notification Service Client Library

HTTP client for the notification service that provides a simple interface
for sending notifications. Includes retry logic, circuit breaker patterns,
and both synchronous and asynchronous interfaces.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

import aiohttp
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.model.notification import NotificationType, NotificationPriority
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Circuit breaker for handling service failures."""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 3

    state: CircuitBreakerState = field(default=CircuitBreakerState.CLOSED)
    failure_count: int = field(default=0)
    last_failure_time: Optional[float] = field(default=None)
    half_open_calls: int = field(default=0)

    def can_execute(self) -> bool:
        """Check if request can be executed."""
        if self.state == CircuitBreakerState.CLOSED:
            return True

        if self.state == CircuitBreakerState.OPEN:
            if self.last_failure_time and time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False

        if self.state == CircuitBreakerState.HALF_OPEN:
            return self.half_open_calls < self.half_open_max_calls

        return False

    def record_success(self):
        """Record successful request."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0

    def record_failure(self):
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitBreakerState.CLOSED and self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN


@dataclass
class NotificationRequest:
    """Notification request data structure."""
    message_type: str
    priority: str = "NORMAL"
    channels: List[str] = field(default_factory=list)
    recipient_id: Optional[str] = None
    template_name: Optional[str] = None
    content: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    scheduled_for: Optional[datetime] = None
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request."""
        data = {
            "message_type": self.message_type,
            "priority": self.priority,
            "channels": self.channels,
            "content": self.content,
            "metadata": self.metadata,
            "max_retries": self.max_retries
        }

        if self.recipient_id:
            data["recipient_id"] = self.recipient_id

        if self.template_name:
            data["template_name"] = self.template_name

        if self.scheduled_for:
            data["scheduled_for"] = self.scheduled_for.isoformat()

        return data


@dataclass
class NotificationResponse:
    """Notification response from the service."""
    message_id: int
    status: str
    channels: List[str]
    priority: str


class NotificationServiceError(Exception):
    """Base exception for notification service errors."""
    pass


class NotificationServiceUnavailableError(NotificationServiceError):
    """Exception raised when notification service is unavailable."""
    pass


class NotificationServiceClient:
    """
    HTTP client for the notification service.

    Provides both synchronous and asynchronous interfaces for sending notifications.
    Includes retry logic, circuit breaker patterns, and proper error handling.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: int = 30,
        max_retries: int = 3,
        retry_backoff_factor: float = 0.3,
        circuit_breaker_enabled: bool = True,
        api_key: Optional[str] = None
    ):
        """
        Initialize the notification service client.

        Args:
            base_url: Base URL of the notification service
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_backoff_factor: Backoff factor for retries
            circuit_breaker_enabled: Whether to enable circuit breaker
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor
        self.api_key = api_key

        # Circuit breaker
        self.circuit_breaker = CircuitBreaker() if circuit_breaker_enabled else None

        # Setup synchronous session with retry strategy
        self._setup_sync_session()

        # Async session will be created when needed
        self._async_session: Optional[aiohttp.ClientSession] = None

        self._logger = setup_logger(f"{__name__}.NotificationServiceClient")

    def _setup_sync_session(self):
        """Setup synchronous requests session with retry strategy."""
        self.session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.retry_backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set default headers
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "NotificationServiceClient/1.0"
        })

        if self.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}"
            })

    async def _get_async_session(self) -> aiohttp.ClientSession:
        """Get or create async session."""
        if self._async_session is None or self._async_session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "NotificationServiceClient/1.0"
            }

            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            self._async_session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers
            )

        return self._async_session

    def _check_circuit_breaker(self):
        """Check circuit breaker before making request."""
        if self.circuit_breaker and not self.circuit_breaker.can_execute():
            raise NotificationServiceUnavailableError(
                f"Circuit breaker is {self.circuit_breaker.state.value}"
            )

    def _record_success(self):
        """Record successful request."""
        if self.circuit_breaker:
            self.circuit_breaker.record_success()

    def _record_failure(self):
        """Record failed request."""
        if self.circuit_breaker:
            self.circuit_breaker.record_failure()

    def send_notification(
        self,
        notification_type: Union[NotificationType, str],
        title: str,
        message: str,
        priority: Union[NotificationPriority, str] = NotificationPriority.NORMAL,
        channels: Optional[List[str]] = None,
        recipient_id: Optional[str] = None,
        attachments: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> NotificationResponse:
        """
        Send a notification synchronously.

        Args:
            notification_type: Type of notification
            title: Notification title (used as subject for email)
            message: Notification message content
            priority: Message priority
            channels: List of channels to send to
            recipient_id: Recipient identifier
            attachments: Attachments to include
            metadata: Additional metadata
            **kwargs: Additional arguments for backward compatibility

        Returns:
            NotificationResponse with message ID and status

        Raises:
            NotificationServiceError: If the request fails
            NotificationServiceUnavailableError: If service is unavailable
        """
        self._check_circuit_breaker()

        try:
            # Convert enums to strings
            if isinstance(notification_type, NotificationType):
                notification_type = notification_type.value
            if isinstance(priority, NotificationPriority):
                priority = priority.value

            # Prepare content
            content = {
                "text": message,
                "subject": title
            }

            # Handle attachments
            if attachments:
                content["attachments"] = attachments

            # Prepare metadata
            request_metadata = metadata or {}

            # Handle backward compatibility arguments
            if kwargs.get("telegram_chat_id"):
                request_metadata["telegram_chat_id"] = kwargs["telegram_chat_id"]
            if kwargs.get("reply_to_message_id"):
                request_metadata["reply_to_message_id"] = kwargs["reply_to_message_id"]
            if kwargs.get("email_receiver"):
                recipient_id = kwargs["email_receiver"]

            # Use default channels if none specified
            if not channels:
                channels = ["telegram", "email"]

            # Create request
            request = NotificationRequest(
                message_type=notification_type,
                priority=priority,
                channels=channels,
                recipient_id=recipient_id,
                content=content,
                metadata=request_metadata
            )

            # Send request
            response = self.session.post(
                f"{self.base_url}/api/v1/messages",
                json=request.to_dict(),
                timeout=self.timeout
            )

            response.raise_for_status()
            data = response.json()

            self._record_success()

            return NotificationResponse(
                message_id=data["message_id"],
                status=data["status"],
                channels=data["channels"],
                priority=data["priority"]
            )

        except requests.exceptions.RequestException as e:
            self._record_failure()
            self._logger.error("Failed to send notification: %s", e)
            raise NotificationServiceError(f"Failed to send notification: {str(e)}")
        except Exception as e:
            self._record_failure()
            self._logger.error("Unexpected error sending notification: %s", e)
            raise NotificationServiceError(f"Unexpected error: {str(e)}")

    async def send_notification_async(
        self,
        notification_type: Union[NotificationType, str],
        title: str,
        message: str,
        priority: Union[NotificationPriority, str] = NotificationPriority.NORMAL,
        channels: Optional[List[str]] = None,
        recipient_id: Optional[str] = None,
        attachments: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> NotificationResponse:
        """
        Send a notification asynchronously.

        Args:
            notification_type: Type of notification
            title: Notification title (used as subject for email)
            message: Notification message content
            priority: Message priority
            channels: List of channels to send to
            recipient_id: Recipient identifier
            attachments: Attachments to include
            metadata: Additional metadata
            **kwargs: Additional arguments for backward compatibility

        Returns:
            NotificationResponse with message ID and status

        Raises:
            NotificationServiceError: If the request fails
            NotificationServiceUnavailableError: If service is unavailable
        """
        self._check_circuit_breaker()

        try:
            # Convert enums to strings
            if isinstance(notification_type, NotificationType):
                notification_type = notification_type.value
            if isinstance(priority, NotificationPriority):
                priority = priority.value

            # Prepare content
            content = {
                "text": message,
                "subject": title
            }

            # Handle attachments
            if attachments:
                content["attachments"] = attachments

            # Prepare metadata
            request_metadata = metadata or {}

            # Handle backward compatibility arguments
            if kwargs.get("telegram_chat_id"):
                request_metadata["telegram_chat_id"] = kwargs["telegram_chat_id"]
            if kwargs.get("reply_to_message_id"):
                request_metadata["reply_to_message_id"] = kwargs["reply_to_message_id"]
            if kwargs.get("email_receiver"):
                recipient_id = kwargs["email_receiver"]

            # Use default channels if none specified
            if not channels:
                channels = ["telegram", "email"]

            # Create request
            request = NotificationRequest(
                message_type=notification_type,
                priority=priority,
                channels=channels,
                recipient_id=recipient_id,
                content=content,
                metadata=request_metadata
            )

            # Send request
            session = await self._get_async_session()
            async with session.post(
                f"{self.base_url}/api/v1/messages",
                json=request.to_dict()
            ) as response:
                response.raise_for_status()
                data = await response.json()

            self._record_success()

            return NotificationResponse(
                message_id=data["message_id"],
                status=data["status"],
                channels=data["channels"],
                priority=data["priority"]
            )

        except aiohttp.ClientError as e:
            self._record_failure()
            self._logger.error("Failed to send notification: %s", e)
            raise NotificationServiceError(f"Failed to send notification: {str(e)}")
        except Exception as e:
            self._record_failure()
            self._logger.error("Unexpected error sending notification: %s", e)
            raise NotificationServiceError(f"Unexpected error: {str(e)}")

    def send_trade_notification(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        entry_price: Optional[float] = None,
        pnl: Optional[float] = None,
        exit_type: Optional[str] = None,
        **kwargs
    ) -> NotificationResponse:
        """
        Send a trade notification synchronously.

        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            price: Trade price
            quantity: Trade quantity
            entry_price: Entry price (for exits)
            pnl: Profit/loss percentage
            exit_type: Exit type (TP/SL)
            **kwargs: Additional arguments

        Returns:
            NotificationResponse with message ID and status
        """
        if side.upper() == "BUY":
            notification_type = NotificationType.TRADE_ENTRY
            title = f"Buy Order: {symbol}"
            message = f"Buy {quantity} {symbol} at {price}"
        else:
            notification_type = NotificationType.TRADE_EXIT
            title = f"Sell Order: {symbol}"
            message = f"Sell {quantity} {symbol} at {price}"
            if pnl is not None:
                message += f" (PnL: {pnl:.2f}%)"
            if exit_type:
                message += f" ({exit_type})"

        metadata = {
            "symbol": symbol,
            "side": side.upper(),
            "price": price,
            "quantity": quantity,
            "entry_price": entry_price,
            "pnl": pnl,
            "exit_type": exit_type,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        return self.send_notification(
            notification_type=notification_type,
            title=title,
            message=message,
            priority=NotificationPriority.HIGH,
            metadata=metadata,
            **kwargs
        )

    async def send_trade_notification_async(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        entry_price: Optional[float] = None,
        pnl: Optional[float] = None,
        exit_type: Optional[str] = None,
        **kwargs
    ) -> NotificationResponse:
        """
        Send a trade notification asynchronously.

        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            price: Trade price
            quantity: Trade quantity
            entry_price: Entry price (for exits)
            pnl: Profit/loss percentage
            exit_type: Exit type (TP/SL)
            **kwargs: Additional arguments

        Returns:
            NotificationResponse with message ID and status
        """
        if side.upper() == "BUY":
            notification_type = NotificationType.TRADE_ENTRY
            title = f"Buy Order: {symbol}"
            message = f"Buy {quantity} {symbol} at {price}"
        else:
            notification_type = NotificationType.TRADE_EXIT
            title = f"Sell Order: {symbol}"
            message = f"Sell {quantity} {symbol} at {price}"
            if pnl is not None:
                message += f" (PnL: {pnl:.2f}%)"
            if exit_type:
                message += f" ({exit_type})"

        metadata = {
            "symbol": symbol,
            "side": side.upper(),
            "price": price,
            "quantity": quantity,
            "entry_price": entry_price,
            "pnl": pnl,
            "exit_type": exit_type,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        return await self.send_notification_async(
            notification_type=notification_type,
            title=title,
            message=message,
            priority=NotificationPriority.HIGH,
            metadata=metadata,
            **kwargs
        )

    def send_error_notification(
        self,
        error_message: str,
        source: str = "trading_bot",
        **kwargs
    ) -> NotificationResponse:
        """
        Send an error notification synchronously.

        Args:
            error_message: Error message
            source: Source of the error
            **kwargs: Additional arguments

        Returns:
            NotificationResponse with message ID and status
        """
        metadata = {
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        return self.send_notification(
            notification_type=NotificationType.ERROR,
            title="Error Alert",
            message=error_message,
            priority=NotificationPriority.CRITICAL,
            metadata=metadata,
            **kwargs
        )

    async def send_error_notification_async(
        self,
        error_message: str,
        source: str = "trading_bot",
        **kwargs
    ) -> NotificationResponse:
        """
        Send an error notification asynchronously.

        Args:
            error_message: Error message
            source: Source of the error
            **kwargs: Additional arguments

        Returns:
            NotificationResponse with message ID and status
        """
        metadata = {
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        return await self.send_notification_async(
            notification_type=NotificationType.ERROR,
            title="Error Alert",
            message=error_message,
            priority=NotificationPriority.CRITICAL,
            metadata=metadata,
            **kwargs
        )

    def get_message_status(self, message_id: int) -> Dict[str, Any]:
        """
        Get the status of a message synchronously.

        Args:
            message_id: Message ID to check

        Returns:
            Message status information

        Raises:
            NotificationServiceError: If the request fails
        """
        self._check_circuit_breaker()

        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/messages/{message_id}/status",
                timeout=self.timeout
            )

            response.raise_for_status()
            self._record_success()

            return response.json()

        except requests.exceptions.RequestException as e:
            self._record_failure()
            self._logger.error("Failed to get message status: %s", e)
            raise NotificationServiceError(f"Failed to get message status: {str(e)}")

    async def get_message_status_async(self, message_id: int) -> Dict[str, Any]:
        """
        Get the status of a message asynchronously.

        Args:
            message_id: Message ID to check

        Returns:
            Message status information

        Raises:
            NotificationServiceError: If the request fails
        """
        self._check_circuit_breaker()

        try:
            session = await self._get_async_session()
            async with session.get(
                f"{self.base_url}/api/v1/messages/{message_id}/status"
            ) as response:
                response.raise_for_status()
                self._record_success()

                return await response.json()

        except aiohttp.ClientError as e:
            self._record_failure()
            self._logger.error("Failed to get message status: %s", e)
            raise NotificationServiceError(f"Failed to get message status: {str(e)}")

    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the notification service synchronously.

        Returns:
            Service health information

        Raises:
            NotificationServiceError: If the request fails
        """
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/health",
                timeout=self.timeout
            )

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            self._logger.error("Health check failed: %s", e)
            raise NotificationServiceError(f"Health check failed: {str(e)}")

    async def health_check_async(self) -> Dict[str, Any]:
        """
        Check the health of the notification service asynchronously.

        Returns:
            Service health information

        Raises:
            NotificationServiceError: If the request fails
        """
        try:
            session = await self._get_async_session()
            async with session.get(f"{self.base_url}/api/v1/health") as response:
                response.raise_for_status()
                return await response.json()

        except aiohttp.ClientError as e:
            self._logger.error("Health check failed: %s", e)
            raise NotificationServiceError(f"Health check failed: {str(e)}")

    def close(self):
        """Close the client and cleanup resources."""
        if self.session:
            self.session.close()

    async def close_async(self):
        """Close the async client and cleanup resources."""
        if self._async_session and not self._async_session.closed:
            await self._async_session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_async()


# Global client instance for backward compatibility
_notification_client: Optional[NotificationServiceClient] = None


def get_notification_client() -> Optional[NotificationServiceClient]:
    """Get the global notification client instance."""
    return _notification_client


def initialize_notification_client(**kwargs) -> NotificationServiceClient:
    """Initialize the global notification client."""
    global _notification_client

    if _notification_client is None:
        _notification_client = NotificationServiceClient(**kwargs)

    return _notification_client


# Convenience functions for backward compatibility
def send_notification(
    notification_type: Union[NotificationType, str],
    title: str,
    message: str,
    **kwargs
) -> bool:
    """Send a notification using the global client."""
    client = get_notification_client()
    if client is None:
        _logger.warning("Notification client not initialized")
        return False

    try:
        client.send_notification(notification_type, title, message, **kwargs)
        return True
    except Exception as e:
        _logger.error("Failed to send notification: %s", e)
        return False


async def send_notification_async(
    notification_type: Union[NotificationType, str],
    title: str,
    message: str,
    **kwargs
) -> bool:
    """Send a notification using the global client asynchronously."""
    client = get_notification_client()
    if client is None:
        _logger.warning("Notification client not initialized")
        return False

    try:
        await client.send_notification_async(notification_type, title, message, **kwargs)
        return True
    except Exception as e:
        _logger.error("Failed to send notification: %s", e)
        return False


def send_trade_notification(**kwargs) -> bool:
    """Send a trade notification using the global client."""
    client = get_notification_client()
    if client is None:
        _logger.warning("Notification client not initialized")
        return False

    try:
        client.send_trade_notification(**kwargs)
        return True
    except Exception as e:
        _logger.error("Failed to send trade notification: %s", e)
        return False


async def send_trade_notification_async(**kwargs) -> bool:
    """Send a trade notification using the global client asynchronously."""
    client = get_notification_client()
    if client is None:
        _logger.warning("Notification client not initialized")
        return False

    try:
        await client.send_trade_notification_async(**kwargs)
        return True
    except Exception as e:
        _logger.error("Failed to send trade notification: %s", e)
        return False


def send_error_notification(error_message: str, **kwargs) -> bool:
    """Send an error notification using the global client."""
    client = get_notification_client()
    if client is None:
        _logger.warning("Notification client not initialized")
        return False

    try:
        client.send_error_notification(error_message, **kwargs)
        return True
    except Exception as e:
        _logger.error("Failed to send error notification: %s", e)
        return False


async def send_error_notification_async(error_message: str, **kwargs) -> bool:
    """Send an error notification using the global client asynchronously."""
    client = get_notification_client()
    if client is None:
        _logger.warning("Notification client not initialized")
        return False

    try:
        await client.send_error_notification_async(error_message, **kwargs)
        return True
    except Exception as e:
        _logger.error("Failed to send error notification: %s", e)
        return False