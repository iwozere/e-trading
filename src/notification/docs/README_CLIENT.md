# Notification Service Client Library

The Notification Service Client Library provides a simple, robust interface for sending notifications through the centralized notification service. It includes retry logic, circuit breaker patterns, and both synchronous and asynchronous interfaces.

## Features

- **Simple Interface**: Easy-to-use methods for common notification types
- **Retry Logic**: Automatic retry with exponential backoff for failed requests
- **Circuit Breaker**: Prevents cascading failures when the service is down
- **Async/Sync Support**: Both synchronous and asynchronous interfaces
- **Backward Compatibility**: Drop-in replacement for AsyncNotificationManager
- **Type Safety**: Full type hints and validation
- **Error Handling**: Comprehensive error handling with specific exception types

## Installation

The client library is part of the notification service package. No additional installation is required.

## Quick Start

### Basic Usage

```python
from src.notification.service.client import NotificationServiceClient
from src.notification.model import NotificationType, NotificationPriority

# Initialize the client
client = NotificationServiceClient(
    base_url="http://localhost:8080",  # Notification service URL
    timeout=30,
    max_retries=3
)

# Send a basic notification
response = client.send_notification(
    notification_type=NotificationType.INFO,
    title="System Alert",
    message="Trading system is online",
    channels=["telegram", "email"]
)

print(f"Notification sent with ID: {response.message_id}")
```

### Async Usage

```python
import asyncio
from src.notification.service.client import NotificationServiceClient

async def send_async_notification():
    client = NotificationServiceClient(base_url="http://localhost:8080")
    
    try:
        response = await client.send_notification_async(
            notification_type="info",
            title="Async Alert",
            message="This was sent asynchronously",
            priority="HIGH"
        )
        print(f"Async notification sent: {response.message_id}")
    finally:
        await client.close_async()

# Run the async function
asyncio.run(send_async_notification())
```

## API Reference

### NotificationServiceClient

The main client class for interacting with the notification service.

#### Constructor

```python
NotificationServiceClient(
    base_url: str = "http://localhost:8080",
    timeout: int = 30,
    max_retries: int = 3,
    retry_backoff_factor: float = 0.3,
    circuit_breaker_enabled: bool = True,
    api_key: Optional[str] = None
)
```

**Parameters:**
- `base_url`: Base URL of the notification service
- `timeout`: Request timeout in seconds
- `max_retries`: Maximum number of retry attempts
- `retry_backoff_factor`: Backoff factor for retries
- `circuit_breaker_enabled`: Whether to enable circuit breaker
- `api_key`: Optional API key for authentication

#### Methods

##### send_notification()

Send a notification synchronously.

```python
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
) -> NotificationResponse
```

**Parameters:**
- `notification_type`: Type of notification (enum or string)
- `title`: Notification title (used as email subject)
- `message`: Notification message content
- `priority`: Message priority (LOW, NORMAL, HIGH, CRITICAL)
- `channels`: List of channels to send to (default: ["telegram", "email"])
- `recipient_id`: Recipient identifier (email, chat ID, etc.)
- `attachments`: Dictionary of filename -> data attachments
- `metadata`: Additional metadata for the notification
- `**kwargs`: Backward compatibility parameters

**Returns:** `NotificationResponse` with message ID and status

##### send_notification_async()

Send a notification asynchronously. Same parameters as `send_notification()`.

##### send_trade_notification()

Send a trade-specific notification synchronously.

```python
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
) -> NotificationResponse
```

##### send_error_notification()

Send an error notification synchronously.

```python
def send_error_notification(
    self,
    error_message: str,
    source: str = "trading_bot",
    **kwargs
) -> NotificationResponse
```

##### get_message_status()

Get the status of a sent message.

```python
def get_message_status(self, message_id: int) -> Dict[str, Any]
```

##### health_check()

Check the health of the notification service.

```python
def health_check(self) -> Dict[str, Any]
```

### Backward Compatibility

For existing code using `AsyncNotificationManager`, you can use the compatibility wrapper:

```python
from src.notification.compatibility import AsyncNotificationManagerCompat

# Drop-in replacement for AsyncNotificationManager
manager = AsyncNotificationManagerCompat(
    telegram_chat_id="123456789",
    email_receiver="trader@example.com",
    notification_service_url="http://localhost:8080"
)

await manager.start()

# Use existing AsyncNotificationManager interface
success = await manager.send_notification(
    notification_type=NotificationType.INFO,
    title="Test",
    message="This works with existing code!",
    priority=NotificationPriority.NORMAL
)
```

## Configuration

### Environment Variables

The client can be configured using environment variables:

```bash
# Notification service URL
NOTIFICATION_SERVICE_URL=http://notification-service:8080

# API key for authentication (optional)
NOTIFICATION_API_KEY=your-api-key-here

# Request timeout
NOTIFICATION_TIMEOUT=30

# Maximum retries
NOTIFICATION_MAX_RETRIES=3
```

### Service Configuration

The notification service itself needs to be configured with channel settings:

```yaml
# notification_service.yaml
channels:
  telegram:
    enabled: true
    bot_token: "${TELEGRAM_BOT_TOKEN}"
    default_chat_id: "${TELEGRAM_CHAT_ID}"
  
  email:
    enabled: true
    smtp_host: "smtp.gmail.com"
    smtp_port: 587
    smtp_username: "${SMTP_USERNAME}"
    smtp_password: "${SMTP_PASSWORD}"
    from_email: "${FROM_EMAIL}"
```

## Error Handling

The client provides specific exception types for different error conditions:

```python
from src.notification.service.client import (
    NotificationServiceError,
    NotificationServiceUnavailableError
)

try:
    response = client.send_notification(
        notification_type="info",
        title="Test",
        message="Test message"
    )
except NotificationServiceUnavailableError:
    # Service is down or circuit breaker is open
    print("Notification service is currently unavailable")
except NotificationServiceError as e:
    # Other service errors (network, validation, etc.)
    print(f"Failed to send notification: {e}")
```

## Circuit Breaker

The client includes a circuit breaker to prevent cascading failures:

- **Closed**: Normal operation, requests are sent
- **Open**: Service is failing, requests are blocked
- **Half-Open**: Testing if service has recovered

The circuit breaker automatically transitions between states based on success/failure rates.

## Best Practices

### 1. Use Context Managers

```python
# Synchronous
with NotificationServiceClient() as client:
    response = client.send_notification(...)

# Asynchronous
async with NotificationServiceClient() as client:
    response = await client.send_notification_async(...)
```

### 2. Handle Errors Gracefully

```python
try:
    response = client.send_notification(...)
    logger.info("Notification sent: %s", response.message_id)
except NotificationServiceUnavailableError:
    logger.warning("Notification service unavailable, will retry later")
except NotificationServiceError as e:
    logger.exception("Failed to send notification:")
```

### 3. Use Appropriate Priorities

```python
# Critical errors
client.send_error_notification("Database connection lost!")

# High priority trades
client.send_trade_notification(
    symbol="BTCUSDT",
    side="BUY",
    price=50000,
    quantity=1.0
)

# Normal info messages
client.send_notification(
    notification_type="info",
    title="Daily Report",
    message="Trading completed successfully",
    priority="NORMAL"
)
```

### 4. Specify Recipients

```python
# Email notification
client.send_notification(
    notification_type="info",
    title="Report",
    message="Daily trading report",
    channels=["email"],
    recipient_id="trader@example.com"
)

# Telegram notification
client.send_notification(
    notification_type="alert",
    title="Price Alert",
    message="BTC reached $50,000!",
    channels=["telegram"],
    metadata={"telegram_chat_id": "123456789"}
)
```

## Migration Guide

### From AsyncNotificationManager

1. **Replace imports:**
   ```python
   # Old
   from src.notification.async_notification_manager import AsyncNotificationManager
   
   # New
   from src.notification.service.client import NotificationServiceClient
   ```

2. **Update initialization:**
   ```python
   # Old
   manager = AsyncNotificationManager(
       telegram_token="...",
       telegram_chat_id="...",
       email_sender="...",
       email_receiver="..."
   )
   
   # New
   client = NotificationServiceClient(
       base_url="http://notification-service:8080"
   )
   ```

3. **Update method calls:**
   ```python
   # Old
   await manager.send_notification(
       notification_type=NotificationType.INFO,
       title="Test",
       message="Test message"
   )
   
   # New
   await client.send_notification_async(
       notification_type=NotificationType.INFO,
       title="Test",
       message="Test message"
   )
   ```

### Gradual Migration

For gradual migration, use the compatibility wrapper:

```python
# Use compatibility wrapper initially
from src.notification.compatibility import AsyncNotificationManagerCompat as AsyncNotificationManager

# Existing code works unchanged
manager = AsyncNotificationManager(...)
await manager.send_notification(...)

# Later, migrate to direct client usage
from src.notification.service.client import NotificationServiceClient
client = NotificationServiceClient(...)
await client.send_notification_async(...)
```

## Testing

The client library includes comprehensive unit tests:

```bash
# Run client tests
python -m pytest src/notification/tests/test_client.py -v

# Run all notification tests
python -m pytest src/notification/tests/ -v
```

For testing your code that uses the client, you can mock the client:

```python
from unittest.mock import Mock, patch
from src.notification.service.client import NotificationServiceClient

@patch('src.notification.service.client.NotificationServiceClient.send_notification')
def test_my_function(mock_send):
    mock_send.return_value = Mock(message_id=123, status="enqueued")
    
    # Test your code that uses the client
    result = my_function_that_sends_notifications()
    
    assert mock_send.called
    assert result is not None
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Ensure the notification service is running
   - Check the base_url configuration
   - Verify network connectivity

2. **Authentication Errors**
   - Check if API key is required
   - Verify API key is correct
   - Ensure proper headers are set

3. **Circuit Breaker Open**
   - Service is experiencing failures
   - Wait for recovery timeout
   - Check service health

4. **Timeout Errors**
   - Increase timeout value
   - Check service performance
   - Verify network latency

### Debug Logging

Enable debug logging to troubleshoot issues:

```python
import logging
logging.getLogger('src.notification').setLevel(logging.DEBUG)

# Client will now log detailed information
client = NotificationServiceClient(...)
```

### Health Monitoring

Monitor the notification service health:

```python
try:
    health = client.health_check()
    if health['status'] != 'healthy':
        logger.warning("Service health: %s", health)
except Exception as e:
    logger.exception("Health check failed:")
```

## Examples

See `src/notification/examples/client_usage.py` for comprehensive examples of all client features.

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the notification service logs
3. Verify service configuration
4. Check network connectivity
5. Review the examples and tests for usage patterns