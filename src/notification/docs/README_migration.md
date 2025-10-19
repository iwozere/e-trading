# Migration Guide: AsyncNotificationManager to NotificationServiceClient

This guide explains how to migrate from the legacy `AsyncNotificationManager` to the new `NotificationServiceClient`.

## Overview

The notification system has been refactored from an embedded component (`AsyncNotificationManager`) to a dedicated service architecture with a client library (`NotificationServiceClient`). This provides better scalability, monitoring, and maintainability.

## Key Changes

### Before (AsyncNotificationManager)
```python
from src.notification.async_notification_manager import AsyncNotificationManager

class MyService:
    def __init__(self):
        self.notification_manager = AsyncNotificationManager(
            telegram_token="token",
            telegram_chat_id="chat_id"
        )
    
    async def start(self):
        await self.notification_manager.start()
    
    async def send_alert(self):
        await self.notification_manager.send_notification(
            notification_type="alert",
            title="Alert",
            message="Something happened"
        )
```

### After (NotificationServiceClient)
```python
from src.notification.service.client import NotificationServiceClient, MessageType, MessagePriority

class MyService:
    def __init__(self, notification_client: NotificationServiceClient):
        self.notification_client = notification_client
    
    async def start(self):
        # No need to start client - managed externally
        pass
    
    async def send_alert(self):
        success = await self.notification_client.send_notification(
            notification_type=MessageType.ALERT,
            title="Alert",
            message="Something happened",
            priority=MessagePriority.HIGH,
            channels=["telegram"],
            recipient_id="user_123"
        )
```

## Migration Steps

### 1. Update Service Dependencies

**Old way:**
```python
# Service creates and manages its own notification manager
class TradingService:
    def __init__(self, config):
        self.notification_manager = AsyncNotificationManager(**config)
```

**New way:**
```python
# Service receives notification client as dependency
class TradingService:
    def __init__(self, config, notification_client: NotificationServiceClient):
        self.notification_client = notification_client
```

### 2. Update Application Initialization

**Old way:**
```python
# Each service initializes its own notification manager
trading_service = TradingService(config)
await trading_service.start()  # Starts embedded notification manager
```

**New way:**
```python
# Initialize shared notification client
from src.notification.service.client import initialize_notification_client

notification_client = await initialize_notification_client("http://localhost:8000")

# Pass client to services
trading_service = TradingService(config, notification_client)
await trading_service.start()  # No notification manager to start
```

### 3. Update Notification Calls

**Old way:**
```python
await self.notification_manager.send_notification(
    notification_type="trade_entry",
    title="Trade Executed",
    message="Bought 0.1 BTC at $50,000",
    priority="high"
)
```

**New way:**
```python
success = await self.notification_client.send_notification(
    notification_type=MessageType.TRADE_ENTRY,
    title="Trade Executed",
    message="Bought 0.1 BTC at $50,000",
    priority=MessagePriority.HIGH,
    channels=["telegram", "email"],
    recipient_id="trader_123"
)

if not success:
    logger.warning("Failed to send trade notification")
```

### 4. Update Error Handling

**Old way:**
```python
# AsyncNotificationManager handled errors internally
await self.notification_manager.send_error_notification("Database error")
```

**New way:**
```python
# Check return value and handle failures
success = await self.notification_client.send_error_notification(
    error_message="Database error",
    source="trading_service",
    recipient_id="admin"
)

if not success:
    # Implement fallback or retry logic
    logger.error("Critical: Failed to send error notification")
```

## Service-Specific Migrations

### SchedulerService Migration

```python
# Before
class SchedulerService:
    def __init__(self, jobs_service, alert_evaluator, database_url):
        self.alert_evaluator = alert_evaluator
        # Notification handled by AlertEvaluator internally

# After
class SchedulerService:
    def __init__(self, jobs_service, alert_evaluator, notification_client, database_url):
        self.alert_evaluator = alert_evaluator
        self.notification_client = notification_client
    
    async def _send_notification(self, notification_data):
        # Implement notification sending using client
        success = await self.notification_client.send_notification(...)
```

### Trading Broker Migration

```python
# Before
class BaseBroker:
    def __init__(self, config):
        self.notification_manager = PositionNotificationManager(config)

# After
class BaseBroker:
    def __init__(self, config, notification_client=None):
        self.notification_manager = PositionNotificationManager(config, notification_client)
```

### AlertEvaluator Integration

The `AlertEvaluator` doesn't need direct changes - it prepares notification data that the `SchedulerService` sends using the client.

## Configuration Changes

### Old Configuration
```yaml
notifications:
  telegram_token: "your_token"
  telegram_chat_id: "your_chat_id"
  email_api_key: "your_key"
  email_sender: "sender@example.com"
```

### New Configuration
```yaml
notification_service:
  url: "http://localhost:8000"
  timeout: 30
  max_retries: 3

# Service-level configuration
notifications:
  channels: ["telegram", "email"]
  recipient_id: "default_user"
```

## Testing Migration

### 1. Unit Tests
```python
@pytest.fixture
def mock_notification_client():
    client = AsyncMock(spec=NotificationServiceClient)
    client.send_notification.return_value = True
    return client

async def test_service_with_notifications(mock_notification_client):
    service = MyService(mock_notification_client)
    await service.send_alert()
    
    mock_notification_client.send_notification.assert_called_once()
```

### 2. Integration Tests
```python
async def test_notification_integration():
    # Start notification service
    # Initialize client
    # Test end-to-end notification flow
    pass
```

## Rollback Plan

If issues arise during migration:

1. **Keep legacy code temporarily:**
   ```python
   class MyService:
       def __init__(self, notification_client=None):
           self.notification_client = notification_client
           self.legacy_manager = None
           
           if not notification_client:
               # Fallback to legacy system
               self.legacy_manager = AsyncNotificationManager(...)
   ```

2. **Feature flag approach:**
   ```python
   USE_NEW_NOTIFICATIONS = os.getenv("USE_NEW_NOTIFICATIONS", "false").lower() == "true"
   
   if USE_NEW_NOTIFICATIONS and self.notification_client:
       await self.notification_client.send_notification(...)
   else:
       await self.legacy_manager.send_notification(...)
   ```

## Benefits After Migration

1. **Decoupled Architecture:** Services no longer manage notification infrastructure
2. **Centralized Monitoring:** All notifications tracked in one place
3. **Better Error Handling:** Comprehensive retry and fallback mechanisms
4. **Scalability:** Notification service can be scaled independently
5. **Extensibility:** Easy to add new notification channels
6. **Analytics:** Detailed delivery tracking and performance metrics

## Troubleshooting

### Common Issues

1. **Service not running:** Ensure notification service is started before clients
2. **Connection errors:** Check service URL and network connectivity
3. **Authentication:** Verify API keys and channel configurations
4. **Rate limiting:** Monitor rate limits and adjust sending patterns

### Debugging

```python
# Enable debug logging
import logging
logging.getLogger("src.notification.service.client").setLevel(logging.DEBUG)

# Check service health
health = await notification_client.get_health_status()
print(f"Service status: {health}")

# Monitor message delivery
status = await notification_client.get_message_status(message_id)
print(f"Message status: {status}")
```

## Next Steps

1. Start with low-risk services (development/testing environments)
2. Migrate one service at a time
3. Monitor notification delivery and performance
4. Gradually migrate production services
5. Remove legacy AsyncNotificationManager code once all services are migrated

For questions or issues during migration, refer to the notification service documentation or create an issue in the project repository.