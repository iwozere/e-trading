# Notification System

## Overview
The notification system provides a comprehensive, multi-channel messaging infrastructure for the e-trading platform. It supports email, SMS, Telegram, and other notification channels with advanced features like rate limiting, priority handling, batching, and delivery tracking.

## Features
- **Multi-channel support**: Email, SMS, Telegram, and extensible plugin system
- **Priority-based delivery**: High, medium, low priority message handling
- **Rate limiting**: Configurable rate limits per channel and user
- **Batch processing**: Efficient bulk message delivery
- **Delivery tracking**: Complete audit trail and delivery status monitoring
- **Fallback system**: Automatic failover between channels
- **Health monitoring**: System health checks and performance metrics
- **Analytics**: Comprehensive delivery statistics and reporting

## Quick Start

### Basic Usage
```python
from src.notification.async_notification_manager import AsyncNotificationManager

# Initialize the notification manager
manager = AsyncNotificationManager()

# Send a simple notification
await manager.send_notification(
    user_id="user123",
    message="Your trade has been executed",
    channel="telegram",
    priority="high"
)
```

### Using the Service Layer
```python
from src.notification.service.client import NotificationClient

client = NotificationClient()

# Send with delivery tracking
message_id = await client.send_message(
    recipient="user@example.com",
    subject="Trade Alert",
    body="Your stop loss has been triggered",
    channel="email",
    priority="high"
)

# Check delivery status
status = await client.get_delivery_status(message_id)
```

## Architecture

### Module Structure
```
src/notification/
├── __init__.py                     # Main module exports
├── alert_system.py                 # Alert management
├── async_notification_manager.py   # Core async notification manager
├── emailer.py                      # Email utilities
├── logger.py                       # Logging configuration
├── batching/                       # Message batching
├── channels/                       # Channel implementations
├── priority/                       # Priority queue management
├── rate_limiting/                  # Rate limiting logic
├── service/                        # Service layer components
├── docs/                          # Documentation and examples
└── tests/                         # Test suite
```

### Core Components

#### Channels (`channels/`)
- **Base Channel**: Abstract base class for all notification channels
- **Email Channel**: SMTP-based email delivery
- **SMS Channel**: SMS gateway integration
- **Telegram Channel**: Telegram bot API integration
- **Plugin System**: Extensible channel loader

#### Service Layer (`service/`)
- **Client**: High-level notification client API
- **Processor**: Message processing engine
- **Analytics**: Delivery metrics and reporting
- **Health Monitor**: System health and performance monitoring
- **Delivery Tracker**: Message delivery status tracking
- **Fallback Manager**: Channel failover logic

#### Supporting Systems
- **Priority Queue**: Message prioritization and queuing
- **Rate Limiter**: Configurable rate limiting per channel/user
- **Batch Processor**: Efficient bulk message handling

## Configuration

### Environment Variables
```bash
# Database
DB_URL=postgresql://user:pass@localhost/notifications

# Email Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# Telegram Configuration
TELEGRAM_BOT_TOKEN=your-bot-token

# SMS Configuration (Twilio)
TWILIO_ACCOUNT_SID=your-account-sid
TWILIO_AUTH_TOKEN=your-auth-token
TWILIO_PHONE_NUMBER=+1234567890
```

### Rate Limiting
```python
# Configure rate limits in service/config.py
RATE_LIMITS = {
    "email": {"per_minute": 60, "per_hour": 1000},
    "sms": {"per_minute": 10, "per_hour": 100},
    "telegram": {"per_minute": 30, "per_hour": 1000}
}
```

## Integration

### With Trading System
```python
from src.notification.service.client import NotificationClient

class TradingBot:
    def __init__(self):
        self.notifications = NotificationClient()
    
    async def on_trade_executed(self, trade):
        await self.notifications.send_message(
            recipient=trade.user_id,
            subject=f"Trade Executed: {trade.symbol}",
            body=f"Bought {trade.quantity} shares at ${trade.price}",
            channel="telegram",
            priority="high"
        )
```

### With Alert System
```python
from src.notification.alert_system import AlertSystem

alerts = AlertSystem()

# Register price alerts
alerts.add_price_alert(
    symbol="AAPL",
    threshold=150.00,
    condition="above",
    user_id="user123",
    channels=["email", "telegram"]
)
```

## Testing

### Running Tests
```bash
# Run all notification tests
python -m pytest src/notification/tests/

# Run specific test categories
python -m pytest src/notification/tests/test_channels.py
python -m pytest src/notification/tests/test_delivery_*.py

# Run end-to-end tests
python src/notification/tests/run_e2e_tests.py
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: Cross-component functionality
- **End-to-End Tests**: Full system workflow testing
- **Performance Tests**: Load and stress testing

## Documentation

### Available Documentation
- [Analytics System](docs/README_analytics.md) - Metrics and reporting
- [Delivery History API](docs/README_delivery_history_api.md) - Delivery tracking
- [Fallback System](docs/README_fallback_system.md) - Channel failover
- [Migration Guide](docs/README_migration.md) - Database migrations
- [Optimization Summary](docs/OPTIMIZATION_SUMMARY.md) - Performance optimizations
- [Channel System](docs/channels_README.md) - Channel implementation guide

### Examples
- [Archival Example](docs/archival_example.py) - Message archival
- [Health Integration](docs/health_integration_example.py) - Health monitoring
- [Migration Example](docs/migration_example.py) - Database migration

## Performance

### Optimizations
- **Database indexing** for fast message lookups
- **Connection pooling** for database and external services
- **Async processing** for non-blocking operations
- **Batch processing** for bulk operations
- **Caching** for frequently accessed data

### Monitoring
- Message delivery rates and success rates
- Channel-specific performance metrics
- Error rates and failure analysis
- System resource utilization

## Security

### Best Practices
- **API key management** through environment variables
- **Rate limiting** to prevent abuse
- **Input validation** for all message content
- **Audit logging** for compliance and debugging
- **Secure credential storage** for channel authentication

## Troubleshooting

### Common Issues
1. **Database connection errors**: Check DB_URL and database availability
2. **Channel authentication failures**: Verify API keys and credentials
3. **Rate limit exceeded**: Check rate limiting configuration
4. **Message delivery failures**: Review channel-specific logs

### Debugging
```python
# Enable debug logging
import logging
logging.getLogger('src.notification').setLevel(logging.DEBUG)

# Check system health
from src.notification.service.health_monitor import HealthMonitor
health = HealthMonitor()
status = await health.check_all_systems()
```

## Contributing

### Adding New Channels
1. Extend `channels/base.py` abstract class
2. Implement required methods: `send_message()`, `validate_config()`
3. Add channel configuration to `channels/config.py`
4. Register channel in `channels/loader.py`
5. Add comprehensive tests

### Performance Improvements
1. Profile code using built-in performance monitoring
2. Add database indexes for new query patterns
3. Implement caching for frequently accessed data
4. Optimize batch processing algorithms

## Related Documentation
- [Requirements](docs/Requirements.md) - Technical requirements
- [Design](docs/Design.md) - Architecture and design decisions
- [Tasks](docs/Tasks.md) - Implementation roadmap and tasks