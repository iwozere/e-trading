# Notification Service Design

## Purpose

The Notification Service is a dedicated, autonomous service that centralizes all outbound communications for the Advanced Trading Framework. It decouples notification logic from business services, provides unified APIs, and implements advanced features like queuing, batching, rate limiting, and delivery tracking.

## Architecture

### High-Level Architecture

```mermaid
graph TB
    subgraph "Service Consumers"
        AlertEvaluator[Alert Evaluator]
        SchedulerService[Scheduler Service]
        TradingBot[Trading Bot]
        TelegramBot[Telegram Bot]
        WebUI[Web UI]
    end

    subgraph "Notification Service"
        API[REST API]
        MessageQueue[Message Queue]
        Processor[Message Processor]
        RateLimiter[Rate Limiter]
        ChannelManager[Channel Manager]
        HealthMonitor[Health Monitor]
    end

    subgraph "Channel Plugins"
        TelegramChannel[Telegram Plugin]
        EmailChannel[Email Plugin]
        SMSChannel[SMS Plugin]
        SlackChannel[Slack Plugin]
    end

    subgraph "External APIs"
        TelegramAPI[Telegram Bot API]
        SMTPServer[SMTP Server]
        SMSProvider[SMS Provider]
        SlackAPI[Slack API]
    end

    subgraph "Database"
        Messages[(msg_messages)]
        DeliveryStatus[(msg_delivery_status)]
        ChannelHealth[(msg_channel_health)]
        RateLimits[(msg_rate_limits)]
    end

    AlertEvaluator --> API
    SchedulerService --> API
    TradingBot --> API
    TelegramBot --> API
    WebUI --> API

    API --> MessageQueue
    MessageQueue --> Processor
    Processor --> RateLimiter
    RateLimiter --> ChannelManager
    ChannelManager --> TelegramChannel
    ChannelManager --> EmailChannel
    ChannelManager --> SMSChannel
    ChannelManager --> SlackChannel

    TelegramChannel --> TelegramAPI
    EmailChannel --> SMTPServer
    SMSChannel --> SMSProvider
    SlackChannel --> SlackAPI

    Processor --> Messages
    Processor --> DeliveryStatus
    HealthMonitor --> ChannelHealth
    RateLimiter --> RateLimits
```

### Component Design

#### 1. REST API Layer
- **Purpose**: Provides HTTP endpoints for enqueueing messages
- **Endpoints**:
  - `POST /api/v1/messages` - Enqueue message
  - `GET /api/v1/messages/{id}/status` - Get delivery status
  - `GET /api/v1/health` - Service health check
  - `GET /api/v1/channels` - List available channels
  - `GET /api/v1/stats` - Delivery statistics

#### 2. Message Queue
- **Purpose**: Database-backed queue for reliable message storage
- **Features**:
  - Priority-based processing
  - Persistence across service restarts
  - Transaction support
  - Dead letter queue for failed messages

#### 3. Message Processor
- **Purpose**: Core processing engine for message delivery
- **Features**:
  - Asynchronous processing
  - Priority queue handling
  - Batch processing for low-priority messages
  - Retry mechanism with exponential backoff

#### 4. Rate Limiter
- **Purpose**: Per-user rate limiting to prevent spam
- **Features**:
  - Configurable limits per channel
  - Token bucket algorithm
  - User-specific tracking
  - Bypass for high-priority messages

#### 5. Channel Manager
- **Purpose**: Manages channel plugins and routing
- **Features**:
  - Dynamic plugin loading
  - Channel health monitoring
  - Fallback routing
  - Configuration management

#### 6. Channel Plugins
- **Purpose**: Channel-specific message delivery implementations
- **Standard Interface**:
  ```python
  class NotificationChannel:
      async def send(self, message: Message) -> DeliveryResult
      def get_health(self) -> ChannelHealth
      def get_config(self) -> ChannelConfig
  ```

### Database Schema

#### msg_messages
```sql
CREATE TABLE msg_messages (
    id BIGSERIAL PRIMARY KEY,
    message_type VARCHAR(50) NOT NULL,
    priority VARCHAR(20) NOT NULL DEFAULT 'NORMAL',
    channels TEXT[] NOT NULL,
    recipient_id VARCHAR(100),
    template_name VARCHAR(100),
    content JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    scheduled_for TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status VARCHAR(20) DEFAULT 'PENDING',
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    last_error TEXT,
    processed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_msg_messages_status ON msg_messages(status);
CREATE INDEX idx_msg_messages_priority ON msg_messages(priority);
CREATE INDEX idx_msg_messages_scheduled ON msg_messages(scheduled_for);
CREATE INDEX idx_msg_messages_recipient ON msg_messages(recipient_id);
```

#### msg_delivery_status
```sql
CREATE TABLE msg_delivery_status (
    id BIGSERIAL PRIMARY KEY,
    message_id BIGINT REFERENCES msg_messages(id),
    channel VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    delivered_at TIMESTAMP WITH TIME ZONE,
    response_time_ms INTEGER,
    error_message TEXT,
    external_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_delivery_status_message ON msg_delivery_status(message_id);
CREATE INDEX idx_delivery_status_channel ON msg_delivery_status(channel);
CREATE INDEX idx_delivery_status_status ON msg_delivery_status(status);
```

#### msg_channel_health
```sql
CREATE TABLE msg_channel_health (
    id BIGSERIAL PRIMARY KEY,
    channel VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    last_success TIMESTAMP WITH TIME ZONE,
    last_failure TIMESTAMP WITH TIME ZONE,
    failure_count INTEGER DEFAULT 0,
    avg_response_time_ms INTEGER,
    error_message TEXT,
    checked_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_channel_health_channel ON msg_channel_health(channel);
```

#### msg_rate_limits
```sql
CREATE TABLE msg_rate_limits (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    channel VARCHAR(50) NOT NULL,
    tokens INTEGER NOT NULL,
    last_refill TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    max_tokens INTEGER NOT NULL,
    refill_rate INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_rate_limits_user_channel ON msg_rate_limits(user_id, channel);
```

#### msg_channel_configs
```sql
CREATE TABLE msg_channel_configs (
    id BIGSERIAL PRIMARY KEY,
    channel VARCHAR(50) NOT NULL UNIQUE,
    enabled BOOLEAN DEFAULT true,
    config JSONB NOT NULL,
    rate_limit_per_minute INTEGER DEFAULT 60,
    max_retries INTEGER DEFAULT 3,
    timeout_seconds INTEGER DEFAULT 30,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Data Flow

#### Message Processing Flow
1. **Enqueue**: Client sends message via REST API
2. **Validate**: API validates message format and channels
3. **Store**: Message stored in database with PENDING status
4. **Process**: Message processor picks up pending messages
5. **Rate Check**: Rate limiter checks user limits
6. **Route**: Channel manager routes to appropriate plugins
7. **Deliver**: Channel plugin delivers message
8. **Track**: Delivery status recorded in database

#### Priority Handling
- **CRITICAL/HIGH**: Immediate processing, bypass batching and rate limits
- **NORMAL**: Standard processing with batching and rate limits
- **LOW**: Batched processing during low-traffic periods

#### Error Handling
- **Temporary Errors**: Retry with exponential backoff
- **Rate Limit Errors**: Queue for later delivery
- **Permanent Errors**: Move to dead letter queue
- **Channel Down**: Route to fallback channels

### Integration Patterns

#### Service Consumer Integration
```python
# Example: Alert Evaluator sending notification
async def send_alert_notification(alert_data):
    message = {
        "message_type": "trade_alert",
        "priority": "HIGH",
        "channels": ["telegram", "email"],
        "recipient_id": alert_data["user_id"],
        "template_name": "trade_alert",
        "content": {
            "symbol": alert_data["symbol"],
            "action": alert_data["action"],
            "price": alert_data["price"]
        },
        "metadata": {
            "alert_id": alert_data["id"],
            "telegram_chat_id": alert_data["telegram_chat_id"]
        }
    }
    
    response = await http_client.post(
        "http://notification-service:8080/api/v1/messages",
        json=message
    )
    return response.json()["message_id"]
```

#### Channel Plugin Interface
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class DeliveryResult:
    success: bool
    external_id: Optional[str] = None
    response_time_ms: Optional[int] = None
    error_message: Optional[str] = None

@dataclass
class ChannelHealth:
    status: str  # "healthy", "degraded", "down"
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    avg_response_time_ms: Optional[int] = None
    error_message: Optional[str] = None

class NotificationChannel(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config["name"]
    
    @abstractmethod
    async def send(self, message: Message) -> DeliveryResult:
        """Send message through this channel"""
        pass
    
    @abstractmethod
    async def get_health(self) -> ChannelHealth:
        """Get current channel health status"""
        pass
    
    def format_message(self, template: str, data: Dict[str, Any]) -> str:
        """Format message using channel-specific formatting"""
        # Default implementation - channels can override
        return template.format(**data)
```

### Configuration Management

#### Service Configuration
```yaml
# notification_service.yaml
server:
  host: "0.0.0.0"
  port: 8080
  workers: 4

database:
  url: "postgresql://user:pass@localhost/trading_db"
  pool_size: 10

processing:
  batch_size: 10
  batch_timeout_seconds: 30
  max_workers: 20
  cleanup_interval_hours: 24

rate_limiting:
  default_limits:
    telegram: 30  # messages per minute
    email: 10
    sms: 5
  
channels:
  telegram:
    enabled: true
    plugin: "telegram_plugin"
    config:
      bot_token: "${TELEGRAM_BOT_TOKEN}"
      timeout: 30
  
  email:
    enabled: true
    plugin: "email_plugin"
    config:
      smtp_server: "smtp.gmail.com"
      smtp_port: 587
      username: "${SMTP_USER}"
      password: "${SMTP_PASSWORD}"
```

### Security Considerations

#### Authentication & Authorization
- **API Keys**: Service-to-service authentication
- **Rate Limiting**: Prevent abuse and DoS attacks
- **Input Validation**: Sanitize all message content
- **Credential Management**: Secure storage of channel credentials

#### Data Protection
- **Encryption**: Encrypt sensitive message content
- **PII Handling**: Careful handling of user data
- **Audit Logging**: Log all message processing activities
- **Data Retention**: Implement proper data lifecycle management

### Performance Considerations

#### Scalability
- **Horizontal Scaling**: Multiple service instances
- **Database Sharding**: Partition messages by user or time
- **Connection Pooling**: Efficient database connections
- **Caching**: Cache channel configurations and templates

#### Monitoring
- **Metrics**: Message throughput, delivery rates, response times
- **Alerting**: Channel health, queue depth, error rates
- **Logging**: Structured logging for debugging and analysis
- **Tracing**: Distributed tracing for message flow

### Migration Strategy

#### Phase 1: Service Setup
- Deploy notification service alongside existing system
- Implement database schema and basic APIs
- Create channel plugins for existing channels

#### Phase 2: Gradual Migration
- Migrate AlertEvaluator to use notification service
- Migrate SchedulerService for report notifications
- Update TelegramBot for non-interactive notifications

#### Phase 3: Full Migration
- Migrate all remaining service consumers
- Deprecate AsyncNotificationManager
- Remove old notification code

#### Backward Compatibility
- Maintain existing AsyncNotificationManager APIs
- Proxy calls to notification service
- Gradual deprecation with migration guides

This design provides a robust, scalable, and maintainable notification service that addresses all the identified requirements while maintaining clean separation of concerns and supporting future extensibility.