# Channel Fallback and Recovery System

## Overview

The Channel Fallback and Recovery System provides automatic failover capabilities for the notification service, ensuring message delivery even when primary channels are unavailable. The system includes intelligent routing, retry mechanisms, and dead letter queue management.

## Key Features

### 1. Automatic Channel Fallback
- **Primary Channel Failure Detection**: Automatically detects when primary channels are unavailable
- **Intelligent Routing**: Routes messages to healthy fallback channels based on configurable strategies
- **Health-Based Decisions**: Integrates with the health monitoring system to make routing decisions
- **Multiple Fallback Strategies**: Supports priority order, round-robin, health-based, and load-balanced routing

### 2. Retry Queue Management
- **Automatic Retry**: Failed messages are automatically queued for retry with exponential backoff
- **Configurable Retry Limits**: Maximum retry attempts and delay intervals are configurable
- **Smart Retry Logic**: Only retries messages that have a chance of succeeding
- **Background Processing**: Retry queue is processed by dedicated background workers

### 3. Dead Letter Queue
- **Permanent Failure Handling**: Messages that cannot be delivered are moved to a dead letter queue
- **Manual Reprocessing**: Administrators can manually reprocess dead letter messages
- **Retention Management**: Automatic cleanup of old dead letter messages
- **Detailed Failure Tracking**: Comprehensive failure reason and attempt history

### 4. Comprehensive Statistics
- **Fallback Analytics**: Tracks fallback attempt success rates and patterns
- **Channel Performance**: Monitors individual channel success rates and response times
- **Failure Analysis**: Detailed analysis of failure reasons and recovery patterns
- **Real-time Monitoring**: Live statistics and health dashboards

## Architecture

### Core Components

#### FallbackManager
The central component that orchestrates all fallback operations:
- Manages fallback rules and strategies
- Coordinates with health monitor for routing decisions
- Handles retry queue and dead letter queue operations
- Provides comprehensive statistics and analytics

#### FallbackRule
Configuration object that defines fallback behavior:
```python
FallbackRule(
    primary_channel='telegram',
    fallback_channels=['email', 'sms'],
    strategy=FallbackStrategy.PRIORITY_ORDER,
    max_attempts=3,
    retry_delay_seconds=60
)
```

#### Message Processing Flow
1. **Primary Attempt**: Try to deliver via primary channel
2. **Health Check**: Verify channel availability using health monitor
3. **Fallback Routing**: If primary fails, route to fallback channels
4. **Retry Logic**: Queue failed messages for retry if appropriate
5. **Dead Letter**: Move permanently failed messages to dead letter queue

### Integration Points

#### Health Monitor Integration
- Real-time channel health status
- Automatic channel disable/enable based on health
- Health-based routing decisions
- Performance metrics integration

#### Message Processor Integration
- Seamless integration with existing message processing
- Background retry processing
- Statistics collection and reporting
- Graceful error handling

## Configuration

### Fallback Rules
Configure fallback behavior for specific channels:

```python
# Priority-based fallback
telegram_rule = FallbackRule(
    primary_channel='telegram',
    fallback_channels=['email', 'sms'],
    strategy=FallbackStrategy.PRIORITY_ORDER,
    max_attempts=2,
    retry_delay_seconds=60
)

# Health-based fallback
email_rule = FallbackRule(
    primary_channel='email',
    fallback_channels=['sms', 'telegram'],
    strategy=FallbackStrategy.HEALTH_BASED,
    max_attempts=3,
    retry_delay_seconds=120
)
```

### Global Fallback Channels
Set default fallback channels for channels without specific rules:

```python
fallback_manager.set_global_fallback_channels(['email', 'sms', 'telegram'])
```

### Retry Configuration
Configure retry behavior:
- `max_retry_attempts`: Maximum number of retry attempts (default: 5)
- `base_retry_delay`: Base delay between retries in seconds (default: 60)
- `retry_backoff_multiplier`: Exponential backoff multiplier (default: 2.0)

## API Endpoints

### Dead Letter Queue Management
- `GET /api/v1/fallback/dead-letters` - List dead letter messages
- `POST /api/v1/fallback/dead-letters/{id}/reprocess` - Reprocess a dead letter message

### Retry Queue Status
- `GET /api/v1/fallback/retry-queue` - Get retry queue status

### Statistics and Analytics
- `GET /api/v1/fallback/statistics` - Get comprehensive fallback statistics

### Rule Management
- `POST /api/v1/fallback/rules/{channel}` - Configure fallback rule
- `DELETE /api/v1/fallback/rules/{channel}` - Remove fallback rule
- `POST /api/v1/fallback/global-channels` - Set global fallback channels

## Usage Examples

### Basic Fallback Configuration
```python
from src.notification.service.fallback_manager import FallbackManager, FallbackRule, FallbackStrategy

# Create fallback manager
fallback_manager = FallbackManager(health_monitor)

# Configure Telegram -> Email fallback
rule = FallbackRule(
    primary_channel='telegram',
    fallback_channels=['email'],
    strategy=FallbackStrategy.PRIORITY_ORDER
)
fallback_manager.configure_fallback_rule(rule)

# Attempt delivery with fallback
success, results, failed_msg = await fallback_manager.attempt_delivery_with_fallback(
    message_id=123,
    channels=['telegram'],
    recipient='user@example.com',
    content=message_content,
    channel_instances=channel_instances
)
```

### Manual Dead Letter Reprocessing
```python
# Reprocess a specific dead letter message
success, message = await fallback_manager.reprocess_dead_letter_message(
    message_id=123,
    channel_instances=channel_instances,
    force_channels=['email']  # Force specific channels
)
```

### Statistics Monitoring
```python
# Get comprehensive statistics
stats = fallback_manager.get_fallback_statistics()

print(f"Total fallback attempts: {stats['statistics']['total_fallback_attempts']}")
print(f"Success rate: {stats['statistics']['successful_fallbacks'] / stats['statistics']['total_fallback_attempts'] * 100:.1f}%")

# Channel-specific success rates
for channel, channel_stats in stats['channel_success_rates'].items():
    print(f"{channel}: {channel_stats['success_rate']:.1f}% success rate")
```

## Monitoring and Alerting

### Key Metrics to Monitor
- **Fallback Success Rate**: Percentage of successful fallback attempts
- **Dead Letter Queue Size**: Number of permanently failed messages
- **Retry Queue Size**: Number of messages awaiting retry
- **Channel Success Rates**: Individual channel performance metrics
- **Average Recovery Time**: Time from failure to successful delivery

### Recommended Alerts
- Dead letter queue size exceeding threshold
- Fallback success rate dropping below acceptable level
- Specific channels showing high failure rates
- Retry queue growing without processing

## Testing

### Unit Tests
Comprehensive unit tests cover:
- Fallback rule configuration and validation
- Message routing logic and strategy implementation
- Retry queue processing and dead letter management
- Statistics collection and reporting

### Integration Tests
Integration tests verify:
- Health monitor integration
- End-to-end message delivery with fallback
- API endpoint functionality
- Performance under load

### Running Tests
```bash
# Run unit tests
python -m pytest src/notification/service/test_fallback_manager.py -v

# Run integration tests
python -m pytest src/notification/service/test_fallback_integration.py -v
```

## Performance Considerations

### Scalability
- **Concurrent Processing**: Multiple fallback attempts can be processed concurrently
- **Memory Management**: Configurable queue sizes and retention policies
- **Database Optimization**: Efficient queries and indexing for large message volumes

### Optimization Tips
- Configure appropriate retry delays to avoid overwhelming external services
- Use health-based routing to minimize failed attempts
- Monitor and tune fallback strategies based on actual performance data
- Implement proper cleanup policies for dead letter and retry queues

## Troubleshooting

### Common Issues

#### High Dead Letter Queue Size
- Check channel health and configuration
- Review failure reasons in dead letter messages
- Consider adjusting retry limits or fallback rules

#### Slow Message Processing
- Monitor retry queue processing performance
- Check for bottlenecks in channel implementations
- Consider increasing worker pool sizes

#### Fallback Not Working
- Verify fallback rules are properly configured
- Check health monitor integration
- Review channel availability status

### Debug Information
The system provides detailed logging and statistics for troubleshooting:
- Fallback attempt logs with timing and results
- Channel health status and availability
- Detailed failure reasons and retry attempts
- Performance metrics and trends

## Future Enhancements

### Planned Features
- **Machine Learning**: Predictive fallback routing based on historical patterns
- **Advanced Analytics**: More sophisticated failure analysis and reporting
- **Dynamic Configuration**: Runtime configuration updates without service restart
- **Multi-Region Support**: Geographic fallback routing for global deployments

### Extension Points
- **Custom Strategies**: Plugin system for custom fallback strategies
- **External Integrations**: Integration with external monitoring and alerting systems
- **Advanced Routing**: Content-based and recipient-based routing rules
- **Performance Optimization**: Caching and pre-computation for faster routing decisions