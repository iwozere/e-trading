# Notification Channel Plugin System

This package provides a comprehensive plugin system for notification channels in the notification service. It enables easy extension and customization of delivery channels while maintaining consistency and reliability.

## Architecture Overview

The plugin system consists of several key components:

### 1. Base Classes (`base.py`)
- **`NotificationChannel`**: Abstract base class that all channel plugins must inherit from
- **`DeliveryResult`**: Data class for delivery attempt results
- **`ChannelHealth`**: Data class for channel health information
- **`MessageContent`**: Structured message content for delivery
- **`ChannelRegistry`**: Registry for managing channel plugin instances

### 2. Plugin Loading (`loader.py`)
- **`PluginLoader`**: Discovers and loads channel plugins automatically
- Supports both built-in and external plugins
- Provides plugin validation and hot-reloading capabilities
- Automatic discovery from the channels package

### 3. Configuration Validation (`config.py`)
- **`ConfigValidator`**: Type-safe configuration validation
- **`ValidationRule`**: Flexible validation rules with constraints
- **`CommonValidationRules`**: Pre-defined rules for common patterns
- Comprehensive error reporting and schema generation

### 4. Test Plugin (`test_plugin.py`)
- Example implementation showing how to create a channel plugin
- Simulates message delivery for testing purposes
- Demonstrates all required methods and best practices

## Key Features

### ✅ **Plugin Interface**
- Abstract base class with clear contract
- Standardized delivery results and health monitoring
- Support for message formatting and splitting
- Feature capability detection

### ✅ **Automatic Discovery**
- Discovers plugins from the channels package
- Supports external plugin loading from files/directories
- Plugin validation before registration
- Hot-reloading for development

### ✅ **Configuration Management**
- Type-safe configuration validation
- Flexible validation rules with constraints
- Common validation patterns (URLs, emails, timeouts)
- Schema generation for documentation

### ✅ **Error Handling**
- Comprehensive error reporting
- Graceful failure handling
- Retry mechanisms with exponential backoff
- Health monitoring and recovery

### ✅ **Message Processing**
- Automatic message splitting for length limits
- Channel-specific message formatting
- Priority handling and feature detection
- Metadata and attachment support

## Usage Examples

### Creating a Channel Plugin

```python
from src.notification.channels.base import NotificationChannel, DeliveryResult, ChannelHealth
from src.notification.channels.config import ConfigValidator, CommonValidationRules

class MyChannel(NotificationChannel):
    def validate_config(self, config):
        validator = ConfigValidator(self.channel_name)
        validator.require_field("api_key", str, "API authentication key")
        validator.add_rule(CommonValidationRules.timeout_seconds())
        self.config.update(validator.validate(config))
    
    async def send_message(self, recipient, content, message_id=None, priority="NORMAL"):
        # Implement message delivery logic
        return DeliveryResult(
            success=True,
            status=DeliveryStatus.DELIVERED,
            external_id="msg_123"
        )
    
    async def check_health(self):
        # Implement health check logic
        return ChannelHealth(
            status=ChannelHealthStatus.HEALTHY,
            last_check=datetime.now(timezone.utc)
        )
    
    def get_rate_limit(self):
        return self.config.get("rate_limit_per_minute", 60)
    
    def supports_feature(self, feature):
        return feature in ["html", "attachments"]
```

### Loading and Using Plugins

```python
from src.notification.channels import load_all_channels, channel_registry, MessageContent

# Load all available plugins
plugins = load_all_channels()

# Get a channel instance
config = {"api_key": "your_key", "timeout_seconds": 30}
channel = channel_registry.get_channel("my_channel", config)

# Send a message
content = MessageContent(text="Hello, World!", subject="Test")
result = await channel.send_message("recipient@example.com", content)

if result.is_successful:
    print(f"Message delivered: {result.external_id}")
else:
    print(f"Delivery failed: {result.error_message}")
```

### Configuration Validation

```python
from src.notification.channels.config import ConfigValidator, ValidationRule

validator = ConfigValidator("my_channel")

# Add custom validation rules
validator.require_field("api_key", str, min_length=10)
validator.optional_field("timeout", int, min_value=1, max_value=300)

# Validate configuration
try:
    validated_config = validator.validate(user_config)
except ConfigValidationError as e:
    print(f"Configuration error: {e}")
```

## Plugin Development Guidelines

### Required Methods
All channel plugins must implement these abstract methods:

1. **`validate_config(config)`**: Validate and normalize configuration
2. **`send_message(recipient, content, message_id, priority)`**: Send a message
3. **`check_health()`**: Check channel health status
4. **`get_rate_limit()`**: Return rate limit in messages per minute
5. **`supports_feature(feature)`**: Check if feature is supported

### Optional Methods
These methods have default implementations but can be overridden:

- **`format_message(content)`**: Apply channel-specific formatting
- **`get_max_message_length()`**: Return maximum message length
- **`split_long_message(content)`**: Split long messages into parts

### Best Practices

1. **Configuration Validation**: Always validate configuration in `validate_config()`
2. **Error Handling**: Use proper exception handling and return meaningful error messages
3. **Health Monitoring**: Implement realistic health checks that detect actual issues
4. **Rate Limiting**: Respect and enforce rate limits to avoid API throttling
5. **Feature Detection**: Accurately report supported features
6. **Logging**: Use the provided logger for debugging and monitoring
7. **Async/Await**: Use proper async patterns for I/O operations

## Testing

The plugin system includes comprehensive tests:

```bash
# Run plugin system tests
python src/notification/channels/test_plugin_system.py
```

Tests cover:
- Plugin discovery and loading
- Configuration validation
- Message delivery simulation
- Health monitoring
- Feature support detection

## Integration with Notification Service

The plugin system integrates seamlessly with the notification service:

1. **Automatic Loading**: Plugins are discovered and loaded at service startup
2. **Configuration Management**: Channel configs are stored in the database
3. **Health Monitoring**: Channel health is tracked and updated automatically
4. **Message Processing**: The message processor uses plugins for delivery
5. **API Integration**: REST API endpoints expose plugin capabilities

## Extending the System

### Adding Built-in Plugins
1. Create a new Python file in `src/notification/channels/`
2. Implement a class inheriting from `NotificationChannel`
3. The plugin will be automatically discovered and loaded

### Adding External Plugins
1. Create a Python file with your channel implementation
2. Use `register_external_plugin()` to register it manually
3. Or add the file path to the plugin loader search paths

### Plugin Validation
The system validates plugins before registration:
- Checks inheritance from `NotificationChannel`
- Verifies all required methods are implemented
- Validates method signatures and return types

This ensures all plugins follow the same interface and maintain system reliability.