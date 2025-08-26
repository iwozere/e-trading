# Context-Aware Logging System

This document describes the context-aware logging system that allows child modules to inherit logging context from their parent modules, ensuring that logs are written to the appropriate files.

## Overview

The context-aware logging system solves the problem where child modules (like `async_notification_manager.py`) were writing logs to `app.log` instead of the same log file as their parent module (like `telegram_screener_bot.log`).

## Problem Solved

**Before:**
- `telegram_screener_bot.py` logs to `telegram_screener_bot.log`
- `async_notification_manager.py` logs to `app.log` (unwanted)
- Logs from the same operation are scattered across different files

**After:**
- `telegram_screener_bot.py` logs to `telegram_screener_bot.log`
- `async_notification_manager.py` logs to `telegram_screener_bot.log` (inherited)
- All logs from the same operation are in the same file

## How It Works

### 1. Context Setting

Parent modules set a logging context before calling child modules:

```python
from src.notification.logger import set_logging_context

# Set context before calling notification manager
set_logging_context("telegram_screener_bot")

# Now any child modules will inherit this context
notification_manager = await initialize_notification_manager(...)
```

### 2. Automatic Inheritance

Child modules automatically inherit the logging context:

```python
# In async_notification_manager.py
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)  # Automatically inherits context if set
```

### 3. Context Detection

The system detects when a notification module is being used and automatically inherits the parent's file handlers.

## Usage Examples

### Basic Context Setting

```python
from src.notification.logger import setup_logger, set_logging_context

# Set context
set_logging_context("telegram_screener_bot")

# Create logger (will inherit context)
logger = setup_logger("my_module")
logger.info("This goes to telegram_screener_bot.log")
```

### Direct Inheritance

```python
from src.notification.logger import setup_context_logger

# Directly inherit from a specific logger
logger = setup_context_logger("my_module", "telegram_background_services")
logger.info("This goes to telegram_background_services.log")
```

### Notification Manager Integration

```python
# In telegram bot
set_logging_context("telegram_screener_bot")
notification_manager = await initialize_notification_manager(...)

# In async_notification_manager.py
_logger = setup_logger(__name__)  # Inherits telegram_screener_bot context
_logger.info("Notification sent")  # Goes to telegram_screener_bot.log
```

## Available Log Files

The system supports the following dedicated log files:

- `telegram_screener_bot.log` - Main telegram bot
- `telegram_background_services.log` - Background services
- `telegram_alert_monitor.log` - Alert monitoring
- `telegram_schedule_processor.log` - Schedule processing
- `app.log` - General application logs
- `app_errors.log` - Error logs only

## Implementation Details

### Context Variables

The system uses Python's `contextvars` to maintain logging context across async operations:

```python
from contextvars import ContextVar
_logging_context = ContextVar('logging_context', default=None)
```

### Handler Inheritance

When a context is set, child loggers inherit file handlers from the parent:

```python
def _inherit_parent_handlers(self, parent_name: str):
    parent_logger = logging.getLogger(parent_name)
    for handler in parent_logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            # Create new handler with same configuration
            new_handler = RotatingFileHandler(...)
            self.logger.addHandler(new_handler)
```

### Automatic Detection

The system automatically detects notification modules and applies context inheritance:

```python
# Check if we have a logging context set
context = get_logging_context()
if context and name.startswith('src.notification.'):
    # Inherit from the context logger
    return ContextAwareLogger(name, context).logger
```

## Migration Guide

### For Existing Code

1. **No changes needed** for most modules
2. **Add context setting** in parent modules that call notification manager
3. **Test** that logs appear in the correct files

### For New Code

1. **Set context** before calling child modules
2. **Use standard setup_logger** in child modules
3. **Context is automatically inherited**

### Example Migration

**Before:**
```python
# In telegram bot
notification_manager = await initialize_notification_manager(...)

# In notification manager
_logger = setup_logger(__name__)  # Goes to app.log
```

**After:**
```python
# In telegram bot
set_logging_context("telegram_screener_bot")
notification_manager = await initialize_notification_manager(...)

# In notification manager
_logger = setup_logger(__name__)  # Goes to telegram_screener_bot.log
```

## Testing

Run the test script to verify the system works:

```bash
python test_context_logging.py
```

This will create test messages in different log files to demonstrate the functionality.

## Benefits

1. **Unified Logging**: All logs from the same operation are in one file
2. **Better Debugging**: Easier to trace operations across modules
3. **Cleaner Organization**: Logs are properly categorized by service
4. **Backward Compatible**: Existing code continues to work
5. **Automatic**: No manual log file management needed

## Troubleshooting

### Logs Still Going to app.log

1. Check that `set_logging_context()` is called before creating child loggers
2. Verify the context name matches a configured logger
3. Ensure the child module uses `setup_logger(__name__)`

### Context Not Inherited

1. Check that the parent logger has file handlers configured
2. Verify the context is set in the same async context
3. Ensure the child module name starts with `src.notification.`

### Performance Impact

The context-aware logging has minimal performance impact:
- Context variables are fast
- Handler inheritance happens once per logger
- No runtime overhead for log messages
