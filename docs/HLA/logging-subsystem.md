# Logging Subsystem Architecture

## Overview

The logging subsystem provides comprehensive, multiprocessing-safe logging infrastructure for the e-trading platform. It handles logging across single-process applications, multi-threaded services, and parallel optimization workloads with Optuna.

## 🔗 Quick Navigation
- **[📖 Documentation Index](INDEX.md)** - Complete documentation guide
- **[🏗️ Infrastructure Module](modules/infrastructure.md)** - Parent infrastructure module
- **[🔧 Multiprocessing Logging Guide](../MULTIPROCESSING_LOGGING.md)** - Detailed implementation guide
- **[📝 Coding Conventions](../.claude/CLAUDE.md#3-logging)** - Logging standards

## Architecture Overview

### System Components

```
┌────────────────────────────────────────────────────────────────┐
│                    Logging Subsystem                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────┐     │
│  │         Logger Configuration (LOG_CONFIG)            │     │
│  │  • Formatters (standard, detailed)                  │     │
│  │  • Handlers (console, file, rotating)               │     │
│  │  • Log levels per module                            │     │
│  │  • UTF-8 encoding                                   │     │
│  └──────────────────────────────────────────────────────┘     │
│                          │                                     │
│              ┌───────────┴───────────┐                         │
│              ▼                       ▼                         │
│  ┌─────────────────────┐  ┌──────────────────────┐           │
│  │  Single-Process     │  │  Multiprocessing     │           │
│  │  Logging            │  │  Logging             │           │
│  │                     │  │                      │           │
│  │  RotatingFile       │  │  QueueHandler +      │           │
│  │  Handler            │  │  QueueListener       │           │
│  └─────────────────────┘  └──────────────────────┘           │
│           │                         │                         │
│           └────────┬────────────────┘                         │
│                    ▼                                           │
│  ┌──────────────────────────────────────────────────────┐     │
│  │              Log Files                               │     │
│  │  • app.log (main)                                    │     │
│  │  • orders.log (order execution)                      │     │
│  │  • trades.log (trade lifecycle)                      │     │
│  │  • app_errors.log (errors only)                      │     │
│  │  • telegram_*.log (bot logs)                         │     │
│  └──────────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Logger Module (`src/notification/logger.py`)

The central logging module provides:

#### Configuration Constants
```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_BYTES = 500 * 1024 * 1024  # 500MB per log file
BACKUP_COUNT = 99               # 99 backup files (≈50GB total)
```

#### Global State
```python
_queue_listener = None  # QueueListener for multiprocessing
_log_queue = None       # Multiprocessing queue
_listener_lock = threading.Lock()  # Thread-safe initialization
_logging_context = ContextVar('logging_context', default=None)
```

### 2. Log Configuration Dictionary

```python
LOG_CONFIG = {
    "version": 1,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s"
        },
        "standard": {
            "format": "%(asctime)s - %(levelname)s - %(message)s"
        },
    },
    "handlers": {
        "console": {...},           # Console output
        "file": {...},              # Main app.log
        "trade_file": {...},        # trades.log
        "order_file": {...},        # orders.log
        "error_file": {...},        # app_errors.log
        "telegram_*_file": {...},   # Telegram bot logs
    },
    "loggers": {
        "orders": {...},            # Order-specific logger
        "trades": {...},            # Trade-specific logger
        "telegram_*": {...},        # Telegram bot loggers
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "DEBUG"
    }
}
```

## Logging Modes

### Mode 1: Standard Logging (Single Process)

**Use Case:** Single-process applications, Telegram bots, live trading

```python
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# Usage
_logger.debug("Detailed debug information")
_logger.info("Trade executed for %s", symbol)
_logger.warning("Low balance: %.2f USDT", balance)
_logger.error("Failed to connect to exchange")
_logger.exception("Critical error")  # Includes full stack trace
```

**Architecture:**
```
Application Code
      ↓
Logger Instance
      ↓
RotatingFileHandler
      ↓
Log File (app.log)
```

### Mode 2: Multiprocessing-Safe Logging

**Use Case:** Parallel optimization (Optuna with `n_jobs>1`), distributed backtesting

```python
# Main Process
from src.notification.logger import setup_multiprocessing_logging, setup_logger

def main():
    # STEP 1: Initialize multiprocessing logging
    setup_multiprocessing_logging()
    _logger = setup_logger(__name__)
    _logger.info("Multiprocessing logging enabled")

    # STEP 2: Run parallel workers
    study.optimize(objective, n_trials=100, n_jobs=-1)

# Worker Process Module
from src.notification.logger import setup_logger

# STEP 3: Use multiprocessing-aware logger
_logger = setup_logger(__name__, use_multiprocessing=True)
_logger.info("Worker %d started", worker_id)
```

**Architecture:**
```
Main Process:
    setup_multiprocessing_logging()
         ↓
    Creates QueueListener
         ↓
    Listener Thread (writes to files)
         ↓
    [app.log, orders.log, trades.log]

Worker Process 1-N:
    setup_logger(use_multiprocessing=True)
         ↓
    QueueHandler
         ↓
    Multiprocessing Queue → Main Process Listener
```

## API Reference

### Public Functions

#### `setup_logger(name, log_file=None, level=DEBUG, inherit_from=None, use_multiprocessing=False)`

Set up a logger with custom configuration.

**Parameters:**
- `name` (str): Logger name (typically `__name__`)
- `log_file` (str, optional): Custom log file path
- `level` (int, optional): Logging level (default: `logging.DEBUG`)
- `inherit_from` (str, optional): Parent logger to inherit handlers from
- `use_multiprocessing` (bool, optional): Enable multiprocessing-safe logging

**Returns:** `logging.Logger` instance

**Examples:**
```python
# Standard logger
logger = setup_logger(__name__)

# Multiprocessing-safe logger
logger = setup_logger(__name__, use_multiprocessing=True)

# Custom log file
logger = setup_logger(__name__, log_file="/path/to/custom.log")

# Context-aware logger
logger = setup_logger("child", inherit_from="parent")
```

#### `setup_multiprocessing_logging()`

Initialize multiprocessing-safe logging infrastructure.

**Usage:**
- Call ONCE in the main process BEFORE spawning workers
- Automatically registers cleanup via `atexit`
- Thread-safe (uses lock to prevent duplicate initialization)

**Returns:** `multiprocessing.Queue` (the log queue)

**Example:**
```python
if __name__ == "__main__":
    setup_multiprocessing_logging()  # First line in main()
    # ... rest of code
```

#### `get_multiprocessing_logger(name, level=DEBUG)`

Get a logger configured for multiprocessing.

**Parameters:**
- `name` (str): Logger name
- `level` (int, optional): Logging level

**Returns:** `logging.Logger` configured with `QueueHandler`

**Note:** Usually called internally by `setup_logger(..., use_multiprocessing=True)`

#### `shutdown_multiprocessing_logging()`

Shut down the queue listener.

**Usage:**
- Automatically called via `atexit.register()`
- Can be called manually for controlled shutdown
- Thread-safe (uses lock)

#### `log_exception(logger, exc_info=None)`

Log an exception with full stack trace.

**Parameters:**
- `logger`: Logger instance
- `exc_info`: Exception info (optional, defaults to current exception)

**Example:**
```python
try:
    risky_operation()
except Exception as e:
    log_exception(_logger, sys.exc_info())
```

### Specialized Logger Names

| Logger Name | Log File | Purpose |
|------------|----------|---------|
| `__name__` | `app.log` | General application logging |
| `orders` | `orders.log` | Order execution tracking |
| `trades` | `trades.log` | Trade lifecycle events |
| `telegram_screener_bot` | `telegram_screener_bot.log` | Telegram screener bot |
| `telegram_background_services` | `telegram_background_services.log` | Background services |
| `telegram_alert_monitor` | `telegram_alert_monitor.log` | Alert monitoring |
| `telegram_schedule_processor` | `telegram_schedule_processor.log` | Scheduled tasks |

## Implementation Details

### Multiprocessing Queue-Based Architecture

The multiprocessing-safe logging uses Python's `QueueHandler` and `QueueListener`:

```python
def setup_multiprocessing_logging():
    global _queue_listener, _log_queue

    with _listener_lock:
        if _queue_listener is not None:
            return _log_queue  # Already initialized

        # Create queue for log messages
        _log_queue = Queue(-1)  # Unlimited size

        # Create file handlers
        file_handlers = _create_file_handlers()

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)

        # Start queue listener
        _queue_listener = QueueListener(
            _log_queue,
            console_handler,
            *file_handlers,
            respect_handler_level=True
        )
        _queue_listener.start()

        # Register cleanup
        atexit.register(shutdown_multiprocessing_logging)

        return _log_queue
```

### Worker Process Configuration

```python
def get_multiprocessing_logger(name, level=logging.DEBUG):
    global _log_queue

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Initialize queue if needed
    if _log_queue is None:
        setup_multiprocessing_logging()

    # Clear existing handlers
    logger.handlers.clear()

    # Add queue handler
    if _log_queue is not None:
        queue_handler = QueueHandler(_log_queue)
        queue_handler.setLevel(level)
        logger.addHandler(queue_handler)
        logger.propagate = False

    return logger
```

## Log File Rotation

### Configuration

```python
file_handler = RotatingFileHandler(
    filename=log_file_path,
    maxBytes=500 * 1024 * 1024,  # 500MB
    backupCount=99,               # 99 backups
    encoding="utf-8"
)
```

### Rotation Behavior

- **Trigger**: When log file reaches 500MB
- **Action**: Current file renamed to `app.log.1`, previous backups shifted
- **Sequence**: `app.log` → `app.log.1` → `app.log.2` → ... → `app.log.99`
- **Oldest**: `app.log.99` is deleted when rotation occurs
- **Total Size**: ≈50GB (100 files × 500MB)

### Rotation in Multiprocessing

With multiprocessing-safe logging:
- ✅ **Safe**: Only the QueueListener thread writes to files
- ✅ **No Conflicts**: Worker processes never access files directly
- ✅ **Atomic**: Rotation happens in single thread (no race conditions)

Without multiprocessing-safe logging:
- ❌ **Unsafe**: Multiple processes compete for file access
- ❌ **Corruption**: Rotation can corrupt files
- ❌ **Lost Messages**: Race conditions drop log records

## Usage Patterns

### Pattern 1: Single-Process Application

```python
# src/web_ui/app.py
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

@app.route('/trade')
def execute_trade():
    _logger.info("Trade request received")
    try:
        result = process_trade()
        _logger.info("Trade executed successfully")
        return result
    except Exception:
        _logger.exception("Trade execution failed")
        raise
```

### Pattern 2: Optimization Script (Multiprocessing)

```python
# src/backtester/optimizer/walk_forward_optimizer.py
from src.notification.logger import setup_multiprocessing_logging, setup_logger

_logger = setup_logger(__name__)

def main():
    # Initialize multiprocessing logging
    setup_multiprocessing_logging()
    _logger.info("Starting walk-forward optimization")

    # Run Optuna with parallel workers
    study.optimize(objective, n_trials=100, n_jobs=-1)
```

### Pattern 3: Strategy Module (Worker Process)

```python
# src/strategy/base_strategy.py
from src.notification.logger import setup_logger

# Enable multiprocessing mode for all loggers
_logger = setup_logger(__name__, use_multiprocessing=True)
_order_logger = setup_logger('orders', use_multiprocessing=True)
_trade_logger = setup_logger('trades', use_multiprocessing=True)

class BaseStrategy(bt.Strategy):
    def notify_order(self, order):
        _order_logger.info(
            "ORDER - %s | Type: %s | Status: %s",
            self.symbol, order_type, status
        )
```

### Pattern 4: Context-Aware Logging

```python
from src.notification.logger import setup_context_logger, set_logging_context

# Set context for telegram bot
set_logging_context('telegram_screener_bot')

# Child modules inherit context
logger = setup_context_logger(__name__, 'telegram_screener_bot')
```

## Performance Considerations

### Queue Size

Default: Unlimited (`Queue(-1)`)

**Tuning:**
```python
# In src/notification/logger.py
_log_queue = Queue(maxsize=10000)  # Limit queue size
```

**Trade-offs:**
- Unlimited: No backpressure, but can consume memory
- Limited: Backpressure prevents memory overflow, but may block workers

### Logging Overhead

| Operation | Time (μs) | Impact |
|-----------|-----------|--------|
| Log with lazy formatting | 1-5 | ✅ Negligible |
| Log with f-string | 5-10 | ⚠️ Wasteful if disabled |
| Queue put (multiprocessing) | 10-50 | ✅ Acceptable |
| File write (single process) | 50-200 | ⚠️ Can block |
| File write (multiprocessing) | 10-50 | ✅ Non-blocking |

### Best Practices for Performance

```python
# ✅ GOOD: Lazy formatting (deferred)
_logger.debug("Processing %d items for %s", count, symbol)

# ❌ BAD: Eager formatting (always executed)
_logger.debug(f"Processing {count} items for {symbol}")

# ✅ GOOD: Guard expensive operations
if _logger.isEnabledFor(logging.DEBUG):
    expensive_data = compute_expensive_debug_info()
    _logger.debug("Debug data: %s", expensive_data)
```

## Troubleshooting

### Issue: Logs Not Appearing

**Symptoms:**
- No log output during optimization
- Empty log files

**Diagnosis:**
```python
# Check if multiprocessing logging is initialized
from src.notification import logger
print(logger._queue_listener)  # Should not be None
print(logger._log_queue)  # Should not be None
```

**Solution:**
```python
# Ensure setup is called FIRST
setup_multiprocessing_logging()
_logger = setup_logger(__name__)
```

### Issue: Duplicate Log Messages

**Symptoms:**
- Each message appears twice
- Console and file have duplicates

**Cause:** Multiple handler registration

**Solution:**
```python
# Clear existing handlers
logger.handlers.clear()

# Or use setup_logger which handles this
logger = setup_logger(__name__, use_multiprocessing=True)
```

### Issue: Lost Log Messages

**Symptoms:**
- Missing log entries
- Inconsistent logging

**Cause:** Not using multiprocessing-safe logging with `n_jobs>1`

**Solution:**
```python
# Main process
setup_multiprocessing_logging()

# Worker modules
_logger = setup_logger(__name__, use_multiprocessing=True)
```

### Issue: Log Messages Out of Order

**Symptoms:**
- Timestamps not sequential
- Messages from different workers interleaved

**Explanation:** This is **expected** with multiprocessing. Messages arrive in queue order, not timestamp order.

**Not a Bug:** Each worker has its own timeline. Use correlation IDs if order matters:

```python
import uuid
correlation_id = str(uuid.uuid4())
_logger.info("Step 1 [%s]", correlation_id)
_logger.info("Step 2 [%s]", correlation_id)
```

## Testing

### Unit Tests

```python
# tests/test_logger.py
import pytest
from src.notification.logger import setup_logger, setup_multiprocessing_logging

def test_standard_logger():
    logger = setup_logger("test")
    assert logger is not None
    assert logger.name == "test"

def test_multiprocessing_logger():
    setup_multiprocessing_logging()
    logger = setup_logger("test", use_multiprocessing=True)
    assert logger is not None
    assert len(logger.handlers) > 0
    assert isinstance(logger.handlers[0], QueueHandler)
```

### Integration Tests

```python
# Test with actual Optuna run
def test_multiprocessing_logging_with_optuna():
    setup_multiprocessing_logging()

    def objective(trial):
        logger = setup_logger(__name__, use_multiprocessing=True)
        logger.info("Trial %d started", trial.number)
        return trial.suggest_float("x", 0, 1)

    study = optuna.create_study()
    study.optimize(objective, n_trials=10, n_jobs=2)

    # Verify all log messages appear
    with open("logs/log/app.log") as f:
        log_content = f.read()
        assert "Trial" in log_content
        # Count log entries
        assert log_content.count("Trial") == 10
```

## Migration Guide

### From Standard to Multiprocessing Logging

**Before:**
```python
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# Run optimization
study.optimize(objective, n_trials=100, n_jobs=-1)
```

**After:**
```python
from src.notification.logger import setup_multiprocessing_logging, setup_logger

# In main process
setup_multiprocessing_logging()
_logger = setup_logger(__name__)

# In worker modules
_logger = setup_logger(__name__, use_multiprocessing=True)

# Run optimization (now safe)
study.optimize(objective, n_trials=100, n_jobs=-1)
```

## Related Documentation

- [Infrastructure Module](modules/infrastructure.md#6-logging--observability) - Parent module overview
- [Multiprocessing Logging Guide](../MULTIPROCESSING_LOGGING.md) - Detailed implementation guide
- [Coding Conventions](../.claude/CLAUDE.md#3-logging) - Logging standards
- [Python Logging Cookbook](https://docs.python.org/3/howto/logging-cookbook.html) - Official Python docs

## Changelog

### 2025-01-27 - Multiprocessing-Safe Logging
- ✅ Added `setup_multiprocessing_logging()` for parallel workloads
- ✅ Added `use_multiprocessing` parameter to `setup_logger()`
- ✅ Implemented `QueueHandler` and `QueueListener` pattern
- ✅ Fixed log message loss with `n_jobs=-1`
- ✅ Added automatic cleanup via `atexit`
- ✅ Thread-safe initialization with `threading.Lock()`

### 2024-12-15 - Context-Aware Logging
- Added `ContextAwareLogger` for hierarchical logging
- Added `set_logging_context()` and `get_logging_context()`
- Telegram bot logging context support

### 2024-11-01 - Initial Implementation
- Base logging infrastructure
- Rotating file handlers
- Separate log files for orders and trades
- UTF-8 encoding support

---

**Last Updated:** 2025-01-27
**Maintainer:** Infrastructure Team
**Status:** Production Ready ✅
