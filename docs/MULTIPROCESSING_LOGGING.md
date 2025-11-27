# Multiprocessing-Safe Logging

## Overview

This document explains the multiprocessing-safe logging system implemented for the e-trading platform, specifically designed to handle concurrent logging from multiple processes during optimization runs.

## Problem Statement

### The Issue

When running optimizations with `n_jobs=-1` (using all CPU cores), Optuna spawns multiple worker processes that execute trials in parallel. Python's standard `RotatingFileHandler` is **NOT process-safe**, leading to:

1. **Lost log messages** - Race conditions cause messages to be dropped
2. **Corrupted log files** - Multiple processes writing simultaneously
3. **File handle conflicts** - File rotation issues across processes
4. **Inconsistent ordering** - Messages appear out of sequence

### Example of the Problem

```python
# Before: Standard logging (NOT multiprocessing-safe)
from src.notification.logger import setup_logger
_logger = setup_logger(__name__)

# With n_jobs=-1, multiple processes try to write to app.log simultaneously
# Result: Lost messages, corrupted files
```

## Solution: Queue-Based Logging

### Architecture

The solution uses Python's `QueueHandler` and `QueueListener` pattern:

```
┌─────────────────────────────────────────────────────────────┐
│                     Main Process                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  QueueListener (single writer thread)                │   │
│  │  • Consumes messages from queue                      │   │
│  │  • Writes to file handlers (app.log, trades.log)    │   │
│  │  • Thread-safe, no file conflicts                   │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ▲                                  │
│                          │ Queue                           │
│                          │                                  │
├──────────────────────────┼──────────────────────────────────┤
│  Worker Process 1        │  Worker Process 2                │
│  ┌────────────────┐     │     ┌────────────────┐          │
│  │ QueueHandler   │─────┴────▶│ QueueHandler   │          │
│  │ (logger)       │            │ (logger)       │          │
│  └────────────────┘            └────────────────┘          │
│  Optuna Trial #1               Optuna Trial #2             │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **QueueHandler**: Each worker process sends log messages to a shared queue
2. **QueueListener**: Single listener thread in main process writes to files
3. **Multiprocessing Queue**: Process-safe message passing
4. **Thread Lock**: Ensures single QueueListener initialization

## Implementation

### 1. Logger Setup Module

The `src/notification/logger.py` module provides:

```python
# New functions for multiprocessing-safe logging
setup_multiprocessing_logging()      # Initialize in main process
get_multiprocessing_logger(name)     # Get logger for worker processes
shutdown_multiprocessing_logging()   # Cleanup (automatic via atexit)
```

### 2. Main Process Setup

In the optimizer's main entry point:

```python
# src/backtester/optimizer/walk_forward_optimizer.py
from src.notification.logger import setup_multiprocessing_logging

def main():
    # MUST be called FIRST, before spawning workers
    setup_multiprocessing_logging()

    # ... rest of optimization code
```

### 3. Worker Process Logging

In modules that run inside worker processes:

```python
# src/strategy/base_strategy.py
from src.notification.logger import setup_logger

# use_multiprocessing=True enables queue-based logging
_logger = setup_logger(__name__, use_multiprocessing=True)
_order_logger = setup_logger('orders', use_multiprocessing=True)
_trade_logger = setup_logger('trades', use_multiprocessing=True)
```

### 4. Backward Compatibility

Standard logging still works for non-multiprocessing scenarios:

```python
# Single-process mode (default)
_logger = setup_logger(__name__)  # Works as before

# Multiprocessing mode
_logger = setup_logger(__name__, use_multiprocessing=True)
```

## Usage

### For Optimization Scripts

**Always call `setup_multiprocessing_logging()` BEFORE running Optuna:**

```python
from src.notification.logger import setup_multiprocessing_logging, setup_logger

def main():
    # Step 1: Initialize multiprocessing logging
    setup_multiprocessing_logging()

    # Step 2: Get your logger (standard usage)
    _logger = setup_logger(__name__)
    _logger.info("Starting optimization")

    # Step 3: Run Optuna with n_jobs=-1 (safe now!)
    study.optimize(objective, n_trials=100, n_jobs=-1)
```

### For Strategy Modules

**Use `use_multiprocessing=True` in modules that run in workers:**

```python
from src.notification.logger import setup_logger

# This module runs in Optuna worker processes
_logger = setup_logger(__name__, use_multiprocessing=True)
```

### Automatic Cleanup

Cleanup happens automatically via `atexit.register()`:

```python
# No manual cleanup needed - happens automatically on exit
# But you can manually trigger it if needed:
from src.notification.logger import shutdown_multiprocessing_logging
shutdown_multiprocessing_logging()
```

## Files Modified

### Core Logging Module
- `src/notification/logger.py`
  - Added `setup_multiprocessing_logging()`
  - Added `get_multiprocessing_logger()`
  - Added `shutdown_multiprocessing_logging()`
  - Added `use_multiprocessing` parameter to `setup_logger()`

### Optimizer Scripts
- `src/backtester/optimizer/walk_forward_optimizer.py`
  - Added `setup_multiprocessing_logging()` call in `main()`

- `src/backtester/optimizer/run_optimizer.py`
  - Added `setup_multiprocessing_logging()` call in `__main__`

### Strategy Modules (Worker Process Code)
- `src/strategy/base_strategy.py`
  - Updated loggers to use `use_multiprocessing=True`

- `src/backtester/optimizer/custom_optimizer.py`
  - Updated logger to use `use_multiprocessing=True`

## Testing

### Verify Multiprocessing Logging

1. **Check for initialization message:**
```bash
python -m src.backtester.optimizer.walk_forward_optimizer
# Should see: "Multiprocessing-safe logging enabled for walk-forward optimization"
```

2. **Monitor log files during parallel runs:**
```bash
# Terminal 1: Run optimization
python -m src.backtester.optimizer.walk_forward_optimizer

# Terminal 2: Watch logs in real-time
tail -f logs/log/app.log
```

3. **Verify no lost messages:**
   - All log messages should appear in `app.log`
   - No corrupted log entries
   - Proper sequencing (timestamps may be out of order due to concurrency)

### Performance Impact

The queue-based approach has minimal performance impact:
- ✅ Queue operations are very fast (in-memory)
- ✅ Single writer thread prevents file contention
- ✅ Worker processes don't block on I/O
- ⚠️ Slight memory overhead for queue buffer

## Troubleshooting

### Logs Not Appearing

**Problem:** No log output during optimization

**Solution:**
```python
# Ensure setup is called FIRST
setup_multiprocessing_logging()  # Add this line
_logger = setup_logger(__name__)
```

### Duplicate Log Messages

**Problem:** Messages appear twice in logs

**Solution:**
```python
# Don't call setup_multiprocessing_logging() multiple times
# It's protected by a lock, but avoid redundant calls
```

### Messages Out of Order

**Problem:** Log timestamps are not sequential

**Explanation:** This is **expected** with multiprocessing. Messages from different processes arrive in queue order, not timestamp order. Each process has its own timeline.

### Import Errors

**Problem:** `ImportError: cannot import name 'setup_multiprocessing_logging'`

**Solution:**
```python
# Make sure you're importing from the updated logger module
from src.notification.logger import setup_multiprocessing_logging, setup_logger
```

## Configuration

### Adjust Queue Size

If you need to handle very high logging volume:

```python
# In src/notification/logger.py
_log_queue = Queue(maxsize=10000)  # Default: -1 (unlimited)
```

### Log Levels in Workers

Control verbosity in worker processes:

```python
_logger = setup_logger(__name__, use_multiprocessing=True, level=logging.INFO)
```

## Best Practices

1. ✅ **Always** call `setup_multiprocessing_logging()` before spawning workers
2. ✅ **Use** `use_multiprocessing=True` in worker process modules
3. ✅ **Avoid** direct file handlers in worker processes
4. ✅ **Test** with `n_jobs=1` first, then enable parallelization
5. ✅ **Monitor** log files during development to verify correctness

## Migration Guide

### From Old Logging to Multiprocessing-Safe

**Before:**
```python
from src.notification.logger import setup_logger
_logger = setup_logger(__name__)

# Optimization with n_jobs=-1
study.optimize(objective, n_trials=100, n_jobs=-1)  # LOSES LOGS
```

**After:**
```python
from src.notification.logger import setup_multiprocessing_logging, setup_logger

# In main process
setup_multiprocessing_logging()  # ADD THIS

# In worker modules
_logger = setup_logger(__name__, use_multiprocessing=True)  # ADD FLAG

# Optimization with n_jobs=-1
study.optimize(objective, n_trials=100, n_jobs=-1)  # NOW SAFE
```

## References

- [Python Logging Cookbook - Multiprocessing](https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes)
- [PEP 391 - Dictionary Based Logging Configuration](https://peps.python.org/pep-0391/)
- [Optuna Parallelization](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html)

## Changelog

### 2025-01-27
- Initial implementation of queue-based multiprocessing-safe logging
- Added `setup_multiprocessing_logging()` function
- Updated walk-forward optimizer and run_optimizer
- Updated base_strategy.py and custom_optimizer.py to use multiprocessing logging
- Fixed symbol/timeframe parameter passing issue

---

**Last Updated:** 2025-01-27
**Author:** AI Assistant with Human Review
**Status:** Production Ready ✅
