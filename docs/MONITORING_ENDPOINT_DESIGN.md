# Channel Health Monitoring - Database-Centric Design

## Overview

This document describes how the Notification Service monitors and reports the health of its
notification channels. Unlike a REST-endpoint approach, the system uses **database-centric health
reporting**: health state is written to the `msg_system_health` table and can be queried via SQL
or `SystemHealthService`.

There are **no HTTP health endpoints** in the Notification Service or Telegram Bot — both services
are standalone processes that report health to the shared database.

---

## Architecture

```
notification_db_centric_bot.py
│
├── HealthReporter (reports every 60s)
│   ├── _report_service_health()      → msg_system_health [system='notification']
│   └── _report_channel_health()      → msg_system_health [system='notification', component='email']
│
└── health_monitor (HealthMonitor)    → in-memory status + configures per-channel checks
    └── _monitor_channel_health()     → polls channel.check_health() every 60s
```

---

## Components

### 1. HealthReporter (`notification_db_centric_bot.py`)

The `HealthReporter` class runs as a periodic background task that writes health state to the
database every `health_check_interval_seconds` (default: 60 seconds).

**Service-level health** (`system='notification'`, `component=None`):
```python
health_service.update_system_health(
    system='notification',
    component=None,
    status=SystemHealthStatus.HEALTHY,  # or DOWN if processor is not running
    metadata={
        'service': config.service_name,
        'message_processor_running': message_processor.is_running,
        'message_poller_running': message_poller.running,
    }
)
```

**Channel-level health** (`system='notification'`, `component='email'`):
```python
health_service.update_notification_channel_health(
    channel='email',
    status=SystemHealthStatus.HEALTHY,  # or DOWN
    response_time_ms=response_time_ms,
    error_message=error_message,
    metadata={'last_check': datetime.now(timezone.utc).isoformat()}
)
```

### 2. HealthMonitor (`src/notification/service/health_monitor.py`)

The `HealthMonitor` manages in-memory health state for each active channel. It runs periodic
background tasks that:

1. Call `channel_instance.check_health()` (connectivity, response time, success rate)
2. Evaluate results against configurable thresholds
3. Auto-disable channels after 3 consecutive failures
4. Auto-enable channels after 2 consecutive successes
5. Trigger callbacks on status changes (for fallback routing)

Each channel is configured with a `HealthCheckConfig`:
```python
from src.notification.service.health_monitor import HealthCheckConfig, HealthCheckType

health_config = HealthCheckConfig(
    channel='email',
    check_interval_seconds=60,
    auto_disable_threshold=3,   # disable after 3 consecutive failures
    auto_enable_threshold=2,    # re-enable after 2 consecutive successes
    enabled_checks={HealthCheckType.CONNECTIVITY, HealthCheckType.RESPONSE_TIME}
)
health_monitor.configure_channel(health_config)
```

### 3. Channel Health Checks

Each channel plugin implements `check_health() -> ChannelHealth`:

- **Email**: Tests SMTP connectivity + authentication
- **Telegram** (future): Tests Telegram Bot API via `get_me()`
- **SMS** (future): Tests provider API connectivity

Health status levels: `HEALTHY`, `DEGRADED`, `DOWN`

---

## Database Schema for Health State

Health data is stored in `msg_system_health`:

```sql
-- Notification service overall health
SELECT * FROM msg_system_health
WHERE system = 'notification' AND component IS NULL;

-- Channel-level health (e.g., email)
SELECT * FROM msg_system_health
WHERE system = 'notification' AND component = 'email';

-- Backward-compatible view across all notification channels
SELECT channel, status, last_success, last_failure, failure_count,
       avg_response_time_ms, error_message, checked_at
FROM v_channel_health;
-- (v_channel_health is a view filtering WHERE system='notification' AND component IS NOT NULL)
```

Status values: `HEALTHY`, `DEGRADED`, `DOWN`, `UNKNOWN`

---

## Querying Health State

### Via SQL (Operational Dashboard)

```sql
-- Current health status of all managed components
SELECT system, component, status, failure_count, avg_response_time_ms,
       last_success, error_message, checked_at
FROM msg_system_health
WHERE system = 'notification'
ORDER BY component NULLS FIRST;

-- Channel message delivery performance (last 24 hours)
SELECT
    channels,
    status,
    COUNT(*) AS message_count,
    AVG(EXTRACT(EPOCH FROM (delivered_at - created_at))) AS avg_delivery_time_sec,
    MIN(created_at) AS first_message,
    MAX(created_at) AS last_message
FROM messages
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY channels, status
ORDER BY channels, status;

-- Results example:
-- channels       | status    | msg_count | avg_delivery_time_sec
-- {email}        | DELIVERED | 45        | 12.5
-- {email}        | PENDING   | 2         | NULL
-- {telegram}     | DELIVERED | 245       | 6.2
-- {telegram}     | FAILED    | 3         | NULL
```

### Via Python (`SystemHealthService`)

```python
from src.data.db.services.system_health_service import SystemHealthService

health_service = SystemHealthService()

# Read service health
# (Low-level DB query — use uow pattern if needed)
```

### Via In-Memory State (`HealthMonitor`)

```python
from src.notification.service.health_monitor import health_monitor

# Get status of a specific channel
status = health_monitor.get_channel_status('email')
print(f"Email: {status.overall_status.value}, uptime: {status.uptime_percentage:.1f}%")

# Get all channel statuses
all_statuses = health_monitor.get_all_statuses()
for channel, status in all_statuses.items():
    print(f"{channel}: {status.overall_status.value}")

# Get overall health summary
summary = health_monitor.get_health_summary()
print(summary)  # total_channels, status_distribution, healthy_percentage, etc.
```

---

## Startup Flow

When `notification_db_centric_bot.py` starts:

```
startup()
├── init_databases()
├── _register_channel_plugins()         ← Only registers enabled channels (from config.enabled_channels)
├── message_processor.start()
│   └── _initialize_channel_instances() ← Only initializes enabled_channels=['email']
│       └── health_monitor.configure_channel(HealthCheckConfig(...))
├── health_monitor.start()              ← Starts periodic checks for each configured channel
├── MessagePoller.start()               ← Polls DB every 5s for pending messages
└── HealthReporter.start()              ← Reports health to DB every 60s
```

---

## Channel Ownership and Health Visibility

| Channel  | Owned By             | Health Reported By   | Where Visible                    |
|----------|----------------------|----------------------|----------------------------------|
| telegram | Telegram Bot         | Telegram Bot         | `msg_system_health` (future)     |
| email    | Notification Service | `HealthReporter`     | `msg_system_health`, `v_channel_health` |
| sms      | Notification Service | `HealthReporter`     | `msg_system_health`, `v_channel_health` |

See [CHANNEL_OWNERSHIP.md](CHANNEL_OWNERSHIP.md) for the full ownership architecture.

---

## Troubleshooting

### Channel Showing as DOWN in database

1. Check if `HealthReporter` is running — look for "Health reporter started" in logs
2. Check the channel's `check_health()` result in logs: `"Error checking health for channel email"`
3. Query the raw health record:
   ```sql
   SELECT error_message, checked_at FROM msg_system_health
   WHERE system='notification' AND component='email';
   ```
4. Check `HealthMonitor` in-memory status for more context (e.g., consecutive_failures count)

### Health not being written to database

1. Verify `SystemHealthService` can connect to DB
2. Check `health_check_interval_seconds` in `NotificationServiceConfig` (default: 60)
3. Look for `"Error reporting channel health"` in notification service logs

### Channel auto-disabled

`HealthMonitor` auto-disables a channel after `auto_disable_threshold=3` consecutive failures.
The channel remains in `_channel_instances` but `is_enabled=False`. Messages for that channel
will be routed to fallback channels by `FallbackManager`.

To manually re-enable:
```python
health_monitor.manually_enable_channel('email', reason='SMTP issue resolved')
```
