# Scheduler Module

## Overview

The `src/scheduler/` module is the centralized job-scheduling engine for the e-trading platform.
It persists scheduled jobs in PostgreSQL via APScheduler's SQLAlchemy job store, executes alert
evaluations and data-processing jobs asynchronously, and delivers notifications through the
notification service.

## Features

- **APScheduler-backed persistence** — jobs survive restarts; state stored in PostgreSQL.
- **Cron-based triggers** — 5-field and 6-field cron expressions supported.
- **Real-time DB synchronization** — listens for `NOTIFY scheduler_updates` via asyncpg and
  debounces burst notifications before reloading the schedule.
- **Atomic job reload** — add/replace first, remove stale jobs last; no zero-jobs window.
- **Per-job timeout enforcement** — wraps every job in `asyncio.wait_for()` using the
  configurable `job_timeout_seconds` parameter.
- **Missed-run protection** — `misfire_grace_time=300` discards executions missed by more than
  5 minutes; `max_instances=1` per job prevents concurrent self-overlap.
- **Structured alert notifications** — formats Markdown messages with OHLCV context, indicator
  bucketing, rule snapshots, and rearm status, then persists to the notification DB.
- **CLI management** — `cli.py` provides `start`, `status`, `reload`, and `validate` commands.

## Quick Start

```python
from src.scheduler.config import SchedulerServiceConfig
from src.scheduler.main import SchedulerApplication
import asyncio

config = SchedulerServiceConfig()
app = SchedulerApplication(config)
asyncio.run(app.start())
```

Or via the CLI:

```bash
python -m src.scheduler.cli start
python -m src.scheduler.cli status
python -m src.scheduler.cli reload
python -m src.scheduler.cli validate
```

## Module Layout

```
src/scheduler/
├── cli.py                  # CLI entry point (start / status / reload / validate)
├── config.py               # Configuration dataclasses + lazy get_config() singleton
├── main.py                 # SchedulerApplication lifecycle (start / stop / reload)
├── scheduler_service.py    # SchedulerService — core APScheduler integration
├── docs/
│   ├── README.md           # Internal design notes
│   ├── MONITORING.md       # Metrics, health checks, alerting
│   ├── BACKUP_RECOVERY.md  # Backup and disaster-recovery procedures
│   └── TROUBLESHOOTING.md  # Common issues and debugging guide
└── tests/                  # Module-specific unit tests
```

## Integration

This module integrates with:

- `src.data.db.services.jobs_service` — reads `Schedule` records to register jobs.
- `src.data.db.services.notification_service` — persists alert notification messages.
- `src.common.alerts.alert_evaluator` — evaluates triggered alert conditions.
- `src.common.alerts.cron_parser` — validates cron expressions before registration.
- `src.notification.service.client` — `MessagePriority` enum for notification priority.

## Configuration

Key environment variables (resolved by `SchedulerServiceConfig`):

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | — | PostgreSQL connection string |
| `TRADING_ENV` | `development` | Environment name |
| `SCHEDULER_MAX_WORKERS` | `10` | Maximum concurrent job slots |
| `SCHEDULER_JOB_TIMEOUT` | `300` | Per-job timeout in seconds |
| `LOG_LEVEL` | `INFO` | Logging level |

Use `get_config()` (lazy singleton) instead of instantiating `SchedulerServiceConfig()` at
import time to avoid side effects during testing.

## Related Documentation

- [Internal Design Notes](docs/README.md)
- [Monitoring Guide](docs/MONITORING.md)
- [Backup & Recovery](docs/BACKUP_RECOVERY.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
