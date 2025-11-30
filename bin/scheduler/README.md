# Scheduler Service Scripts

This directory contains scripts and configuration for managing the E-Trading Scheduler Service.

## Overview

The Scheduler Service runs scheduled jobs for:
- **VIX Daily Monitor** - Monitors VIX index and sends notifications
- **EMPS2 Scanner** - Runs morning and evening scans for trading opportunities
- **Fundamentals Refresh** - Updates fundamentals cache on weekends

## Quick Start

### 1. Insert Schedules into Database

First, populate the database with the 4 scheduled jobs:

```bash
# PostgreSQL
psql -d your_database_name < bin/scheduler/insert_schedules.sql

# SQLite
sqlite3 your_database.db < bin/scheduler/insert_schedules.sql
```

### 2. Start the Scheduler

**Linux/Mac:**
```bash
./bin/scheduler/start.sh
```

**Windows:**
```cmd
bin\scheduler\start.bat
```

### 3. Check Status

**Linux/Mac:**
```bash
./bin/scheduler/status.sh
```

**Windows:**
```cmd
bin\scheduler\status.bat
```

### 4. Stop the Scheduler

**Linux/Mac:**
```bash
./bin/scheduler/stop.sh
```

**Windows:**
```cmd
bin\scheduler\stop.bat
```

## Files

### Configuration
- `insert_schedules.sql` - SQL script to create 4 scheduled jobs

### Linux/Mac Scripts
- `start.sh` - Start scheduler service
- `stop.sh` - Stop scheduler service
- `status.sh` - Check service status
- `reload.sh` - Reload schedules from database

### Windows Scripts
- `start.bat` - Start scheduler service
- `stop.bat` - Stop scheduler service
- `status.bat` - Check service status

## Scheduled Jobs

### 1. VIX Daily Monitor
- **Schedule**: Daily at 9:30 AM ET (weekdays)
- **Timeout**: 10 minutes
- **Notifications**:
  - Email when VIX ≥ 20
  - Email + Telegram when VIX ≥ 25

### 2. EMPS2 Morning Scan
- **Schedule**: Daily at 9:35 AM ET (weekdays)
- **Timeout**: 4 hours
- **Notifications**:
  - Email for Phase 1 candidates
  - Email + Telegram for Phase 2 candidates

### 3. EMPS2 Evening Scan
- **Schedule**: Daily at 2:00 PM ET (weekdays) ≈ 8 PM CET
- **Timeout**: 4 hours
- **Notifications**:
  - Email for Phase 1 candidates
  - Email + Telegram for Phase 2 candidates

### 4. Fundamentals Cache Refresh
- **Schedule**: Saturday at 2:00 AM UTC
- **Timeout**: 1 hour
- **Notifications**:
  - Email on completion (success or error)

## Production Deployment (Linux)

### Option 1: Systemd Service (Recommended)

1. **Update service file** with your paths:
   ```bash
   nano src/scheduler/deployment/scheduler.service
   # Update User, Group, WorkingDirectory, and ExecStart paths
   ```

2. **Install service**:
   ```bash
   sudo cp src/scheduler/deployment/scheduler.service /etc/systemd/system/e-trading-scheduler.service
   sudo systemctl daemon-reload
   ```

3. **Enable and start**:
   ```bash
   sudo systemctl enable e-trading-scheduler
   sudo systemctl start e-trading-scheduler
   ```

4. **Check status**:
   ```bash
   sudo systemctl status e-trading-scheduler
   sudo journalctl -u e-trading-scheduler -f
   ```

### Option 2: Manual Background Process

Use the provided shell scripts:
```bash
./bin/scheduler/start.sh
```

The process will run in the background and log to `logs/scheduler.log`.

## Notification Configuration

Notifications are sent to **user_id = 2**:
- **Email**: akossyrev@gmail.com
- **Telegram**: 859865894

To change recipients, update the user record in the `usr_users` and `usr_auth_identities` tables.

## Troubleshooting

### Check Logs

**Linux/Mac:**
```bash
tail -f logs/scheduler.log
```

**Windows:**
```cmd
type logs\scheduler.log
```

### Common Issues

1. **Database Connection Error**
   - Check `DATABASE_URL` environment variable
   - Verify database is running and accessible

2. **Script Not Found**
   - Verify paths in `task_params.script_path`
   - Check project root is correct

3. **No Notifications**
   - Verify notification service is running at `http://localhost:8000`
   - Check user has telegram_chat_id in `usr_auth_identities`
   - Check user has email in `usr_users`

4. **Job Timeout**
   - Increase `timeout_seconds` in `task_params`
   - Check script execution time in logs

### Verify Schedules

```sql
-- Check all schedules
SELECT id, name, job_type, cron, enabled, next_run_at
FROM job_schedules
WHERE user_id = 2;

-- Check recent job runs
SELECT id, job_type, status, started_at, finished_at, error
FROM job_schedule_runs
ORDER BY started_at DESC
LIMIT 10;
```

### Reload Schedules

If you modify schedules in the database, reload them without restarting:

**Linux/Mac:**
```bash
./bin/scheduler/reload.sh
```

**Python CLI:**
```bash
python -m src.scheduler.cli reload
```

## Development

### Run in Foreground

**Linux/Mac:**
```bash
./bin/scheduler/start.sh --foreground
```

**Python:**
```bash
python -m src.scheduler.main
```

### Environment Variables

Set these in `.env` or export them:

```bash
export TRADING_ENV=development
export LOG_LEVEL=DEBUG
export DATABASE_URL=postgresql://user:pass@localhost/e_trading
export NOTIFICATION_SERVICE_URL=http://localhost:8000
```

## Architecture

```
Scheduler Service
├── APScheduler (job orchestration)
├── Database (PostgreSQL/SQLite)
│   ├── job_schedules (job definitions)
│   └── job_schedule_runs (execution history)
├── Job Execution
│   └── Subprocess execution of Python scripts
├── Notification Service
│   ├── Email Channel
│   └── Telegram Channel
└── User Management
    ├── usr_users (user details)
    └── usr_auth_identities (telegram IDs)
```

## Support

For issues or questions:
1. Check logs: `logs/scheduler.log`
2. Review job runs in database: `job_schedule_runs` table
3. Verify notification service is running
4. Check script execution manually:
   ```bash
   python src/data/vix.py
   python src/ml/pipeline/p06_emps2/run_emps2_scan.py
   python src/data/utils/refresh_fundamentals_cache.py
   ```
