# System Scheduler Troubleshooting Guide

## Overview

This guide provides solutions for common issues encountered with the System Scheduler service. The scheduler is responsible for executing alerts, screeners, and other scheduled jobs in the Advanced Trading Framework.

## Common Issues and Solutions

### 1. Service Startup Issues

#### Issue: Scheduler fails to start
**Symptoms:**
- Service exits immediately after startup
- Error messages about database connection
- APScheduler initialization failures

**Diagnosis:**
```bash
# Check service status
sudo systemctl status trading-scheduler

# View recent logs
sudo journalctl -u trading-scheduler -n 50

# Test database connection
python -m src.scheduler.cli test-db
```

**Solutions:**

**Database Connection Issues:**
```bash
# Check database connectivity
psql -h localhost -U trading_user -d trading_db -c "SELECT 1;"

# Verify database URL in configuration
grep -r "DATABASE_URL" config/scheduler/

# Test with correct credentials
export SCHEDULER_DATABASE_URL="postgresql+asyncpg://user:pass@localhost/trading_db"
python -m src.scheduler.cli test-db
```

**Configuration Issues:**
```bash
# Validate configuration
python -m src.scheduler.cli validate-config

# Check for missing environment variables
env | grep SCHEDULER_

# Verify configuration file syntax
python -c "import json; json.load(open('config/scheduler/scheduler.json'))"
```

**Permission Issues:**
```bash
# Check file permissions
ls -la /opt/trading-framework/src/scheduler/
sudo chown -R trading:trading /opt/trading-framework/

# Check systemd service permissions
sudo systemctl cat trading-scheduler
```

#### Issue: APScheduler job store initialization fails
**Symptoms:**
- Error: "Could not create job store"
- Database table creation failures
- SQLAlchemy connection errors

**Solutions:**
```bash
# Create APScheduler tables manually
python -c "
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from sqlalchemy import create_engine
engine = create_engine('postgresql://user:pass@localhost/trading_db')
jobstore = SQLAlchemyJobStore(engine=engine)
jobstore._get_jobs_tablename()  # This creates the table
"

# Check database permissions
psql -h localhost -U trading_user -d trading_db -c "
CREATE TABLE IF NOT EXISTS apscheduler_jobs (
    id VARCHAR(191) PRIMARY KEY,
    next_run_time DOUBLE PRECISION,
    job_state BYTEA NOT NULL
);
"
```

### 2. Job Execution Issues

#### Issue: Jobs not executing
**Symptoms:**
- Schedules exist but jobs don't run
- No job_schedule_runs records created
- APScheduler shows no active jobs

**Diagnosis:**
```bash
# Check active schedules
python -c "
from src.data.db.services.jobs_service import JobsService
from src.data.db.database import get_session
with get_session() as session:
    jobs_service = JobsService(session)
    schedules = jobs_service.get_enabled_schedules()
    print(f'Found {len(schedules)} enabled schedules')
    for s in schedules[:5]:
        print(f'  {s.id}: {s.name} - {s.cron}')
"

# Check APScheduler job registration
curl http://localhost:8002/stats | jq '.job_counts'

# Verify cron expressions
python -m src.scheduler.cli validate-cron "0 */5 * * * *"
```

**Solutions:**

**Invalid Cron Expressions:**
```bash
# Test cron parsing
python -c "
from src.common.alerts.cron_parser import CronParser
try:
    result = CronParser.parse_cron('0 */5 * * * *')
    print(f'Valid cron: {result}')
except Exception as e:
    print(f'Invalid cron: {e}')
"

# Fix common cron issues
# Wrong: "*/5 * * * *" (5-field for seconds)
# Right: "0 */5 * * * *" (6-field with seconds)
# Right: "*/5 * * * *" (5-field for minutes)
```

**Schedule Not Enabled:**
```sql
-- Check schedule status
SELECT id, name, enabled, next_run_at FROM job_schedules WHERE enabled = false;

-- Enable schedules
UPDATE job_schedules SET enabled = true WHERE id = 123;
```

**Next Run Time Issues:**
```sql
-- Check next_run_at values
SELECT id, name, cron, next_run_at, 
       CASE WHEN next_run_at < NOW() THEN 'OVERDUE' ELSE 'SCHEDULED' END as status
FROM job_schedules WHERE enabled = true;

-- Reset next_run_at for overdue jobs
UPDATE job_schedules SET next_run_at = NULL WHERE next_run_at < NOW() - INTERVAL '1 hour';
```

#### Issue: Jobs failing during execution
**Symptoms:**
- job_schedule_runs records with FAILED status
- Error messages in logs
- Alerts not triggering

**Diagnosis:**
```sql
-- Check recent failures
SELECT run_id, job_type, error, started_at 
FROM job_schedule_runs 
WHERE status = 'FAILED' 
AND started_at > NOW() - INTERVAL '24 hours'
ORDER BY started_at DESC;

-- Check error patterns
SELECT error, COUNT(*) as count
FROM job_schedule_runs 
WHERE status = 'FAILED' 
AND started_at > NOW() - INTERVAL '24 hours'
GROUP BY error
ORDER BY count DESC;
```

**Solutions:**

**Market Data Issues:**
```bash
# Test data availability
python -c "
from src.data.data_manager import DataManager
dm = DataManager()
try:
    data = dm.get_data('AAPL', '1h', limit=10)
    print(f'Data available: {len(data)} rows')
except Exception as e:
    print(f'Data error: {e}')
"

# Check data providers
python -c "
from src.data.providers.factory import DataProviderFactory
factory = DataProviderFactory()
providers = factory.get_available_providers()
print(f'Available providers: {providers}')
"
```

**Indicator Calculation Issues:**
```bash
# Test indicator service
python -c "
from src.indicators.service import IndicatorService
import pandas as pd
import numpy as np

# Create sample data
data = pd.DataFrame({
    'close': np.random.randn(100).cumsum() + 100,
    'high': np.random.randn(100).cumsum() + 102,
    'low': np.random.randn(100).cumsum() + 98,
    'volume': np.random.randint(1000, 10000, 100)
})

service = IndicatorService()
try:
    rsi = service.calculate_rsi(data['close'])
    print(f'RSI calculated: {len(rsi)} values')
except Exception as e:
    print(f'Indicator error: {e}')
"
```

**Alert Configuration Issues:**
```bash
# Validate alert configuration
python -c "
from src.common.alerts.schema_validator import AlertSchemaValidator
validator = AlertSchemaValidator()

config = {
    'ticker': 'AAPL',
    'timeframe': '1h',
    'rule': {'indicator': 'rsi', 'operator': '>', 'value': 70}
}

try:
    result = validator.validate_alert_config(config)
    print(f'Valid config: {result.is_valid}')
except Exception as e:
    print(f'Validation error: {e}')
"
```

### 3. Performance Issues

#### Issue: Slow job execution
**Symptoms:**
- Jobs taking longer than expected
- High CPU/memory usage
- Timeout errors

**Diagnosis:**
```bash
# Check execution times
curl http://localhost:8002/stats | jq '.execution_stats'

# Monitor resource usage
top -p $(pgrep -f "scheduler")
htop -p $(pgrep -f "scheduler")

# Check database performance
psql -h localhost -U trading_user -d trading_db -c "
SELECT query, mean_time, calls 
FROM pg_stat_statements 
WHERE query LIKE '%job_schedule%' 
ORDER BY mean_time DESC LIMIT 10;
"
```

**Solutions:**

**Database Optimization:**
```sql
-- Add indexes for job queries
CREATE INDEX IF NOT EXISTS idx_job_schedules_enabled_next_run 
ON job_schedules(enabled, next_run_at) WHERE enabled = true;

CREATE INDEX IF NOT EXISTS idx_job_schedule_runs_status_started 
ON job_schedule_runs(status, started_at);

-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM job_schedules 
WHERE enabled = true AND next_run_at <= NOW() 
ORDER BY next_run_at LIMIT 10;
```

**Memory Optimization:**
```bash
# Adjust worker pool size
export SCHEDULER_MAX_WORKERS=5  # Reduce if memory constrained

# Enable data caching
export SCHEDULER_DATA_CACHE_TTL=300  # 5 minutes

# Limit concurrent jobs
# Edit config/scheduler/scheduler.json:
{
  "job_defaults": {
    "max_instances": 1,
    "coalesce": true
  }
}
```

**Data Processing Optimization:**
```python
# In alert configurations, limit data fetching:
{
  "ticker": "AAPL",
  "timeframe": "1h",
  "data_limit": 100,  # Limit historical data
  "cache_indicators": true
}
```

#### Issue: High memory usage
**Symptoms:**
- Memory usage continuously growing
- Out of memory errors
- System becoming unresponsive

**Solutions:**
```bash
# Monitor memory usage
watch -n 5 'ps aux | grep scheduler | grep -v grep'

# Check for memory leaks
python -c "
import psutil
import time
process = psutil.Process($(pgrep -f scheduler))
for i in range(10):
    print(f'Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB')
    time.sleep(30)
"

# Restart service periodically (temporary fix)
sudo systemctl restart trading-scheduler

# Reduce data cache size
export SCHEDULER_DATA_CACHE_TTL=60  # Reduce cache time
export SCHEDULER_MAX_WORKERS=3      # Reduce worker count
```

### 4. Notification Issues

#### Issue: Notifications not being sent
**Symptoms:**
- Alerts trigger but no messages received
- Notification client errors
- TelegramBot not receiving messages

**Diagnosis:**
```bash
# Check notification client
python -c "
from src.notification.service.client import NotificationServiceClient
client = NotificationServiceClient()
try:
    result = client.send_message(
        user_id=123,
        message='Test message',
        message_type='alert'
    )
    print(f'Notification sent: {result}')
except Exception as e:
    print(f'Notification error: {e}')
"

# Check TelegramBot status
curl http://localhost:8001/health

# Test notification endpoint
curl -X POST http://localhost:8001/api/notify \
  -H "Content-Type: application/json" \
  -d '{"user_id": 123, "message": "Test notification"}'
```

**Solutions:**

**TelegramBot Connection Issues:**
```bash
# Check TelegramBot service
sudo systemctl status trading-telegram

# Verify TelegramBot configuration
grep -r "TELEGRAM_BOT_TOKEN" config/

# Test Telegram API connectivity
python -c "
import requests
token = 'YOUR_BOT_TOKEN'
response = requests.get(f'https://api.telegram.org/bot{token}/getMe')
print(response.json())
"
```

**Notification Client Configuration:**
```bash
# Check notification service configuration
python -c "
from src.notification.service.client import NotificationServiceClient
client = NotificationServiceClient()
print(f'Client config: {client.config}')
"

# Verify service endpoints
export NOTIFICATION_SERVICE_URL="http://localhost:8001"
export NOTIFICATION_TIMEOUT=30
```

### 5. Database Issues

#### Issue: Database connection failures
**Symptoms:**
- "Connection refused" errors
- "Too many connections" errors
- Job execution failures

**Solutions:**
```bash
# Check database status
sudo systemctl status postgresql

# Check connection limits
psql -h localhost -U postgres -c "
SELECT setting FROM pg_settings WHERE name = 'max_connections';
SELECT count(*) FROM pg_stat_activity;
"

# Optimize connection pooling
# Edit config/scheduler/scheduler.json:
{
  "database": {
    "pool_size": 5,
    "max_overflow": 10,
    "pool_timeout": 30,
    "pool_recycle": 3600
  }
}
```

#### Issue: Database locks and deadlocks
**Symptoms:**
- Jobs hanging during execution
- "Deadlock detected" errors
- Slow database queries

**Solutions:**
```sql
-- Check for locks
SELECT 
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_statement,
    blocking_activity.query AS current_statement_in_blocking_process
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;

-- Kill blocking queries (use with caution)
SELECT pg_terminate_backend(pid) FROM pg_stat_activity 
WHERE state = 'idle in transaction' 
AND state_change < NOW() - INTERVAL '5 minutes';
```

## Monitoring and Alerting

### Health Check Commands

```bash
# Service health
curl http://localhost:8002/health | jq '.'

# Detailed statistics
curl http://localhost:8002/stats | jq '.'

# Database connectivity
python -m src.scheduler.cli test-db

# Configuration validation
python -m src.scheduler.cli validate-config

# Schema validation
python -m src.scheduler.cli validate-schemas
```

### Log Analysis

```bash
# Recent errors
sudo journalctl -u trading-scheduler --since "1 hour ago" | grep ERROR

# Job execution patterns
sudo journalctl -u trading-scheduler --since "24 hours ago" | grep "Job.*executed" | wc -l

# Performance metrics
sudo journalctl -u trading-scheduler --since "1 hour ago" | grep "execution time" | tail -10

# Alert patterns
sudo journalctl -u trading-scheduler --since "24 hours ago" | grep "Alert triggered" | wc -l
```

### Automated Monitoring Scripts

**Health Check Script** (`scripts/check_scheduler_health.sh`):
```bash
#!/bin/bash
set -e

# Check service status
if ! systemctl is-active --quiet trading-scheduler; then
    echo "ERROR: Scheduler service is not running"
    exit 1
fi

# Check health endpoint
if ! curl -f -s http://localhost:8002/health > /dev/null; then
    echo "ERROR: Health endpoint not responding"
    exit 1
fi

# Check recent job executions
recent_jobs=$(curl -s http://localhost:8002/stats | jq -r '.execution_stats.runs_24h')
if [ "$recent_jobs" -eq 0 ]; then
    echo "WARNING: No jobs executed in last 24 hours"
    exit 1
fi

echo "OK: Scheduler service is healthy"
```

**Performance Monitor Script** (`scripts/monitor_scheduler_performance.sh`):
```bash
#!/bin/bash

# Get performance metrics
stats=$(curl -s http://localhost:8002/stats)
success_rate=$(echo "$stats" | jq -r '.execution_stats.success_rate')
avg_duration=$(echo "$stats" | jq -r '.execution_stats.avg_duration')
failed_runs=$(echo "$stats" | jq -r '.error_stats.failed_runs_24h')

# Check thresholds
if (( $(echo "$success_rate < 0.95" | bc -l) )); then
    echo "ALERT: Success rate below 95%: $success_rate"
fi

if (( $(echo "$avg_duration > 10.0" | bc -l) )); then
    echo "ALERT: Average duration above 10s: $avg_duration"
fi

if [ "$failed_runs" -gt 50 ]; then
    echo "ALERT: Too many failed runs in 24h: $failed_runs"
fi
```

## Recovery Procedures

### Service Recovery

```bash
# Graceful restart
sudo systemctl restart trading-scheduler

# Force restart if hung
sudo systemctl kill trading-scheduler
sudo systemctl start trading-scheduler

# Reset service state
sudo systemctl reset-failed trading-scheduler
sudo systemctl start trading-scheduler
```

### Database Recovery

```bash
# Reset stuck jobs
psql -h localhost -U trading_user -d trading_db -c "
UPDATE job_schedule_runs 
SET status = 'FAILED', 
    error = 'Service restart cleanup',
    finished_at = NOW()
WHERE status = 'RUNNING' 
AND started_at < NOW() - INTERVAL '1 hour';
"

# Clean up old runs
psql -h localhost -U trading_user -d trading_db -c "
DELETE FROM job_schedule_runs 
WHERE started_at < NOW() - INTERVAL '30 days';
"

# Rebuild APScheduler job store
python -c "
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from sqlalchemy import create_engine

engine = create_engine('postgresql://user:pass@localhost/trading_db')
jobstore = SQLAlchemyJobStore(engine=engine)
scheduler = AsyncIOScheduler()
scheduler.add_jobstore(jobstore)
scheduler.start()
scheduler.shutdown()
"
```

### Configuration Recovery

```bash
# Restore default configuration
cp config/scheduler/scheduler.json.example config/scheduler/scheduler.json

# Validate restored configuration
python -m src.scheduler.cli validate-config

# Test with minimal configuration
cat > /tmp/minimal_scheduler.json << EOF
{
  "database_url": "postgresql+asyncpg://user:pass@localhost/trading_db",
  "max_workers": 5,
  "alert_timeout": 300
}
EOF

SCHEDULER_CONFIG_FILE=/tmp/minimal_scheduler.json python -m src.scheduler.main
```

## Getting Help

### Log Collection

When reporting issues, collect the following information:

```bash
# Service status
sudo systemctl status trading-scheduler > scheduler_status.txt

# Recent logs
sudo journalctl -u trading-scheduler --since "24 hours ago" > scheduler_logs.txt

# Configuration
cp config/scheduler/scheduler.json scheduler_config.json

# System information
uname -a > system_info.txt
python --version >> system_info.txt
psql --version >> system_info.txt

# Database status
psql -h localhost -U trading_user -d trading_db -c "
SELECT 
  schemaname,
  tablename,
  n_tup_ins as inserts,
  n_tup_upd as updates,
  n_tup_del as deletes
FROM pg_stat_user_tables 
WHERE tablename IN ('job_schedules', 'job_schedule_runs')
ORDER BY tablename;
" > db_stats.txt
```

### Support Contacts

- **Technical Issues**: Check GitHub issues and documentation
- **Configuration Help**: Review configuration examples in `config/scheduler/`
- **Performance Issues**: Use monitoring scripts and performance analysis tools
- **Database Issues**: Consult PostgreSQL documentation and logs

### Useful Resources

- **APScheduler Documentation**: https://apscheduler.readthedocs.io/
- **PostgreSQL Performance**: https://wiki.postgresql.org/wiki/Performance_Optimization
- **Python Async Programming**: https://docs.python.org/3/library/asyncio.html
- **Systemd Service Management**: https://www.freedesktop.org/software/systemd/man/systemctl.html