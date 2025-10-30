# System Scheduler Monitoring and Alerting Guide

## Overview

This guide provides comprehensive monitoring and alerting strategies for the System Scheduler service. Proper monitoring ensures reliable operation, early problem detection, and optimal performance of scheduled jobs and alerts.

## Monitoring Architecture

### 1. Service Health Monitoring

The scheduler service provides built-in health endpoints and metrics for monitoring:

**Health Check Endpoint**: `GET /health`
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "scheduler": "running",
    "database": "connected",
    "job_store": "operational",
    "alert_evaluator": "ready"
  },
  "metrics": {
    "active_jobs": 15,
    "pending_runs": 3,
    "failed_jobs_24h": 1,
    "avg_execution_time": 2.5
  }
}
```

**Statistics Endpoint**: `GET /stats`
```json
{
  "job_counts": {
    "total_schedules": 25,
    "enabled_schedules": 20,
    "active_jobs": 15
  },
  "execution_stats": {
    "runs_24h": 1440,
    "success_rate": 0.995,
    "avg_duration": 2.3,
    "max_duration": 45.2
  },
  "error_stats": {
    "failed_runs_24h": 7,
    "common_errors": [
      "Data unavailable: 3",
      "Timeout: 2",
      "Network error: 2"
    ]
  }
}
```

### 2. System-Level Monitoring

**Systemd Service Monitoring**:
```bash
# Service status
systemctl is-active trading-scheduler
systemctl is-enabled trading-scheduler

# Service metrics
systemctl show trading-scheduler --property=ActiveState,SubState,LoadState,UnitFileState

# Resource usage
systemctl status trading-scheduler
```

**Process Monitoring**:
```bash
# Process information
ps aux | grep scheduler
pgrep -f scheduler

# Resource usage
top -p $(pgrep -f scheduler)
htop -p $(pgrep -f scheduler)

# Memory usage over time
watch -n 30 'ps -p $(pgrep -f scheduler) -o pid,ppid,cmd,%mem,%cpu,etime'
```

### 3. Database Monitoring

**Job Execution Monitoring**:
```sql
-- Recent job execution summary
SELECT 
    job_type,
    status,
    COUNT(*) as count,
    AVG(EXTRACT(EPOCH FROM (finished_at - started_at))) as avg_duration
FROM job_schedule_runs 
WHERE started_at > NOW() - INTERVAL '24 hours'
GROUP BY job_type, status
ORDER BY job_type, status;

-- Failed jobs analysis
SELECT 
    error,
    COUNT(*) as count,
    MAX(started_at) as last_occurrence
FROM job_schedule_runs 
WHERE status = 'FAILED' 
AND started_at > NOW() - INTERVAL '24 hours'
GROUP BY error
ORDER BY count DESC;

-- Long-running jobs
SELECT 
    run_id,
    job_type,
    started_at,
    EXTRACT(EPOCH FROM (NOW() - started_at)) as duration_seconds
FROM job_schedule_runs 
WHERE status = 'RUNNING'
AND started_at < NOW() - INTERVAL '10 minutes'
ORDER BY started_at;
```

**Schedule Health Monitoring**:
```sql
-- Schedules that haven't run recently
SELECT 
    s.id,
    s.name,
    s.job_type,
    s.cron,
    s.next_run_at,
    MAX(r.started_at) as last_run
FROM job_schedules s
LEFT JOIN job_schedule_runs r ON s.id = r.job_id
WHERE s.enabled = true
GROUP BY s.id, s.name, s.job_type, s.cron, s.next_run_at
HAVING MAX(r.started_at) < NOW() - INTERVAL '2 hours' OR MAX(r.started_at) IS NULL
ORDER BY last_run NULLS FIRST;

-- Overdue schedules
SELECT 
    id,
    name,
    job_type,
    cron,
    next_run_at,
    EXTRACT(EPOCH FROM (NOW() - next_run_at)) as overdue_seconds
FROM job_schedules 
WHERE enabled = true 
AND next_run_at < NOW() - INTERVAL '5 minutes'
ORDER BY overdue_seconds DESC;
```

## Monitoring Tools and Scripts

### 1. Health Check Scripts

**Basic Health Check** (`scripts/health_check.sh`):
```bash
#!/bin/bash
set -e

SCHEDULER_URL="http://localhost:8002"
TIMEOUT=10

# Function to check HTTP endpoint
check_endpoint() {
    local url=$1
    local expected_status=${2:-200}
    
    response=$(curl -s -w "%{http_code}" -o /tmp/response.json --max-time $TIMEOUT "$url" || echo "000")
    
    if [ "$response" != "$expected_status" ]; then
        echo "ERROR: $url returned status $response (expected $expected_status)"
        return 1
    fi
    
    return 0
}

# Check service status
if ! systemctl is-active --quiet trading-scheduler; then
    echo "CRITICAL: Scheduler service is not running"
    exit 2
fi

# Check health endpoint
if ! check_endpoint "$SCHEDULER_URL/health"; then
    echo "CRITICAL: Health endpoint not responding"
    exit 2
fi

# Parse health response
health_status=$(jq -r '.status' /tmp/response.json 2>/dev/null || echo "unknown")
if [ "$health_status" != "healthy" ]; then
    echo "WARNING: Service reports unhealthy status: $health_status"
    exit 1
fi

# Check statistics
if ! check_endpoint "$SCHEDULER_URL/stats"; then
    echo "WARNING: Stats endpoint not responding"
    exit 1
fi

# Parse statistics
success_rate=$(jq -r '.execution_stats.success_rate // 0' /tmp/response.json)
failed_jobs=$(jq -r '.error_stats.failed_runs_24h // 0' /tmp/response.json)

# Check success rate
if (( $(echo "$success_rate < 0.95" | bc -l) )); then
    echo "WARNING: Success rate below 95%: $success_rate"
    exit 1
fi

# Check failed jobs
if [ "$failed_jobs" -gt 20 ]; then
    echo "WARNING: High number of failed jobs in 24h: $failed_jobs"
    exit 1
fi

echo "OK: Scheduler service is healthy (success_rate: $success_rate, failed_jobs: $failed_jobs)"
exit 0
```

**Comprehensive Monitoring Script** (`scripts/monitor_scheduler.sh`):
```bash
#!/bin/bash

SCHEDULER_URL="http://localhost:8002"
LOG_FILE="/var/log/scheduler_monitor.log"
ALERT_THRESHOLD_SUCCESS_RATE=0.90
ALERT_THRESHOLD_AVG_DURATION=15.0
ALERT_THRESHOLD_FAILED_JOBS=50

# Logging function
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Get metrics
get_metrics() {
    curl -s --max-time 10 "$SCHEDULER_URL/stats" || echo "{}"
}

# Check service health
check_service_health() {
    if ! systemctl is-active --quiet trading-scheduler; then
        log_message "ALERT: Scheduler service is not running"
        return 1
    fi
    
    if ! curl -s --max-time 10 "$SCHEDULER_URL/health" > /dev/null; then
        log_message "ALERT: Health endpoint not responding"
        return 1
    fi
    
    return 0
}

# Check performance metrics
check_performance() {
    local metrics=$(get_metrics)
    
    if [ -z "$metrics" ] || [ "$metrics" = "{}" ]; then
        log_message "WARNING: Unable to retrieve metrics"
        return 1
    fi
    
    local success_rate=$(echo "$metrics" | jq -r '.execution_stats.success_rate // 0')
    local avg_duration=$(echo "$metrics" | jq -r '.execution_stats.avg_duration // 0')
    local failed_runs=$(echo "$metrics" | jq -r '.error_stats.failed_runs_24h // 0')
    local active_jobs=$(echo "$metrics" | jq -r '.job_counts.active_jobs // 0')
    
    # Check success rate
    if (( $(echo "$success_rate < $ALERT_THRESHOLD_SUCCESS_RATE" | bc -l) )); then
        log_message "ALERT: Success rate below threshold: $success_rate < $ALERT_THRESHOLD_SUCCESS_RATE"
    fi
    
    # Check average duration
    if (( $(echo "$avg_duration > $ALERT_THRESHOLD_AVG_DURATION" | bc -l) )); then
        log_message "ALERT: Average duration above threshold: $avg_duration > $ALERT_THRESHOLD_AVG_DURATION"
    fi
    
    # Check failed jobs
    if [ "$failed_runs" -gt "$ALERT_THRESHOLD_FAILED_JOBS" ]; then
        log_message "ALERT: Too many failed jobs: $failed_runs > $ALERT_THRESHOLD_FAILED_JOBS"
    fi
    
    # Check if no jobs are active (might indicate scheduling issues)
    if [ "$active_jobs" -eq 0 ]; then
        log_message "WARNING: No active jobs scheduled"
    fi
    
    log_message "INFO: Metrics - Success: $success_rate, Duration: ${avg_duration}s, Failed: $failed_runs, Active: $active_jobs"
}

# Check database connectivity
check_database() {
    if ! python3 -c "
from src.scheduler.cli import test_database_connection
import sys
try:
    test_database_connection()
    print('Database connection OK')
except Exception as e:
    print(f'Database connection failed: {e}')
    sys.exit(1)
" 2>/dev/null; then
        log_message "ALERT: Database connection failed"
        return 1
    fi
    
    return 0
}

# Main monitoring function
main() {
    log_message "Starting scheduler monitoring check"
    
    local exit_code=0
    
    # Check service health
    if ! check_service_health; then
        exit_code=2
    fi
    
    # Check performance metrics
    if ! check_performance; then
        exit_code=1
    fi
    
    # Check database connectivity
    if ! check_database; then
        exit_code=2
    fi
    
    if [ $exit_code -eq 0 ]; then
        log_message "INFO: All checks passed"
    fi
    
    return $exit_code
}

# Run monitoring
main "$@"
```

### 2. Performance Monitoring

**Resource Usage Monitor** (`scripts/resource_monitor.sh`):
```bash
#!/bin/bash

INTERVAL=60  # seconds
DURATION=3600  # 1 hour
LOG_FILE="/var/log/scheduler_resources.log"

# Get scheduler process PID
get_scheduler_pid() {
    pgrep -f "scheduler" | head -1
}

# Monitor resources
monitor_resources() {
    local pid=$1
    local start_time=$(date +%s)
    local end_time=$((start_time + DURATION))
    
    echo "timestamp,cpu_percent,memory_mb,memory_percent,threads,open_files" > "$LOG_FILE"
    
    while [ $(date +%s) -lt $end_time ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "Process $pid no longer exists"
            break
        fi
        
        local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        local stats=$(ps -p "$pid" -o %cpu,%mem,nlwp --no-headers 2>/dev/null || echo "0 0 0")
        local cpu_percent=$(echo "$stats" | awk '{print $1}')
        local mem_percent=$(echo "$stats" | awk '{print $2}')
        local threads=$(echo "$stats" | awk '{print $3}')
        
        local memory_kb=$(ps -p "$pid" -o rss --no-headers 2>/dev/null || echo "0")
        local memory_mb=$((memory_kb / 1024))
        
        local open_files=$(lsof -p "$pid" 2>/dev/null | wc -l || echo "0")
        
        echo "$timestamp,$cpu_percent,$memory_mb,$mem_percent,$threads,$open_files" >> "$LOG_FILE"
        
        sleep $INTERVAL
    done
}

# Main function
main() {
    local pid=$(get_scheduler_pid)
    
    if [ -z "$pid" ]; then
        echo "Scheduler process not found"
        exit 1
    fi
    
    echo "Monitoring scheduler process $pid for $((DURATION / 60)) minutes"
    echo "Logging to $LOG_FILE"
    
    monitor_resources "$pid"
}

main "$@"
```

### 3. Database Monitoring Queries

**Performance Analysis Queries** (`scripts/db_monitor.sql`):
```sql
-- Job execution performance over time
WITH hourly_stats AS (
    SELECT 
        DATE_TRUNC('hour', started_at) as hour,
        job_type,
        COUNT(*) as total_jobs,
        COUNT(*) FILTER (WHERE status = 'COMPLETED') as successful_jobs,
        COUNT(*) FILTER (WHERE status = 'FAILED') as failed_jobs,
        AVG(EXTRACT(EPOCH FROM (finished_at - started_at))) as avg_duration,
        MAX(EXTRACT(EPOCH FROM (finished_at - started_at))) as max_duration
    FROM job_schedule_runs 
    WHERE started_at > NOW() - INTERVAL '24 hours'
    GROUP BY DATE_TRUNC('hour', started_at), job_type
)
SELECT 
    hour,
    job_type,
    total_jobs,
    successful_jobs,
    failed_jobs,
    ROUND((successful_jobs::float / total_jobs * 100)::numeric, 2) as success_rate_percent,
    ROUND(avg_duration::numeric, 2) as avg_duration_seconds,
    ROUND(max_duration::numeric, 2) as max_duration_seconds
FROM hourly_stats
ORDER BY hour DESC, job_type;

-- Alert evaluation performance
SELECT 
    job_type,
    COUNT(*) as total_evaluations,
    AVG(EXTRACT(EPOCH FROM (finished_at - started_at))) as avg_evaluation_time,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (finished_at - started_at))) as p95_evaluation_time,
    COUNT(*) FILTER (WHERE EXTRACT(EPOCH FROM (finished_at - started_at)) > 10) as slow_evaluations
FROM job_schedule_runs 
WHERE job_type = 'alert'
AND started_at > NOW() - INTERVAL '24 hours'
AND status IN ('COMPLETED', 'FAILED')
GROUP BY job_type;

-- Database connection and lock analysis
SELECT 
    datname,
    numbackends,
    xact_commit,
    xact_rollback,
    blks_read,
    blks_hit,
    tup_returned,
    tup_fetched,
    tup_inserted,
    tup_updated,
    tup_deleted
FROM pg_stat_database 
WHERE datname = current_database();

-- Table statistics for scheduler tables
SELECT 
    schemaname,
    tablename,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes,
    n_live_tup as live_tuples,
    n_dead_tup as dead_tuples,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables 
WHERE tablename IN ('job_schedules', 'job_schedule_runs', 'apscheduler_jobs')
ORDER BY tablename;
```

## Alerting Configuration

### 1. Nagios/Icinga Configuration

**Service Definition** (`nagios/scheduler_service.cfg`):
```
define service {
    use                     generic-service
    host_name               trading-server
    service_description     Scheduler Service Health
    check_command           check_scheduler_health
    check_interval          5
    retry_interval          1
    max_check_attempts      3
    notification_interval   30
    notification_period     24x7
    contact_groups          trading-admins
}

define service {
    use                     generic-service
    host_name               trading-server
    service_description     Scheduler Performance
    check_command           check_scheduler_performance
    check_interval          10
    retry_interval          2
    max_check_attempts      3
    notification_interval   60
    notification_period     24x7
    contact_groups          trading-admins
}
```

**Command Definitions** (`nagios/scheduler_commands.cfg`):
```
define command {
    command_name    check_scheduler_health
    command_line    /opt/trading-framework/scripts/health_check.sh
}

define command {
    command_name    check_scheduler_performance
    command_line    /opt/trading-framework/scripts/monitor_scheduler.sh
}
```

### 2. Prometheus Configuration

**Metrics Exporter** (`scripts/prometheus_exporter.py`):
```python
#!/usr/bin/env python3
"""
Prometheus metrics exporter for System Scheduler
"""

import time
import requests
import json
from prometheus_client import start_http_server, Gauge, Counter, Histogram
from datetime import datetime

# Metrics definitions
scheduler_health = Gauge('scheduler_health_status', 'Scheduler health status (1=healthy, 0=unhealthy)')
active_jobs = Gauge('scheduler_active_jobs', 'Number of active scheduled jobs')
pending_runs = Gauge('scheduler_pending_runs', 'Number of pending job runs')
success_rate = Gauge('scheduler_success_rate', 'Job success rate (0-1)')
avg_duration = Gauge('scheduler_avg_duration_seconds', 'Average job execution duration')
failed_jobs_24h = Gauge('scheduler_failed_jobs_24h', 'Number of failed jobs in last 24 hours')

job_executions = Counter('scheduler_job_executions_total', 'Total job executions', ['job_type', 'status'])
job_duration = Histogram('scheduler_job_duration_seconds', 'Job execution duration', ['job_type'])

class SchedulerMetricsExporter:
    def __init__(self, scheduler_url='http://localhost:8002', port=9090):
        self.scheduler_url = scheduler_url
        self.port = port
    
    def collect_metrics(self):
        try:
            # Get health status
            health_response = requests.get(f'{self.scheduler_url}/health', timeout=10)
            if health_response.status_code == 200:
                health_data = health_response.json()
                scheduler_health.set(1 if health_data.get('status') == 'healthy' else 0)
                
                metrics = health_data.get('metrics', {})
                active_jobs.set(metrics.get('active_jobs', 0))
                pending_runs.set(metrics.get('pending_runs', 0))
            else:
                scheduler_health.set(0)
            
            # Get detailed statistics
            stats_response = requests.get(f'{self.scheduler_url}/stats', timeout=10)
            if stats_response.status_code == 200:
                stats_data = stats_response.json()
                
                exec_stats = stats_data.get('execution_stats', {})
                success_rate.set(exec_stats.get('success_rate', 0))
                avg_duration.set(exec_stats.get('avg_duration', 0))
                
                error_stats = stats_data.get('error_stats', {})
                failed_jobs_24h.set(error_stats.get('failed_runs_24h', 0))
        
        except Exception as e:
            print(f"Error collecting metrics: {e}")
            scheduler_health.set(0)
    
    def run(self):
        start_http_server(self.port)
        print(f"Prometheus metrics server started on port {self.port}")
        
        while True:
            self.collect_metrics()
            time.sleep(30)  # Collect metrics every 30 seconds

if __name__ == '__main__':
    exporter = SchedulerMetricsExporter()
    exporter.run()
```

**Prometheus Configuration** (`prometheus/scheduler.yml`):
```yaml
global:
  scrape_interval: 30s
  evaluation_interval: 30s

rule_files:
  - "scheduler_alerts.yml"

scrape_configs:
  - job_name: 'scheduler'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 30s
    metrics_path: /metrics

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

**Alert Rules** (`prometheus/scheduler_alerts.yml`):
```yaml
groups:
  - name: scheduler_alerts
    rules:
      - alert: SchedulerDown
        expr: scheduler_health_status == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Scheduler service is down"
          description: "The scheduler service has been down for more than 1 minute"
      
      - alert: SchedulerLowSuccessRate
        expr: scheduler_success_rate < 0.95
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Scheduler success rate is low"
          description: "Scheduler success rate is {{ $value }} (below 95%)"
      
      - alert: SchedulerHighFailureRate
        expr: scheduler_failed_jobs_24h > 50
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High number of failed jobs"
          description: "{{ $value }} jobs failed in the last 24 hours"
      
      - alert: SchedulerSlowExecution
        expr: scheduler_avg_duration_seconds > 15
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Scheduler jobs are running slowly"
          description: "Average job execution time is {{ $value }} seconds"
      
      - alert: SchedulerNoActiveJobs
        expr: scheduler_active_jobs == 0
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "No active scheduled jobs"
          description: "The scheduler has no active jobs for 30 minutes"
```

### 3. Custom Alerting Scripts

**Email Alert Script** (`scripts/send_alert.sh`):
```bash
#!/bin/bash

ALERT_TYPE=$1
ALERT_MESSAGE=$2
EMAIL_RECIPIENTS="admin@trading-company.com,ops@trading-company.com"
SMTP_SERVER="smtp.company.com"

send_email_alert() {
    local subject="[SCHEDULER ALERT] $ALERT_TYPE"
    local body="$ALERT_MESSAGE

Timestamp: $(date)
Server: $(hostname)
Service: trading-scheduler

Please check the scheduler service status and logs.

Health Check: curl http://localhost:8002/health
Service Status: systemctl status trading-scheduler
Recent Logs: journalctl -u trading-scheduler -n 50
"

    echo "$body" | mail -s "$subject" -S smtp="$SMTP_SERVER" "$EMAIL_RECIPIENTS"
}

# Slack notification (if webhook configured)
send_slack_alert() {
    local webhook_url="$SLACK_WEBHOOK_URL"
    if [ -n "$webhook_url" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚨 Scheduler Alert: $ALERT_TYPE\\n$ALERT_MESSAGE\"}" \
            "$webhook_url"
    fi
}

# Send alerts
send_email_alert
send_slack_alert

echo "Alert sent: $ALERT_TYPE - $ALERT_MESSAGE"
```

**Automated Recovery Script** (`scripts/auto_recovery.sh`):
```bash
#!/bin/bash

LOG_FILE="/var/log/scheduler_recovery.log"

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Check if service is running
check_service() {
    systemctl is-active --quiet trading-scheduler
}

# Attempt service restart
restart_service() {
    log_message "Attempting to restart scheduler service"
    
    if systemctl restart trading-scheduler; then
        log_message "Service restart successful"
        sleep 30  # Wait for service to stabilize
        
        if check_service; then
            log_message "Service is running after restart"
            return 0
        else
            log_message "Service failed to start after restart"
            return 1
        fi
    else
        log_message "Service restart failed"
        return 1
    fi
}

# Clean up stuck jobs
cleanup_stuck_jobs() {
    log_message "Cleaning up stuck jobs"
    
    psql -h localhost -U trading_user -d trading_db -c "
    UPDATE job_schedule_runs 
    SET status = 'FAILED', 
        error = 'Auto-recovery cleanup',
        finished_at = NOW()
    WHERE status = 'RUNNING' 
    AND started_at < NOW() - INTERVAL '1 hour';
    " 2>&1 | tee -a "$LOG_FILE"
}

# Main recovery function
main() {
    log_message "Starting automated recovery check"
    
    if ! check_service; then
        log_message "Service is not running, attempting recovery"
        
        # Clean up stuck jobs first
        cleanup_stuck_jobs
        
        # Attempt restart
        if restart_service; then
            log_message "Recovery successful"
            /opt/trading-framework/scripts/send_alert.sh "RECOVERY" "Scheduler service automatically recovered"
        else
            log_message "Recovery failed"
            /opt/trading-framework/scripts/send_alert.sh "CRITICAL" "Scheduler service recovery failed - manual intervention required"
            exit 1
        fi
    else
        log_message "Service is running normally"
    fi
}

main "$@"
```

## Log Analysis and Monitoring

### 1. Log Aggregation

**Rsyslog Configuration** (`/etc/rsyslog.d/50-scheduler.conf`):
```
# Scheduler service logs
:programname, isequal, "trading-scheduler" /var/log/scheduler/scheduler.log
& stop

# Rotate logs
$outchannel scheduler_log,/var/log/scheduler/scheduler.log,52428800,/usr/sbin/logrotate /etc/logrotate.d/scheduler
```

**Logrotate Configuration** (`/etc/logrotate.d/scheduler`):
```
/var/log/scheduler/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 trading trading
    postrotate
        systemctl reload rsyslog
    endscript
}
```

### 2. Log Analysis Scripts

**Error Pattern Analysis** (`scripts/analyze_logs.py`):
```python
#!/usr/bin/env python3
"""
Analyze scheduler logs for patterns and issues
"""

import re
import sys
from collections import defaultdict, Counter
from datetime import datetime, timedelta

def analyze_log_file(log_file):
    error_patterns = defaultdict(int)
    performance_data = []
    job_stats = defaultdict(lambda: {'success': 0, 'failed': 0, 'total': 0})
    
    with open(log_file, 'r') as f:
        for line in f:
            # Parse timestamp
            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if not timestamp_match:
                continue
            
            timestamp = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
            
            # Error pattern analysis
            if 'ERROR' in line:
                # Extract error type
                error_match = re.search(r'ERROR.*?([A-Za-z]+Error|Exception|Failed|Timeout)', line)
                if error_match:
                    error_patterns[error_match.group(1)] += 1
            
            # Performance data
            duration_match = re.search(r'executed.*?in (\d+\.?\d*)s', line)
            if duration_match:
                duration = float(duration_match.group(1))
                performance_data.append((timestamp, duration))
            
            # Job statistics
            job_match = re.search(r'Job (\w+)_\d+ (completed|failed)', line)
            if job_match:
                job_type, status = job_match.groups()
                job_stats[job_type]['total'] += 1
                if status == 'completed':
                    job_stats[job_type]['success'] += 1
                else:
                    job_stats[job_type]['failed'] += 1
    
    return error_patterns, performance_data, job_stats

def generate_report(error_patterns, performance_data, job_stats):
    print("=== Scheduler Log Analysis Report ===")
    print(f"Analysis time: {datetime.now()}")
    print()
    
    # Error patterns
    print("Top Error Patterns:")
    for error, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {error}: {count} occurrences")
    print()
    
    # Performance analysis
    if performance_data:
        durations = [d for _, d in performance_data]
        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)
        
        print("Performance Statistics:")
        print(f"  Average execution time: {avg_duration:.2f}s")
        print(f"  Maximum execution time: {max_duration:.2f}s")
        print(f"  Total executions analyzed: {len(durations)}")
        
        # Slow jobs (> 10 seconds)
        slow_jobs = [d for d in durations if d > 10]
        if slow_jobs:
            print(f"  Slow executions (>10s): {len(slow_jobs)} ({len(slow_jobs)/len(durations)*100:.1f}%)")
        print()
    
    # Job statistics
    print("Job Type Statistics:")
    for job_type, stats in job_stats.items():
        success_rate = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {job_type}:")
        print(f"    Total: {stats['total']}")
        print(f"    Success: {stats['success']} ({success_rate:.1f}%)")
        print(f"    Failed: {stats['failed']}")
    print()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: analyze_logs.py <log_file>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    error_patterns, performance_data, job_stats = analyze_log_file(log_file)
    generate_report(error_patterns, performance_data, job_stats)
```

## Backup and Recovery Procedures

### 1. Configuration Backup

**Backup Script** (`scripts/backup_config.sh`):
```bash
#!/bin/bash

BACKUP_DIR="/opt/backups/scheduler"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="scheduler_config_$TIMESTAMP.tar.gz"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup configuration files
tar -czf "$BACKUP_DIR/$BACKUP_FILE" \
    config/scheduler/ \
    src/scheduler/deployment/ \
    src/common/alerts/schemas/ \
    /etc/systemd/system/trading-scheduler.service

echo "Configuration backup created: $BACKUP_DIR/$BACKUP_FILE"

# Keep only last 30 backups
find "$BACKUP_DIR" -name "scheduler_config_*.tar.gz" -mtime +30 -delete
```

### 2. Database Backup

**Job Data Backup** (`scripts/backup_job_data.sh`):
```bash
#!/bin/bash

BACKUP_DIR="/opt/backups/scheduler"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DB_HOST="localhost"
DB_USER="trading_user"
DB_NAME="trading_db"

# Backup job tables
pg_dump -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" \
    --table=job_schedules \
    --table=job_schedule_runs \
    --table=apscheduler_jobs \
    --data-only \
    --file="$BACKUP_DIR/job_data_$TIMESTAMP.sql"

echo "Job data backup created: $BACKUP_DIR/job_data_$TIMESTAMP.sql"

# Compress backup
gzip "$BACKUP_DIR/job_data_$TIMESTAMP.sql"

# Keep only last 7 days of backups
find "$BACKUP_DIR" -name "job_data_*.sql.gz" -mtime +7 -delete
```

## Best Practices

### 1. Monitoring Frequency

- **Health checks**: Every 1-5 minutes
- **Performance metrics**: Every 30 seconds to 1 minute
- **Database monitoring**: Every 5-10 minutes
- **Log analysis**: Every hour or daily
- **Resource monitoring**: Every 30 seconds

### 2. Alert Thresholds

- **Service down**: Immediate alert
- **Success rate < 95%**: Alert after 5 minutes
- **Average duration > 15s**: Alert after 10 minutes
- **Failed jobs > 50/day**: Alert after 1 hour
- **No active jobs**: Alert after 30 minutes

### 3. Retention Policies

- **Logs**: 30 days
- **Metrics**: 90 days
- **Job run data**: 30 days
- **Configuration backups**: 1 year
- **Database backups**: 30 days

### 4. Escalation Procedures

1. **Level 1**: Automated recovery attempts
2. **Level 2**: Email/Slack notifications to operations team
3. **Level 3**: Page on-call engineer for critical issues
4. **Level 4**: Escalate to development team for persistent issues

This comprehensive monitoring and alerting setup ensures reliable operation of the System Scheduler service with early detection of issues and automated recovery where possible.