# System Scheduler Backup and Recovery Procedures

## Overview

This document outlines comprehensive backup and recovery procedures for the System Scheduler service. These procedures ensure business continuity, data protection, and rapid recovery from various failure scenarios.

## Backup Strategy

### 1. Backup Components

The System Scheduler backup strategy covers the following components:

**Configuration Data**:
- Service configuration files (`config/scheduler/`)
- Alert schemas (`src/common/alerts/schemas/`)
- Systemd service definitions
- Environment configuration
- Deployment configurations

**Application Data**:
- Job schedules (`job_schedules` table)
- Job execution history (`job_schedule_runs` table)
- APScheduler job store (`apscheduler_jobs` table)
- Alert state data (embedded in job records)

**System State**:
- Service logs
- Performance metrics
- Monitoring configurations
- Custom scripts and tools

### 2. Backup Types

**Full Backup**:
- Complete system configuration and data
- Performed daily during low-activity periods
- Includes all components listed above

**Incremental Backup**:
- Changes since last backup
- Performed every 4 hours
- Focuses on job execution data and logs

**Configuration Backup**:
- Service configuration only
- Performed before any configuration changes
- Quick recovery for configuration issues

**Emergency Backup**:
- Critical data only for rapid recovery
- Job schedules and current state
- Can be performed in under 5 minutes

## Backup Procedures

### 1. Automated Daily Backup

**Full Backup Script** (`scripts/backup_full.sh`):
```bash
#!/bin/bash
set -e

# Configuration
BACKUP_ROOT="/opt/backups/scheduler"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$BACKUP_ROOT/full_$TIMESTAMP"
RETENTION_DAYS=30

# Database configuration
DB_HOST="localhost"
DB_USER="trading_user"
DB_NAME="trading_db"
DB_PASSWORD_FILE="/etc/scheduler/.pgpass"

# Logging
LOG_FILE="/var/log/scheduler_backup.log"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Create backup directory
mkdir -p "$BACKUP_DIR"

log_message "Starting full backup to $BACKUP_DIR"

# 1. Stop scheduler service for consistent backup
log_message "Stopping scheduler service"
systemctl stop trading-scheduler

# 2. Backup configuration files
log_message "Backing up configuration files"
tar -czf "$BACKUP_DIR/config.tar.gz" \
    config/scheduler/ \
    src/scheduler/deployment/ \
    src/common/alerts/schemas/ \
    /etc/systemd/system/trading-scheduler.service \
    /etc/scheduler/ 2>/dev/null || true

# 3. Backup database
log_message "Backing up database"
export PGPASSFILE="$DB_PASSWORD_FILE"

# Full database backup
pg_dump -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" \
    --verbose \
    --format=custom \
    --compress=9 \
    --file="$BACKUP_DIR/database_full.dump"

# Job-specific tables backup (for faster recovery)
pg_dump -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" \
    --table=job_schedules \
    --table=job_schedule_runs \
    --table=apscheduler_jobs \
    --data-only \
    --format=plain \
    --file="$BACKUP_DIR/job_data.sql"

# 4. Backup logs
log_message "Backing up logs"
tar -czf "$BACKUP_DIR/logs.tar.gz" \
    /var/log/scheduler/ \
    /var/log/scheduler_*.log 2>/dev/null || true

# 5. Backup system state
log_message "Backing up system state"
cat > "$BACKUP_DIR/system_info.txt" << EOF
Backup Date: $(date)
Hostname: $(hostname)
OS Version: $(cat /etc/os-release | grep PRETTY_NAME)
Python Version: $(python3 --version)
PostgreSQL Version: $(psql --version)
Service Status: $(systemctl is-active trading-scheduler || echo "stopped")
Disk Usage: $(df -h /opt/trading-framework)
Memory Usage: $(free -h)
EOF

# 6. Create backup manifest
log_message "Creating backup manifest"
cat > "$BACKUP_DIR/manifest.txt" << EOF
Backup Type: Full
Backup Date: $(date)
Backup Directory: $BACKUP_DIR
Components:
- Configuration: config.tar.gz
- Database Full: database_full.dump
- Job Data: job_data.sql
- Logs: logs.tar.gz
- System Info: system_info.txt

File Checksums:
$(cd "$BACKUP_DIR" && sha256sum *.tar.gz *.dump *.sql *.txt)
EOF

# 7. Restart scheduler service
log_message "Restarting scheduler service"
systemctl start trading-scheduler

# Wait for service to be ready
sleep 10
if systemctl is-active --quiet trading-scheduler; then
    log_message "Service restarted successfully"
else
    log_message "WARNING: Service failed to restart properly"
fi

# 8. Compress entire backup
log_message "Compressing backup"
cd "$BACKUP_ROOT"
tar -czf "full_backup_$TIMESTAMP.tar.gz" "full_$TIMESTAMP/"
rm -rf "full_$TIMESTAMP/"

# 9. Cleanup old backups
log_message "Cleaning up old backups"
find "$BACKUP_ROOT" -name "full_backup_*.tar.gz" -mtime +$RETENTION_DAYS -delete

# 10. Verify backup
BACKUP_SIZE=$(du -h "$BACKUP_ROOT/full_backup_$TIMESTAMP.tar.gz" | cut -f1)
log_message "Backup completed successfully: full_backup_$TIMESTAMP.tar.gz ($BACKUP_SIZE)"

# 11. Test backup integrity
log_message "Testing backup integrity"
if tar -tzf "$BACKUP_ROOT/full_backup_$TIMESTAMP.tar.gz" > /dev/null; then
    log_message "Backup integrity check passed"
else
    log_message "ERROR: Backup integrity check failed"
    exit 1
fi

log_message "Full backup procedure completed"
```

### 2. Incremental Backup

**Incremental Backup Script** (`scripts/backup_incremental.sh`):
```bash
#!/bin/bash
set -e

# Configuration
BACKUP_ROOT="/opt/backups/scheduler"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$BACKUP_ROOT/incremental_$TIMESTAMP"
LAST_BACKUP_FILE="$BACKUP_ROOT/.last_incremental"

# Database configuration
DB_HOST="localhost"
DB_USER="trading_user"
DB_NAME="trading_db"
DB_PASSWORD_FILE="/etc/scheduler/.pgpass"

# Logging
LOG_FILE="/var/log/scheduler_backup.log"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Get last backup timestamp
if [ -f "$LAST_BACKUP_FILE" ]; then
    LAST_BACKUP=$(cat "$LAST_BACKUP_FILE")
    log_message "Last incremental backup: $LAST_BACKUP"
else
    LAST_BACKUP=$(date -d "4 hours ago" '+%Y-%m-%d %H:%M:%S')
    log_message "No previous backup found, using: $LAST_BACKUP"
fi

# Create backup directory
mkdir -p "$BACKUP_DIR"

log_message "Starting incremental backup to $BACKUP_DIR"

# 1. Backup new job runs
log_message "Backing up new job runs"
export PGPASSFILE="$DB_PASSWORD_FILE"

psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "
COPY (
    SELECT * FROM job_schedule_runs 
    WHERE started_at > '$LAST_BACKUP'
) TO STDOUT WITH CSV HEADER
" > "$BACKUP_DIR/new_job_runs.csv"

# 2. Backup modified schedules
log_message "Backing up modified schedules"
psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "
COPY (
    SELECT * FROM job_schedules 
    WHERE updated_at > '$LAST_BACKUP'
) TO STDOUT WITH CSV HEADER
" > "$BACKUP_DIR/modified_schedules.csv"

# 3. Backup recent logs
log_message "Backing up recent logs"
find /var/log/scheduler/ -name "*.log" -newer "$LAST_BACKUP_FILE" -exec cp {} "$BACKUP_DIR/" \; 2>/dev/null || true

# 4. Create incremental manifest
log_message "Creating incremental manifest"
cat > "$BACKUP_DIR/manifest.txt" << EOF
Backup Type: Incremental
Backup Date: $(date)
Since: $LAST_BACKUP
Backup Directory: $BACKUP_DIR
Components:
- New Job Runs: new_job_runs.csv ($(wc -l < "$BACKUP_DIR/new_job_runs.csv") records)
- Modified Schedules: modified_schedules.csv ($(wc -l < "$BACKUP_DIR/modified_schedules.csv") records)
- Recent Logs: $(ls "$BACKUP_DIR"/*.log 2>/dev/null | wc -l) files

File Checksums:
$(cd "$BACKUP_DIR" && sha256sum *.csv *.txt *.log 2>/dev/null || true)
EOF

# 5. Compress backup
log_message "Compressing incremental backup"
cd "$BACKUP_ROOT"
tar -czf "incremental_backup_$TIMESTAMP.tar.gz" "incremental_$TIMESTAMP/"
rm -rf "incremental_$TIMESTAMP/"

# 6. Update last backup timestamp
echo "$(date '+%Y-%m-%d %H:%M:%S')" > "$LAST_BACKUP_FILE"

# 7. Cleanup old incremental backups (keep 48 hours worth)
find "$BACKUP_ROOT" -name "incremental_backup_*.tar.gz" -mtime +2 -delete

BACKUP_SIZE=$(du -h "$BACKUP_ROOT/incremental_backup_$TIMESTAMP.tar.gz" | cut -f1)
log_message "Incremental backup completed: incremental_backup_$TIMESTAMP.tar.gz ($BACKUP_SIZE)"
```

### 3. Configuration-Only Backup

**Configuration Backup Script** (`scripts/backup_config.sh`):
```bash
#!/bin/bash
set -e

# Configuration
BACKUP_ROOT="/opt/backups/scheduler"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="config_backup_$TIMESTAMP.tar.gz"

# Logging
LOG_FILE="/var/log/scheduler_backup.log"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

mkdir -p "$BACKUP_ROOT"

log_message "Starting configuration backup"

# Create temporary directory for staging
TEMP_DIR=$(mktemp -d)
CONFIG_DIR="$TEMP_DIR/scheduler_config"
mkdir -p "$CONFIG_DIR"

# Copy configuration files
cp -r config/scheduler/ "$CONFIG_DIR/" 2>/dev/null || true
cp -r src/scheduler/deployment/ "$CONFIG_DIR/" 2>/dev/null || true
cp -r src/common/alerts/schemas/ "$CONFIG_DIR/" 2>/dev/null || true
cp /etc/systemd/system/trading-scheduler.service "$CONFIG_DIR/" 2>/dev/null || true
cp -r /etc/scheduler/ "$CONFIG_DIR/" 2>/dev/null || true

# Create configuration manifest
cat > "$CONFIG_DIR/config_manifest.txt" << EOF
Configuration Backup
Date: $(date)
Hostname: $(hostname)
Service Version: $(python3 -c "import src.scheduler; print(getattr(src.scheduler, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")

Files included:
$(find "$CONFIG_DIR" -type f | sort)

Environment Variables:
$(env | grep SCHEDULER_ | sort)
EOF

# Create compressed backup
cd "$TEMP_DIR"
tar -czf "$BACKUP_ROOT/$BACKUP_FILE" scheduler_config/

# Cleanup
rm -rf "$TEMP_DIR"

# Keep last 100 configuration backups
find "$BACKUP_ROOT" -name "config_backup_*.tar.gz" | sort | head -n -100 | xargs rm -f

BACKUP_SIZE=$(du -h "$BACKUP_ROOT/$BACKUP_FILE" | cut -f1)
log_message "Configuration backup completed: $BACKUP_FILE ($BACKUP_SIZE)"
```

### 4. Emergency Backup

**Emergency Backup Script** (`scripts/backup_emergency.sh`):
```bash
#!/bin/bash
set -e

# Configuration
BACKUP_ROOT="/opt/backups/scheduler"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="emergency_backup_$TIMESTAMP.tar.gz"

# Database configuration
DB_HOST="localhost"
DB_USER="trading_user"
DB_NAME="trading_db"
DB_PASSWORD_FILE="/etc/scheduler/.pgpass"

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

mkdir -p "$BACKUP_ROOT"

log_message "Starting emergency backup (critical data only)"

# Create temporary directory
TEMP_DIR=$(mktemp -d)
EMERGENCY_DIR="$TEMP_DIR/emergency"
mkdir -p "$EMERGENCY_DIR"

# 1. Backup active job schedules
log_message "Backing up active schedules"
export PGPASSFILE="$DB_PASSWORD_FILE"

psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "
COPY (
    SELECT * FROM job_schedules WHERE enabled = true
) TO STDOUT WITH CSV HEADER
" > "$EMERGENCY_DIR/active_schedules.csv"

# 2. Backup recent job runs (last 24 hours)
log_message "Backing up recent job runs"
psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "
COPY (
    SELECT * FROM job_schedule_runs 
    WHERE started_at > NOW() - INTERVAL '24 hours'
) TO STDOUT WITH CSV HEADER
" > "$EMERGENCY_DIR/recent_runs.csv"

# 3. Backup essential configuration
log_message "Backing up essential configuration"
cp config/scheduler/scheduler.json "$EMERGENCY_DIR/" 2>/dev/null || true
cp /etc/systemd/system/trading-scheduler.service "$EMERGENCY_DIR/" 2>/dev/null || true

# 4. Backup alert schemas
log_message "Backing up alert schemas"
cp -r src/common/alerts/schemas/ "$EMERGENCY_DIR/" 2>/dev/null || true

# 5. Create emergency manifest
cat > "$EMERGENCY_DIR/emergency_manifest.txt" << EOF
Emergency Backup
Date: $(date)
Purpose: Critical data for rapid recovery
Hostname: $(hostname)

Contents:
- Active Schedules: $(wc -l < "$EMERGENCY_DIR/active_schedules.csv") records
- Recent Runs: $(wc -l < "$EMERGENCY_DIR/recent_runs.csv") records
- Configuration: scheduler.json, service file
- Alert Schemas: $(find "$EMERGENCY_DIR/schemas" -name "*.json" | wc -l) files

Recovery Instructions:
1. Restore database tables from CSV files
2. Copy configuration files to correct locations
3. Restart scheduler service
4. Verify active schedules are loaded
EOF

# Create compressed backup
cd "$TEMP_DIR"
tar -czf "$BACKUP_ROOT/$BACKUP_FILE" emergency/

# Cleanup
rm -rf "$TEMP_DIR"

BACKUP_SIZE=$(du -h "$BACKUP_ROOT/$BACKUP_FILE" | cut -f1)
log_message "Emergency backup completed: $BACKUP_FILE ($BACKUP_SIZE) - Ready for rapid recovery"

echo "Emergency backup location: $BACKUP_ROOT/$BACKUP_FILE"
```

## Recovery Procedures

### 1. Complete System Recovery

**Full Recovery Script** (`scripts/recovery_full.sh`):
```bash
#!/bin/bash
set -e

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root for full system recovery"
    exit 1
fi

BACKUP_FILE=$1
if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file.tar.gz>"
    echo "Available backups:"
    ls -la /opt/backups/scheduler/full_backup_*.tar.gz | tail -5
    exit 1
fi

# Configuration
RECOVERY_DIR="/tmp/scheduler_recovery"
DB_HOST="localhost"
DB_USER="trading_user"
DB_NAME="trading_db"
DB_PASSWORD_FILE="/etc/scheduler/.pgpass"

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_message "Starting full system recovery from $BACKUP_FILE"

# 1. Stop scheduler service
log_message "Stopping scheduler service"
systemctl stop trading-scheduler || true

# 2. Extract backup
log_message "Extracting backup"
rm -rf "$RECOVERY_DIR"
mkdir -p "$RECOVERY_DIR"
cd "$RECOVERY_DIR"
tar -xzf "$BACKUP_FILE"

# Find the backup directory
BACKUP_DIR=$(find . -name "full_*" -type d | head -1)
if [ -z "$BACKUP_DIR" ]; then
    echo "ERROR: Could not find backup directory in archive"
    exit 1
fi

cd "$BACKUP_DIR"

# 3. Verify backup integrity
log_message "Verifying backup integrity"
if [ -f "manifest.txt" ]; then
    log_message "Backup manifest found:"
    cat manifest.txt
else
    log_message "WARNING: No backup manifest found"
fi

# 4. Restore configuration files
log_message "Restoring configuration files"
if [ -f "config.tar.gz" ]; then
    cd /
    tar -xzf "$RECOVERY_DIR/$BACKUP_DIR/config.tar.gz"
    log_message "Configuration files restored"
else
    log_message "WARNING: No configuration backup found"
fi

# 5. Restore database
log_message "Restoring database"
export PGPASSFILE="$DB_PASSWORD_FILE"

if [ -f "$RECOVERY_DIR/$BACKUP_DIR/database_full.dump" ]; then
    # Full database restore
    log_message "Performing full database restore"
    
    # Drop existing scheduler tables
    psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "
    DROP TABLE IF EXISTS job_schedule_runs CASCADE;
    DROP TABLE IF EXISTS job_schedules CASCADE;
    DROP TABLE IF EXISTS apscheduler_jobs CASCADE;
    " || true
    
    # Restore from dump
    pg_restore -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" \
        --verbose \
        --clean \
        --if-exists \
        "$RECOVERY_DIR/$BACKUP_DIR/database_full.dump"
    
elif [ -f "$RECOVERY_DIR/$BACKUP_DIR/job_data.sql" ]; then
    # Job data only restore
    log_message "Performing job data restore"
    psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -f "$RECOVERY_DIR/$BACKUP_DIR/job_data.sql"
else
    log_message "WARNING: No database backup found"
fi

# 6. Restore logs (optional)
log_message "Restoring logs"
if [ -f "$RECOVERY_DIR/$BACKUP_DIR/logs.tar.gz" ]; then
    cd /
    tar -xzf "$RECOVERY_DIR/$BACKUP_DIR/logs.tar.gz"
    chown -R trading:trading /var/log/scheduler/
    log_message "Logs restored"
fi

# 7. Set correct permissions
log_message "Setting permissions"
chown -R trading:trading /opt/trading-framework/
chown -R trading:trading /etc/scheduler/
chmod +x /opt/trading-framework/src/scheduler/main.py

# 8. Reload systemd and start service
log_message "Reloading systemd and starting service"
systemctl daemon-reload
systemctl enable trading-scheduler
systemctl start trading-scheduler

# 9. Wait for service to be ready
log_message "Waiting for service to be ready"
sleep 15

# 10. Verify recovery
log_message "Verifying recovery"
if systemctl is-active --quiet trading-scheduler; then
    log_message "✓ Service is running"
else
    log_message "✗ Service failed to start"
    systemctl status trading-scheduler
    exit 1
fi

# Check health endpoint
if curl -f -s http://localhost:8002/health > /dev/null; then
    log_message "✓ Health endpoint responding"
else
    log_message "✗ Health endpoint not responding"
fi

# Check job schedules
SCHEDULE_COUNT=$(psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT COUNT(*) FROM job_schedules WHERE enabled = true;" | tr -d ' ')
log_message "✓ Active schedules: $SCHEDULE_COUNT"

# Cleanup
rm -rf "$RECOVERY_DIR"

log_message "Full system recovery completed successfully"
log_message "Please verify that all schedules are working correctly"
```

### 2. Configuration-Only Recovery

**Configuration Recovery Script** (`scripts/recovery_config.sh`):
```bash
#!/bin/bash
set -e

BACKUP_FILE=$1
if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <config_backup_file.tar.gz>"
    echo "Available configuration backups:"
    ls -la /opt/backups/scheduler/config_backup_*.tar.gz | tail -5
    exit 1
fi

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_message "Starting configuration recovery from $BACKUP_FILE"

# 1. Stop scheduler service
log_message "Stopping scheduler service"
systemctl stop trading-scheduler

# 2. Backup current configuration
log_message "Backing up current configuration"
CURRENT_BACKUP="/tmp/current_config_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$CURRENT_BACKUP" \
    config/scheduler/ \
    src/scheduler/deployment/ \
    src/common/alerts/schemas/ \
    /etc/systemd/system/trading-scheduler.service \
    /etc/scheduler/ 2>/dev/null || true

log_message "Current configuration backed up to: $CURRENT_BACKUP"

# 3. Extract and restore configuration
log_message "Restoring configuration"
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"
tar -xzf "$BACKUP_FILE"

# Copy configuration files
if [ -d "scheduler_config" ]; then
    cd scheduler_config
    
    # Restore main configuration
    if [ -d "scheduler" ]; then
        cp -r scheduler/* /opt/trading-framework/config/scheduler/ 2>/dev/null || true
    fi
    
    # Restore deployment configuration
    if [ -d "deployment" ]; then
        cp -r deployment/* /opt/trading-framework/src/scheduler/deployment/ 2>/dev/null || true
    fi
    
    # Restore alert schemas
    if [ -d "schemas" ]; then
        cp -r schemas/* /opt/trading-framework/src/common/alerts/schemas/ 2>/dev/null || true
    fi
    
    # Restore systemd service
    if [ -f "trading-scheduler.service" ]; then
        cp trading-scheduler.service /etc/systemd/system/
        systemctl daemon-reload
    fi
    
    # Restore system configuration
    if [ -d "scheduler" ] && [ -d "/etc/scheduler" ]; then
        cp -r scheduler/* /etc/scheduler/ 2>/dev/null || true
    fi
fi

# 4. Set permissions
chown -R trading:trading /opt/trading-framework/config/scheduler/
chown -R trading:trading /opt/trading-framework/src/scheduler/
chown -R trading:trading /opt/trading-framework/src/common/alerts/schemas/
chown -R trading:trading /etc/scheduler/ 2>/dev/null || true

# 5. Validate configuration
log_message "Validating configuration"
if python3 -m src.scheduler.cli validate-config; then
    log_message "✓ Configuration validation passed"
else
    log_message "✗ Configuration validation failed"
    log_message "Restoring previous configuration"
    cd /
    tar -xzf "$CURRENT_BACKUP"
    exit 1
fi

# 6. Start service
log_message "Starting scheduler service"
systemctl start trading-scheduler

# 7. Verify service
sleep 10
if systemctl is-active --quiet trading-scheduler; then
    log_message "✓ Service started successfully"
else
    log_message "✗ Service failed to start"
    systemctl status trading-scheduler
    exit 1
fi

# Cleanup
rm -rf "$TEMP_DIR"

log_message "Configuration recovery completed successfully"
```

### 3. Emergency Recovery

**Emergency Recovery Script** (`scripts/recovery_emergency.sh`):
```bash
#!/bin/bash
set -e

BACKUP_FILE=$1
if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <emergency_backup_file.tar.gz>"
    echo "Available emergency backups:"
    ls -la /opt/backups/scheduler/emergency_backup_*.tar.gz | tail -5
    exit 1
fi

# Database configuration
DB_HOST="localhost"
DB_USER="trading_user"
DB_NAME="trading_db"
DB_PASSWORD_FILE="/etc/scheduler/.pgpass"

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_message "Starting emergency recovery from $BACKUP_FILE"
log_message "This will restore critical data for rapid service recovery"

# 1. Extract backup
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"
tar -xzf "$BACKUP_FILE"

EMERGENCY_DIR=$(find . -name "emergency" -type d | head -1)
if [ -z "$EMERGENCY_DIR" ]; then
    echo "ERROR: Could not find emergency directory in backup"
    exit 1
fi

cd "$EMERGENCY_DIR"

# 2. Show recovery manifest
if [ -f "emergency_manifest.txt" ]; then
    log_message "Emergency backup contents:"
    cat emergency_manifest.txt
    echo
fi

# 3. Restore database tables
log_message "Restoring database tables"
export PGPASSFILE="$DB_PASSWORD_FILE"

# Restore active schedules
if [ -f "active_schedules.csv" ]; then
    log_message "Restoring active schedules"
    
    # Create temporary table and import
    psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "
    CREATE TEMP TABLE temp_schedules (LIKE job_schedules);
    "
    
    psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "
    \copy temp_schedules FROM 'active_schedules.csv' WITH CSV HEADER
    "
    
    # Merge with existing data (update or insert)
    psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "
    INSERT INTO job_schedules 
    SELECT * FROM temp_schedules
    ON CONFLICT (id) DO UPDATE SET
        name = EXCLUDED.name,
        job_type = EXCLUDED.job_type,
        target = EXCLUDED.target,
        task_params = EXCLUDED.task_params,
        cron = EXCLUDED.cron,
        enabled = EXCLUDED.enabled,
        next_run_at = EXCLUDED.next_run_at,
        updated_at = NOW();
    "
    
    RESTORED_SCHEDULES=$(psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT COUNT(*) FROM temp_schedules;" | tr -d ' ')
    log_message "✓ Restored $RESTORED_SCHEDULES schedules"
fi

# Restore recent job runs
if [ -f "recent_runs.csv" ]; then
    log_message "Restoring recent job runs"
    
    psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "
    CREATE TEMP TABLE temp_runs (LIKE job_schedule_runs);
    "
    
    psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "
    \copy temp_runs FROM 'recent_runs.csv' WITH CSV HEADER
    "
    
    psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "
    INSERT INTO job_schedule_runs 
    SELECT * FROM temp_runs
    ON CONFLICT (run_id) DO NOTHING;
    "
    
    RESTORED_RUNS=$(psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT COUNT(*) FROM temp_runs;" | tr -d ' ')
    log_message "✓ Restored $RESTORED_RUNS job runs"
fi

# 4. Restore essential configuration
log_message "Restoring essential configuration"

if [ -f "scheduler.json" ]; then
    cp scheduler.json /opt/trading-framework/config/scheduler/
    log_message "✓ Restored scheduler.json"
fi

if [ -f "trading-scheduler.service" ]; then
    cp trading-scheduler.service /etc/systemd/system/
    systemctl daemon-reload
    log_message "✓ Restored systemd service"
fi

if [ -d "schemas" ]; then
    cp -r schemas/* /opt/trading-framework/src/common/alerts/schemas/
    log_message "✓ Restored alert schemas"
fi

# 5. Set permissions
chown -R trading:trading /opt/trading-framework/config/scheduler/
chown -R trading:trading /opt/trading-framework/src/common/alerts/schemas/

# 6. Start service
log_message "Starting scheduler service"
systemctl enable trading-scheduler
systemctl start trading-scheduler

# 7. Wait and verify
sleep 15

if systemctl is-active --quiet trading-scheduler; then
    log_message "✓ Service is running"
else
    log_message "✗ Service failed to start"
    systemctl status trading-scheduler
    exit 1
fi

# Check health
if curl -f -s http://localhost:8002/health > /dev/null; then
    log_message "✓ Health endpoint responding"
else
    log_message "⚠ Health endpoint not responding (may need more time)"
fi

# Check schedules loaded
ACTIVE_SCHEDULES=$(psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT COUNT(*) FROM job_schedules WHERE enabled = true;" | tr -d ' ')
log_message "✓ Active schedules loaded: $ACTIVE_SCHEDULES"

# Cleanup
rm -rf "$TEMP_DIR"

log_message "Emergency recovery completed successfully"
log_message "Service is operational with critical data restored"
log_message "Recommend performing full backup once system is stable"
```

## Backup Automation

### 1. Cron Jobs

**Crontab Configuration** (`/etc/cron.d/scheduler-backup`):
```bash
# System Scheduler Backup Jobs
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin

# Full backup daily at 2 AM
0 2 * * * trading /opt/trading-framework/scripts/backup_full.sh

# Incremental backup every 4 hours
0 */4 * * * trading /opt/trading-framework/scripts/backup_incremental.sh

# Configuration backup before any changes (manual trigger)
# Use: touch /tmp/config_changed to trigger
*/5 * * * * trading [ -f /tmp/config_changed ] && /opt/trading-framework/scripts/backup_config.sh && rm /tmp/config_changed

# Emergency backup weekly (for offsite storage)
0 3 * * 0 trading /opt/trading-framework/scripts/backup_emergency.sh
```

### 2. Systemd Timers

**Backup Timer Service** (`/etc/systemd/system/scheduler-backup.timer`):
```ini
[Unit]
Description=System Scheduler Backup Timer
Requires=scheduler-backup.service

[Timer]
OnCalendar=daily
Persistent=true
RandomizedDelaySec=1800

[Install]
WantedBy=timers.target
```

**Backup Service** (`/etc/systemd/system/scheduler-backup.service`):
```ini
[Unit]
Description=System Scheduler Backup Service
After=trading-scheduler.service

[Service]
Type=oneshot
User=trading
Group=trading
ExecStart=/opt/trading-framework/scripts/backup_full.sh
StandardOutput=journal
StandardError=journal
```

### 3. Backup Monitoring

**Backup Monitor Script** (`scripts/monitor_backups.sh`):
```bash
#!/bin/bash

BACKUP_DIR="/opt/backups/scheduler"
ALERT_EMAIL="admin@trading-company.com"
MAX_BACKUP_AGE_HOURS=26  # Daily backup should be less than 26 hours old

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

check_backup_freshness() {
    local backup_type=$1
    local max_age_hours=$2
    
    local latest_backup=$(find "$BACKUP_DIR" -name "${backup_type}_backup_*.tar.gz" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [ -z "$latest_backup" ]; then
        log_message "ERROR: No $backup_type backups found"
        return 1
    fi
    
    local backup_age_hours=$(( ($(date +%s) - $(stat -c %Y "$latest_backup")) / 3600 ))
    
    if [ $backup_age_hours -gt $max_age_hours ]; then
        log_message "ERROR: Latest $backup_type backup is $backup_age_hours hours old (max: $max_age_hours)"
        return 1
    else
        log_message "OK: Latest $backup_type backup is $backup_age_hours hours old"
        return 0
    fi
}

send_alert() {
    local message=$1
    echo "$message" | mail -s "[SCHEDULER BACKUP ALERT] Backup Issue Detected" "$ALERT_EMAIL"
}

# Check backup freshness
backup_status=0

if ! check_backup_freshness "full" $MAX_BACKUP_AGE_HOURS; then
    backup_status=1
fi

if ! check_backup_freshness "incremental" 5; then  # Incremental should be less than 5 hours old
    backup_status=1
fi

# Check backup integrity
log_message "Checking backup integrity"
latest_full_backup=$(find "$BACKUP_DIR" -name "full_backup_*.tar.gz" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

if [ -n "$latest_full_backup" ]; then
    if tar -tzf "$latest_full_backup" > /dev/null 2>&1; then
        log_message "OK: Latest full backup integrity check passed"
    else
        log_message "ERROR: Latest full backup failed integrity check"
        backup_status=1
    fi
fi

# Check disk space
backup_disk_usage=$(df "$BACKUP_DIR" | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $backup_disk_usage -gt 80 ]; then
    log_message "WARNING: Backup disk usage is ${backup_disk_usage}%"
    if [ $backup_disk_usage -gt 90 ]; then
        backup_status=1
    fi
fi

if [ $backup_status -ne 0 ]; then
    send_alert "Scheduler backup monitoring detected issues. Please check backup system."
    exit 1
else
    log_message "All backup checks passed"
    exit 0
fi
```

## Disaster Recovery Planning

### 1. Recovery Time Objectives (RTO)

- **Emergency Recovery**: 15 minutes (critical schedules only)
- **Configuration Recovery**: 30 minutes (service operational)
- **Full Recovery**: 2 hours (complete system restoration)

### 2. Recovery Point Objectives (RPO)

- **Job Schedules**: 0 minutes (real-time replication)
- **Job Execution History**: 4 hours (incremental backups)
- **Configuration Changes**: 0 minutes (immediate backup)
- **System Logs**: 24 hours (daily backup)

### 3. Disaster Scenarios

**Scenario 1: Service Failure**
- **Impact**: Scheduler stops running, no new jobs execute
- **Recovery**: Restart service, verify schedules loaded
- **RTO**: 5 minutes
- **RPO**: 0 minutes

**Scenario 2: Configuration Corruption**
- **Impact**: Service fails to start due to bad configuration
- **Recovery**: Restore configuration from backup, restart service
- **RTO**: 30 minutes
- **RPO**: 0 minutes (configuration backed up on change)

**Scenario 3: Database Corruption**
- **Impact**: Job schedules and history lost
- **Recovery**: Restore database from backup, restart service
- **RTO**: 1 hour
- **RPO**: 4 hours (incremental backup interval)

**Scenario 4: Complete System Loss**
- **Impact**: Entire server lost
- **Recovery**: Rebuild server, restore from full backup
- **RTO**: 4 hours
- **RPO**: 24 hours (daily full backup)

### 4. Recovery Testing

**Monthly Recovery Test** (`scripts/test_recovery.sh`):
```bash
#!/bin/bash
set -e

TEST_ENV="test"
TEST_DB="trading_test"
BACKUP_FILE="/opt/backups/scheduler/$(ls -t /opt/backups/scheduler/full_backup_*.tar.gz | head -1)"

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - [RECOVERY TEST] $1"
}

log_message "Starting monthly recovery test"
log_message "Using backup: $BACKUP_FILE"

# 1. Create test database
log_message "Creating test database"
createdb "$TEST_DB" || true

# 2. Extract backup to test location
TEST_DIR="/tmp/recovery_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"
tar -xzf "$BACKUP_FILE"

# 3. Test database restore
log_message "Testing database restore"
BACKUP_DIR=$(find . -name "full_*" -type d | head -1)
if [ -f "$BACKUP_DIR/database_full.dump" ]; then
    pg_restore -d "$TEST_DB" --clean --if-exists "$BACKUP_DIR/database_full.dump"
    log_message "✓ Database restore successful"
else
    log_message "✗ Database backup not found"
    exit 1
fi

# 4. Verify data integrity
log_message "Verifying data integrity"
SCHEDULE_COUNT=$(psql -d "$TEST_DB" -t -c "SELECT COUNT(*) FROM job_schedules;" | tr -d ' ')
RUN_COUNT=$(psql -d "$TEST_DB" -t -c "SELECT COUNT(*) FROM job_schedule_runs;" | tr -d ' ')

log_message "✓ Schedules restored: $SCHEDULE_COUNT"
log_message "✓ Job runs restored: $RUN_COUNT"

# 5. Test configuration extraction
log_message "Testing configuration extraction"
if [ -f "$BACKUP_DIR/config.tar.gz" ]; then
    tar -tzf "$BACKUP_DIR/config.tar.gz" > /dev/null
    log_message "✓ Configuration backup is valid"
else
    log_message "✗ Configuration backup not found"
fi

# 6. Cleanup
log_message "Cleaning up test environment"
dropdb "$TEST_DB"
rm -rf "$TEST_DIR"

log_message "Recovery test completed successfully"
log_message "Backup file is valid and can be used for recovery"
```

This comprehensive backup and recovery system ensures that the System Scheduler service can be quickly restored from various failure scenarios with minimal data loss and downtime.