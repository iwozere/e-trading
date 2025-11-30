#!/usr/bin/env bash
# Check Scheduler Service Status
#
# Usage:
#   ./bin/scheduler/status.sh

set -e

# Change to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================"
echo "Scheduler Service Status"
echo "========================================"

# Check PID file
if [ ! -f "scheduler.pid" ]; then
    echo "Status: NOT RUNNING (no PID file found)"
    echo ""
    echo "Start with: ./bin/scheduler/start.sh"
    exit 1
fi

PID=$(cat scheduler.pid)

# Check if process is running
if ! ps -p $PID > /dev/null 2>&1; then
    echo "Status: NOT RUNNING (stale PID file)"
    echo "PID file exists but process $PID is not running"
    echo ""
    echo "Removing stale PID file..."
    rm -f scheduler.pid
    echo "Start with: ./bin/scheduler/start.sh"
    exit 1
fi

# Get process info
PROCESS_INFO=$(ps -p $PID -o pid,ppid,user,%cpu,%mem,etime,command | tail -n 1)

echo "Status: RUNNING"
echo "PID: $PID"
echo ""
echo "Process Details:"
echo "$PROCESS_INFO"
echo ""

# Check log file
if [ -f "logs/scheduler.log" ]; then
    LOG_SIZE=$(du -h logs/scheduler.log | cut -f1)
    LOG_LINES=$(wc -l < logs/scheduler.log)
    echo "Log File: logs/scheduler.log"
    echo "Log Size: $LOG_SIZE"
    echo "Log Lines: $LOG_LINES"
    echo ""
    echo "Last 10 log lines:"
    echo "----------------------------------------"
    tail -n 10 logs/scheduler.log
fi

echo ""
echo "========================================"
echo "Use './bin/scheduler/stop.sh' to stop"
echo "Use 'tail -f logs/scheduler.log' to watch logs"
echo "========================================"
