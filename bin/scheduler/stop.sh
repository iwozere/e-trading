#!/usr/bin/env bash
# Stop the Scheduler Service
#
# Usage:
#   ./bin/scheduler/stop.sh

set -e

# Change to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================"
echo "Stopping Scheduler Service"
echo "========================================"

if [ ! -f "scheduler.pid" ]; then
    echo "ERROR: scheduler.pid not found. Is the service running?"
    exit 1
fi

PID=$(cat scheduler.pid)

if ! ps -p $PID > /dev/null 2>&1; then
    echo "WARNING: Process $PID is not running"
    rm -f scheduler.pid
    exit 0
fi

echo "Sending SIGTERM to process $PID..."
kill -TERM $PID

# Wait for graceful shutdown (up to 30 seconds)
for i in {1..30}; do
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "Service stopped successfully"
        rm -f scheduler.pid
        exit 0
    fi
    sleep 1
done

echo "WARNING: Service did not stop gracefully, sending SIGKILL..."
kill -KILL $PID
rm -f scheduler.pid
echo "Service forcefully stopped"
