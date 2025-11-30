#!/usr/bin/env bash
# Start the Scheduler Service
#
# Usage:
#   ./bin/scheduler/start.sh [--foreground]
#
# Options:
#   --foreground    Run in foreground (don't daemonize)
#
# Environment Variables:
#   TRADING_ENV     Environment (development/staging/production)
#   DATABASE_URL    Database connection URL

set -e

# Change to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Load environment if .env exists
if [ -f ".env" ]; then
    echo "Loading environment from .env"
    export $(grep -v '^#' .env | xargs)
fi

# Set defaults
TRADING_ENV="${TRADING_ENV:-development}"
FOREGROUND=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --foreground)
            FOREGROUND=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--foreground]"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Starting Scheduler Service"
echo "========================================"
echo "Environment: $TRADING_ENV"
echo "Project Root: $PROJECT_ROOT"
echo "Python: $(which python)"
echo "========================================"

if [ "$FOREGROUND" = true ]; then
    echo "Running in foreground mode..."
    python -m src.scheduler.main
else
    echo "Starting as background service..."

    # Check if already running
    if [ -f "scheduler.pid" ]; then
        PID=$(cat scheduler.pid)
        if ps -p $PID > /dev/null 2>&1; then
            echo "ERROR: Scheduler service is already running (PID: $PID)"
            exit 1
        else
            echo "Removing stale PID file"
            rm -f scheduler.pid
        fi
    fi

    # Start in background
    nohup python -m src.scheduler.main > logs/scheduler.log 2>&1 &
    echo $! > scheduler.pid

    echo "Scheduler service started (PID: $(cat scheduler.pid))"
    echo "Log file: logs/scheduler.log"
    echo ""
    echo "Use './bin/scheduler/status.sh' to check status"
    echo "Use './bin/scheduler/stop.sh' to stop the service"
fi
