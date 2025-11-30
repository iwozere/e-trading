#!/bin/bash
# Start Notification Service (Database-Centric Bot)
# This script starts the notification service that polls the database for messages
# and delivers them through channel plugins (Telegram, Email, SMS)
#
# Usage:
#   ./start_notification_bot.sh              # Run in foreground
#   ./start_notification_bot.sh --background # Run in background
#   ./start_notification_bot.sh --stop       # Stop background process

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Create logs directory if it doesn't exist
mkdir -p logs

# PID file location
PID_FILE="$PROJECT_ROOT/logs/notification_bot.pid"
LOG_FILE="$PROJECT_ROOT/logs/notification_bot.log"

# Function to start in foreground
start_foreground() {
    echo "Starting Notification Service (Database-Centric)..."
    echo ""
    echo "Service will poll the database for pending messages"
    echo "Press Ctrl+C to stop the service"
    echo ""

    python -m src.notification.notification_db_centric_bot
}

# Function to start in background
start_background() {
    # Check if already running
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Notification service is already running (PID: $PID)"
            exit 1
        else
            echo "Removing stale PID file"
            rm -f "$PID_FILE"
        fi
    fi

    echo "Starting Notification Service in background..."
    echo "Logs: $LOG_FILE"

    # Start the service in background
    nohup python -m src.notification.notification_db_centric_bot >> "$LOG_FILE" 2>&1 &

    # Save PID
    echo $! > "$PID_FILE"

    echo "Notification service started (PID: $(cat "$PID_FILE"))"
    echo ""
    echo "Commands:"
    echo "  Stop:   $0 --stop"
    echo "  Status: $0 --status"
    echo "  Logs:   tail -f $LOG_FILE"
}

# Function to stop background process
stop_background() {
    if [ ! -f "$PID_FILE" ]; then
        echo "Notification service is not running (no PID file found)"
        exit 1
    fi

    PID=$(cat "$PID_FILE")

    if ! ps -p "$PID" > /dev/null 2>&1; then
        echo "Notification service is not running (process $PID not found)"
        rm -f "$PID_FILE"
        exit 1
    fi

    echo "Stopping Notification Service (PID: $PID)..."

    # Send SIGTERM for graceful shutdown
    kill -TERM "$PID"

    # Wait for process to stop (max 30 seconds)
    for i in {1..30}; do
        if ! ps -p "$PID" > /dev/null 2>&1; then
            echo "Notification service stopped successfully"
            rm -f "$PID_FILE"
            exit 0
        fi
        sleep 1
    done

    # Force kill if still running
    echo "Process did not stop gracefully, forcing shutdown..."
    kill -KILL "$PID"
    rm -f "$PID_FILE"
    echo "Notification service stopped (forced)"
}

# Function to check status
check_status() {
    if [ ! -f "$PID_FILE" ]; then
        echo "Notification service is NOT running"
        exit 1
    fi

    PID=$(cat "$PID_FILE")

    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Notification service is RUNNING (PID: $PID)"
        echo ""
        ps -p "$PID" -o pid,ppid,user,%cpu,%mem,etime,cmd
        exit 0
    else
        echo "Notification service is NOT running (stale PID file)"
        rm -f "$PID_FILE"
        exit 1
    fi
}

# Parse command line arguments
case "${1:-}" in
    --background|-b)
        start_background
        ;;
    --stop|-s)
        stop_background
        ;;
    --status)
        check_status
        ;;
    --help|-h)
        echo "Usage: $0 [OPTION]"
        echo ""
        echo "Options:"
        echo "  (no option)      Run in foreground (default)"
        echo "  --background, -b Run in background"
        echo "  --stop, -s       Stop background process"
        echo "  --status         Check if service is running"
        echo "  --help, -h       Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0                    # Run in foreground"
        echo "  $0 --background       # Run in background"
        echo "  $0 --stop             # Stop background service"
        echo "  tail -f $LOG_FILE     # View logs"
        ;;
    "")
        start_foreground
        ;;
    *)
        echo "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac
