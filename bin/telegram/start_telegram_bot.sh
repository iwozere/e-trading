#!/bin/bash
# Start Telegram Bot
# This script starts the Telegram bot that provides screener alerts,
# reports, and user interaction through Telegram
#
# Usage:
#   ./start_telegram_bot.sh              # Run in foreground
#   ./start_telegram_bot.sh --background # Run in background
#   ./start_telegram_bot.sh --stop       # Stop background process

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
PID_FILE="$PROJECT_ROOT/logs/telegram_bot.pid"
LOG_FILE="$PROJECT_ROOT/logs/telegram_bot.log"

# Function to start in foreground
start_foreground() {
    echo "Starting Telegram Bot..."
    echo ""
    echo "Bot will handle Telegram messages and commands"
    echo "Press Ctrl+C to stop the bot"
    echo ""

    python -m src.telegram.telegram_bot
}

# Function to start in background
start_background() {
    # Check if already running
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Telegram bot is already running (PID: $PID)"
            exit 1
        else
            echo "Removing stale PID file"
            rm -f "$PID_FILE"
        fi
    fi

    echo "Starting Telegram Bot in background..."
    echo "Logs: $LOG_FILE"

    # Start the bot in background
    nohup python -m src.telegram.telegram_bot >> "$LOG_FILE" 2>&1 &

    # Save PID
    echo $! > "$PID_FILE"

    echo "Telegram bot started (PID: $(cat "$PID_FILE"))"
    echo ""
    echo "Commands:"
    echo "  Stop:   $0 --stop"
    echo "  Status: $0 --status"
    echo "  Logs:   tail -f $LOG_FILE"
}

# Function to stop background process
stop_background() {
    if [ ! -f "$PID_FILE" ]; then
        echo "Telegram bot is not running (no PID file found)"
        exit 1
    fi

    PID=$(cat "$PID_FILE")

    if ! ps -p "$PID" > /dev/null 2>&1; then
        echo "Telegram bot is not running (process $PID not found)"
        rm -f "$PID_FILE"
        exit 1
    fi

    echo "Stopping Telegram Bot (PID: $PID)..."

    # Send SIGTERM for graceful shutdown
    kill -TERM "$PID"

    # Wait for process to stop (max 30 seconds)
    for i in {1..30}; do
        if ! ps -p "$PID" > /dev/null 2>&1; then
            echo "Telegram bot stopped successfully"
            rm -f "$PID_FILE"
            exit 0
        fi
        sleep 1
    done

    # Force kill if still running
    echo "Process did not stop gracefully, forcing shutdown..."
    kill -KILL "$PID"
    rm -f "$PID_FILE"
    echo "Telegram bot stopped (forced)"
}

# Function to check status
check_status() {
    if [ ! -f "$PID_FILE" ]; then
        echo "Telegram bot is NOT running"
        exit 1
    fi

    PID=$(cat "$PID_FILE")

    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Telegram bot is RUNNING (PID: $PID)"
        echo ""
        ps -p "$PID" -o pid,ppid,user,%cpu,%mem,etime,cmd
        exit 0
    else
        echo "Telegram bot is NOT running (stale PID file)"
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
        echo "  --status         Check if bot is running"
        echo "  --help, -h       Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0                    # Run in foreground"
        echo "  $0 --background       # Run in background"
        echo "  $0 --stop             # Stop background bot"
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
