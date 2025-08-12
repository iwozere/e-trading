#!/bin/bash
# Script to run the Telegram Screener Bot background services (alerts & schedules)

# Get the absolute path to the project root
PROJECT_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"

mkdir -p "$PROJECT_ROOT/logs/log"

if [ ! -f "$PROJECT_ROOT/.venv/bin/python" ]; then
    echo "Error: Python venv not found at $PROJECT_ROOT/.venv/bin/python"
    exit 1
fi

source "$PROJECT_ROOT/.venv/bin/activate"

echo "Starting Telegram Screener Background Services..."
"$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/src/frontend/telegram/screener/background_services.py"
