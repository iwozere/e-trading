#!/bin/bash
# Script to run the Telegram Screener Bot admin panel

# Get the absolute path to the project root
PROJECT_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"

mkdir -p "$PROJECT_ROOT/logs/log"

if [ ! -f "$PROJECT_ROOT/.venv/bin/python" ]; then
    echo "Error: Python venv not found at $PROJECT_ROOT/.venv/bin/python"
    exit 1
fi

source "$PROJECT_ROOT/.venv/bin/activate"

echo "Starting Telegram Screener Admin Panel..."
echo "Admin panel will be available at http://localhost:5000"
"$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/src/frontend/telegram/screener/admin_panel.py"
