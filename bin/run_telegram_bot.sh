#!/bin/bash
# Run the Telegram Screener Bot (Linux)
# Activate virtual environment if needed

# Get the absolute path to the project root
PROJECT_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"

mkdir -p "$PROJECT_ROOT/logs/log"

source "$PROJECT_ROOT/.venv/bin/activate"

nohup "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/src/screener/telegram/bot.py" > "$PROJECT_ROOT/logs/log/telegram_bot.log" 2>&1 & 