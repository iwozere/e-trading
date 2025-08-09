#!/bin/bash
# Script to run the Telegram Screener Bot background services (alerts & schedules)

cd "$(dirname "$0")/.."

echo "Starting Telegram Screener Background Services..."
python3 src/frontend/telegram/screener/background_services.py
