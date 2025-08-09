#!/bin/bash
# Script to run the Telegram Screener Bot admin panel

cd "$(dirname "$0")/.."

echo "Starting Telegram Screener Admin Panel..."
echo "Admin panel will be available at http://localhost:5001"
python3 src/frontend/telegram/screener/admin_panel.py
