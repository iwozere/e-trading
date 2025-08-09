#!/bin/bash
# Script to run the Telegram Screener Bot

cd "$(dirname "$0")/.."

echo "Starting Telegram Screener Bot..."
python3 src/frontend/telegram/bot.py
