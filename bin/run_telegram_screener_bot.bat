@echo off
REM Script to run the Telegram Screener Bot

cd /d "%~dp0\.."

echo Starting Telegram Screener Bot...
python src/frontend/telegram/bot.py
pause
