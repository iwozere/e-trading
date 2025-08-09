@echo off
REM Script to run the Telegram Screener Bot background services

cd /d "%~dp0\.."

echo Starting Telegram Screener Background Services...
python src/frontend/telegram/screener/background_services.py
pause
