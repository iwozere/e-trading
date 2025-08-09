@echo off
REM Script to run the Telegram Screener Bot admin panel

cd /d "%~dp0\.."

echo Starting Telegram Screener Admin Panel...
echo Admin panel will be available at http://localhost:5001
python src/frontend/telegram/screener/admin_panel.py
pause
