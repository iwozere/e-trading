@echo off
REM Start Notification Service (Database-Centric Bot)
REM This script starts the notification service that polls the database for messages
REM and delivers them through channel plugins (Telegram, Email, SMS)

setlocal

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..\..

REM Change to project root
cd /d "%PROJECT_ROOT%"

REM Set environment variables
set PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%

REM Create logs directory if it doesn't exist
if not exist logs mkdir logs

REM Start the notification service
echo Starting Notification Service (Database-Centric)...
echo.
echo Service will poll the database for pending messages
echo Press Ctrl+C to stop the service
echo.

python -m src.notification.notification_db_centric_bot

echo.
echo Notification service stopped
