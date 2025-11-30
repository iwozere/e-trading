@echo off
REM Start Telegram Bot
REM This script starts the Telegram bot that provides screener alerts,
REM reports, and user interaction through Telegram

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

REM Start the Telegram bot
echo Starting Telegram Bot...
echo.
echo Bot will handle Telegram messages and commands
echo Press Ctrl+C to stop the bot
echo.

python -m src.telegram.telegram_bot

echo.
echo Telegram bot stopped
