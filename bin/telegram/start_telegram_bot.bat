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
set TELEGRAM_API_PORT=5004

REM Create logs directory if it doesn't exist
if not exist logs mkdir logs

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found at .venv\Scripts\activate.bat
    echo Please run: python -m venv .venv
    echo Then install dependencies: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Start the Telegram bot
echo Starting Telegram Bot...
echo.
echo Bot will handle Telegram messages and commands
echo Press Ctrl+C to stop the bot
echo.

python -m src.telegram.telegram_bot

echo.
echo Telegram bot stopped
