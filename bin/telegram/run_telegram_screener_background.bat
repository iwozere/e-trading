@echo off
REM Script to run the Telegram Screener Bot background services

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..\\..

REM Change to project root directory
cd /d "%PROJECT_ROOT%"

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

REM Check if activation was successful
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

echo Starting Telegram Screener Background Services...
python src\frontend\telegram\screener\background_services.py

REM Check if script ran successfully
if errorlevel 1 (
    echo Error: Background services script failed
    pause
    exit /b 1
)

pause
