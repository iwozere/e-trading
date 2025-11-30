@echo off
REM Start the Scheduler Service (Windows)
REM
REM Usage:
REM   bin\scheduler\start.bat

setlocal

REM Change to project root
cd /d "%~dp0\..\..\"

echo ========================================
echo Starting Scheduler Service
echo ========================================

REM Check if already running
if exist scheduler.pid (
    echo ERROR: Scheduler service appears to be running
    echo If you're sure it's not running, delete scheduler.pid and try again
    exit /b 1
)

REM Set environment
if exist .env (
    echo Loading environment from .env
    for /f "usebackq tokens=* delims=" %%a in (".env") do (
        set "%%a"
    )
)

REM Check Python
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    exit /b 1
)

echo Python:
where python
echo.

REM Create logs directory if it doesn't exist
if not exist logs mkdir logs

echo Starting scheduler service...
echo.

REM Start Python process in background
start /B python -m src.scheduler.main > logs\scheduler.log 2>&1

REM Wait a moment for process to start
timeout /t 2 /nobreak >nul

echo Scheduler service started
echo Log file: logs\scheduler.log
echo.
echo Use 'bin\scheduler\status.bat' to check status
echo Use 'bin\scheduler\stop.bat' to stop the service
echo.
pause
