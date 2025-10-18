@echo off
REM Start Job Scheduler Process
REM This script starts the APScheduler process for cron-based job triggering

setlocal

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

REM Change to project root
cd /d "%PROJECT_ROOT%"

REM Set environment variables
set PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%

REM Create logs directory if it doesn't exist
if not exist logs mkdir logs

REM Start the scheduler process
echo Starting job scheduler process...
python -m src.backend.scheduler.scheduler_process

echo Job scheduler process started successfully


