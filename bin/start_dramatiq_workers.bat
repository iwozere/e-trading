@echo off
REM Start Dramatiq Workers
REM This script starts the Dramatiq workers for job execution

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

REM Start Dramatiq workers
echo Starting Dramatiq workers...
echo Workers will process jobs from the 'reports' and 'screeners' queues

REM Start workers with proper configuration
dramatiq src.backend.workers.report_worker src.backend.workers.screener_worker ^
    --processes 4 ^
    --threads 2 ^
    --queues reports,screeners ^
    --log-file logs/dramatiq_workers.log ^
    --pid-file logs/dramatiq_workers.pid

echo Dramatiq workers started successfully


