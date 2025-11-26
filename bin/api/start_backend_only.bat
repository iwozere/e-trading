@echo off
REM Start Backend Only
REM ==================

setlocal

echo Starting Backend API Server...
echo.

REM Get project root
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%..\\..\"

REM Change to project root
cd /d "%PROJECT_ROOT%"

REM Set environment
set "PYTHONPATH=%PROJECT_ROOT%"
set "PYTHONUNBUFFERED=1"

echo Backend API: http://localhost:5003
echo API Docs: http://localhost:5003/docs
echo.
echo Press Ctrl+C to stop
echo.

REM Start backend
".venv\Scripts\python.exe" -m uvicorn src.api.main:app --host 0.0.0.0 --port 5003 --reload

pause