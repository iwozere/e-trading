@echo off
REM Start Frontend Only
REM ===================

setlocal

echo Starting Frontend Development Server...
echo.

REM Get project root
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%..\..\"

REM Change to project root
cd /d "%PROJECT_ROOT%"

REM Change to frontend directory
cd src\web_ui\frontend

REM Check if package.json exists
if not exist "package.json" (
    echo ERROR: package.json not found
    pause
    exit /b 1
)

REM Install dependencies if needed
if not exist "node_modules" (
    echo Installing dependencies...
    npm install
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Set environment variables
set "VITE_API_BASE_URL=http://localhost:5003"
set "VITE_WS_URL=ws://localhost:5003"

echo Frontend UI: http://localhost:5002
echo Backend API: http://localhost:5003 (must be running separately)
echo.
echo Press Ctrl+C to stop
echo.

REM Start frontend
npm run dev

pause