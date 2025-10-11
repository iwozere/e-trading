@echo off
REM Trading Web UI Development Startup Script
REM ==========================================
REM
REM This script starts the Trading Web UI in development mode on Windows.
REM It handles both backend and frontend with auto-reload capabilities.
REM
REM Usage:
REM   start_webui_dev.bat [--port PORT]
REM
REM Default ports:
REM   Backend API: 5003
REM   Frontend UI: 5002

setlocal enabledelayedexpansion

REM Enable ANSI color support (Windows 10+)
if not defined ANSICON (
    REM Try to enable ANSI colors
    reg add HKCU\Console /v VirtualTerminalLevel /t REG_DWORD /d 1 /f >nul 2>&1
)

REM Colors for Windows (simplified)
set "GREEN=*** "
set "RED=!!! "
set "YELLOW=>>> "
set "BLUE=--- "
set "NC="

REM Default configuration
set "DEFAULT_PORT=5003"
set "FRONTEND_PORT=5002"
set "HOST=0.0.0.0"

REM Get script directory and project root
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%..\..\"

REM Parse command line arguments
set "PORT=%DEFAULT_PORT%"
:parse_args
if "%~1"=="--port" (
    set "PORT=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--help" goto show_help
if "%~1"=="-h" goto show_help
if not "%~1"=="" (
    echo %RED%Unknown option: %~1%NC%
    exit /b 1
)

goto main

:show_help
echo Usage: %0 [--port PORT]
echo.
echo Options:
echo   --port PORT    Backend API port (default: %DEFAULT_PORT%)
echo   --help, -h     Show this help message
echo.
echo Default Configuration:
echo   Backend API: http://localhost:%DEFAULT_PORT%
echo   Frontend UI: http://localhost:%FRONTEND_PORT%
echo   API Docs: http://localhost:%DEFAULT_PORT%/docs
exit /b 0

:main
REM Print startup banner
echo.
echo %BLUE%======================================================
echo %BLUE%    Trading Web UI Development Startup
echo %BLUE%======================================================%NC%
echo.
echo Project Root: %PROJECT_ROOT%
echo Backend Port: %PORT%
echo Frontend Port: %FRONTEND_PORT%
echo.

REM Change to project root
cd /d "%PROJECT_ROOT%"

REM Check if virtual environment exists
echo %BLUE%Checking environment...%NC%
if not exist ".venv" (
    echo %RED%ERROR: Virtual environment not found at .venv%NC%
    echo %RED%Please create a virtual environment first:%NC%
    echo %RED%  python -m venv .venv%NC%
    echo %RED%  .venv\Scripts\activate%NC%
    echo %RED%  pip install -r requirements.txt%NC%
    pause
    exit /b 1
)

REM Check if Python executable exists in venv
set "PYTHON_EXEC=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXEC%" (
    echo %RED%ERROR: Python executable not found at %PYTHON_EXEC%%NC%
    pause
    exit /b 1
)

echo %GREEN%OK: Virtual environment found%NC%

REM Check if main script exists
set "MAIN_SCRIPT=src\web_ui\run_web_ui.py"
if not exist "%MAIN_SCRIPT%" (
    echo %RED%ERROR: Main script not found at %MAIN_SCRIPT%%NC%
    echo Current directory: %CD%
    pause
    exit /b 1
)

echo %GREEN%OK: Main script found%NC%

REM Check if Node.js is available
echo %BLUE%Checking Node.js...%NC%
node --version >nul 2>&1
if errorlevel 1 (
    echo %RED%ERROR: Node.js is not installed%NC%
    echo %RED%Please install Node.js >= 18.0.0 from https://nodejs.org/%NC%
    pause
    exit /b 1
)

echo %GREEN%OK: Node.js is available%NC%

REM Check if npm is available
echo %BLUE%Checking npm...%NC%
npm --version >nul 2>&1
if errorlevel 1 (
    echo %RED%ERROR: npm is not installed%NC%
    pause
    exit /b 1
)

echo %GREEN%OK: npm is available%NC%

echo %BLUE%Setting up environment...%NC%
REM Set environment variables
set "PYTHONPATH=%PROJECT_ROOT%"
set "PYTHONUNBUFFERED=1"
echo %GREEN%OK: Environment variables set%NC%

REM Create logs directory if it doesn't exist
if not exist "logs\web_ui" mkdir "logs\web_ui"

REM Check if frontend dependencies are installed
echo %BLUE%Checking frontend dependencies...%NC%
if not exist "src\web_ui\frontend\node_modules" (
    echo %YELLOW%WARNING: Frontend dependencies not found%NC%
    echo Installing frontend dependencies...
    cd src\web_ui\frontend
    if not exist "package.json" (
        echo %RED%ERROR: package.json not found in frontend directory%NC%
        cd /d "%PROJECT_ROOT%"
        pause
        exit /b 1
    )
    npm install
    if errorlevel 1 (
        echo %RED%ERROR: Failed to install frontend dependencies%NC%
        cd /d "%PROJECT_ROOT%"
        pause
        exit /b 1
    )
    cd /d "%PROJECT_ROOT%"
    echo %GREEN%OK: Frontend dependencies installed%NC%
) else (
    echo %GREEN%OK: Frontend dependencies found%NC%
)

echo.
echo %GREEN%======================================================%NC%
echo %GREEN%    Starting Trading Web UI in development mode%NC%
echo %GREEN%======================================================%NC%
echo.
echo Backend API: http://localhost:%PORT%
echo Frontend UI: http://localhost:%FRONTEND_PORT%
echo API Docs: http://localhost:%PORT%/docs
echo.
echo %YELLOW%INFO: Both services will auto-reload when you make changes%NC%
echo %YELLOW%INFO: Default login: admin/admin%NC%
echo %YELLOW%INFO: Press Ctrl+C to stop both services%NC%
echo.
echo %BLUE%Starting services...%NC%
echo.

REM Start the Web UI in development mode
echo %BLUE%Executing command:%NC%
echo "%PYTHON_EXEC%" "%MAIN_SCRIPT%" --dev --host %HOST% --port %PORT%
echo.
echo %BLUE%Starting Web UI...%NC%
"%PYTHON_EXEC%" "%MAIN_SCRIPT%" --dev --host %HOST% --port %PORT%
set WEBUI_EXIT_CODE=%ERRORLEVEL%
echo.
echo %BLUE%Web UI process exited with code: %WEBUI_EXIT_CODE%%NC%

echo.
echo %GREEN%Web UI stopped%NC%
echo.
pause