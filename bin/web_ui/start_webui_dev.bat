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

REM Colors (limited in Windows batch)
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

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
echo %BLUE%
echo 🚀 Trading Web UI Development Startup
echo =====================================
echo %NC%
echo Project Root: %PROJECT_ROOT%
echo Backend Port: %PORT%
echo Frontend Port: %FRONTEND_PORT%
echo.

REM Change to project root
cd /d "%PROJECT_ROOT%"

REM Check if virtual environment exists
if not exist ".venv" (
    echo %RED%❌ Virtual environment not found at .venv%NC%
    echo %RED%Please create a virtual environment first:%NC%
    echo %RED%  python -m venv .venv%NC%
    echo %RED%  .venv\Scripts\activate%NC%
    echo %RED%  pip install -r requirements.txt%NC%
    exit /b 1
)

REM Check if Python executable exists in venv
set "PYTHON_EXEC=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXEC%" (
    echo %RED%❌ Python executable not found at %PYTHON_EXEC%%NC%
    exit /b 1
)

echo %GREEN%✅ Virtual environment found%NC%

REM Check if main script exists
set "MAIN_SCRIPT=src\web_ui\run_web_ui.py"
if not exist "%MAIN_SCRIPT%" (
    echo %RED%❌ Main script not found at %MAIN_SCRIPT%%NC%
    exit /b 1
)

echo %GREEN%✅ Main script found%NC%

REM Check if Node.js is available
node --version >nul 2>&1
if errorlevel 1 (
    echo %RED%❌ Node.js is not installed%NC%
    echo %RED%Please install Node.js >= 18.0.0 from https://nodejs.org/%NC%
    exit /b 1
)

echo %GREEN%✅ Node.js is available%NC%

REM Check if npm is available
npm --version >nul 2>&1
if errorlevel 1 (
    echo %RED%❌ npm is not installed%NC%
    exit /b 1
)

echo %GREEN%✅ npm is available%NC%

REM Set environment variables
set "PYTHONPATH=%PROJECT_ROOT%"
set "PYTHONUNBUFFERED=1"

REM Create logs directory if it doesn't exist
if not exist "logs\web_ui" mkdir "logs\web_ui"

REM Check if frontend dependencies are installed
if not exist "src\web_ui\frontend\node_modules" (
    echo %YELLOW%⚠️ Frontend dependencies not found%NC%
    echo Installing frontend dependencies...
    cd src\web_ui\frontend
    npm install
    if errorlevel 1 (
        echo %RED%❌ Failed to install frontend dependencies%NC%
        exit /b 1
    )
    cd /d "%PROJECT_ROOT%"
    echo %GREEN%✅ Frontend dependencies installed%NC%
) else (
    echo %GREEN%✅ Frontend dependencies found%NC%
)

echo.
echo %GREEN%🚀 Starting Trading Web UI in development mode...%NC%
echo %GREEN%📡 Backend API: http://localhost:%PORT%%NC%
echo %GREEN%🎨 Frontend UI: http://localhost:%FRONTEND_PORT%%NC%
echo %GREEN%📚 API Docs: http://localhost:%PORT%/docs%NC%
echo.
echo %YELLOW%💡 Both services will auto-reload when you make changes%NC%
echo %YELLOW%🔑 Default login: admin/admin%NC%
echo %YELLOW%🛑 Press Ctrl+C to stop both services%NC%
echo.

REM Start the Web UI in development mode
"%PYTHON_EXEC%" "%MAIN_SCRIPT%" --dev --host %HOST% --port %PORT%

echo.
echo %GREEN%✅ Web UI stopped%NC%
pause