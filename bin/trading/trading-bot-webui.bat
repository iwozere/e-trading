@echo off
REM Trading Bot Web UI - Windows Development Launcher
REM ================================================
REM This batch file starts the trading web UI in development mode on Windows
REM It handles environment setup, dependency checks, and launches both backend and frontend

setlocal enabledelayedexpansion

REM Colors for output (Windows 10+ with ANSI support)
REM Note: These are set to empty by default for maximum compatibility 
REM unless the terminal is known to support them.
set "RED="
set "GREEN="
set "YELLOW="
set "BLUE="
set "PURPLE="
set "CYAN="
set "NC="



REM Configuration
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%..\\.."
set "VENV_DIR=%PROJECT_ROOT%\.venv"
set "LOG_DIR=%PROJECT_ROOT%\logs\webui"
set "BACKEND_PORT=5003"
set "FRONTEND_PORT=5002"

goto :main

REM Functions
:print_header
echo.
echo ------------------------------------------------
echo  Trading Bot Web UI - Windows Development Mode
echo ------------------------------------------------
echo.
goto :eof

:print_system_info
echo [System Information]
echo Date: %date% %time%
echo OS: %OS%
echo Architecture: %PROCESSOR_ARCHITECTURE%
echo Python: 
python --version 2>nul || echo Not found
echo Node.js: 
node --version 2>nul || echo Not found
echo Working Directory: %PROJECT_ROOT%
echo.
goto :eof

:check_python
echo [Checking Python...]
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found in path.
    echo TIP: Please install Python and ensure it is in your PATH.
    pause
    exit /b 1
)
echo Python found.
goto :eof

:check_node
echo [Checking Node.js...]
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js not found in path.
    echo TIP: Please install Node.js and ensure it is in your PATH.
    pause
    exit /b 1
)
echo Node.js found.

where npm >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: npm not found in path.
    pause
    exit /b 1
)
echo npm found.
goto :eof

:setup_environment
echo [Setting up Environment...]

REM Create directories
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%PROJECT_ROOT%\config\enhanced_trading" mkdir "%PROJECT_ROOT%\config\enhanced_trading"
if not exist "%PROJECT_ROOT%\data" mkdir "%PROJECT_ROOT%\data"
if not exist "%PROJECT_ROOT%\db" mkdir "%PROJECT_ROOT%\db"

REM Check virtual environment
if not exist "%VENV_DIR%" (
    echo Virtual environment not found, creating...
    python -m venv "%VENV_DIR%"
    echo Virtual environment created.
) else (
    echo Virtual environment found and verified.
)

REM Set Python path
set "PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%"

REM Load environment variables if .env exists
if exist "%PROJECT_ROOT%\.env" (
    echo Loading .env file...
) else (
    echo No .env file found in root.
    if exist "%PROJECT_ROOT%\config\donotshare\.env" (
        echo Copying .env from config/donotshare...
        copy "%PROJECT_ROOT%\config\donotshare\.env" "%PROJECT_ROOT%\.env"
        echo .env file created.
    )
)

echo Environment setup complete.
echo.
goto :eof

:install_python_deps
echo [Installing Python dependencies...]

REM Activate virtual environment
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

REM Upgrade pip
python -m pip install --upgrade pip

REM Install web UI requirements
if exist "%PROJECT_ROOT%\requirements-webui.txt" (
    pip install -r "%PROJECT_ROOT%\requirements-webui.txt"
    echo Python dependencies installed.
) else (
    echo requirements-webui.txt not found, installing basic dependencies...
    pip install fastapi uvicorn python-socketio
)

echo.
goto :eof

:install_frontend_deps
echo [Installing Frontend dependencies...]

cd /d "%PROJECT_ROOT%\src\web_ui\frontend"

if not exist "node_modules" (
    echo Installing npm packages...
    call npm install
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install frontend dependencies
        pause
        exit /b 1
    )
    echo Frontend dependencies installed.
) else (
    echo Frontend dependencies already installed.
)

cd /d "%PROJECT_ROOT%"
echo.
goto :eof

:check_config
echo [Checking Configuration...]

if not exist "%PROJECT_ROOT%\config\enhanced_trading\raspberry_pi_multi_strategy.json" (
    echo Strategy configuration not found.
    echo Running setup...
    python "%PROJECT_ROOT%\setup_enhanced_trading.py"
    if %errorlevel% neq 0 (
        echo ERROR: Setup failed
        pause
        exit /b 1
    )
    echo Setup completed.
) else (
    echo Configuration found.
)

echo.
goto :eof

:start_backend
echo [Starting Backend Server...]
echo Backend available at: http://localhost:%BACKEND_PORT%
echo API docs available at: http://localhost:%BACKEND_PORT%/docs
echo.

REM Activate virtual environment
call "%VENV_DIR%\Scripts\activate.bat"

REM Start backend in development mode
python "%PROJECT_ROOT%\src\web_ui\run_web_ui.py" --dev --port %BACKEND_PORT%

goto :eof

:start_frontend
echo [Starting Frontend Development Server...]
echo Frontend available at: http://localhost:%FRONTEND_PORT%
echo.

cd /d "%PROJECT_ROOT%\src\web_ui\frontend"

REM Set environment variables for frontend
set "VITE_API_BASE_URL=http://localhost:%BACKEND_PORT%"
set "VITE_WS_URL=ws://localhost:%BACKEND_PORT%"

REM Start frontend development server
call npm run dev

cd /d "%PROJECT_ROOT%"
goto :eof

:show_usage_info
echo --- Trading Bot Web UI - Development Mode ---
echo.
echo Access Points:
echo   Backend API: http://localhost:%BACKEND_PORT%
echo   Frontend UI: http://localhost:%FRONTEND_PORT%
echo   API Docs: http://localhost:%BACKEND_PORT%/docs
echo.
echo Default Login:
echo   Username: admin ^| Password: admin
echo   Username: trader ^| Password: trader
echo.
echo Note: To stop servers, press Ctrl+C in each window.
echo ------------------------------------------------
echo.
goto :eof

:show_menu
echo --- Development Menu ---
echo 1. Start Backend Only
echo 2. Start Frontend Only
echo 3. Install Dependencies
echo 4. Check Configuration
echo 5. System Information
echo 6. Exit
echo.
set /p choice="Select option (1-6): "
goto :eof

REM Main execution
:main
call :print_header
call :print_system_info

REM Check dependencies
call :check_python
if %errorlevel% neq 0 exit /b 1

call :check_node
if %errorlevel% neq 0 exit /b 1

REM Setup environment
call :setup_environment

REM Handle command line arguments
if "%1"=="--menu" goto :interactive_mode
if "%1"=="--backend" goto :backend_only
if "%1"=="--frontend" goto :frontend_only
if "%1"=="--setup" goto :setup_only

REM Default: Full setup and backend start
call :install_python_deps
call :install_frontend_deps
call :check_config
call :show_usage_info

echo Starting Backend Server...
echo Start Frontend in another window with: %0 --frontend
echo.
call :start_backend
goto :end

:interactive_mode
:menu_loop
call :show_menu

if "%choice%"=="1" (
    call :install_python_deps
    call :check_config
    call :start_backend
    goto :menu_loop
)
if "%choice%"=="2" (
    call :install_frontend_deps
    call :start_frontend
    goto :menu_loop
)
if "%choice%"=="3" (
    call :install_python_deps
    call :install_frontend_deps
    echo Dependencies installed.
    pause
    goto :menu_loop
)
if "%choice%"=="4" (
    call :check_config
    pause
    goto :menu_loop
)
if "%choice%"=="5" (
    call :print_system_info
    pause
    goto :menu_loop
)
if "%choice%"=="6" (
    echo Goodbye!
    goto :end
)

echo Invalid option
pause
goto :menu_loop

:backend_only
call :install_python_deps
call :check_config
echo Starting Backend Only...
call :start_backend
goto :end

:frontend_only
call :install_frontend_deps
echo Starting Frontend Only...
call :start_frontend
goto :end

:setup_only
call :install_python_deps
call :install_frontend_deps
call :check_config
echo Setup completed.
pause
goto :end

:end
echo.
echo Trading Bot Web UI stopped.
pause
exit /b 0

REM Start main execution
call :main %*