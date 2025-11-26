@echo off
REM Trading Bot Web UI - Windows Development Launcher
REM ================================================
REM This batch file starts the trading web UI in development mode on Windows
REM It handles environment setup, dependency checks, and launches both backend and frontend

setlocal enabledelayedexpansion

REM Colors for output (Windows 10+ with ANSI support)
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "PURPLE=[95m"
set "CYAN=[96m"
set "NC=[0m"

REM Configuration
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%..\\.."
set "VENV_DIR=%PROJECT_ROOT%\.venv"
set "LOG_DIR=%PROJECT_ROOT%\logs\webui"
set "BACKEND_PORT=8000"
set "FRONTEND_PORT=5173"

REM Functions
:print_header
echo %BLUE%
echo 🚀 Trading Bot Web UI - Windows Development Mode
echo ================================================
echo %NC%
goto :eof

:print_system_info
echo %CYAN%📊 System Information:%NC%
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
echo %CYAN%🔍 Checking Python...%NC%
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ Python not found%NC%
    echo %YELLOW%💡 Please install Python from https://python.org%NC%
    pause
    exit /b 1
)
echo %GREEN%✅ Python found%NC%
goto :eof

:check_node
echo %CYAN%🔍 Checking Node.js...%NC%
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ Node.js not found%NC%
    echo %YELLOW%💡 Please install Node.js from https://nodejs.org%NC%
    pause
    exit /b 1
)
echo %GREEN%✅ Node.js found%NC%

npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ npm not found%NC%
    pause
    exit /b 1
)
echo %GREEN%✅ npm found%NC%
goto :eof

:setup_environment
echo %CYAN%🔧 Setting up Environment...%NC%

REM Create directories
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%PROJECT_ROOT%\config\enhanced_trading" mkdir "%PROJECT_ROOT%\config\enhanced_trading"
if not exist "%PROJECT_ROOT%\data" mkdir "%PROJECT_ROOT%\data"
if not exist "%PROJECT_ROOT%\db" mkdir "%PROJECT_ROOT%\db"

REM Check virtual environment
if not exist "%VENV_DIR%" (
    echo %YELLOW%⚠️  Virtual environment not found, creating...%NC%
    python -m venv "%VENV_DIR%"
    echo %GREEN%✅ Virtual environment created%NC%
) else (
    echo %GREEN%✅ Virtual environment found%NC%
)

REM Set Python path
set "PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%"

REM Load environment variables if .env exists
if exist "%PROJECT_ROOT%\.env" (
    echo %GREEN%✅ Loading .env file%NC%
    REM Note: Windows batch doesn't have a direct equivalent to source
    REM Environment variables should be set manually or via Python script
) else (
    echo %YELLOW%⚠️  No .env file found%NC%
    if exist "%PROJECT_ROOT%\config\donotshare\.env" (
        echo %YELLOW%📝 Copying .env.example to .env%NC%
        copy "%PROJECT_ROOT%\.env.example" "%PROJECT_ROOT%\.env"
        echo %GREEN%✅ .env file created from template%NC%
        echo %YELLOW%💡 Please edit .env file with your API keys%NC%
    )
)

echo %GREEN%✅ Environment setup complete%NC%
echo.
goto :eof

:install_python_deps
echo %CYAN%📦 Installing Python dependencies...%NC%

REM Activate virtual environment
call "%VENV_DIR%\Scripts\activate.bat"

REM Upgrade pip
python -m pip install --upgrade pip

REM Install web UI requirements
if exist "%PROJECT_ROOT%\requirements-webui.txt" (
    pip install -r "%PROJECT_ROOT%\requirements-webui.txt"
    echo %GREEN%✅ Python dependencies installed%NC%
) else (
    echo %YELLOW%⚠️  requirements-webui.txt not found, installing basic deps%NC%
    pip install fastapi uvicorn python-socketio
)

echo.
goto :eof

:install_frontend_deps
echo %CYAN%📦 Installing Frontend dependencies...%NC%

cd /d "%PROJECT_ROOT%\src\web_ui\frontend"

if not exist "node_modules" (
    echo %YELLOW%📦 Installing npm packages...%NC%
    npm install
    if %errorlevel% neq 0 (
        echo %RED%❌ Failed to install frontend dependencies%NC%
        pause
        exit /b 1
    )
    echo %GREEN%✅ Frontend dependencies installed%NC%
) else (
    echo %GREEN%✅ Frontend dependencies already installed%NC%
)

cd /d "%PROJECT_ROOT%"
echo.
goto :eof

:check_config
echo %CYAN%🔍 Checking Configuration...%NC%

if not exist "%PROJECT_ROOT%\config\enhanced_trading\raspberry_pi_multi_strategy.json" (
    echo %YELLOW%⚠️  Strategy configuration not found%NC%
    echo %YELLOW%💡 Running setup...%NC%
    python "%PROJECT_ROOT%\setup_enhanced_trading.py"
    if %errorlevel% neq 0 (
        echo %RED%❌ Setup failed%NC%
        pause
        exit /b 1
    )
    echo %GREEN%✅ Setup completed%NC%
) else (
    echo %GREEN%✅ Configuration found%NC%
)

echo.
goto :eof

:start_backend
echo %CYAN%🚀 Starting Backend Server...%NC%
echo Backend will be available at: http://localhost:%BACKEND_PORT%
echo API docs will be available at: http://localhost:%BACKEND_PORT%/docs
echo.

REM Activate virtual environment
call "%VENV_DIR%\Scripts\activate.bat"

REM Start backend in development mode
python "%PROJECT_ROOT%\start_trading_webui.py" --dev --port %BACKEND_PORT%

goto :eof

:start_frontend
echo %CYAN%🎨 Starting Frontend Development Server...%NC%
echo Frontend will be available at: http://localhost:%FRONTEND_PORT%
echo.

cd /d "%PROJECT_ROOT%\src\web_ui\frontend"

REM Set environment variables for frontend
set "VITE_API_BASE_URL=http://localhost:%BACKEND_PORT%"
set "VITE_WS_URL=ws://localhost:%BACKEND_PORT%"

REM Start frontend development server
npm run dev

cd /d "%PROJECT_ROOT%"
goto :eof

:show_usage_info
echo %PURPLE%🎯 Trading Bot Web UI - Development Mode%NC%
echo ===============================================
echo.
echo %GREEN%🌐 Access Points:%NC%
echo   Backend API: http://localhost:%BACKEND_PORT%
echo   Frontend UI: http://localhost:%FRONTEND_PORT%
echo   API Docs: http://localhost:%BACKEND_PORT%/docs
echo.
echo %GREEN%🔐 Default Login:%NC%
echo   Username: admin ^| Password: admin
echo   Username: trader ^| Password: trader
echo.
echo %GREEN%💡 Development Features:%NC%
echo   • Auto-reload on code changes
echo   • Real-time debugging
echo   • Hot module replacement
echo.
echo %YELLOW%🛑 To stop servers, press Ctrl+C in each window%NC%
echo ===============================================
echo.
goto :eof

:show_menu
echo %PURPLE%🎛️  Development Menu:%NC%
echo 1. 🚀 Start Backend Only
echo 2. 🎨 Start Frontend Only  
echo 3. 🔧 Install Dependencies
echo 4. 📋 Check Configuration
echo 5. 🔍 System Information
echo 6. ❌ Exit
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

echo %YELLOW%🔄 Starting Backend Server...%NC%
echo %YELLOW%💡 Start Frontend in another window with: %0 --frontend%NC%
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
    echo %GREEN%✅ Dependencies installed%NC%
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
    echo %GREEN%👋 Goodbye!%NC%
    goto :end
)

echo %RED%❌ Invalid option%NC%
pause
goto :menu_loop

:backend_only
call :install_python_deps
call :check_config
echo %CYAN%🚀 Starting Backend Only...%NC%
call :start_backend
goto :end

:frontend_only
call :install_frontend_deps
echo %CYAN%🎨 Starting Frontend Only...%NC%
call :start_frontend
goto :end

:setup_only
call :install_python_deps
call :install_frontend_deps
call :check_config
echo %GREEN%✅ Setup completed%NC%
pause
goto :end

:end
echo.
echo %GREEN%👋 Trading Bot Web UI stopped%NC%
pause
exit /b 0

REM Start main execution
call :main %*