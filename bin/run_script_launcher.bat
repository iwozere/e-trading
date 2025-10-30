@echo off
REM ========================================
REM Crypto Trading Platform - Script Launcher
REM ========================================

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

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

:menu
cls
echo.
echo ========================================
echo    Crypto Trading Platform - Scripts
echo ========================================
echo.
echo Available scripts:
echo.
echo [1]  Telegram Bot
echo [2]  Telegram Admin Panel
echo [3]  Telegram Screener Bot
echo [4]  Telegram Background Services
echo [5]  JSON to CSV Converter
echo [6]  Optimizer
echo [7]  Plotter
echo [8]  LSTM Optimizer
echo [9]  Exit
echo.
set /p choice="Select a script to run (1-9): "

if "%choice%"=="1" goto telegram_bot
if "%choice%"=="2" goto admin_panel
if "%choice%"=="3" goto screener_bot
if "%choice%"=="4" goto background_services
if "%choice%"=="5" goto json2csv
if "%choice%"=="6" goto optimizer
if "%choice%"=="7" goto plotter
if "%choice%"=="8" goto lstm_optimizer
if "%choice%"=="9" goto exit
echo Invalid choice. Please try again.
pause
goto menu

:telegram_bot
echo.
echo Starting Telegram Bot...
python src\frontend\telegram\bot.py
pause
goto menu

:admin_panel
echo.
echo Starting Telegram Admin Panel...
echo Admin panel will be available at http://localhost:5000
python src\frontend\telegram\screener\admin_panel.py
pause
goto menu

:screener_bot
echo.
echo Starting Telegram Screener Bot...
python src\frontend\telegram\bot.py
pause
goto menu

:background_services
echo.
echo Starting Telegram Background Services...
python src\frontend\telegram\screener\background_services.py
pause
goto menu

:json2csv
echo.
echo Starting JSON to CSV Converter...
python src\backtester\optimizer\run_json2csv.py
pause
goto menu

:optimizer
echo.
echo Starting Optimizer...
python src\backtester\optimizer\run_optimizer.py
pause
goto menu

:plotter
echo.
echo Starting Plotter...
python src\backtester\plotter\run_plotter.py
pause
goto menu

:lstm_optimizer
echo.
echo Starting LSTM Optimizer...
python src\ml\lstm\lstm_optuna_log_return_from_csv.py
pause
goto menu

:exit
echo.
echo Goodbye!
exit /b 0
