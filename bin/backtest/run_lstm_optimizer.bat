@echo off
REM ========================================
REM Crypto Trading Platform - Optimizer Runner
REM ========================================

echo Starting Crypto Trading Optimizer...

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..\\..

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

REM Run the optimizer script
echo Running optimizer...
python src\ml\lstm\lstm_optuna_log_return_from_csv.py

REM Check if script ran successfully
if errorlevel 1 (
    echo Error: Optimizer script failed
    pause
    exit /b 1
)

echo Optimizer completed successfully!
pause
