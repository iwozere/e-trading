@echo off
REM HMM-LSTM Backtesting Runner for Windows
REM This script runs the HMM-LSTM backtesting optimizer

echo Starting HMM-LSTM Backtesting...
echo.

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..\..

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

REM Set Python path and run the optimizer
python src/backtester/optimizer/hmm_lstm.py %*

echo.
echo HMM-LSTM Backtesting completed.
pause
