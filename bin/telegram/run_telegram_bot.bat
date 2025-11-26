@echo off
REM Run the Telegram Screener Bot (Windows)

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..\\..

REM Activate virtual environment
call %PROJECT_ROOT%\.venv\Scripts\activate.bat

python %PROJECT_ROOT%\src\frontend\telegram\telegram_bot.py

 