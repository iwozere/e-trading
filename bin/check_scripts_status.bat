@echo off
REM ========================================
REM Crypto Trading Platform - Script Status Checker (Windows)
REM ========================================

setlocal enabledelayedexpansion

REM Get the absolute path to the project root
for %%i in ("%~dp0..") do set "PROJECT_ROOT=%%~fi"

echo 📊 Crypto Trading Platform - Script Status
echo ==========================================
echo.

REM Function to check if a process is running
:is_running
set "script_name=%~1"
set "pid_file=%PROJECT_ROOT%\logs\pids\%script_name%.pid"

if exist "!pid_file!" (
    for /f "usebackq delims=" %%i in ("!pid_file!") do set "pid=%%i"
    tasklist /FI "PID eq !pid!" 2>nul | find /I "!pid!" >nul
    if !errorlevel! equ 0 (
        exit /b 0
    ) else (
        del "!pid_file!" 2>nul
    )
)
exit /b 1

REM Function to show logs
:show_logs
set "script_name=%~1"
set "log_file=%PROJECT_ROOT%\logs\log\%script_name%.log"

if exist "!log_file!" (
    echo 📋 Recent logs for !script_name!:
    echo ----------------------------------------
    powershell -Command "Get-Content '!log_file!' | Select-Object -Last 10"
    echo ----------------------------------------
) else (
    echo ❌ No log file found for !script_name!
)
goto :eof

REM Function to stop a script
:stop_script
set "script_name=%~1"
set "pid_file=%PROJECT_ROOT%\logs\pids\%script_name%.pid"

if exist "!pid_file!" (
    for /f "usebackq delims=" %%i in ("!pid_file!") do set "pid=%%i"
    tasklist /FI "PID eq !pid!" 2>nul | find /I "!pid!" >nul
    if !errorlevel! equ 0 (
        echo 🛑 Stopping !script_name! (PID: !pid!)...
        taskkill /PID !pid! /F >nul 2>&1
        del "!pid_file!" 2>nul
        echo ✅ !script_name! stopped
    ) else (
        echo ⚠️  !script_name! was not running (stale PID file)
        del "!pid_file!" 2>nul
    )
) else (
    echo ❌ !script_name! is not running
)
goto :eof

REM Show status of all scripts
set "running_count=0"
set "total_count=0"

for %%s in (
    "telegram_bot:Telegram Bot"
    "admin_panel:Admin Panel"
    "background_services:Background Services"
    "json2csv:JSON to CSV Converter"
    "optimizer:Optimizer"
    "plotter:Plotter"
    "lstm_optimizer:LSTM Optimizer"
) do (
    for /f "tokens=1,2 delims=:" %%a in (%%s) do (
        set "script_name=%%a"
        set "display_name=%%b"
        
        call :is_running "!script_name!"
        if !errorlevel! equ 0 (
            for /f "usebackq delims=" %%i in ("%PROJECT_ROOT%\logs\pids\!script_name!.pid") do set "pid=%%i"
            echo 🟢 !display_name!: Running (PID: !pid!)
            set /a "running_count+=1"
        ) else (
            echo 🔴 !display_name!: Stopped
        )
        set /a "total_count+=1"
    )
)

echo.
echo Summary: !running_count!/!total_count! scripts running
echo.

REM Interactive menu
if "%1"=="--interactive" goto :interactive
if "%1"=="-i" goto :interactive
goto :eof

:interactive
echo Available actions:
echo [1] Show logs for a script
echo [2] Stop a script
echo [3] Stop all scripts
echo [4] Exit
echo.

set /p "choice=Select an action (1-4): "

if "!choice!"=="1" goto :show_logs_menu
if "!choice!"=="2" goto :stop_script_menu
if "!choice!"=="3" goto :stop_all
if "!choice!"=="4" goto :exit
echo Invalid choice
goto :interactive

:show_logs_menu
echo.
echo Select a script to show logs:
echo [1] Telegram Bot
echo [2] Admin Panel
echo [3] Background Services
echo [4] JSON to CSV Converter
echo [5] Optimizer
echo [6] Plotter
echo [7] LSTM Optimizer
echo.

set /p "script_choice=Select script (1-7): "

if "!script_choice!"=="1" call :show_logs "telegram_bot"
if "!script_choice!"=="2" call :show_logs "admin_panel"
if "!script_choice!"=="3" call :show_logs "background_services"
if "!script_choice!"=="4" call :show_logs "json2csv"
if "!script_choice!"=="5" call :show_logs "optimizer"
if "!script_choice!"=="6" call :show_logs "plotter"
if "!script_choice!"=="7" call :show_logs "lstm_optimizer"
goto :interactive

:stop_script_menu
echo.
echo Select a script to stop:
echo [1] Telegram Bot
echo [2] Admin Panel
echo [3] Background Services
echo [4] JSON to CSV Converter
echo [5] Optimizer
echo [6] Plotter
echo [7] LSTM Optimizer
echo.

set /p "script_choice=Select script (1-7): "

if "!script_choice!"=="1" call :stop_script "telegram_bot"
if "!script_choice!"=="2" call :stop_script "admin_panel"
if "!script_choice!"=="3" call :stop_script "background_services"
if "!script_choice!"=="4" call :stop_script "json2csv"
if "!script_choice!"=="5" call :stop_script "optimizer"
if "!script_choice!"=="6" call :stop_script "plotter"
if "!script_choice!"=="7" call :stop_script "lstm_optimizer"
goto :interactive

:stop_all
echo 🛑 Stopping all scripts...
call :stop_script "telegram_bot"
call :stop_script "admin_panel"
call :stop_script "background_services"
call :stop_script "json2csv"
call :stop_script "optimizer"
call :stop_script "plotter"
call :stop_script "lstm_optimizer"
echo ✅ All scripts stopped
goto :interactive

:exit
echo Goodbye!
exit /b 0
