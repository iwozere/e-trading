@echo off
REM Stop the Scheduler Service (Windows)
REM
REM Usage:
REM   bin\scheduler\stop.bat

setlocal

REM Change to project root
cd /d "%~dp0\..\..\"

echo ========================================
echo Stopping Scheduler Service
echo ========================================

REM Find Python process running scheduler
for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FO LIST ^| findstr /C:"PID:"') do (
    wmic process where "ProcessId=%%a" get CommandLine 2>nul | findstr /C:"src.scheduler.main" >nul
    if not errorlevel 1 (
        echo Found scheduler process: %%a
        echo Stopping process...
        taskkill /PID %%a /F
        echo Process stopped
        goto :cleanup
    )
)

echo WARNING: No scheduler process found
echo.

:cleanup
REM Clean up PID file if exists
if exist scheduler.pid (
    del /Q scheduler.pid
)

echo.
echo Scheduler service stopped
pause
