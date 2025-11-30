@echo off
REM Check Scheduler Service Status (Windows)
REM
REM Usage:
REM   bin\scheduler\status.bat

setlocal

REM Change to project root
cd /d "%~dp0\..\..\"

echo ========================================
echo Scheduler Service Status
echo ========================================
echo.

REM Find Python process running scheduler
set FOUND=0
for /f "tokens=2,*" %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FO LIST ^| findstr /C:"PID:"') do (
    wmic process where "ProcessId=%%a" get CommandLine 2>nul | findstr /C:"src.scheduler.main" >nul
    if not errorlevel 1 (
        set FOUND=1
        echo Status: RUNNING
        echo PID: %%a
        echo.
        echo Process Details:
        wmic process where "ProcessId=%%a" get ProcessId,CreationDate,WorkingSetSize,CommandLine
        echo.
        goto :checklog
    )
)

if %FOUND%==0 (
    echo Status: NOT RUNNING
    echo.
    echo Start with: bin\scheduler\start.bat
    goto :end
)

:checklog
REM Check log file
if exist logs\scheduler.log (
    echo Log File: logs\scheduler.log
    for %%A in (logs\scheduler.log) do echo Log Size: %%~zA bytes
    echo.
    echo Last 10 log lines:
    echo ----------------------------------------
    powershell -Command "Get-Content logs\scheduler.log -Tail 10"
    echo.
)

echo ========================================
echo Use 'bin\scheduler\stop.bat' to stop
echo ========================================

:end
pause
