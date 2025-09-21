@echo off
echo Starting Trading Web UI - Backend and Frontend
echo ================================================

echo Starting Backend API on port 5003...
start "Trading API Backend" cmd /k "python start_trading_webui.py --backend-only"

echo Waiting 5 seconds for backend to start...
timeout /t 5 /nobreak > nul

echo Starting Frontend on port 5002...
start "Trading Web UI Frontend" cmd /k "python start_trading_webui.py --frontend-only"

echo.
echo Both services are starting...
echo Backend API: http://localhost:5003
echo Frontend UI: http://localhost:5002
echo API Docs: http://localhost:5003/docs
echo.
echo Press any key to exit...
pause > nul