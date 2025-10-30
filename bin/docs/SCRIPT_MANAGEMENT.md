# Script Management System

This document describes the improved script management system that allows running scripts in the background to avoid SSH session timeout issues.

## Overview

The script management system provides:
- **Background execution**: Scripts run independently of SSH sessions
- **Process management**: Start, stop, and monitor running scripts
- **Duplicate prevention**: Prevents multiple instances of the same script
- **Logging**: All script output is captured in log files
- **Status monitoring**: Real-time status of all scripts

## Files

### Main Scripts
- `run_script_launcher.sh` - Main interactive script launcher (Linux/macOS)
- `run_script_launcher.bat` - Main interactive script launcher (Windows)
- `check_scripts_status.sh` - Status checker and process manager (Linux/macOS)
- `check_scripts_status.bat` - Status checker and process manager (Windows)

### Individual Scripts
All individual scripts in the `bin/` folder can be run directly or through the launcher.

## Usage

### Starting the Script Launcher

**Linux/macOS:**
```bash
./bin/run_script_launcher.sh
```

**Windows:**
```cmd
bin\run_script_launcher.bat
```

### Checking Script Status

**Quick status check:**
```bash
./bin/check_scripts_status.sh
```

**Interactive status management:**
```bash
./bin/check_scripts_status.sh --interactive
```

**Windows:**
```cmd
bin\check_scripts_status.bat
bin\check_scripts_status.bat --interactive
```

## Features

### 1. Background Execution
- All scripts run using `nohup` (Linux) or background processes (Windows)
- Scripts continue running even if SSH session disconnects
- Each script has its own log file and PID tracking

### 2. Process Management
- **Start scripts**: Launch any script in the background
- **Stop scripts**: Gracefully terminate running scripts
- **Status monitoring**: See which scripts are running/stopped
- **Bulk operations**: Start/stop all scripts at once

### 3. Logging System
- All script output is captured in `logs/log/` directory
- Individual log files for each script (e.g., `telegram_bot.log`)
- PID files for process tracking in `logs/pids/` directory (e.g., `telegram_bot.pid`)
- Recent log viewing through the interface

### 4. Duplicate Prevention
- System checks if a script is already running before starting
- Prevents multiple instances of the same script
- Automatic cleanup of stale PID files

## Script Status Indicators

- 🟢 **Running**: Script is active and running in background
- 🔴 **Stopped**: Script is not currently running
- ❌ **Error**: Failed to start or encountered an error
- ⚠️ **Warning**: Stale PID file detected and cleaned up

## Log Files

All logs are stored in `logs/log/` with the following naming convention:
- `{script_name}.log` - Script output and error logs

PID files for process tracking are stored in `logs/pids/`:
- `{script_name}.pid` - Process ID for tracking

### Available Scripts

1. **Telegram Bot** (`telegram_bot`)
   - File: `src/frontend/telegram/bot.py`
   - Purpose: Main Telegram bot interface
   - Service script: `svc_telegram_bot.sh`

2. **Admin Panel** (`admin_panel`)
   - File: `src/frontend/telegram/screener/admin_panel.py`
   - Purpose: Web-based admin interface (port 5000)
   - Service script: `svc_telegram_admin_panel.sh`

3. **Background Services** (`background_services`)
   - File: `src/frontend/telegram/screener/background_services.py`
   - Purpose: Background monitoring, alerts, and scheduled services
   - Service script: `svc_telegram_screener_background.sh`

4. **JSON to CSV Converter** (`json2csv`)
   - File: `src/backtester/optimizer/run_json2csv.py`
   - Purpose: Data format conversion

5. **Optimizer** (`optimizer`)
   - File: `src/backtester/optimizer/run_optimizer.py`
   - Purpose: Strategy optimization

6. **Plotter** (`plotter`)
   - File: `src/backtester/plotter/run_plotter.py`
   - Purpose: Chart and visualization generation

7. **LSTM Optimizer** (`lstm_optimizer`)
   - File: `src/ml/lstm/lstm_optuna_log_return_from_csv.py`
   - Purpose: Machine learning optimization

8. **HMM LSTM Backtest** (`hmm_lstm_backtest`)
   - File: `src/ml/hmm_lstm/backtest.py`
   - Purpose: Hidden Markov Model LSTM backtesting

## Troubleshooting

### Script Won't Start
1. Check if virtual environment is activated
2. Verify Python dependencies are installed
3. Check log files for error messages
4. Ensure no other instance is running

### Script Won't Stop
1. Check if PID file exists
2. Verify process is actually running
3. Use `kill -9` as last resort (Linux)
4. Use Task Manager to force stop (Windows)

### Stale PID Files
- System automatically cleans up stale PID files
- Manual cleanup: Delete `logs/pids/*.pid` files

### Log File Issues
- Check disk space
- Verify write permissions to `logs/log/` directory
- Rotate large log files if needed

## Best Practices

1. **Always use the launcher** for starting scripts in production
2. **Monitor logs regularly** to catch issues early
3. **Stop scripts gracefully** using the management interface
4. **Check status before starting** to avoid duplicates
5. **Keep SSH sessions alive** with `tmux` or `screen` for long operations

## Security Considerations

- PID files contain process IDs only (not sensitive data)
- Log files may contain sensitive information
- Ensure proper file permissions on log directory
- Consider log rotation for long-running scripts
- Monitor disk usage for log files

## Systemd Service Integration

For production deployments, systemd service scripts are available for the main components:

### Service Scripts
- `svc_telegram_bot.sh` - Telegram bot service
- `svc_telegram_admin_panel.sh` - Admin panel service  
- `svc_telegram_screener_background.sh` - Background services

### Key Differences from Regular Scripts
- **Foreground execution**: Use `exec` instead of `nohup &` for proper systemd management
- **Shell compatibility**: Use `/bin/sh` instead of `/bin/bash` for better portability
- **Robust path resolution**: Handle symlinks and absolute paths correctly
- **Logging**: Output to both systemd journal and log files
- **Process management**: Designed for systemd's process supervision

### Example Systemd Service File
```ini
[Unit]
Description=E-Trading Telegram Bot
After=network.target

[Service]
Type=exec
User=trading
Group=trading
WorkingDirectory=/opt/apps/e-trading
ExecStart=/opt/apps/e-trading/bin/svc_telegram_bot.sh
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Service Management Commands
```bash
# Enable and start service
sudo systemctl enable e-trading-bot.service
sudo systemctl start e-trading-bot.service

# Check status
sudo systemctl status e-trading-bot.service

# View logs
journalctl -u e-trading-bot.service -f

# Stop service
sudo systemctl stop e-trading-bot.service
```

## Migration from Old System

If you were using the old script launcher:
1. The new system is backward compatible
2. Individual scripts still work as before
3. New features are additive (no breaking changes)
4. Old log files are preserved
5. Gradual migration is supported

### Migration to Systemd Services
1. Use `svc_*.sh` scripts for systemd service definitions
2. Update existing service files to use new service scripts
3. Service scripts automatically handle logging and process management
4. Test services in development before production deployment
