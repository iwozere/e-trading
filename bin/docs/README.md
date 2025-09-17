# Script Launcher & Management System

This directory contains a comprehensive script management system for the E-Trading Platform, including interactive launchers, individual script runners, and systemd service scripts.

## Files Overview

### Interactive Launchers
- `run_script_launcher.bat` - Windows interactive script launcher
- `run_script_launcher.sh` - Linux/Mac interactive script launcher
- `check_scripts_status.bat` - Windows script status checker and manager
- `check_scripts_status.sh` - Linux/Mac script status checker and manager

### Individual Script Runners
- `run_telegram_bot.bat/.sh` - Main Telegram bot (background mode)
- `run_telegram_admin_panel.bat/.sh` - Admin panel (foreground mode)
- `run_telegram_screener_background.bat/.sh` - Background services (foreground mode)
- `run_json2csv.bat/.sh` - JSON to CSV converter
- `run_optimizer.bat/.sh` - Strategy optimizer
- `run_plotter.bat/.sh` - Chart plotter
- `run_lstm_optimizer.bat` - LSTM optimizer (Windows only)
- `run_hmm_lstm_backtest.bat/.sh` - HMM LSTM backtester

### Systemd Service Scripts (Linux)
- `svc_telegram_bot.sh` - Telegram bot service script
- `svc_telegram_admin_panel.sh` - Admin panel service script
- `svc_telegram_screener_background.sh` - Background services service script

### Setup & Installation
- `setup_raspberry_pi.sh` - Raspberry Pi setup script
- `install-ibkr/` - Interactive Brokers installation scripts

## Usage

### Interactive Script Launcher

**Windows:**
```cmd
bin\run_script_launcher.bat
```

**Linux/Mac:**
```bash
./bin/run_script_launcher.sh
```

### Script Status Management

**Quick status check:**
```bash
./bin/check_scripts_status.sh
```

**Interactive management:**
```bash
./bin/check_scripts_status.sh --interactive
```

## Available Scripts

The launcher provides access to the following scripts:

1. **Telegram Bot** - Main Telegram bot for user interactions
2. **Telegram Admin Panel** - Web-based admin interface (http://localhost:5000)
3. **Telegram Background Services** - Background alert and schedule processing
4. **JSON to CSV Converter** - Convert optimization results from JSON to CSV
5. **Optimizer** - Run strategy optimization
6. **Plotter** - Generate performance plots
7. **LSTM Optimizer** - Run LSTM model optimization

## Script Types & Execution Modes

### Development Scripts (Foreground)
These scripts run in the foreground and are suitable for development and testing:
- `run_telegram_admin_panel.sh` - Admin panel
- `run_telegram_screener_background.sh` - Background services
- All other individual scripts

### Background Scripts (Daemon Mode)
These scripts run in the background using `nohup`:
- `run_telegram_bot.sh` - Main bot (uses `nohup &`)

### Service Scripts (Systemd)
These scripts are designed for systemd service management:
- `svc_telegram_bot.sh` - Runs bot in foreground for systemd
- `svc_telegram_admin_panel.sh` - Runs admin panel in foreground for systemd
- `svc_telegram_screener_background.sh` - Runs background services in foreground for systemd

## Systemd Service Integration

The service scripts (`svc_*.sh`) are specifically designed for systemd integration:

### Features
- **Foreground execution**: Uses `exec` for proper systemd process management
- **Robust path resolution**: Handles symlinks and absolute paths correctly
- **Logging**: Outputs to both systemd journal and log files
- **Shell compatibility**: Uses `/bin/sh` for maximum compatibility

### Example Systemd Service File
```ini
[Unit]
Description=E-Trading Telegram Bot
After=network.target

[Service]
Type=exec
User=trading
WorkingDirectory=/opt/apps/e-trading
ExecStart=/opt/apps/e-trading/bin/svc_telegram_bot.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Features

### Virtual Environment Management
- Automatically activates the `.venv` environment
- Checks for virtual environment existence
- Provides helpful error messages for missing dependencies

### Process Management
- **Background execution**: Scripts run independently of SSH sessions
- **Duplicate prevention**: Prevents multiple instances of the same script
- **PID tracking**: Maintains process IDs in `logs/pids/` directory
- **Graceful shutdown**: Proper process termination

### Logging System
- **Individual log files**: Each script has its own log in `logs/log/`
- **Real-time monitoring**: View recent logs through the interface
- **Systemd integration**: Service scripts log to both journal and files

### Error Handling
- Comprehensive error checking and reporting
- Virtual environment validation
- Process state verification
- Automatic cleanup of stale PID files

## Requirements

- Python virtual environment (`.venv`) must be set up
- All dependencies must be installed (`pip install -r requirements.txt`)
- Proper file permissions (for shell scripts on Unix systems)
- For systemd services: systemd-compatible Linux distribution

## Troubleshooting

### Virtual Environment Not Found
If you get an error about the virtual environment not being found:
1. Create the virtual environment: `python -m venv .venv`
2. Install dependencies: `pip install -r requirements.txt`

### Permission Denied (Linux/Mac)
If you get a permission denied error on shell scripts:
```bash
chmod +x bin/*.sh
```

### Script Execution Fails
- Check that all dependencies are installed
- Verify that the virtual environment is properly set up
- Check the individual script logs for specific error messages
- For service scripts, check systemd journal: `journalctl -u your-service-name`

### Process Management Issues
- **Script won't start**: Check if already running, verify dependencies
- **Script won't stop**: Use interactive status manager or manual kill
- **Stale PID files**: System automatically cleans these up

## Individual Script Details

### Telegram Scripts
- **`run_telegram_bot.sh`**: Main bot with background execution (`nohup &`)
- **`run_telegram_admin_panel.sh`**: Web interface on port 5000
- **`run_telegram_screener_background.sh`**: Alert monitoring and scheduled tasks

### Analysis Scripts
- **`run_optimizer.sh`**: Strategy optimization with configurable parameters
- **`run_plotter.sh`**: Visual charts from optimization results
- **`run_json2csv.sh`**: Convert optimization JSON results to CSV format
- **`run_lstm_optimizer.bat`**: Machine learning model optimization (Windows)
- **`run_hmm_lstm_backtest.sh`**: Hidden Markov Model LSTM backtesting

### Service Scripts
- **`svc_telegram_bot.sh`**: Systemd-compatible bot service
- **`svc_telegram_admin_panel.sh`**: Systemd-compatible admin panel service
- **`svc_telegram_screener_background.sh`**: Systemd-compatible background services

## Environment Variables

The scripts automatically handle:
- **PYTHONPATH**: Adds project root to Python path
- **Virtual Environment**: Activates `.venv` with all dependencies
- **Working Directory**: Uses absolute paths for all execution

## Security Notes

- Scripts only activate the local `.venv` virtual environment
- No external network calls or installations during script execution
- All paths are absolute and relative to the project structure
- Error handling prevents execution with missing dependencies
- Service scripts run with specified user permissions in systemd

## Migration Notes

### From Old System
- The new system is backward compatible
- Individual scripts still work as before
- New features are additive (no breaking changes)
- Old log files are preserved

### To Systemd Services
- Use `svc_*.sh` scripts instead of `run_*.sh` for systemd
- Update service files to use the new service scripts
- Service scripts handle logging and process management automatically

## Best Practices

1. **Use the interactive launcher** for development and testing
2. **Use service scripts** for production systemd deployments
3. **Monitor logs regularly** to catch issues early
4. **Stop scripts gracefully** using the management interface
5. **Keep SSH sessions alive** with `tmux` or `screen` for long operations
6. **Use systemd services** for production deployments requiring auto-restart