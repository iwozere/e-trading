# Notification Service Scripts

This directory contains scripts to run the E-Trading Notification Service (Database-Centric Bot).

## Overview

The notification service is a database-centric message delivery engine that:
- Polls the database for pending messages
- Delivers messages through channel plugins (Telegram, Email, SMS)
- Reports health status to the database
- Runs independently without REST endpoints

## Files

- `start_notification_bot.bat` - Windows batch file for foreground execution
- `start_notification_bot.sh` - Linux shell script for foreground/background execution
- `notification-bot.service` - systemd service file for running as a Linux service

---

## Windows Usage

### Running in Foreground

```cmd
cd c:\path\to\e-trading
bin\notification\start_notification_bot.bat
```

Press `Ctrl+C` to stop the service.

---

## Linux Usage

### Running in Foreground

```bash
cd /path/to/e-trading
./bin/notification/start_notification_bot.sh
```

Press `Ctrl+C` to stop the service.

### Running in Background

Start in background:
```bash
./bin/notification/start_notification_bot.sh --background
```

Stop background process:
```bash
./bin/notification/start_notification_bot.sh --stop
```

Check status:
```bash
./bin/notification/start_notification_bot.sh --status
```

View logs:
```bash
tail -f logs/notification_bot.log
```

### Command Options

- `(no option)` - Run in foreground (default)
- `--background`, `-b` - Run in background
- `--stop`, `-s` - Stop background process
- `--status` - Check if service is running
- `--help`, `-h` - Show help message

---

## Linux Service Installation

### 1. Edit the Service File

Edit `notification-bot.service` and update the following:

- `User=your-username` - Replace with your Linux username
- `Group=your-username` - Replace with your Linux group
- `WorkingDirectory=/path/to/e-trading` - Replace with actual project path
- `Environment="PYTHONPATH=/path/to/e-trading"` - Replace with actual project path
- `ReadWritePaths=/path/to/e-trading/logs /path/to/e-trading/data` - Replace with actual paths

**Optional**: Uncomment and configure:
- `EnvironmentFile=/path/to/e-trading/.env` - If you use a .env file
- `MemoryLimit=1G` - To limit memory usage
- `CPUQuota=50%` - To limit CPU usage

### 2. Install the Service

```bash
# Copy the service file to systemd directory
sudo cp bin/notification/notification-bot.service /etc/systemd/system/

# Reload systemd to recognize the new service
sudo systemctl daemon-reload

# Enable the service to start on boot
sudo systemctl enable notification-bot
```

### 3. Service Commands

Start the service:
```bash
sudo systemctl start notification-bot
```

Stop the service:
```bash
sudo systemctl stop notification-bot
```

Restart the service:
```bash
sudo systemctl restart notification-bot
```

Check status:
```bash
sudo systemctl status notification-bot
```

View logs:
```bash
# View all logs
sudo journalctl -u notification-bot

# Follow logs in real-time
sudo journalctl -u notification-bot -f

# View recent logs
sudo journalctl -u notification-bot -n 100

# View logs since boot
sudo journalctl -u notification-bot -b
```

Disable auto-start on boot:
```bash
sudo systemctl disable notification-bot
```

### 4. Uninstall the Service

```bash
# Stop and disable the service
sudo systemctl stop notification-bot
sudo systemctl disable notification-bot

# Remove the service file
sudo rm /etc/systemd/system/notification-bot.service

# Reload systemd
sudo systemctl daemon-reload
```

---

## Configuration

The notification service uses configuration from:
- `src/notification/service/config.py` - Service configuration
- Environment variables (if using `.env` file)
- Database settings from `src/data/db/services/database_service.py`

### Key Configuration Options

- **Poll Interval**: How often to check for pending messages (default: 5 seconds)
- **Health Check Interval**: How often to report health status (default: 60 seconds)
- **Channel Plugins**: Telegram, Email, SMS channel configurations

---

## Troubleshooting

### Service Won't Start

1. Check the Python path:
   ```bash
   which python3
   ```
   Update `ExecStart` in the service file if needed.

2. Check permissions:
   ```bash
   # Ensure logs directory exists and is writable
   mkdir -p logs
   chmod 755 logs
   ```

3. Check database connectivity:
   ```bash
   # Test database connection
   python3 -c "from src.data.db.services.database_service import get_database_service; get_database_service()"
   ```

### Service Keeps Restarting

Check the logs:
```bash
sudo journalctl -u notification-bot -n 50
```

Common issues:
- Database connection errors
- Missing environment variables
- Channel plugin configuration errors

### Background Process Won't Stop

```bash
# Find the process
ps aux | grep notification_db_centric_bot

# Force kill if needed
kill -9 <PID>

# Clean up PID file
rm -f logs/notification_bot.pid
```

---

## Monitoring

### Using systemd

```bash
# Watch service status
watch -n 2 'sudo systemctl status notification-bot'

# Monitor resource usage
sudo systemd-cgtop
```

### Using Logs

```bash
# Monitor application logs
tail -f logs/notification_bot.log

# Monitor system logs
sudo journalctl -u notification-bot -f

# Check for errors
sudo journalctl -u notification-bot -p err
```

### Database Health Monitoring

The service reports health status to the database:

```sql
-- Check service health
SELECT * FROM system_health
WHERE system = 'notification'
ORDER BY last_check_at DESC
LIMIT 10;

-- Check channel health
SELECT * FROM notification_channel_health
ORDER BY last_check_at DESC
LIMIT 10;
```

---

## Security Recommendations

1. **Run as non-root user**: The service file is configured to run as a regular user
2. **Limit permissions**: Use `ProtectSystem`, `ProtectHome`, and `ReadWritePaths`
3. **Resource limits**: Configure `MemoryLimit` and `CPUQuota` to prevent resource exhaustion
4. **Secure credentials**: Use environment files with proper permissions (600)
5. **Network isolation**: Consider using `PrivateNetwork=true` if the service only needs database access

---

## Support

For issues or questions:
1. Check the logs first
2. Review the service configuration
3. Test the notification bot manually before using as a service
4. Consult the main project documentation

---

*Last updated: 2025-11-30*
