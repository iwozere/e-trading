# Systemd Service Integration

This document describes the systemd service scripts and how to deploy the E-Trading Platform as managed system services.

## Overview

The E-Trading Platform includes systemd-compatible service scripts that enable running the application components as managed system services with automatic restart, logging, and process supervision.

## Service Scripts

### Available Service Scripts

| Service Script | Target Application | Purpose |
|----------------|-------------------|---------|
| `svc_telegram_bot.sh` | `src/frontend/telegram/bot.py` | Main Telegram bot interface |
| `svc_telegram_admin_panel.sh` | `src/frontend/telegram/screener/admin_panel.py` | Web admin interface (port 5000) |
| `svc_telegram_screener_background.sh` | `src/frontend/telegram/screener/background_services.py` | Background alerts and schedules |

### Key Features

- **Foreground execution**: Uses `exec` for proper systemd process management
- **Robust path resolution**: Handles symlinks and works from any installation path
- **Shell compatibility**: Uses `/bin/sh` for maximum portability
- **Dual logging**: Outputs to both systemd journal and log files
- **Virtual environment**: Automatically activates the project's `.venv`
- **Error handling**: Comprehensive error checking and reporting

## Service File Examples

### Telegram Bot Service

Create `/etc/systemd/system/e-trading-bot.service`:

```ini
[Unit]
Description=E-Trading Telegram Bot
Documentation=https://github.com/your-org/e-trading
After=network.target
Wants=network.target

[Service]
Type=exec
User=trading
Group=trading
WorkingDirectory=/opt/apps/e-trading
ExecStart=/opt/apps/e-trading/bin/svc_telegram_bot.sh
Restart=always
RestartSec=10
TimeoutStartSec=30
TimeoutStopSec=30

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=e-trading-bot

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/opt/apps/e-trading/logs

# Environment
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

### Admin Panel Service

Create `/etc/systemd/system/e-trading-admin.service`:

```ini
[Unit]
Description=E-Trading Admin Panel
Documentation=https://github.com/your-org/e-trading
After=network.target
Wants=network.target

[Service]
Type=exec
User=trading
Group=trading
WorkingDirectory=/opt/apps/e-trading
ExecStart=/opt/apps/e-trading/bin/svc_telegram_admin_panel.sh
Restart=always
RestartSec=10
TimeoutStartSec=30
TimeoutStopSec=30

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=e-trading-admin

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/opt/apps/e-trading/logs

# Environment
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

### Background Services

Create `/etc/systemd/system/e-trading-background.service`:

```ini
[Unit]
Description=E-Trading Background Services
Documentation=https://github.com/your-org/e-trading
After=network.target
Wants=network.target

[Service]
Type=exec
User=trading
Group=trading
WorkingDirectory=/opt/apps/e-trading
ExecStart=/opt/apps/e-trading/bin/svc_telegram_screener_background.sh
Restart=always
RestartSec=10
TimeoutStartSec=30
TimeoutStopSec=30

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=e-trading-background

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/opt/apps/e-trading/logs

# Environment
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

## Installation & Setup

### 1. System User Setup

Create a dedicated user for the trading application:

```bash
# Create user and group
sudo useradd -r -s /bin/false -d /opt/apps/e-trading trading
sudo usermod -a -G trading trading

# Set up directory permissions
sudo chown -R trading:trading /opt/apps/e-trading
sudo chmod -R 755 /opt/apps/e-trading
sudo chmod +x /opt/apps/e-trading/bin/svc_*.sh
```

### 2. Service Installation

```bash
# Copy service files to systemd directory
sudo cp /path/to/your/service/files/*.service /etc/systemd/system/

# Reload systemd configuration
sudo systemctl daemon-reload

# Enable services (auto-start on boot)
sudo systemctl enable e-trading-bot.service
sudo systemctl enable e-trading-admin.service
sudo systemctl enable e-trading-background.service
```

### 3. Service Management

```bash
# Start all services
sudo systemctl start e-trading-bot.service
sudo systemctl start e-trading-admin.service
sudo systemctl start e-trading-background.service

# Check status
sudo systemctl status e-trading-bot.service
sudo systemctl status e-trading-admin.service
sudo systemctl status e-trading-background.service

# Stop services
sudo systemctl stop e-trading-bot.service
sudo systemctl stop e-trading-admin.service
sudo systemctl stop e-trading-background.service

# Restart services
sudo systemctl restart e-trading-bot.service
```

## Logging & Monitoring

### Systemd Journal Logs

```bash
# View real-time logs
journalctl -u e-trading-bot.service -f
journalctl -u e-trading-admin.service -f
journalctl -u e-trading-background.service -f

# View recent logs
journalctl -u e-trading-bot.service -n 50
journalctl -u e-trading-admin.service -n 50
journalctl -u e-trading-background.service -n 50

# View logs since specific time
journalctl -u e-trading-bot.service --since "2024-01-01 00:00:00"
```

### Application Log Files

Service scripts also write to application log files:

```bash
# View application logs
tail -f /opt/apps/e-trading/logs/log/telegram_bot.out
tail -f /opt/apps/e-trading/logs/log/telegram_admin_panel.out
tail -f /opt/apps/e-trading/logs/log/telegram_screener_background.out
```

### Log Rotation

Configure logrotate for application logs:

Create `/etc/logrotate.d/e-trading`:

```
/opt/apps/e-trading/logs/log/*.out {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 trading trading
    postrotate
        systemctl reload e-trading-bot.service
        systemctl reload e-trading-admin.service
        systemctl reload e-trading-background.service
    endscript
}
```

## Troubleshooting

### Service Won't Start

1. **Check service status**:
   ```bash
   sudo systemctl status e-trading-bot.service
   ```

2. **Check journal logs**:
   ```bash
   journalctl -u e-trading-bot.service -n 20
   ```

3. **Test script manually**:
   ```bash
   sudo -u trading /opt/apps/e-trading/bin/svc_telegram_bot.sh
   ```

4. **Check permissions**:
   ```bash
   ls -la /opt/apps/e-trading/bin/svc_*.sh
   ls -la /opt/apps/e-trading/.venv/
   ```

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Permission denied | Incorrect file permissions | `chmod +x bin/svc_*.sh` |
| Virtual environment not found | Missing or incorrect .venv | Recreate virtual environment |
| Service fails to start | Missing dependencies | Check application logs |
| Port already in use | Another instance running | Stop conflicting services |

### Performance Monitoring

```bash
# Monitor resource usage
systemctl show e-trading-bot.service --property=MainPID
ps -p $(systemctl show e-trading-bot.service --property=MainPID --value) -o pid,ppid,cmd,%mem,%cpu

# Monitor service restarts
journalctl -u e-trading-bot.service | grep -i restart
```

## Security Considerations

### Service Security Features

- **User isolation**: Services run as dedicated `trading` user
- **Filesystem protection**: `ProtectSystem=strict` prevents system modifications
- **Privilege restriction**: `NoNewPrivileges=true` prevents privilege escalation
- **Temporary filesystem**: `PrivateTmp=true` provides isolated /tmp

### Additional Security Measures

1. **Firewall configuration**: Restrict access to admin panel port
2. **Log monitoring**: Monitor for suspicious activity
3. **Regular updates**: Keep dependencies updated
4. **Backup strategy**: Regular backups of configuration and data

## Best Practices

1. **Test in development**: Always test service scripts before production
2. **Monitor logs**: Set up log monitoring and alerting
3. **Resource limits**: Configure memory and CPU limits if needed
4. **Health checks**: Implement application health checks
5. **Graceful shutdown**: Ensure applications handle SIGTERM properly
6. **Documentation**: Keep service documentation updated

## Migration from Manual Scripts

### From Background Scripts

Replace manual `nohup` execution:

```bash
# Old way
nohup ./bin/run_telegram_bot.sh &

# New way
sudo systemctl start e-trading-bot.service
```

### From Screen/Tmux Sessions

Replace screen/tmux sessions with systemd services:

```bash
# Old way
screen -S telegram-bot ./bin/run_telegram_bot.sh

# New way
sudo systemctl start e-trading-bot.service
sudo systemctl enable e-trading-bot.service  # Auto-start on boot
```

### Benefits of Migration

- **Automatic restart**: Services restart on failure
- **Boot integration**: Services start automatically on system boot
- **Better logging**: Integrated with systemd journal
- **Resource management**: Better process and resource control
- **Security**: Enhanced security through systemd features