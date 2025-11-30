# Telegram Bot Scripts

This directory contains scripts to run the E-Trading Telegram Bot.

## Overview

The Telegram bot provides:
- Stock screener alerts and notifications
- Interactive commands for market data and reports
- User registration and authentication
- Admin panel for managing users and settings
- Integration with the notification service

## Files

- `start_telegram_bot.bat` - Windows batch file for foreground execution
- `start_telegram_bot.sh` - Linux shell script for foreground/background execution
- `telegram-bot.service` - systemd service file for running as a Linux service

---

## Windows Usage

### Running in Foreground

```cmd
cd c:\path\to\e-trading
bin\telegram\start_telegram_bot.bat
```

Press `Ctrl+C` to stop the bot.

---

## Linux Usage

### Running in Foreground

```bash
cd /path/to/e-trading
./bin/telegram/start_telegram_bot.sh
```

Press `Ctrl+C` to stop the bot.

### Running in Background

Start in background:
```bash
./bin/telegram/start_telegram_bot.sh --background
```

Stop background process:
```bash
./bin/telegram/start_telegram_bot.sh --stop
```

Check status:
```bash
./bin/telegram/start_telegram_bot.sh --status
```

View logs:
```bash
tail -f logs/telegram_bot.log
```

### Command Options

- `(no option)` - Run in foreground (default)
- `--background`, `-b` - Run in background
- `--stop`, `-s` - Stop background process
- `--status` - Check if bot is running
- `--help`, `-h` - Show help message

---

## Linux Service Installation

### 1. Edit the Service File

Edit `telegram-bot.service` and update the following if needed:

- `User=alkotrader` - Replace with your Linux username (if different)
- `Group=alkotrader` - Replace with your Linux group (if different)
- `WorkingDirectory=/opt/apps/e-trading` - Verify project path
- `Environment="PYTHONPATH=/opt/apps/e-trading"` - Verify project path
- `EnvironmentFile=/opt/apps/e-trading/config/donotshare/.env` - Verify .env path
- `ReadWritePaths=/opt/apps/e-trading/logs /opt/apps/e-trading/data` - Verify paths

**Important**: The service file requires the `.env` file with your `TELEGRAM_BOT_TOKEN`. Make sure:
- The `.env` file exists at the specified path
- It contains: `TELEGRAM_BOT_TOKEN=your_token_here`
- The file has proper permissions (600): `chmod 600 /opt/apps/e-trading/config/donotshare/.env`

**Optional**: Uncomment and configure:
- `MemoryLimit=1G` - To limit memory usage
- `CPUQuota=50%` - To limit CPU usage

### 2. Install the Service

```bash
# Copy the service file to systemd directory
sudo cp bin/telegram/telegram-bot.service /etc/systemd/system/

# Reload systemd to recognize the new service
sudo systemctl daemon-reload

# Enable the service to start on boot
sudo systemctl enable telegram-bot
```

### 3. Service Commands

Start the service:
```bash
sudo systemctl start telegram-bot
```

Stop the service:
```bash
sudo systemctl stop telegram-bot
```

Restart the service:
```bash
sudo systemctl restart telegram-bot
```

Check status:
```bash
sudo systemctl status telegram-bot
```

View logs:
```bash
# View all logs
sudo journalctl -u telegram-bot

# Follow logs in real-time
sudo journalctl -u telegram-bot -f

# View recent logs
sudo journalctl -u telegram-bot -n 100

# View logs since boot
sudo journalctl -u telegram-bot -b
```

Disable auto-start on boot:
```bash
sudo systemctl disable telegram-bot
```

### 4. Uninstall the Service

```bash
# Stop and disable the service
sudo systemctl stop telegram-bot
sudo systemctl disable telegram-bot

# Remove the service file
sudo rm /etc/systemd/system/telegram-bot.service

# Reload systemd
sudo systemctl daemon-reload
```

---

## Configuration

The Telegram bot requires configuration:

### Required Environment Variables

Create or edit `config/donotshare/.env`:

```bash
# Telegram Bot Token (required)
TELEGRAM_BOT_TOKEN=your_bot_token_from_botfather

# Optional: Database connection (if not using defaults)
# DATABASE_URL=postgresql://user:password@localhost/etrading
```

### Getting a Telegram Bot Token

1. Open Telegram and search for [@BotFather](https://t.me/botfather)
2. Send `/newbot` command
3. Follow the instructions to create your bot
4. Copy the token provided by BotFather
5. Add it to your `.env` file

### File Permissions

Ensure the `.env` file is secure:
```bash
chmod 600 /opt/apps/e-trading/config/donotshare/.env
```

---

## Bot Commands

Once running, users can interact with the bot using these commands:

- `/start` - Register and start using the bot
- `/info` - View your account information
- `/help` - List available commands
- `/screener` - Run stock screener and get alerts
- `/report` - Get market analysis reports

Admin users have additional commands for user management and configuration.

---

## Troubleshooting

### Bot Won't Start

1. **Check the Telegram token**:
   ```bash
   # Verify .env file exists and contains token
   cat /opt/apps/e-trading/config/donotshare/.env | grep TELEGRAM_BOT_TOKEN
   ```

2. **Check Python path**:
   ```bash
   which python3
   ```
   Update `ExecStart` in the service file if needed.

3. **Check permissions**:
   ```bash
   # Ensure logs directory exists and is writable
   mkdir -p logs
   chmod 755 logs
   ```

4. **Test bot manually**:
   ```bash
   cd /opt/apps/e-trading
   python3 -m src.telegram.telegram_bot
   ```

### Bot Not Responding to Commands

1. **Check if bot is running**:
   ```bash
   sudo systemctl status telegram-bot
   # or
   ./bin/telegram/start_telegram_bot.sh --status
   ```

2. **Check logs for errors**:
   ```bash
   sudo journalctl -u telegram-bot -n 50
   # or
   tail -f logs/telegram_bot.log
   ```

3. **Verify database connectivity**:
   ```bash
   python3 -c "from src.data.db.services.database_service import get_database_service; get_database_service()"
   ```

### Service Keeps Restarting

Check the logs:
```bash
sudo journalctl -u telegram-bot -n 50
```

Common issues:
- Invalid or missing Telegram bot token
- Database connection errors
- Network connectivity issues
- Missing Python dependencies

### Background Process Won't Stop

```bash
# Find the process
ps aux | grep telegram_bot

# Force kill if needed
kill -9 <PID>

# Clean up PID file
rm -f logs/telegram_bot.pid
```

---

## Integration with Notification Service

The Telegram bot integrates with the Notification Service for sending alerts:

- Bot commands can trigger notifications
- Screener alerts are sent through the notification service
- Reports and updates are delivered via Telegram channel

To use both services together:

1. Start the Notification Service:
   ```bash
   ./bin/notification/start_notification_bot.sh --background
   ```

2. Start the Telegram Bot:
   ```bash
   ./bin/telegram/start_telegram_bot.sh --background
   ```

Or as systemd services:
```bash
sudo systemctl start notification-bot
sudo systemctl start telegram-bot
```

---

## Monitoring

### Using systemd

```bash
# Watch service status
watch -n 2 'sudo systemctl status telegram-bot'

# Monitor resource usage
sudo systemd-cgtop
```

### Using Logs

```bash
# Monitor application logs
tail -f logs/telegram_bot.log

# Monitor system logs
sudo journalctl -u telegram-bot -f

# Check for errors
sudo journalctl -u telegram-bot -p err
```

### Database Monitoring

Check bot activity in the database:

```sql
-- Check user registrations
SELECT * FROM telegram_users
ORDER BY created_at DESC
LIMIT 10;

-- Check sent messages
SELECT * FROM notification_messages
WHERE channel = 'telegram'
ORDER BY created_at DESC
LIMIT 10;
```

---

## Security Recommendations

1. **Protect your bot token**: Never commit `.env` files to git
2. **Use file permissions**: Keep `.env` file permissions at 600
3. **Run as non-root**: The service file is configured to run as `alkotrader` user
4. **Limit permissions**: Use `ProtectSystem`, `ProtectHome`, and `ReadWritePaths`
5. **Resource limits**: Configure `MemoryLimit` and `CPUQuota` to prevent resource exhaustion
6. **Monitor logs**: Regularly check logs for suspicious activity
7. **User verification**: Enable user approval system to control who can use the bot

---

## Development

### Running in Development Mode

For development, run in foreground to see all logs:

```bash
cd /opt/apps/e-trading
python3 -m src.telegram.telegram_bot
```

### Testing

Before deploying as a service, test the bot:

1. Run in foreground
2. Send `/start` command to the bot in Telegram
3. Test various commands
4. Check logs for errors
5. Verify database interactions

---

## Support

For issues or questions:
1. Check the logs first (application and system logs)
2. Review the Telegram bot token configuration
3. Test the bot manually before using as a service
4. Check database connectivity
5. Consult the main project documentation

---

## Related Services

- **Notification Service**: [bin/notification/](../notification/) - Database-centric message delivery
- **API Service**: [bin/api/](../api/) - REST API backend
- **Web UI**: [bin/web_ui/](../web_ui/) - Web interface

---

*Last updated: 2025-11-30*
