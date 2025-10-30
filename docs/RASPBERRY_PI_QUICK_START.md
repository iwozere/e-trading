# Raspberry Pi Quick Start Guide

## Prerequisites
- Raspberry Pi 3B+ or 4 (4GB+ RAM recommended)
- Raspberry Pi OS (Bullseye or newer)
- Internet connection

## Quick Setup (Automated)

### 1. Ensure Project is in Correct Location
```bash
# Verify project is in the expected location
ls -la /opt/apps/e-trading
```

### 2. Run the Setup Script
```bash
# Navigate to project directory
cd /opt/apps/e-trading

# Make script executable (if needed)
chmod +x bin/setup_raspberry_pi.sh

# Run the setup script
./bin/setup_raspberry_pi.sh
```

### 3. Configure Your Credentials
```bash
nano .env
```

Edit the `.env` file with your actual credentials:
```env
# Telegram Bot Token (get from @BotFather)
TELEGRAM_BOT_TOKEN=your_actual_bot_token

# Email Configuration
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# Admin Panel Credentials
WEBGUI_LOGIN=admin
WEBGUI_PASSWORD=your_secure_password

# FMP API Key (optional)
FMP_API_KEY=your_fmp_api_key

# Database
DATABASE_URL=sqlite:///trading_bot.db
```

### 4. Start the Services
```bash
# Start Telegram bot
sudo systemctl start trading-bot.service

# Start admin panel
sudo systemctl start trading-admin.service

# Check status
sudo systemctl status trading-bot.service
sudo systemctl status trading-admin.service
```

### 5. Access Your Services

**Telegram Bot:**
- Open Telegram and search for your bot
- Send `/start` to begin

**Admin Panel:**
- Open web browser
- Navigate to: `http://your-pi-ip:5000`
- Login with credentials from `.env` file

## Manual Setup (Alternative)

If you prefer manual setup, follow the detailed guide in `docs/RASPBERRY_PI_DEPLOYMENT.md`.

## Useful Commands

### Service Management
```bash
# Start services
sudo systemctl start trading-bot.service
sudo systemctl start trading-admin.service

# Stop services
sudo systemctl stop trading-bot.service
sudo systemctl stop trading-admin.service

# Restart services
sudo systemctl restart trading-bot.service
sudo systemctl restart trading-admin.service

# Check status
sudo systemctl status trading-bot.service
sudo systemctl status trading-admin.service

# Enable auto-start on boot
sudo systemctl enable trading-bot.service
sudo systemctl enable trading-admin.service
```

### View Logs
```bash
# View bot logs
sudo journalctl -u trading-bot.service -f

# View admin panel logs
sudo journalctl -u trading-admin.service -f

# View recent logs
sudo journalctl -u trading-bot.service --since "1 hour ago"
```

### Backup
```bash
# Run manual backup
/home/pi/backup-trading.sh

# Schedule daily backups (run once)
crontab -e
# Add this line: 0 2 * * * /home/pi/backup-trading.sh
```

### Troubleshooting
```bash
# Check if ports are in use
sudo netstat -tlnp | grep :5000
sudo netstat -tlnp | grep :8080

# Check disk space
df -h

# Check memory usage
free -h

# Check system resources
htop
```

## Security Checklist

- [ ] Changed default passwords in `.env`
- [ ] Set up firewall (done by setup script)
- [ ] Configured SSL certificate (optional)
- [ ] Set up regular backups
- [ ] Monitored logs regularly

## Performance Tips

1. **Disable unnecessary services:**
   ```bash
   sudo systemctl disable bluetooth
   sudo systemctl disable wolfram-engine
   ```

2. **Optimize memory:**
   ```bash
   # Check if swap is needed
   free -h
   ```

3. **Monitor system:**
   ```bash
   # Install monitoring tools
   sudo apt install htop iotop
   ```

## Getting Help

1. **Check logs first:**
   ```bash
   sudo journalctl -u trading-bot.service -n 50
   ```

2. **Verify configuration:**
   ```bash
   # Check if .env file exists and has correct format
   cat .env
   ```

3. **Test individual components:**
   ```bash
   # Test bot API
   curl http://localhost:8080/api/status
   
   # Test admin panel
   curl http://localhost:5000
   ```

## What's Running

After setup, your Raspberry Pi will have:

- **Telegram Bot** (Port 8080): Handles Telegram commands and API
- **Admin Panel** (Port 5000): Web interface for management
- **Nginx** (Port 80): Reverse proxy (optional)
- **Systemd Services**: Auto-restart on crashes and boot
- **Firewall**: UFW with necessary ports open
- **Log Rotation**: Automatic log management
- **Backup System**: Daily backups with retention

## Next Steps

1. **Test your bot** by sending commands in Telegram
2. **Access admin panel** to manage users and settings
3. **Set up alerts** for price monitoring
4. **Configure schedules** for automated reports
5. **Monitor performance** and logs regularly

Your Raspberry Pi is now running a production-ready trading framework!
