# Raspberry Pi Deployment Guide

This guide will help you deploy the Advanced Trading Framework's Telegram bot and admin panel on your Raspberry Pi.

## Prerequisites

- Raspberry Pi (3B+ or 4 recommended)
- Raspberry Pi OS (Bullseye or newer)
- At least 4GB RAM (8GB recommended)
- At least 16GB SD card
- Internet connection
- Python 3.9+ (included with Raspberry Pi OS)

## Step 1: Initial Setup

### 1.1 Update Raspberry Pi OS
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git curl wget
```

### 1.2 Install System Dependencies
```bash
# Install required system packages
sudo apt install -y build-essential python3-dev libffi-dev libssl-dev
sudo apt install -y libatlas-base-dev gfortran libopenblas-dev liblapack-dev
sudo apt install -y libhdf5-dev libhdf5-serial-dev libhdf5-103 libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5
sudo apt install -y libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio
```

### 1.3 Install TA-Lib (Technical Analysis Library)
```bash
# Download and install TA-Lib
cd /tmp
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
sudo ldconfig
```

## Step 2: Clone and Setup Project

### 2.1 Verify Project Location
```bash
# Ensure project is in the expected location
ls -la /opt/apps/e-trading

# Navigate to project directory
cd /opt/apps/e-trading
```

### 2.2 Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2.3 Install Python Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install additional dependencies for Raspberry Pi
pip install aiohttp flask gunicorn
```

## Step 3: Configuration

### 3.1 Create Configuration Files
```bash
# Create config directory
mkdir -p config/donotshare

# Create configuration file
nano config/donotshare/donotshare.py
```

### 3.2 Add Configuration Content
```python
# config/donotshare/donotshare.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

# Email Configuration (for notifications)
SMTP_USER = os.getenv('SMTP_USER')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')

# Admin Panel Configuration
WEBGUI_LOGIN = os.getenv('WEBGUI_LOGIN', 'admin')
WEBGUI_PASSWORD = os.getenv('WEBGUI_PASSWORD', 'your-secure-password')

# FMP API Key (for financial data)
FMP_API_KEY = os.getenv('FMP_API_KEY')

# Database Configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///trading_bot.db')
```

### 3.3 Create Environment File
```bash
# Create .env file
nano .env
```

Add the following content:
```env
# Telegram Bot Token (get from @BotFather)
TELEGRAM_BOT_TOKEN=your_bot_token_here

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

## Step 4: Create Systemd Services

### 4.1 Create Telegram Bot Service
```bash
sudo nano /etc/systemd/system/trading-bot.service
```

Add the following content:
```ini
[Unit]
Description=Trading Framework Telegram Bot
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/opt/apps/e-trading
Environment=PATH=/opt/apps/e-trading/venv/bin
ExecStart=/opt/apps/e-trading/venv/bin/python src/frontend/telegram/bot.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### 4.2 Create Admin Panel Service
```bash
sudo nano /etc/systemd/system/trading-admin.service
```

Add the following content:
```ini
[Unit]
Description=Trading Framework Admin Panel
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/opt/apps/e-trading
Environment=PATH=/opt/apps/e-trading/venv/bin
ExecStart=/opt/apps/e-trading/venv/bin/python src/frontend/telegram/screener/admin_panel.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### 4.3 Enable and Start Services
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable services to start on boot
sudo systemctl enable trading-bot.service
sudo systemctl enable trading-admin.service

# Start services
sudo systemctl start trading-bot.service
sudo systemctl start trading-admin.service

# Check status
sudo systemctl status trading-bot.service
sudo systemctl status trading-admin.service
```

## Step 5: Firewall Configuration

### 5.1 Configure UFW Firewall
```bash
# Install UFW if not installed
sudo apt install ufw

# Allow SSH
sudo ufw allow ssh

# Allow admin panel port
sudo ufw allow 5000

# Allow bot API port
sudo ufw allow 8080

# Enable firewall
sudo ufw enable
```

## Step 6: Nginx Reverse Proxy (Optional but Recommended)

### 6.1 Install Nginx
```bash
sudo apt install nginx
```

### 6.2 Create Nginx Configuration
```bash
sudo nano /etc/nginx/sites-available/trading-admin
```

Add the following content:
```nginx
server {
    listen 80;
    server_name your-pi-ip-or-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 6.3 Enable Nginx Site
```bash
sudo ln -s /etc/nginx/sites-available/trading-admin /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Step 7: SSL Certificate (Optional but Recommended)

### 7.1 Install Certbot
```bash
sudo apt install certbot python3-certbot-nginx
```

### 7.2 Get SSL Certificate
```bash
sudo certbot --nginx -d your-domain.com
```

## Step 8: Monitoring and Logs

### 8.1 View Service Logs
```bash
# View bot logs
sudo journalctl -u trading-bot.service -f

# View admin panel logs
sudo journalctl -u trading-admin.service -f

# View recent logs
sudo journalctl -u trading-bot.service --since "1 hour ago"
```

### 8.2 Create Log Rotation
```bash
sudo nano /etc/logrotate.d/trading-bot
```

Add the following content:
```
/opt/apps/e-trading/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 pi pi
}
```

## Step 9: Performance Optimization

### 9.1 Disable Unnecessary Services
```bash
# Disable Bluetooth if not needed
sudo systemctl disable bluetooth

# Disable WiFi power management
sudo iwconfig wlan0 power off
```

### 9.2 Configure Swap (if needed)
```bash
# Check current swap
free -h

# Create swap file if needed
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make swap permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## Step 10: Testing

### 10.1 Test Telegram Bot
1. Open Telegram and search for your bot
2. Send `/start` command
3. Test basic commands like `/help`, `/info`

### 10.2 Test Admin Panel
1. Open web browser
2. Navigate to `http://your-pi-ip:5000` or `http://your-domain.com`
3. Login with admin credentials
4. Test dashboard and user management

### 10.3 Test API Endpoints
```bash
# Test bot API health
curl http://localhost:8080/api/status

# Test broadcast message
curl -X POST http://localhost:8080/api/broadcast \
  -H "Content-Type: application/json" \
  -d '{"message": "Test broadcast message", "title": "Test"}'
```

## Troubleshooting

### Common Issues

1. **Service won't start**
   ```bash
   # Check logs
   sudo journalctl -u trading-bot.service -n 50
   
   # Check permissions
   sudo chown -R pi:pi /home/pi/e-trading
   ```

2. **Port already in use**
   ```bash
   # Check what's using the port
   sudo netstat -tlnp | grep :5000
   
   # Kill process if needed
   sudo kill -9 <PID>
   ```

3. **Memory issues**
   ```bash
   # Monitor memory usage
   htop
   
   # Check swap usage
   free -h
   ```

4. **Database issues**
   ```bash
   # Check database file permissions
   ls -la /home/pi/e-trading/*.db
   
   # Reset database if needed
   rm /home/pi/e-trading/trading_bot.db
   ```

### Performance Monitoring

```bash
# Monitor system resources
htop

# Monitor disk usage
df -h

# Monitor network connections
sudo netstat -tlnp

# Monitor service status
sudo systemctl status trading-bot.service trading-admin.service
```

## Security Considerations

1. **Change default passwords**
2. **Use strong admin credentials**
3. **Keep system updated**
4. **Monitor logs regularly**
5. **Use HTTPS in production**
6. **Restrict access to admin panel**
7. **Regular backups**

## Backup Strategy

### 9.1 Create Backup Script
```bash
nano /home/pi/backup-trading.sh
```

Add the following content:
```bash
#!/bin/bash
BACKUP_DIR="/home/pi/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup database
cp /opt/apps/e-trading/trading_bot.db $BACKUP_DIR/trading_bot_$DATE.db

# Backup configuration
tar -czf $BACKUP_DIR/config_$DATE.tar.gz /opt/apps/e-trading/config/

# Backup logs
tar -czf $BACKUP_DIR/logs_$DATE.tar.gz /opt/apps/e-trading/logs/

# Keep only last 7 days of backups
find $BACKUP_DIR -name "*.db" -mtime +7 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed: $DATE"
```

### 9.2 Make Backup Script Executable
```bash
chmod +x /home/pi/backup-trading.sh
```

### 9.3 Schedule Daily Backups
```bash
crontab -e
```

Add the following line:
```
0 2 * * * /home/pi/backup-trading.sh
```

## Conclusion

Your Raspberry Pi is now running the Advanced Trading Framework with:
- Telegram bot accessible via Telegram
- Admin panel accessible via web browser
- Automatic startup on boot
- Log monitoring and rotation
- Backup system
- Security configurations

The system will automatically restart if it crashes and will start on boot. Monitor the logs regularly to ensure everything is running smoothly.
