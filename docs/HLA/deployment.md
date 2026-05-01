# Deployment Guide — Raspberry Pi

**Target Platform:** Raspberry Pi 3B+ or 4  
**OS:** Raspberry Pi OS Bullseye or newer  
**Project Path:** `/opt/apps/e-trading`

## Related Documentation
- **[Infrastructure Module](modules/infrastructure.md)** — Database, scheduling, error handling
- **[Configuration](modules/configuration.md)** — Environment and config management
- **[Documentation Index](INDEX.md)** — Full documentation guide

---

## Quick Start (Automated)

For a fresh Raspberry Pi with the project already cloned to `/opt/apps/e-trading`:

```bash
# 1. Verify project location
ls -la /opt/apps/e-trading

# 2. Run the automated setup script
cd /opt/apps/e-trading
chmod +x bin/setup_raspberry_pi.sh
./bin/setup_raspberry_pi.sh

# 3. Configure credentials
nano .env

# 4. Start services
sudo systemctl start trading-bot.service
sudo systemctl start trading-admin.service

# 5. Verify
sudo systemctl status trading-bot.service
sudo systemctl status trading-admin.service
```

After setup, the Telegram bot is reachable via Telegram (`/start`), and the admin panel at `http://<pi-ip>:5002`.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Model | Raspberry Pi 3B+ | Raspberry Pi 4 |
| RAM | 4 GB | 8 GB |
| SD Card | 16 GB | 32 GB (Class 10 / A1) |
| OS | Raspberry Pi OS Bullseye | Raspberry Pi OS Bookworm |
| Python | 3.9+ | 3.11+ |

---

## Full Deployment Procedure

### Step 1 — OS Packages

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git curl wget
sudo apt install -y build-essential python3-dev libffi-dev libssl-dev
sudo apt install -y libatlas-base-dev gfortran libopenblas-dev liblapack-dev
```

### Step 2 — TA-Lib (C Library)

TA-Lib must be compiled from source on ARM:

```bash
cd /tmp
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
sudo ldconfig
```

### Step 3 — Python Virtual Environment

```bash
cd /opt/apps/e-trading
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install aiohttp flask gunicorn
```

### Step 4 — Configuration

#### 4.1 Create the config directory

```bash
mkdir -p config/donotshare
nano config/donotshare/donotshare.py
```

Minimal `donotshare.py`:

```python
import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
SMTP_USER = os.getenv('SMTP_USER')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
WEBGUI_LOGIN = os.getenv('WEBGUI_LOGIN', 'admin')
WEBGUI_PASSWORD = os.getenv('WEBGUI_PASSWORD', 'change-me')
FMP_API_KEY = os.getenv('FMP_API_KEY')
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///trading_bot.db')
```

#### 4.2 Create the `.env` file

```bash
nano .env
```

```env
# Telegram Bot Token — obtain from @BotFather
TELEGRAM_BOT_TOKEN=your_bot_token_here

# Email (optional, for SMTP notifications)
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# Admin panel credentials
WEBGUI_LOGIN=admin
WEBGUI_PASSWORD=your_secure_password

# Financial data (optional)
FMP_API_KEY=your_fmp_api_key

# Database (SQLite default, PostgreSQL for production)
DATABASE_URL=sqlite:///trading_bot.db
```

### Step 5 — Systemd Services

#### 5.1 Telegram bot service

```bash
sudo nano /etc/systemd/system/trading-bot.service
```

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

#### 5.2 Admin panel service

```bash
sudo nano /etc/systemd/system/trading-admin.service
```

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

#### 5.3 Enable and start

```bash
sudo systemctl daemon-reload
sudo systemctl enable trading-bot.service trading-admin.service
sudo systemctl start trading-bot.service trading-admin.service
sudo systemctl status trading-bot.service trading-admin.service
```

### Step 6 — Firewall

```bash
sudo apt install -y ufw
sudo ufw allow ssh
sudo ufw allow 5000   # admin panel (internal)
sudo ufw allow 8080   # bot API
sudo ufw enable
```

### Step 7 — Nginx Reverse Proxy (Recommended)

```bash
sudo apt install -y nginx
sudo nano /etc/nginx/sites-available/trading-admin
```

```nginx
server {
    listen 80;
    server_name your-pi-ip-or-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5002;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:5003;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/trading-admin /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

Optional SSL via Certbot:

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

## Running Services Summary

| Service | Port | Entry Point |
|---|---|---|
| Telegram Bot | 8080 | `src/frontend/telegram/bot.py` |
| Admin Panel | 5002 | `src/frontend/telegram/screener/admin_panel.py` |
| Bot API | 5003 | internal REST API |
| Nginx (optional) | 80/443 | reverse proxy |

---

## Service Management Reference

```bash
# Start / Stop / Restart
sudo systemctl start   trading-bot.service trading-admin.service
sudo systemctl stop    trading-bot.service trading-admin.service
sudo systemctl restart trading-bot.service trading-admin.service

# Enable / disable auto-start on boot
sudo systemctl enable  trading-bot.service trading-admin.service
sudo systemctl disable trading-bot.service trading-admin.service

# Live log tail
sudo journalctl -u trading-bot.service -f
sudo journalctl -u trading-admin.service -f

# Recent logs
sudo journalctl -u trading-bot.service --since "1 hour ago"
```

---

## Performance Optimisation

```bash
# Disable Bluetooth if unused
sudo systemctl disable bluetooth

# Disable WiFi power management
sudo iwconfig wlan0 power off

# Add swap if RAM is tight (check first)
free -h
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Lightweight monitoring tools
sudo apt install -y htop iotop
```

---

## Backup

### Backup script

```bash
nano /home/pi/backup-trading.sh
chmod +x /home/pi/backup-trading.sh
```

```bash
#!/bin/bash
BACKUP_DIR="/home/pi/backups"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p "$BACKUP_DIR"

cp /opt/apps/e-trading/trading_bot.db "$BACKUP_DIR/trading_bot_$DATE.db"
tar -czf "$BACKUP_DIR/config_$DATE.tar.gz" /opt/apps/e-trading/config/
tar -czf "$BACKUP_DIR/logs_$DATE.tar.gz" /opt/apps/e-trading/logs/

# Retain last 7 days
find "$BACKUP_DIR" -name "*.db" -mtime +7 -delete
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed: $DATE"
```

### Schedule daily at 02:00

```bash
crontab -e
# Add:
0 2 * * * /home/pi/backup-trading.sh
```

### Log rotation

```bash
sudo nano /etc/logrotate.d/trading-bot
```

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

---

## Monitoring

```bash
# System resources
htop
df -h
free -h

# Network connections
sudo netstat -tlnp

# Check port occupancy
sudo netstat -tlnp | grep :5002
sudo netstat -tlnp | grep :5003

# All service status at once
sudo systemctl status trading-bot.service trading-admin.service

# Test API endpoints
curl http://localhost:5003/api/status
curl http://localhost:5002
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Service won't start | Missing deps / config | `sudo journalctl -u trading-bot.service -n 50` |
| Port already in use | Stale process | `sudo netstat -tlnp \| grep :5002` then `sudo kill -9 <PID>` |
| Memory pressure | Insufficient RAM / no swap | Add swap (see Performance section) |
| Database errors | Bad permissions | `ls -la /opt/apps/e-trading/*.db` — fix ownership with `sudo chown pi:pi` |
| No signals from bot | Wrong token | Verify `TELEGRAM_BOT_TOKEN` in `.env` |

---

## Security Checklist

- [ ] Changed default `WEBGUI_PASSWORD` in `.env`
- [ ] Firewall enabled (UFW) with only required ports open
- [ ] SSH key-based auth configured (disable password auth)
- [ ] SSL certificate installed (Certbot) for admin panel
- [ ] Daily backups configured and tested
- [ ] System packages kept up to date (`sudo apt upgrade`)
- [ ] Logs reviewed regularly

---

**Last Updated:** 2026-04-30
