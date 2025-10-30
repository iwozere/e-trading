#!/bin/bash

# Raspberry Pi Setup Script for Advanced Trading Framework
# This script automates the deployment process

set -e  # Exit on any error

echo "🚀 Starting Raspberry Pi setup for Advanced Trading Framework..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root. Please run as pi user."
   exit 1
fi

# Check if we're on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo; then
    print_warning "This script is designed for Raspberry Pi. Continue anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 1: Update system
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Step 2: Install system dependencies
print_status "Installing system dependencies..."
sudo apt install -y python3-pip python3-venv git curl wget build-essential python3-dev
sudo apt install -y libffi-dev libssl-dev libatlas-base-dev gfortran
sudo apt install -y libopenblas-dev liblapack-dev libhdf5-dev libhdf5-serial-dev
sudo apt install -y libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5
sudo apt install -y libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good
sudo apt install -y gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav
sudo apt install -y gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl
sudo apt install -y gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio
sudo apt install -y ufw nginx certbot python3-certbot-nginx

# Step 3: Check and Install TA-Lib
print_status "Checking TA-Lib installation..."
if pkg-config --exists ta-lib; then
    print_success "TA-Lib is already installed"
    TA_LIB_VERSION=$(pkg-config --modversion ta-lib)
    print_status "TA-Lib version: $TA_LIB_VERSION"
else
    print_status "TA-Lib not found, installing..."
    cd /tmp
    if [ ! -f "ta-lib-0.4.0-src.tar.gz" ]; then
        wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    fi
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib/
    ./configure --prefix=/usr
    make -j$(nproc)
    sudo make install
    sudo ldconfig
    print_success "TA-Lib installed successfully"
fi

# Step 4: Setup project directory
print_status "Setting up project directory..."
PROJECT_DIR="/opt/apps/e-trading"

if [ ! -d "$PROJECT_DIR" ]; then
    print_error "Project directory not found at $PROJECT_DIR"
    print_status "Please ensure the project is located at $PROJECT_DIR"
    exit 1
fi

print_success "Found project at $PROJECT_DIR"
cd "$PROJECT_DIR"

# Step 5: Create virtual environment
print_status "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Step 6: Install Python dependencies
print_status "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install aiohttp flask gunicorn python-dotenv

# Step 7: Create configuration directory
print_status "Setting up configuration..."
mkdir -p config/donotshare

# Step 8: Create configuration file
if [ ! -f "config/donotshare/donotshare.py" ]; then
    print_status "Creating configuration file..."
    cat > config/donotshare/donotshare.py << 'EOF'
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
EOF
fi

# Step 9: Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_status "Creating .env file..."
    cat > .env << 'EOF'
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
EOF
    print_warning "Please edit .env file with your actual credentials before starting services"
fi

# Step 10: Create systemd services
print_status "Creating systemd services..."

# Telegram Bot Service
sudo tee /etc/systemd/system/trading-bot.service > /dev/null << EOF
[Unit]
Description=Trading Framework Telegram Bot
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$PROJECT_DIR/venv/bin
ExecStart=$PROJECT_DIR/venv/bin/python src/frontend/telegram/bot.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Admin Panel Service
sudo tee /etc/systemd/system/trading-admin.service > /dev/null << EOF
[Unit]
Description=Trading Framework Admin Panel
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$PROJECT_DIR/venv/bin
ExecStart=$PROJECT_DIR/venv/bin/python src/frontend/telegram/screener/admin_panel.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Step 11: Setup firewall
print_status "Configuring firewall..."
sudo ufw allow ssh
sudo ufw allow 5000
sudo ufw allow 8080
sudo ufw --force enable

# Step 12: Create Nginx configuration
print_status "Setting up Nginx..."
sudo tee /etc/nginx/sites-available/trading-admin > /dev/null << EOF
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/trading-admin /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx

# Step 13: Create log rotation
print_status "Setting up log rotation..."
sudo tee /etc/logrotate.d/trading-bot > /dev/null << EOF
$PROJECT_DIR/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 pi pi
}
EOF

# Step 14: Create backup script
print_status "Creating backup script..."
mkdir -p /home/pi/backups
cat > /home/pi/backup-trading.sh << EOF
#!/bin/bash
BACKUP_DIR="/home/pi/backups"
PROJECT_DIR="$PROJECT_DIR"
DATE=\$(date +%Y%m%d_%H%M%S)

mkdir -p \$BACKUP_DIR

# Backup database
if [ -f "\$PROJECT_DIR/trading_bot.db" ]; then
    cp \$PROJECT_DIR/trading_bot.db \$BACKUP_DIR/trading_bot_\$DATE.db
fi

# Backup configuration
if [ -d "\$PROJECT_DIR/config" ]; then
    tar -czf \$BACKUP_DIR/config_\$DATE.tar.gz -C \$PROJECT_DIR config/
fi

# Backup logs
if [ -d "\$PROJECT_DIR/logs" ]; then
    tar -czf \$BACKUP_DIR/logs_\$DATE.tar.gz -C \$PROJECT_DIR logs/
fi

# Keep only last 7 days of backups
find \$BACKUP_DIR -name "*.db" -mtime +7 -delete
find \$BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed: \$DATE"
EOF

chmod +x /home/pi/backup-trading.sh

# Step 15: Set proper permissions
print_status "Setting permissions..."
sudo chown -R pi:pi "$PROJECT_DIR"
chmod +x "$PROJECT_DIR/venv/bin/python"

# Step 16: Reload systemd and enable services
print_status "Enabling services..."
sudo systemctl daemon-reload
sudo systemctl enable trading-bot.service
sudo systemctl enable trading-admin.service

# Step 17: Create logs directory
mkdir -p logs

print_success "Setup completed successfully!"
echo ""
print_status "Next steps:"
echo "1. Edit .env file with your actual credentials:"
echo "   nano .env"
echo ""
echo "2. Start the services:"
echo "   sudo systemctl start trading-bot.service"
echo "   sudo systemctl start trading-admin.service"
echo ""
echo "3. Check service status:"
echo "   sudo systemctl status trading-bot.service"
echo "   sudo systemctl status trading-admin.service"
echo ""
echo "4. View logs:"
echo "   sudo journalctl -u trading-bot.service -f"
echo "   sudo journalctl -u trading-admin.service -f"
echo ""
echo "5. Access admin panel:"
echo "   http://your-pi-ip:5000"
echo ""
print_warning "Remember to:"
echo "- Change default passwords in .env file"
echo "- Set up SSL certificate if using a domain"
echo "- Configure daily backups: crontab -e and add '0 2 * * * /home/pi/backup-trading.sh'"
echo ""
print_success "Your Raspberry Pi is now ready to run the Advanced Trading Framework!"
