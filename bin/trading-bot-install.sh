#!/bin/bash

# Trading Bot Service Installation Script
# This script installs the trading bot as a systemd service on Raspberry Pi

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="trading-bot"
SERVICE_USER="trading"
PROJECT_DIR="/opt/e-trading"
VENV_DIR="$PROJECT_DIR/.venv"
LOG_DIR="/var/log/trading-bot"
CONFIG_DIR="/etc/trading-bot"

echo -e "${BLUE}🤖 Trading Bot Service Installation${NC}"
echo "=================================="

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}❌ This script must be run as root (use sudo)${NC}"
   exit 1
fi

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Warning: This doesn't appear to be a Raspberry Pi${NC}"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${BLUE}📋 System Information${NC}"
echo "OS: $(lsb_release -d | cut -f2)"
echo "Architecture: $(uname -m)"
echo "Kernel: $(uname -r)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo

# Create service user
echo -e "${BLUE}👤 Creating service user...${NC}"
if ! id "$SERVICE_USER" &>/dev/null; then
    useradd --system --shell /bin/bash --home-dir "$PROJECT_DIR" --create-home "$SERVICE_USER"
    echo -e "${GREEN}✅ Created user: $SERVICE_USER${NC}"
else
    echo -e "${YELLOW}⚠️  User $SERVICE_USER already exists${NC}"
fi

# Create directories
echo -e "${BLUE}📁 Creating directories...${NC}"
mkdir -p "$PROJECT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$CONFIG_DIR"
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/data"
mkdir -p "$PROJECT_DIR/db"

# Set ownership
chown -R "$SERVICE_USER:$SERVICE_USER" "$PROJECT_DIR"
chown -R "$SERVICE_USER:$SERVICE_USER" "$LOG_DIR"
chown -R "$SERVICE_USER:$SERVICE_USER" "$CONFIG_DIR"

echo -e "${GREEN}✅ Directories created${NC}"

# Install system dependencies
echo -e "${BLUE}📦 Installing system dependencies...${NC}"
apt-get update
apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    curl \
    htop \
    logrotate \
    supervisor

echo -e "${GREEN}✅ System dependencies installed${NC}"

# Copy project files
echo -e "${BLUE}📋 Copying project files...${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Copy source code
cp -r "$PROJECT_ROOT/src" "$PROJECT_DIR/"
cp -r "$PROJECT_ROOT/config" "$PROJECT_DIR/"
cp -r "$PROJECT_ROOT/requirements.txt" "$PROJECT_DIR/"
cp -r "$PROJECT_ROOT/bin" "$PROJECT_DIR/"

# Set ownership
chown -R "$SERVICE_USER:$SERVICE_USER" "$PROJECT_DIR"

echo -e "${GREEN}✅ Project files copied${NC}"

# Create Python virtual environment
echo -e "${BLUE}🐍 Setting up Python virtual environment...${NC}"
sudo -u "$SERVICE_USER" python3 -m venv "$VENV_DIR"
sudo -u "$SERVICE_USER" "$VENV_DIR/bin/pip" install --upgrade pip
sudo -u "$SERVICE_USER" "$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt"

echo -e "${GREEN}✅ Virtual environment created${NC}"

# Create systemd service file
echo -e "${BLUE}⚙️  Creating systemd service...${NC}"
cat > "/etc/systemd/system/$SERVICE_NAME.service" << EOF
[Unit]
Description=Trading Bot Paper Trading Service
After=network.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$VENV_DIR/bin
Environment=PYTHONPATH=$PROJECT_DIR
ExecStart=$VENV_DIR/bin/python -m src.trading.run_bot paper_trading_config.json
ExecReload=/bin/kill -HUP \$MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$SERVICE_NAME

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$PROJECT_DIR $LOG_DIR $CONFIG_DIR
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true

# Resource limits (optimized for Raspberry Pi)
MemoryMax=512M
CPUQuota=80%
TasksMax=100

[Install]
WantedBy=multi-user.target
EOF

echo -e "${GREEN}✅ Systemd service created${NC}"

# Create logrotate configuration
echo -e "${BLUE}📝 Setting up log rotation...${NC}"
cat > "/etc/logrotate.d/$SERVICE_NAME" << EOF
$LOG_DIR/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 $SERVICE_USER $SERVICE_USER
    postrotate
        systemctl reload $SERVICE_NAME > /dev/null 2>&1 || true
    endscript
}
EOF

echo -e "${GREEN}✅ Log rotation configured${NC}"

# Create default configuration
echo -e "${BLUE}⚙️  Creating default configuration...${NC}"
if [ ! -f "$PROJECT_DIR/config/trading/paper_trading_config.json" ]; then
    mkdir -p "$PROJECT_DIR/config/trading"
    cat > "$PROJECT_DIR/config/trading/paper_trading_config.json" << EOF
{
  "bot_id": "raspberry_pi_paper_trading",
  "environment": "production",
  "version": "1.0.0",
  "description": "Raspberry Pi paper trading bot with multi-strategy support",
  "symbol": "BTCUSDT",
  "timeframe": "4h",
  "risk_per_trade": 0.02,
  "max_open_trades": 3,
  "position_size": 0.1,
  "broker_type": "binance_paper",
  "initial_balance": 10000.0,
  "commission": 0.001,
  "data_source": "binance",
  "lookback_bars": 1000,
  "retry_interval": 60,
  "strategy_type": "custom",
  "strategy_params": {
    "entry_logic": {
      "name": "RSIOrBBEntryMixin",
      "params": {
        "e_rsi_period": 14,
        "e_rsi_oversold": 30,
        "e_bb_period": 20,
        "e_bb_dev": 2.0
      }
    },
    "exit_logic": {
      "name": "ATRExitMixin",
      "params": {
        "x_atr_period": 14,
        "x_sl_multiplier": 2.0
      }
    },
    "use_talib": true
  },
  "paper_trading": true,
  "log_level": "INFO",
  "log_file": "$LOG_DIR/trading-bot.log",
  "max_daily_loss": 100.0,
  "max_drawdown_pct": 20.0,
  "notifications_enabled": true,
  "telegram_enabled": false,
  "email_enabled": false
}
EOF
    chown "$SERVICE_USER:$SERVICE_USER" "$PROJECT_DIR/config/trading/paper_trading_config.json"
fi

echo -e "${GREEN}✅ Default configuration created${NC}"

# Create environment file
echo -e "${BLUE}🔐 Creating environment file...${NC}"
cat > "$CONFIG_DIR/trading-bot.env" << EOF
# Trading Bot Environment Variables
# Copy your API keys here

# Binance Testnet API Keys (for paper trading)
BINANCE_API_KEY=your_testnet_api_key_here
BINANCE_API_SECRET=your_testnet_secret_here

# Database
DATABASE_URL=sqlite:///$PROJECT_DIR/db/trading.db

# Logging
LOG_LEVEL=INFO
LOG_FILE=$LOG_DIR/trading-bot.log

# Telegram (optional)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Email (optional)
SMTP_SERVER=
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
EOF

chown "$SERVICE_USER:$SERVICE_USER" "$CONFIG_DIR/trading-bot.env"
chmod 600 "$CONFIG_DIR/trading-bot.env"

echo -e "${GREEN}✅ Environment file created${NC}"

# Update systemd service to use environment file
sed -i '/Environment=PYTHONPATH/a EnvironmentFile='$CONFIG_DIR'/trading-bot.env' "/etc/systemd/system/$SERVICE_NAME.service"

# Reload systemd and enable service
echo -e "${BLUE}🔄 Enabling service...${NC}"
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

echo -e "${GREEN}✅ Service enabled${NC}"

# Create management scripts
echo -e "${BLUE}📜 Creating management scripts...${NC}"

# Service management script
cat > "$PROJECT_DIR/bin/trading-bot-management.sh" << 'EOF'
#!/bin/bash

SERVICE_NAME="trading-bot"
LOG_DIR="/var/log/trading-bot"

case "$1" in
    start)
        echo "Starting trading bot service..."
        sudo systemctl start $SERVICE_NAME
        ;;
    stop)
        echo "Stopping trading bot service..."
        sudo systemctl stop $SERVICE_NAME
        ;;
    restart)
        echo "Restarting trading bot service..."
        sudo systemctl restart $SERVICE_NAME
        ;;
    status)
        sudo systemctl status $SERVICE_NAME
        ;;
    logs)
        sudo journalctl -u $SERVICE_NAME -f
        ;;
    logs-tail)
        tail -f $LOG_DIR/trading-bot.log
        ;;
    enable)
        echo "Enabling trading bot service..."
        sudo systemctl enable $SERVICE_NAME
        ;;
    disable)
        echo "Disabling trading bot service..."
        sudo systemctl disable $SERVICE_NAME
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|logs-tail|enable|disable}"
        exit 1
        ;;
esac
EOF

chmod +x "$PROJECT_DIR/bin/trading-bot-management.sh"
chown "$SERVICE_USER:$SERVICE_USER" "$PROJECT_DIR/bin/trading-bot-management.sh"

echo -e "${GREEN}✅ Management scripts created${NC}"

# Create system monitoring script
cat > "$PROJECT_DIR/bin/trading-bot-health.sh" << 'EOF'
#!/bin/bash

# System monitoring script for Raspberry Pi

echo "🤖 Trading Bot System Monitor"
echo "============================="
echo

# System information
echo "📊 System Information:"
echo "Date: $(date)"
echo "Uptime: $(uptime -p)"
echo "Load: $(uptime | awk -F'load average:' '{print $2}')"
echo

# Memory usage
echo "💾 Memory Usage:"
free -h
echo

# Disk usage
echo "💿 Disk Usage:"
df -h / | tail -1
echo

# CPU temperature (Raspberry Pi specific)
if [ -f /sys/class/thermal/thermal_zone0/temp ]; then
    temp=$(cat /sys/class/thermal/thermal_zone0/temp)
    temp_c=$((temp/1000))
    echo "🌡️  CPU Temperature: ${temp_c}°C"
    echo
fi

# Service status
echo "🤖 Trading Bot Service:"
systemctl is-active trading-bot --quiet && echo "Status: ✅ Running" || echo "Status: ❌ Stopped"
echo

# Recent logs
echo "📝 Recent Logs (last 10 lines):"
sudo journalctl -u trading-bot -n 10 --no-pager
EOF

chmod +x "$PROJECT_DIR/bin/trading-bot-health.sh"
chown "$SERVICE_USER:$SERVICE_USER" "$PROJECT_DIR/bin/trading-bot-health.sh"

# Final instructions
echo
echo -e "${GREEN}🎉 Installation Complete!${NC}"
echo "=========================="
echo
echo -e "${BLUE}📋 Next Steps:${NC}"
echo "1. Edit API keys in: $CONFIG_DIR/trading-bot.env"
echo "2. Configure strategies in: $PROJECT_DIR/config/trading/"
echo "3. Start the service: sudo systemctl start $SERVICE_NAME"
echo "4. Check status: sudo systemctl status $SERVICE_NAME"
echo "5. View logs: sudo journalctl -u $SERVICE_NAME -f"
echo
echo -e "${BLUE}🛠️  Management Commands:${NC}"
echo "• Start service: $PROJECT_DIR/bin/trading-bot-management.sh start"
echo "• Stop service: $PROJECT_DIR/bin/trading-bot-management.sh stop"
echo "• View logs: $PROJECT_DIR/bin/trading-bot-management.sh logs"
echo "• System monitor: $PROJECT_DIR/bin/trading-bot-health.sh"
echo
echo -e "${BLUE}📁 Important Paths:${NC}"
echo "• Project directory: $PROJECT_DIR"
echo "• Configuration: $CONFIG_DIR"
echo "• Logs: $LOG_DIR"
echo "• Service file: /etc/systemd/system/$SERVICE_NAME.service"
echo
echo -e "${YELLOW}⚠️  Remember to:${NC}"
echo "• Configure your Binance testnet API keys"
echo "• Test the configuration before starting"
echo "• Monitor system resources (RAM/CPU/Temperature)"
echo "• Set up log monitoring and alerts"
echo
echo -e "${GREEN}✅ Trading bot is ready to run as a system service!${NC}"