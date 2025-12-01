#!/usr/bin/env bash
# Setup Scheduler Service on Raspberry Pi
#
# This script helps install and configure the scheduler systemd service

set -e

echo "=========================================="
echo "E-Trading Scheduler Service Setup"
echo "=========================================="
echo ""

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (use sudo)"
   exit 1
fi

# Detect current user who invoked sudo
REAL_USER="${SUDO_USER:-$USER}"
REAL_GROUP=$(id -gn "$REAL_USER")

echo "Installing for user: $REAL_USER:$REAL_GROUP"
echo ""

# Get project directory
if [ -z "$1" ]; then
    echo "Usage: sudo ./systemd-setup.sh /path/to/e-trading"
    echo ""
    echo "Example:"
    echo "  sudo ./systemd-setup.sh /opt/apps/e-trading"
    echo "  sudo ./systemd-setup.sh /home/alkotrader/e-trading"
    exit 1
fi

PROJECT_DIR="$1"

# Validate project directory
if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory does not exist: $PROJECT_DIR"
    exit 1
fi

if [ ! -f "$PROJECT_DIR/bin/scheduler/scheduler.service" ]; then
    echo "ERROR: scheduler.service not found in $PROJECT_DIR/bin/scheduler/"
    exit 1
fi

echo "Project directory: $PROJECT_DIR"
echo ""

# Check Python virtual environment
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "ERROR: Python virtual environment not found at $PROJECT_DIR/.venv/"
    echo "Please create virtual environment first:"
    echo "  cd $PROJECT_DIR"
    echo "  python -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

echo "✓ Found Python virtual environment"

# Check if src/scheduler/main.py exists
if [ ! -f "$PROJECT_DIR/src/scheduler/main.py" ]; then
    echo "ERROR: Scheduler main.py not found at $PROJECT_DIR/src/scheduler/main.py"
    exit 1
fi

echo "✓ Found scheduler main.py"
echo ""

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_DIR/logs"
chown "$REAL_USER:$REAL_GROUP" "$PROJECT_DIR/logs"
echo "✓ Created logs directory"

# Create data directory if it doesn't exist
mkdir -p "$PROJECT_DIR/data"
chown "$REAL_USER:$REAL_GROUP" "$PROJECT_DIR/data"
echo "✓ Created data directory"

# Create results directory if it doesn't exist
mkdir -p "$PROJECT_DIR/results"
chown "$REAL_USER:$REAL_GROUP" "$PROJECT_DIR/results"
echo "✓ Created results directory"
echo ""

# Create temporary service file with correct paths
SERVICE_FILE="/etc/systemd/system/e-trading-scheduler.service"
TEMP_SERVICE="/tmp/scheduler.service.tmp"

cat > "$TEMP_SERVICE" << EOF
[Unit]
Description=E-Trading Scheduler Service
Documentation=https://github.com/iwozere/e-trading
After=network.target postgresql.service
Wants=postgresql.service

[Service]
Type=simple
User=$REAL_USER
Group=$REAL_GROUP
WorkingDirectory=$PROJECT_DIR
ExecStart=$VENV_PYTHON -m src.scheduler.main
ExecReload=/bin/kill -HUP \$MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=e-trading-scheduler

# Environment variables
Environment="PYTHONPATH=$PROJECT_DIR"
Environment="PYTHONUNBUFFERED=1"
Environment="TRADING_ENV=production"
Environment="LOG_LEVEL=INFO"
Environment="SCHEDULER_MAX_WORKERS=10"
Environment="DATA_CACHE_ENABLED=true"

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$PROJECT_DIR/logs
ReadWritePaths=$PROJECT_DIR/data
ReadWritePaths=$PROJECT_DIR/results

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
EOF

echo "Installing service file to $SERVICE_FILE"
cp "$TEMP_SERVICE" "$SERVICE_FILE"
rm "$TEMP_SERVICE"
echo "✓ Service file installed"
echo ""

# Reload systemd
echo "Reloading systemd daemon..."
systemctl daemon-reload
echo "✓ Systemd reloaded"
echo ""

# Enable service
echo "Enabling service to start on boot..."
systemctl enable e-trading-scheduler.service
echo "✓ Service enabled"
echo ""

# Start service
echo "Starting service..."
systemctl start e-trading-scheduler.service
echo "✓ Service started"
echo ""

# Wait a moment for service to initialize
sleep 2

# Show status
echo "=========================================="
echo "Service Status:"
echo "=========================================="
systemctl status e-trading-scheduler.service --no-pager
echo ""

echo "=========================================="
echo "Recent Logs:"
echo "=========================================="
journalctl -u e-trading-scheduler.service -n 20 --no-pager
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Useful commands:"
echo "  sudo systemctl status e-trading-scheduler    # Check status"
echo "  sudo systemctl stop e-trading-scheduler      # Stop service"
echo "  sudo systemctl start e-trading-scheduler     # Start service"
echo "  sudo systemctl restart e-trading-scheduler   # Restart service"
echo "  sudo journalctl -u e-trading-scheduler -f    # Watch logs"
echo ""
