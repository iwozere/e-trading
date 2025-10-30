#!/bin/bash

# Trading Bot Web UI System Service Script for Ubuntu/Raspberry Pi
# ================================================================
# This script manages the trading web UI as a system service on Ubuntu/Pi5
# It handles installation, configuration, and service management

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SERVICE_NAME="trading-webui"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
SERVICE_USER="pi"
SERVICE_GROUP="pi"
WEB_UI_PORT="8080"
LOG_DIR="/var/log/${SERVICE_NAME}"
PID_FILE="/var/run/${SERVICE_NAME}.pid"

# Functions
print_header() {
    echo -e "${BLUE}"
    echo "🚀 Trading Bot Web UI System Service Manager"
    echo "============================================"
    echo -e "${NC}"
}

print_system_info() {
    echo -e "${CYAN}📊 System Information:${NC}"
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo "OS: $(lsb_release -d 2>/dev/null | cut -f2 || uname -s)"
    echo "Architecture: $(uname -m)"
    echo "Python: $(python3 --version 2>/dev/null || echo 'Not found')"
    echo "Node.js: $(node --version 2>/dev/null || echo 'Not found')"
    
    # Check if on Raspberry Pi
    if grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
        echo "Platform: 🥧 Raspberry Pi"
        if [ -f /sys/class/thermal/thermal_zone0/temp ]; then
            temp=$(cat /sys/class/thermal/thermal_zone0/temp)
            temp_c=$((temp/1000))
            echo "CPU Temperature: ${temp_c}°C"
            if [ $temp_c -gt 70 ]; then
                echo -e "${RED}⚠️  High temperature warning!${NC}"
            fi
        fi
    else
        echo "Platform: 💻 Desktop/Server"
    fi
    
    echo "Memory: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
    echo "Load Average: $(uptime | awk -F'load average:' '{print $2}')"
    echo
}

check_root() {
    if [ "$EUID" -ne 0 ]; then
        echo -e "${RED}❌ This script must be run as root (use sudo)${NC}"
        exit 1
    fi
}

check_dependencies() {
    echo -e "${CYAN}🔍 Checking Dependencies:${NC}"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 not found${NC}"
        echo -e "${YELLOW}💡 Installing Python 3...${NC}"
        apt update && apt install -y python3 python3-pip python3-venv
    fi
    echo -e "${GREEN}✅ Python 3 found${NC}"
    
    # Check Node.js (for building frontend)
    if ! command -v node &> /dev/null; then
        echo -e "${YELLOW}⚠️  Node.js not found, installing...${NC}"
        curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
        apt install -y nodejs
    fi
    echo -e "${GREEN}✅ Node.js found${NC}"
    
    # Check systemctl
    if ! command -v systemctl &> /dev/null; then
        echo -e "${RED}❌ systemctl not found. This script requires systemd.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ systemctl found${NC}"
    
    echo
}

setup_user() {
    echo -e "${CYAN}👤 Setting up Service User:${NC}"
    
    # Create service user if it doesn't exist
    if ! id "$SERVICE_USER" &>/dev/null; then
        echo -e "${YELLOW}Creating user: $SERVICE_USER${NC}"
        useradd -r -s /bin/bash -d "$PROJECT_ROOT" -c "Trading Bot Web UI Service" "$SERVICE_USER"
    else
        echo -e "${GREEN}✅ User $SERVICE_USER already exists${NC}"
    fi
    
    # Set ownership of project directory
    chown -R "$SERVICE_USER:$SERVICE_GROUP" "$PROJECT_ROOT"
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    chown "$SERVICE_USER:$SERVICE_GROUP" "$LOG_DIR"
    
    echo -e "${GREEN}✅ User setup complete${NC}"
    echo
}

install_python_dependencies() {
    echo -e "${CYAN}📦 Installing Python Dependencies:${NC}"
    
    # Switch to service user for installation
    sudo -u "$SERVICE_USER" bash << EOF
cd "$PROJECT_ROOT"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment and install dependencies
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install web UI requirements
if [ -f "requirements-webui.txt" ]; then
    pip install -r requirements-webui.txt
    echo "✅ Web UI dependencies installed"
else
    echo "⚠️  requirements-webui.txt not found, installing basic deps"
    pip install fastapi uvicorn python-socketio
fi

# Install project requirements if they exist
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Project dependencies installed"
fi
EOF
    
    echo -e "${GREEN}✅ Python dependencies installed${NC}"
    echo
}

build_frontend() {
    echo -e "${CYAN}🎨 Building Frontend:${NC}"
    
    # Switch to service user for building
    sudo -u "$SERVICE_USER" bash << EOF
cd "$PROJECT_ROOT/src/web_ui/frontend"

# Install npm dependencies
if [ ! -d "node_modules" ]; then
    echo "Installing npm packages..."
    npm install
fi

# Build for production
echo "Building frontend for production..."
npm run build

if [ -d "dist" ]; then
    echo "✅ Frontend built successfully"
else
    echo "❌ Frontend build failed"
    exit 1
fi
EOF
    
    echo -e "${GREEN}✅ Frontend built successfully${NC}"
    echo
}

create_systemd_service() {
    echo -e "${CYAN}🔧 Creating systemd service:${NC}"
    
    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Trading Bot Web UI Service
Documentation=https://github.com/your-repo/trading-bot
After=network.target
Wants=network.target

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_GROUP
WorkingDirectory=$PROJECT_ROOT
Environment=PYTHONPATH=$PROJECT_ROOT
Environment=PYTHONUNBUFFERED=1
Environment=WEB_UI_PORT=$WEB_UI_PORT
Environment=WEB_UI_HOST=0.0.0.0
Environment=WEB_UI_ENV=production

# Command to run
ExecStart=$PROJECT_ROOT/.venv/bin/python $PROJECT_ROOT/start_trading_webui.py --port $WEB_UI_PORT

# Restart policy
Restart=always
RestartSec=10
StartLimitInterval=60
StartLimitBurst=3

# Output to journal
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$SERVICE_NAME

# Security settings
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$PROJECT_ROOT
ReadWritePaths=$LOG_DIR
ReadWritePaths=/tmp

# Resource limits
MemoryMax=2G
CPUQuota=200%

# Process management
KillMode=mixed
KillSignal=SIGTERM
TimeoutStopSec=30

# PID file
PIDFile=$PID_FILE

[Install]
WantedBy=multi-user.target
EOF
    
    echo -e "${GREEN}✅ Systemd service file created: $SERVICE_FILE${NC}"
    echo
}

setup_nginx_proxy() {
    echo -e "${CYAN}🌐 Setting up Nginx Reverse Proxy:${NC}"
    
    # Check if nginx is installed
    if ! command -v nginx &> /dev/null; then
        echo -e "${YELLOW}Installing Nginx...${NC}"
        apt update && apt install -y nginx
    fi
    
    # Create nginx configuration
    cat > "/etc/nginx/sites-available/$SERVICE_NAME" << EOF
server {
    listen 80;
    server_name localhost $(hostname) $(hostname -I | awk '{print $1}');
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
    
    # Proxy to backend
    location /api/ {
        proxy_pass http://127.0.0.1:$WEB_UI_PORT;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
        proxy_read_timeout 86400;
    }
    
    # WebSocket support
    location /socket.io/ {
        proxy_pass http://127.0.0.1:$WEB_UI_PORT;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    # Serve static files
    location / {
        try_files \$uri \$uri/ @backend;
        root $PROJECT_ROOT/src/web_ui/frontend/dist;
        index index.html;
        
        # Cache static assets
        location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
    
    # Fallback to backend for SPA routing
    location @backend {
        proxy_pass http://127.0.0.1:$WEB_UI_PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    # Logs
    access_log $LOG_DIR/nginx_access.log;
    error_log $LOG_DIR/nginx_error.log;
}
EOF
    
    # Enable the site
    ln -sf "/etc/nginx/sites-available/$SERVICE_NAME" "/etc/nginx/sites-enabled/"
    
    # Remove default site if it exists
    rm -f /etc/nginx/sites-enabled/default
    
    # Test nginx configuration
    if nginx -t; then
        echo -e "${GREEN}✅ Nginx configuration is valid${NC}"
        systemctl reload nginx
    else
        echo -e "${RED}❌ Nginx configuration error${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Nginx reverse proxy configured${NC}"
    echo
}

install_service() {
    echo -e "${CYAN}🔧 Installing Trading Bot Web UI Service...${NC}"
    
    check_root
    print_system_info
    check_dependencies
    setup_user
    install_python_dependencies
    build_frontend
    create_systemd_service
    setup_nginx_proxy
    
    # Reload systemd
    systemctl daemon-reload
    
    # Enable service
    systemctl enable "$SERVICE_NAME"
    echo -e "${GREEN}✅ Service enabled for auto-start${NC}"
    
    # Enable nginx
    systemctl enable nginx
    
    echo -e "${GREEN}✅ Trading Bot Web UI Service installed successfully${NC}"
    echo
    echo -e "${PURPLE}🎯 Service Information:${NC}"
    echo "Service Name: $SERVICE_NAME"
    echo "Service File: $SERVICE_FILE"
    echo "Service User: $SERVICE_USER"
    echo "Web UI Port: $WEB_UI_PORT"
    echo "Web Access: http://$(hostname -I | awk '{print $1}') or http://localhost"
    echo "Logs: $LOG_DIR"
    echo
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Start the service: sudo systemctl start $SERVICE_NAME"
    echo "2. Check status: sudo systemctl status $SERVICE_NAME"
    echo "3. View logs: sudo journalctl -u $SERVICE_NAME -f"
    echo
}

start_service() {
    echo -e "${CYAN}🚀 Starting Trading Bot Web UI Service...${NC}"
    
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        echo -e "${YELLOW}⚠️  Service is already running${NC}"
        return 0
    fi
    
    systemctl start "$SERVICE_NAME"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Service started successfully${NC}"
        sleep 2
        show_service_status
    else
        echo -e "${RED}❌ Failed to start service${NC}"
        echo "Check logs with: sudo journalctl -u $SERVICE_NAME -f"
        exit 1
    fi
}

stop_service() {
    echo -e "${CYAN}🛑 Stopping Trading Bot Web UI Service...${NC}"
    
    if ! systemctl is-active --quiet "$SERVICE_NAME"; then
        echo -e "${YELLOW}⚠️  Service is not running${NC}"
        return 0
    fi
    
    systemctl stop "$SERVICE_NAME"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Service stopped successfully${NC}"
    else
        echo -e "${RED}❌ Failed to stop service${NC}"
        exit 1
    fi
}

restart_service() {
    echo -e "${CYAN}🔄 Restarting Trading Bot Web UI Service...${NC}"
    
    systemctl restart "$SERVICE_NAME"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Service restarted successfully${NC}"
        sleep 2
        show_service_status
    else
        echo -e "${RED}❌ Failed to restart service${NC}"
        exit 1
    fi
}

show_service_status() {
    echo -e "${CYAN}📊 Service Status:${NC}"
    
    if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
        echo -e "Enabled: ${GREEN}✅ Yes${NC}"
    else
        echo -e "Enabled: ${RED}❌ No${NC}"
    fi
    
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        echo -e "Status: ${GREEN}✅ Running${NC}"
        
        # Show uptime
        start_time=$(systemctl show "$SERVICE_NAME" --property=ActiveEnterTimestamp --value)
        if [ -n "$start_time" ]; then
            echo "Started: $start_time"
        fi
        
        # Show memory usage
        memory_usage=$(systemctl show "$SERVICE_NAME" --property=MemoryCurrent --value)
        if [ -n "$memory_usage" ] && [ "$memory_usage" != "[not set]" ]; then
            memory_mb=$((memory_usage / 1024 / 1024))
            echo "Memory Usage: ${memory_mb}MB"
        fi
        
        # Show web access info
        echo -e "${PURPLE}🌐 Web Access:${NC}"
        echo "Local: http://localhost"
        if command -v hostname &> /dev/null; then
            local_ip=$(hostname -I | awk '{print $1}')
            if [ -n "$local_ip" ]; then
                echo "Network: http://$local_ip"
            fi
        fi
        
    else
        echo -e "Status: ${RED}❌ Stopped${NC}"
        
        # Show last exit status
        exit_code=$(systemctl show "$SERVICE_NAME" --property=ExecMainStatus --value)
        if [ -n "$exit_code" ] && [ "$exit_code" != "0" ]; then
            echo -e "Last Exit Code: ${RED}$exit_code${NC}"
        fi
    fi
    
    echo
}

show_logs() {
    echo -e "${CYAN}📝 Service Logs:${NC}"
    echo "Press Ctrl+C to exit log view"
    echo "=================================="
    
    journalctl -u "$SERVICE_NAME" -f --no-pager
}

show_recent_logs() {
    echo -e "${CYAN}📝 Recent Service Logs (last 50 lines):${NC}"
    echo "========================================"
    
    journalctl -u "$SERVICE_NAME" -n 50 --no-pager
    echo
}

uninstall_service() {
    echo -e "${CYAN}🗑️  Uninstalling Trading Bot Web UI Service...${NC}"
    
    check_root
    
    # Stop service if running
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        systemctl stop "$SERVICE_NAME"
    fi
    
    # Disable service
    if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
        systemctl disable "$SERVICE_NAME"
    fi
    
    # Remove service file
    if [ -f "$SERVICE_FILE" ]; then
        rm "$SERVICE_FILE"
        echo -e "${GREEN}✅ Service file removed${NC}"
    fi
    
    # Remove nginx configuration
    if [ -f "/etc/nginx/sites-available/$SERVICE_NAME" ]; then
        rm "/etc/nginx/sites-available/$SERVICE_NAME"
        rm -f "/etc/nginx/sites-enabled/$SERVICE_NAME"
        systemctl reload nginx
        echo -e "${GREEN}✅ Nginx configuration removed${NC}"
    fi
    
    # Reload systemd
    systemctl daemon-reload
    
    echo -e "${GREEN}✅ Service uninstalled successfully${NC}"
    echo -e "${YELLOW}Note: Project files and logs were not removed${NC}"
    echo
}

show_menu() {
    echo -e "${PURPLE}🎛️  Service Management Menu:${NC}"
    echo "1.  🔧 Install service"
    echo "2.  🚀 Start service"
    echo "3.  🛑 Stop service"
    echo "4.  🔄 Restart service"
    echo "5.  📊 Show service status"
    echo "6.  📝 Show live logs"
    echo "7.  📋 Show recent logs"
    echo "8.  🗑️  Uninstall service"
    echo "9.  🔧 Rebuild frontend"
    echo "10. 📦 Update dependencies"
    echo "11. ❌ Exit"
    echo
}

rebuild_frontend() {
    echo -e "${CYAN}🎨 Rebuilding Frontend...${NC}"
    build_frontend
    
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        echo -e "${YELLOW}🔄 Restarting service to use new frontend...${NC}"
        restart_service
    fi
}

update_dependencies() {
    echo -e "${CYAN}📦 Updating Dependencies...${NC}"
    install_python_dependencies
    build_frontend
    
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        echo -e "${YELLOW}🔄 Restarting service...${NC}"
        restart_service
    fi
}

# Main execution
main() {
    print_header
    
    # Handle command line arguments
    if [ $# -eq 0 ]; then
        # Interactive mode
        while true; do
            show_menu
            read -p "Select option (1-11): " choice
            echo
            
            case $choice in
                1) install_service ;;
                2) start_service ;;
                3) stop_service ;;
                4) restart_service ;;
                5) show_service_status ;;
                6) show_logs ;;
                7) show_recent_logs ;;
                8) uninstall_service ;;
                9) rebuild_frontend ;;
                10) update_dependencies ;;
                11)
                    echo -e "${GREEN}👋 Goodbye!${NC}"
                    exit 0
                    ;;
                *)
                    echo -e "${RED}❌ Invalid option${NC}"
                    ;;
            esac
            
            echo
            read -p "Press Enter to continue..."
            echo
        done
    else
        # Command line mode
        case "$1" in
            install) install_service ;;
            start) start_service ;;
            stop) stop_service ;;
            restart) restart_service ;;
            status) show_service_status ;;
            logs) show_logs ;;
            recent-logs) show_recent_logs ;;
            uninstall) uninstall_service ;;
            rebuild) rebuild_frontend ;;
            update) update_dependencies ;;
            *)
                echo "Usage: $0 [install|start|stop|restart|status|logs|recent-logs|uninstall|rebuild|update]"
                exit 1
                ;;
        esac
    fi
}

# Trap Ctrl+C for graceful shutdown
trap 'echo -e "\n${YELLOW}🛑 Interrupted by user${NC}"; exit 0' INT

# Run main function
main "$@"