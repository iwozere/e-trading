#!/bin/bash

# Enhanced Multi-Strategy Trading Service Management Script
# Provides comprehensive management for the enhanced multi-strategy trading service on Raspberry Pi

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
SERVICE_NAME="trading-service"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
CONFIG_FILE="$PROJECT_ROOT/config/enhanced_trading/raspberry_pi_multi_strategy.json"
LOG_FILE="/var/log/${SERVICE_NAME}.log"

# Functions
print_header() {
    echo -e "${BLUE}"
    echo "🤖 Enhanced Multi-Strategy Trading Service Manager"
    echo "================================================="
    echo -e "${NC}"
}

print_system_info() {
    echo -e "${CYAN}📊 System Information:${NC}"
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo "OS: $(uname -s) $(uname -r)"
    echo "Architecture: $(uname -m)"
    
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

check_service_status() {
    if systemctl is-active --quiet $SERVICE_NAME; then
        echo -e "${GREEN}✅ Service is running${NC}"
        return 0
    else
        echo -e "${RED}❌ Service is not running${NC}"
        return 1
    fi
}

show_service_status() {
    echo -e "${CYAN}📊 Service Status:${NC}"
    
    if systemctl is-enabled --quiet $SERVICE_NAME 2>/dev/null; then
        echo -e "Enabled: ${GREEN}✅ Yes${NC}"
    else
        echo -e "Enabled: ${RED}❌ No${NC}"
    fi
    
    if systemctl is-active --quiet $SERVICE_NAME; then
        echo -e "Status: ${GREEN}✅ Running${NC}"
        
        # Show uptime
        start_time=$(systemctl show $SERVICE_NAME --property=ActiveEnterTimestamp --value)
        if [ -n "$start_time" ]; then
            echo "Started: $start_time"
        fi
        
        # Show memory usage
        memory_usage=$(systemctl show $SERVICE_NAME --property=MemoryCurrent --value)
        if [ -n "$memory_usage" ] && [ "$memory_usage" != "[not set]" ]; then
            memory_mb=$((memory_usage / 1024 / 1024))
            echo "Memory Usage: ${memory_mb}MB"
        fi
        
    else
        echo -e "Status: ${RED}❌ Stopped${NC}"
        
        # Show last exit status
        exit_code=$(systemctl show $SERVICE_NAME --property=ExecMainStatus --value)
        if [ -n "$exit_code" ] && [ "$exit_code" != "0" ]; then
            echo -e "Last Exit Code: ${RED}$exit_code${NC}"
        fi
    fi
    
    echo
}

install_service() {
    echo -e "${CYAN}🔧 Installing Enhanced Trading Service...${NC}"
    
    # Check if running as root
    if [ "$EUID" -ne 0 ]; then
        echo -e "${RED}❌ Please run with sudo to install service${NC}"
        exit 1
    fi
    
    # Install the service
    cd "$PROJECT_ROOT"
    python3 raspberry_pi_trading_service.py --install-service
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Service installed successfully${NC}"
        
        # Reload systemd
        systemctl daemon-reload
        
        # Enable service
        systemctl enable $SERVICE_NAME
        echo -e "${GREEN}✅ Service enabled for auto-start${NC}"
        
    else
        echo -e "${RED}❌ Service installation failed${NC}"
        exit 1
    fi
}

start_service() {
    echo -e "${CYAN}🚀 Starting Enhanced Trading Service...${NC}"
    
    if systemctl is-active --quiet $SERVICE_NAME; then
        echo -e "${YELLOW}⚠️  Service is already running${NC}"
        return 0
    fi
    
    systemctl start $SERVICE_NAME
    
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
    echo -e "${CYAN}🛑 Stopping Enhanced Trading Service...${NC}"
    
    if ! systemctl is-active --quiet $SERVICE_NAME; then
        echo -e "${YELLOW}⚠️  Service is not running${NC}"
        return 0
    fi
    
    systemctl stop $SERVICE_NAME
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Service stopped successfully${NC}"
    else
        echo -e "${RED}❌ Failed to stop service${NC}"
        exit 1
    fi
}

restart_service() {
    echo -e "${CYAN}🔄 Restarting Enhanced Trading Service...${NC}"
    
    systemctl restart $SERVICE_NAME
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Service restarted successfully${NC}"
        sleep 2
        show_service_status
    else
        echo -e "${RED}❌ Failed to restart service${NC}"
        exit 1
    fi
}

enable_service() {
    echo -e "${CYAN}⚡ Enabling Enhanced Trading Service for auto-start...${NC}"
    
    systemctl enable $SERVICE_NAME
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Service enabled for auto-start${NC}"
    else
        echo -e "${RED}❌ Failed to enable service${NC}"
        exit 1
    fi
}

disable_service() {
    echo -e "${CYAN}⏸️  Disabling Enhanced Trading Service auto-start...${NC}"
    
    systemctl disable $SERVICE_NAME
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Service disabled from auto-start${NC}"
    else
        echo -e "${RED}❌ Failed to disable service${NC}"
        exit 1
    fi
}

show_logs() {
    echo -e "${CYAN}📝 Service Logs:${NC}"
    echo "Press Ctrl+C to exit log view"
    echo "=================================="
    
    journalctl -u $SERVICE_NAME -f --no-pager
}

show_recent_logs() {
    echo -e "${CYAN}📝 Recent Service Logs (last 50 lines):${NC}"
    echo "========================================"
    
    journalctl -u $SERVICE_NAME -n 50 --no-pager
    echo
}

validate_config() {
    echo -e "${CYAN}🔍 Validating Configuration:${NC}"
    
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${RED}❌ Configuration file not found: $CONFIG_FILE${NC}"
        return 1
    fi
    
    # Basic JSON validation
    if python3 -c "import json; json.load(open('$CONFIG_FILE'))" 2>/dev/null; then
        echo -e "${GREEN}✅ Configuration file is valid JSON${NC}"
    else
        echo -e "${RED}❌ Invalid JSON in configuration file${NC}"
        return 1
    fi
    
    # Check for required sections
    if python3 -c "
import json
config = json.load(open('$CONFIG_FILE'))
required = ['system', 'strategies']
missing = [s for s in required if s not in config]
if missing:
    print('Missing sections:', missing)
    exit(1)
print('All required sections present')
" 2>/dev/null; then
        echo -e "${GREEN}✅ Configuration structure is valid${NC}"
    else
        echo -e "${RED}❌ Configuration structure is invalid${NC}"
        return 1
    fi
    
    echo
}

show_strategy_status() {
    echo -e "${CYAN}📊 Strategy Status:${NC}"
    
    if ! systemctl is-active --quiet $SERVICE_NAME; then
        echo -e "${RED}❌ Service is not running${NC}"
        return 1
    fi
    
    # Try to get status from service (this would need to be implemented)
    echo "Strategy status monitoring would be implemented here"
    echo "This would show individual strategy performance, P&L, etc."
    echo
}

backup_config() {
    echo -e "${CYAN}💾 Backing up Configuration...${NC}"
    
    backup_dir="$PROJECT_ROOT/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    cp "$CONFIG_FILE" "$backup_dir/"
    cp -r "$PROJECT_ROOT/config/enhanced_trading" "$backup_dir/"
    
    echo -e "${GREEN}✅ Configuration backed up to: $backup_dir${NC}"
    echo
}

show_menu() {
    echo -e "${PURPLE}🎛️  Service Management Menu:${NC}"
    echo "1.  📊 Show service status"
    echo "2.  🚀 Start service"
    echo "3.  🛑 Stop service"
    echo "4.  🔄 Restart service"
    echo "5.  ⚡ Enable auto-start"
    echo "6.  ⏸️  Disable auto-start"
    echo "7.  🔧 Install/reinstall service"
    echo "8.  📝 Show live logs"
    echo "9.  📋 Show recent logs"
    echo "10. 🔍 Validate configuration"
    echo "11. 📊 Show strategy status"
    echo "12. 💾 Backup configuration"
    echo "13. 🧹 Clean logs"
    echo "14. 🔧 System maintenance"
    echo "15. ❌ Exit"
    echo
}

clean_logs() {
    echo -e "${CYAN}🧹 Cleaning Service Logs...${NC}"
    
    read -p "This will clear all service logs. Continue? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo journalctl --vacuum-time=1d
        echo -e "${GREEN}✅ Logs cleaned${NC}"
    else
        echo "Cancelled"
    fi
    echo
}

system_maintenance() {
    echo -e "${CYAN}🔧 System Maintenance:${NC}"
    echo "======================"
    
    echo "1. Updating package lists..."
    sudo apt update -qq
    
    echo "2. Checking for system updates..."
    updates=$(apt list --upgradable 2>/dev/null | wc -l)
    if [ $updates -gt 1 ]; then
        echo -e "${YELLOW}⚠️  $((updates-1)) updates available${NC}"
        read -p "Install updates? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sudo apt upgrade -y
        fi
    else
        echo -e "${GREEN}✅ System is up to date${NC}"
    fi
    
    echo "3. Checking disk space..."
    df -h / | tail -1 | awk '{
        if ($5+0 > 80) 
            print "⚠️  Disk usage high: " $5 " used"
        else 
            print "✅ Disk usage OK: " $5 " used"
    }'
    
    echo "4. Checking memory usage..."
    free -h | grep Mem | awk '{
        used_pct = ($3/$2)*100
        if (used_pct > 80)
            print "⚠️  Memory usage high: " used_pct "%"
        else
            print "✅ Memory usage OK: " used_pct "%"
    }'
    
    echo
}

# Main execution
main() {
    print_header
    print_system_info
    
    # Handle command line arguments
    if [ $# -eq 0 ]; then
        # Interactive mode
        while true; do
            show_menu
            read -p "Select option (1-15): " choice
            echo
            
            case $choice in
                1) show_service_status ;;
                2) start_service ;;
                3) stop_service ;;
                4) restart_service ;;
                5) enable_service ;;
                6) disable_service ;;
                7) install_service ;;
                8) show_logs ;;
                9) show_recent_logs ;;
                10) validate_config ;;
                11) show_strategy_status ;;
                12) backup_config ;;
                13) clean_logs ;;
                14) system_maintenance ;;
                15)
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
            status) show_service_status ;;
            start) start_service ;;
            stop) stop_service ;;
            restart) restart_service ;;
            enable) enable_service ;;
            disable) disable_service ;;
            install) install_service ;;
            logs) show_logs ;;
            recent-logs) show_recent_logs ;;
            validate) validate_config ;;
            backup) backup_config ;;
            clean) clean_logs ;;
            maintenance) system_maintenance ;;
            *)
                echo "Usage: $0 [status|start|stop|restart|enable|disable|install|logs|recent-logs|validate|backup|clean|maintenance]"
                exit 1
                ;;
        esac
    fi
}

# Trap Ctrl+C for graceful shutdown
trap 'echo -e "\n${YELLOW}🛑 Interrupted by user${NC}"; exit 0' INT

# Check if systemctl is available
if ! command -v systemctl &> /dev/null; then
    echo -e "${RED}❌ systemctl not found. This script requires systemd.${NC}"
    exit 1
fi

# Run main function
main "$@"