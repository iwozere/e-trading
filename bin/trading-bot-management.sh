#!/bin/bash

# Trading Bot Service Management Script
# Provides easy management of the trading bot systemd service

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="trading-bot"
LOG_DIR="/var/log/trading-bot"
PROJECT_DIR="/opt/e-trading"
CONFIG_DIR="/etc/trading-bot"

# Functions
print_header() {
    echo -e "${BLUE}"
    echo "🤖 Trading Bot Service Manager"
    echo "=============================="
    echo -e "${NC}"
}

check_service_exists() {
    if ! systemctl list-unit-files | grep -q "^$SERVICE_NAME.service"; then
        echo -e "${RED}❌ Service $SERVICE_NAME not found${NC}"
        echo "Run the installation script first: sudo bin/trading-bot-install.sh"
        exit 1
    fi
}

get_service_status() {
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        echo -e "${GREEN}🟢 Running${NC}"
    elif systemctl is-enabled --quiet "$SERVICE_NAME"; then
        echo -e "${YELLOW}🟡 Stopped (but enabled)${NC}"
    else
        echo -e "${RED}🔴 Stopped (disabled)${NC}"
    fi
}

show_status() {
    echo -e "${CYAN}📊 Service Status:${NC}"
    echo "=================="
    
    # Service status
    echo -n "Status: "
    get_service_status
    
    # Detailed systemctl status
    echo
    echo "Detailed Status:"
    systemctl status "$SERVICE_NAME" --no-pager -l
    
    echo
    echo -e "${CYAN}📈 Performance Metrics:${NC}"
    
    # Memory usage
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        local pid=$(systemctl show "$SERVICE_NAME" --property=MainPID --value)
        if [ "$pid" != "0" ] && [ -n "$pid" ]; then
            echo "Memory Usage: $(ps -p $pid -o rss= | awk '{print $1/1024 " MB"}')"
            echo "CPU Usage: $(ps -p $pid -o %cpu= | awk '{print $1"%"}')"
        fi
    fi
    
    # Uptime
    local start_time=$(systemctl show "$SERVICE_NAME" --property=ActiveEnterTimestamp --value)
    if [ -n "$start_time" ] && [ "$start_time" != "n/a" ]; then
        echo "Started: $start_time"
    fi
    
    echo
}

start_service() {
    echo -e "${CYAN}🚀 Starting trading bot service...${NC}"
    
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        echo -e "${YELLOW}⚠️  Service is already running${NC}"
        return 0
    fi
    
    if sudo systemctl start "$SERVICE_NAME"; then
        echo -e "${GREEN}✅ Service started successfully${NC}"
        
        # Wait a moment and check if it's still running
        sleep 2
        if systemctl is-active --quiet "$SERVICE_NAME"; then
            echo -e "${GREEN}✅ Service is running properly${NC}"
        else
            echo -e "${RED}❌ Service failed to start properly${NC}"
            echo "Check logs with: $0 logs"
        fi
    else
        echo -e "${RED}❌ Failed to start service${NC}"
        return 1
    fi
}

stop_service() {
    echo -e "${CYAN}🛑 Stopping trading bot service...${NC}"
    
    if ! systemctl is-active --quiet "$SERVICE_NAME"; then
        echo -e "${YELLOW}⚠️  Service is already stopped${NC}"
        return 0
    fi
    
    if sudo systemctl stop "$SERVICE_NAME"; then
        echo -e "${GREEN}✅ Service stopped successfully${NC}"
    else
        echo -e "${RED}❌ Failed to stop service${NC}"
        return 1
    fi
}

restart_service() {
    echo -e "${CYAN}🔄 Restarting trading bot service...${NC}"
    
    if sudo systemctl restart "$SERVICE_NAME"; then
        echo -e "${GREEN}✅ Service restarted successfully${NC}"
        
        # Wait a moment and check if it's running
        sleep 2
        if systemctl is-active --quiet "$SERVICE_NAME"; then
            echo -e "${GREEN}✅ Service is running properly${NC}"
        else
            echo -e "${RED}❌ Service failed to restart properly${NC}"
            echo "Check logs with: $0 logs"
        fi
    else
        echo -e "${RED}❌ Failed to restart service${NC}"
        return 1
    fi
}

reload_service() {
    echo -e "${CYAN}🔄 Reloading trading bot service configuration...${NC}"
    
    if sudo systemctl reload "$SERVICE_NAME"; then
        echo -e "${GREEN}✅ Service configuration reloaded${NC}"
    else
        echo -e "${YELLOW}⚠️  Reload not supported, restarting instead...${NC}"
        restart_service
    fi
}

enable_service() {
    echo -e "${CYAN}⚡ Enabling trading bot service...${NC}"
    
    if sudo systemctl enable "$SERVICE_NAME"; then
        echo -e "${GREEN}✅ Service enabled (will start on boot)${NC}"
    else
        echo -e "${RED}❌ Failed to enable service${NC}"
        return 1
    fi
}

disable_service() {
    echo -e "${CYAN}⏸️  Disabling trading bot service...${NC}"
    
    if sudo systemctl disable "$SERVICE_NAME"; then
        echo -e "${GREEN}✅ Service disabled (will not start on boot)${NC}"
    else
        echo -e "${RED}❌ Failed to disable service${NC}"
        return 1
    fi
}

show_logs() {
    local lines="${2:-50}"
    local follow="${3:-false}"
    
    echo -e "${CYAN}📝 Service Logs:${NC}"
    echo "================"
    
    if [ "$follow" = "true" ]; then
        echo "Following logs (Press Ctrl+C to stop)..."
        echo
        sudo journalctl -u "$SERVICE_NAME" -f
    else
        echo "Last $lines lines:"
        echo
        sudo journalctl -u "$SERVICE_NAME" -n "$lines" --no-pager
    fi
}

show_file_logs() {
    local lines="${2:-50}"
    local follow="${3:-false}"
    
    echo -e "${CYAN}📄 File Logs:${NC}"
    echo "============="
    
    if [ -f "$LOG_DIR/trading-bot.log" ]; then
        if [ "$follow" = "true" ]; then
            echo "Following file logs (Press Ctrl+C to stop)..."
            echo
            tail -f "$LOG_DIR/trading-bot.log"
        else
            echo "Last $lines lines from file:"
            echo
            tail -n "$lines" "$LOG_DIR/trading-bot.log"
        fi
    else
        echo -e "${YELLOW}⚠️  Log file not found: $LOG_DIR/trading-bot.log${NC}"
    fi
}

show_system_info() {
    echo -e "${CYAN}🖥️  System Information:${NC}"
    echo "======================"
    
    # Basic system info
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo "Uptime: $(uptime -p)"
    echo "Load Average: $(uptime | awk -F'load average:' '{print $2}')"
    echo
    
    # Memory and disk
    echo "Memory Usage:"
    free -h
    echo
    
    echo "Disk Usage:"
    df -h "$PROJECT_DIR" 2>/dev/null || df -h /
    echo
    
    # Temperature (Raspberry Pi)
    if [ -f /sys/class/thermal/thermal_zone0/temp ]; then
        temp=$(cat /sys/class/thermal/thermal_zone0/temp)
        temp_c=$((temp/1000))
        echo "CPU Temperature: ${temp_c}°C"
        
        if [ $temp_c -gt 80 ]; then
            echo -e "${RED}🔥 Critical temperature!${NC}"
        elif [ $temp_c -gt 70 ]; then
            echo -e "${YELLOW}⚠️  High temperature${NC}"
        else
            echo -e "${GREEN}✅ Temperature OK${NC}"
        fi
        echo
    fi
    
    # Network connectivity
    echo "Network Status:"
    if ping -c 1 -W 3 8.8.8.8 &>/dev/null; then
        echo -e "${GREEN}✅ Internet connection OK${NC}"
    else
        echo -e "${RED}❌ No internet connection${NC}"
    fi
    
    # Binance API connectivity
    if curl -s --max-time 5 "https://testnet.binance.vision/api/v3/ping" &>/dev/null; then
        echo -e "${GREEN}✅ Binance testnet reachable${NC}"
    else
        echo -e "${RED}❌ Cannot reach Binance testnet${NC}"
    fi
    echo
}

edit_config() {
    echo -e "${CYAN}⚙️  Configuration Files:${NC}"
    echo "======================="
    
    echo "1. Main configuration: $PROJECT_DIR/config/trading/"
    echo "2. Environment variables: $CONFIG_DIR/trading-bot.env"
    echo "3. Service file: /etc/systemd/system/$SERVICE_NAME.service"
    echo
    
    read -p "Which would you like to edit? (1-3): " choice
    
    case $choice in
        1)
            echo "Available configuration files:"
            ls -la "$PROJECT_DIR/config/trading/"*.json 2>/dev/null || echo "No config files found"
            echo
            read -p "Enter config filename: " config_file
            if [ -n "$config_file" ]; then
                sudo -u trading nano "$PROJECT_DIR/config/trading/$config_file"
            fi
            ;;
        2)
            sudo nano "$CONFIG_DIR/trading-bot.env"
            echo -e "${YELLOW}⚠️  Restart service to apply environment changes${NC}"
            ;;
        3)
            sudo nano "/etc/systemd/system/$SERVICE_NAME.service"
            echo -e "${YELLOW}⚠️  Run 'sudo systemctl daemon-reload' after editing${NC}"
            ;;
        *)
            echo -e "${RED}❌ Invalid choice${NC}"
            ;;
    esac
}

backup_config() {
    local backup_dir="$PROJECT_DIR/backups/$(date +%Y%m%d-%H%M%S)"
    
    echo -e "${CYAN}💾 Creating configuration backup...${NC}"
    
    sudo mkdir -p "$backup_dir"
    
    # Backup configurations
    sudo cp -r "$PROJECT_DIR/config" "$backup_dir/"
    sudo cp "$CONFIG_DIR/trading-bot.env" "$backup_dir/"
    sudo cp "/etc/systemd/system/$SERVICE_NAME.service" "$backup_dir/"
    
    # Backup database
    if [ -f "$PROJECT_DIR/db/trading.db" ]; then
        sudo cp "$PROJECT_DIR/db/trading.db" "$backup_dir/"
    fi
    
    sudo chown -R trading:trading "$backup_dir"
    
    echo -e "${GREEN}✅ Backup created: $backup_dir${NC}"
}

show_menu() {
    echo -e "${PURPLE}🎛️  Service Management Menu:${NC}"
    echo "1.  Start service"
    echo "2.  Stop service"
    echo "3.  Restart service"
    echo "4.  Service status"
    echo "5.  Enable service (auto-start)"
    echo "6.  Disable service"
    echo "7.  View logs (journal)"
    echo "8.  View logs (file)"
    echo "9.  Follow logs (live)"
    echo "10. System information"
    echo "11. Edit configuration"
    echo "12. Backup configuration"
    echo "13. Exit"
    echo
}

# Main execution
main() {
    print_header
    check_service_exists
    
    # Handle command line arguments
    case "${1:-menu}" in
        start)
            start_service
            ;;
        stop)
            stop_service
            ;;
        restart)
            restart_service
            ;;
        reload)
            reload_service
            ;;
        status)
            show_status
            ;;
        enable)
            enable_service
            ;;
        disable)
            disable_service
            ;;
        logs)
            show_logs "$@"
            ;;
        logs-follow|follow)
            show_logs "" "" "true"
            ;;
        file-logs)
            show_file_logs "$@"
            ;;
        file-logs-follow)
            show_file_logs "" "" "true"
            ;;
        system|info)
            show_system_info
            ;;
        config|edit)
            edit_config
            ;;
        backup)
            backup_config
            ;;
        menu)
            # Interactive menu
            while true; do
                show_menu
                read -p "Select option (1-13): " choice
                echo
                
                case $choice in
                    1) start_service ;;
                    2) stop_service ;;
                    3) restart_service ;;
                    4) show_status ;;
                    5) enable_service ;;
                    6) disable_service ;;
                    7) show_logs ;;
                    8) show_file_logs ;;
                    9) show_logs "" "" "true" ;;
                    10) show_system_info ;;
                    11) edit_config ;;
                    12) backup_config ;;
                    13) 
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
            ;;
        help|--help|-h)
            echo "Usage: $0 [COMMAND]"
            echo
            echo "Commands:"
            echo "  start              Start the service"
            echo "  stop               Stop the service"
            echo "  restart            Restart the service"
            echo "  reload             Reload service configuration"
            echo "  status             Show service status"
            echo "  enable             Enable service (auto-start on boot)"
            echo "  disable            Disable service"
            echo "  logs               Show recent logs"
            echo "  logs-follow        Follow logs in real-time"
            echo "  file-logs          Show recent file logs"
            echo "  file-logs-follow   Follow file logs in real-time"
            echo "  system             Show system information"
            echo "  config             Edit configuration"
            echo "  backup             Backup configuration"
            echo "  menu               Show interactive menu (default)"
            echo "  help               Show this help"
            echo
            ;;
        *)
            echo -e "${RED}❌ Unknown command: $1${NC}"
            echo "Use '$0 help' for available commands"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"