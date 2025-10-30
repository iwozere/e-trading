#!/bin/bash

# Trading Bot System Health Monitor
# Comprehensive health monitoring script for Raspberry Pi deployment

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
ALERT_EMAIL=""  # Set this to receive email alerts
TELEGRAM_CHAT_ID=""  # Set this for Telegram alerts

# Thresholds
CPU_TEMP_WARNING=70
CPU_TEMP_CRITICAL=80
MEMORY_WARNING=80
MEMORY_CRITICAL=90
DISK_WARNING=80
DISK_CRITICAL=90
LOAD_WARNING=2.0
LOAD_CRITICAL=4.0

# Functions
print_header() {
    echo -e "${BLUE}"
    echo "🏥 Trading Bot System Health Monitor"
    echo "===================================="
    echo "$(date)"
    echo -e "${NC}"
}

check_service_health() {
    echo -e "${CYAN}🤖 Service Health Check:${NC}"
    echo "========================"
    
    local status="unknown"
    local health_score=0
    
    # Check if service is running
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        echo -e "${GREEN}✅ Service Status: Running${NC}"
        status="running"
        health_score=$((health_score + 25))
    else
        echo -e "${RED}❌ Service Status: Stopped${NC}"
        status="stopped"
        return 1
    fi
    
    # Check service uptime
    local start_time=$(systemctl show "$SERVICE_NAME" --property=ActiveEnterTimestamp --value)
    if [ -n "$start_time" ] && [ "$start_time" != "n/a" ]; then
        echo "Service Started: $start_time"
        health_score=$((health_score + 25))
    fi
    
    # Check memory usage
    local pid=$(systemctl show "$SERVICE_NAME" --property=MainPID --value)
    if [ "$pid" != "0" ] && [ -n "$pid" ]; then
        local memory_mb=$(ps -p $pid -o rss= | awk '{print $1/1024}')
        local cpu_percent=$(ps -p $pid -o %cpu= | awk '{print $1}')
        
        echo "Process ID: $pid"
        echo "Memory Usage: ${memory_mb} MB"
        echo "CPU Usage: ${cpu_percent}%"
        
        # Check if memory usage is reasonable (less than 256MB)
        if (( $(echo "$memory_mb < 256" | bc -l) )); then
            health_score=$((health_score + 25))
        else
            echo -e "${YELLOW}⚠️  High memory usage${NC}"
        fi
        
        # Check if CPU usage is reasonable (less than 50%)
        if (( $(echo "$cpu_percent < 50" | bc -l) )); then
            health_score=$((health_score + 25))
        else
            echo -e "${YELLOW}⚠️  High CPU usage${NC}"
        fi
    fi
    
    # Overall health score
    echo "Health Score: $health_score/100"
    
    if [ $health_score -ge 75 ]; then
        echo -e "${GREEN}✅ Service Health: Excellent${NC}"
    elif [ $health_score -ge 50 ]; then
        echo -e "${YELLOW}⚠️  Service Health: Good${NC}"
    else
        echo -e "${RED}❌ Service Health: Poor${NC}"
    fi
    
    echo
}

check_system_resources() {
    echo -e "${CYAN}💻 System Resources:${NC}"
    echo "===================="
    
    local alerts=()
    
    # CPU Temperature (Raspberry Pi specific)
    if [ -f /sys/class/thermal/thermal_zone0/temp ]; then
        local temp=$(cat /sys/class/thermal/thermal_zone0/temp)
        local temp_c=$((temp/1000))
        
        echo "CPU Temperature: ${temp_c}°C"
        
        if [ $temp_c -gt $CPU_TEMP_CRITICAL ]; then
            echo -e "${RED}🔥 CRITICAL: CPU temperature too high!${NC}"
            alerts+=("CPU temperature critical: ${temp_c}°C")
        elif [ $temp_c -gt $CPU_TEMP_WARNING ]; then
            echo -e "${YELLOW}⚠️  WARNING: CPU temperature high${NC}"
            alerts+=("CPU temperature warning: ${temp_c}°C")
        else
            echo -e "${GREEN}✅ CPU temperature OK${NC}"
        fi
    fi
    
    # Memory Usage
    local memory_info=$(free | grep Mem)
    local total_mem=$(echo $memory_info | awk '{print $2}')
    local used_mem=$(echo $memory_info | awk '{print $3}')
    local memory_percent=$((used_mem * 100 / total_mem))
    
    echo "Memory Usage: ${memory_percent}% ($(echo $memory_info | awk '{print $3/1024/1024}' | cut -d. -f1)MB / $(echo $memory_info | awk '{print $2/1024/1024}' | cut -d. -f1)MB)"
    
    if [ $memory_percent -gt $MEMORY_CRITICAL ]; then
        echo -e "${RED}❌ CRITICAL: Memory usage too high!${NC}"
        alerts+=("Memory usage critical: ${memory_percent}%")
    elif [ $memory_percent -gt $MEMORY_WARNING ]; then
        echo -e "${YELLOW}⚠️  WARNING: Memory usage high${NC}"
        alerts+=("Memory usage warning: ${memory_percent}%")
    else
        echo -e "${GREEN}✅ Memory usage OK${NC}"
    fi
    
    # Disk Usage
    local disk_info=$(df "$PROJECT_DIR" 2>/dev/null || df /)
    local disk_percent=$(echo "$disk_info" | tail -1 | awk '{print $5}' | sed 's/%//')
    
    echo "Disk Usage: ${disk_percent}%"
    
    if [ $disk_percent -gt $DISK_CRITICAL ]; then
        echo -e "${RED}❌ CRITICAL: Disk usage too high!${NC}"
        alerts+=("Disk usage critical: ${disk_percent}%")
    elif [ $disk_percent -gt $DISK_WARNING ]; then
        echo -e "${YELLOW}⚠️  WARNING: Disk usage high${NC}"
        alerts+=("Disk usage warning: ${disk_percent}%")
    else
        echo -e "${GREEN}✅ Disk usage OK${NC}"
    fi
    
    # Load Average
    local load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    echo "Load Average (1min): $load_avg"
    
    if (( $(echo "$load_avg > $LOAD_CRITICAL" | bc -l) )); then
        echo -e "${RED}❌ CRITICAL: System load too high!${NC}"
        alerts+=("System load critical: $load_avg")
    elif (( $(echo "$load_avg > $LOAD_WARNING" | bc -l) )); then
        echo -e "${YELLOW}⚠️  WARNING: System load high${NC}"
        alerts+=("System load warning: $load_avg")
    else
        echo -e "${GREEN}✅ System load OK${NC}"
    fi
    
    echo
    
    # Return alerts for notification
    if [ ${#alerts[@]} -gt 0 ]; then
        return 1
    else
        return 0
    fi
}

check_network_connectivity() {
    echo -e "${CYAN}🌐 Network Connectivity:${NC}"
    echo "======================="
    
    local network_ok=true
    
    # Internet connectivity
    if ping -c 1 -W 5 8.8.8.8 &>/dev/null; then
        echo -e "${GREEN}✅ Internet connection OK${NC}"
    else
        echo -e "${RED}❌ No internet connection${NC}"
        network_ok=false
    fi
    
    # DNS resolution
    if nslookup google.com &>/dev/null; then
        echo -e "${GREEN}✅ DNS resolution OK${NC}"
    else
        echo -e "${RED}❌ DNS resolution failed${NC}"
        network_ok=false
    fi
    
    # Binance API connectivity
    if curl -s --max-time 10 "https://testnet.binance.vision/api/v3/ping" &>/dev/null; then
        echo -e "${GREEN}✅ Binance testnet reachable${NC}"
    else
        echo -e "${RED}❌ Cannot reach Binance testnet${NC}"
        network_ok=false
    fi
    
    # Check API response time
    local response_time=$(curl -s -w "%{time_total}" -o /dev/null --max-time 10 "https://testnet.binance.vision/api/v3/time" 2>/dev/null || echo "timeout")
    if [ "$response_time" != "timeout" ]; then
        echo "Binance API Response Time: ${response_time}s"
        if (( $(echo "$response_time > 2.0" | bc -l) )); then
            echo -e "${YELLOW}⚠️  Slow API response time${NC}"
        fi
    fi
    
    echo
    
    if [ "$network_ok" = false ]; then
        return 1
    else
        return 0
    fi
}

check_log_health() {
    echo -e "${CYAN}📝 Log Health Check:${NC}"
    echo "==================="
    
    local log_issues=()
    
    # Check if log directory exists and is writable
    if [ -d "$LOG_DIR" ] && [ -w "$LOG_DIR" ]; then
        echo -e "${GREEN}✅ Log directory accessible${NC}"
    else
        echo -e "${RED}❌ Log directory not accessible${NC}"
        log_issues+=("Log directory not accessible")
    fi
    
    # Check log file size
    if [ -f "$LOG_DIR/trading-bot.log" ]; then
        local log_size=$(du -h "$LOG_DIR/trading-bot.log" | cut -f1)
        echo "Log file size: $log_size"
        
        # Check for recent log entries (last 5 minutes)
        local recent_logs=$(find "$LOG_DIR/trading-bot.log" -newermt "5 minutes ago" 2>/dev/null)
        if [ -n "$recent_logs" ]; then
            echo -e "${GREEN}✅ Recent log activity detected${NC}"
        else
            echo -e "${YELLOW}⚠️  No recent log activity${NC}"
            log_issues+=("No recent log activity")
        fi
        
        # Check for error patterns in recent logs
        local error_count=$(tail -100 "$LOG_DIR/trading-bot.log" 2>/dev/null | grep -i "error\|exception\|failed" | wc -l)
        if [ $error_count -gt 5 ]; then
            echo -e "${YELLOW}⚠️  High error count in recent logs: $error_count${NC}"
            log_issues+=("High error count: $error_count")
        else
            echo -e "${GREEN}✅ Error count acceptable: $error_count${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Log file not found${NC}"
        log_issues+=("Log file not found")
    fi
    
    # Check systemd journal
    local journal_errors=$(journalctl -u "$SERVICE_NAME" --since "1 hour ago" -p err --no-pager -q | wc -l)
    if [ $journal_errors -gt 0 ]; then
        echo -e "${YELLOW}⚠️  Journal errors in last hour: $journal_errors${NC}"
        log_issues+=("Journal errors: $journal_errors")
    else
        echo -e "${GREEN}✅ No journal errors in last hour${NC}"
    fi
    
    echo
    
    if [ ${#log_issues[@]} -gt 0 ]; then
        return 1
    else
        return 0
    fi
}

check_configuration() {
    echo -e "${CYAN}⚙️  Configuration Health:${NC}"
    echo "======================="
    
    local config_issues=()
    
    # Check main config file
    if [ -f "$PROJECT_DIR/config/trading/paper_trading_config.json" ]; then
        if python3 -c "import json; json.load(open('$PROJECT_DIR/config/trading/paper_trading_config.json'))" 2>/dev/null; then
            echo -e "${GREEN}✅ Main configuration valid${NC}"
        else
            echo -e "${RED}❌ Main configuration invalid JSON${NC}"
            config_issues+=("Invalid main configuration")
        fi
    else
        echo -e "${RED}❌ Main configuration file not found${NC}"
        config_issues+=("Main configuration missing")
    fi
    
    # Check environment file
    if [ -f "$CONFIG_DIR/trading-bot.env" ]; then
        echo -e "${GREEN}✅ Environment file exists${NC}"
        
        # Check for required variables
        source "$CONFIG_DIR/trading-bot.env"
        if [ -n "$BINANCE_API_KEY" ] && [ -n "$BINANCE_API_SECRET" ]; then
            echo -e "${GREEN}✅ API keys configured${NC}"
        else
            echo -e "${YELLOW}⚠️  API keys not configured${NC}"
            config_issues+=("API keys missing")
        fi
    else
        echo -e "${RED}❌ Environment file not found${NC}"
        config_issues+=("Environment file missing")
    fi
    
    # Check database
    if [ -f "$PROJECT_DIR/db/trading.db" ]; then
        echo -e "${GREEN}✅ Database file exists${NC}"
    else
        echo -e "${YELLOW}⚠️  Database file not found (will be created)${NC}"
    fi
    
    echo
    
    if [ ${#config_issues[@]} -gt 0 ]; then
        return 1
    else
        return 0
    fi
}

generate_health_report() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local report_file="/tmp/trading-bot-health-$(date +%Y%m%d-%H%M%S).txt"
    
    {
        echo "Trading Bot Health Report"
        echo "========================="
        echo "Generated: $timestamp"
        echo "Hostname: $(hostname)"
        echo
        
        check_service_health
        check_system_resources
        check_network_connectivity
        check_log_health
        check_configuration
        
    } > "$report_file"
    
    echo "Health report saved: $report_file"
    return "$report_file"
}

send_alert() {
    local message="$1"
    local severity="$2"
    
    # Email alert
    if [ -n "$ALERT_EMAIL" ] && command -v mail &> /dev/null; then
        echo "$message" | mail -s "Trading Bot Alert [$severity]" "$ALERT_EMAIL"
    fi
    
    # Telegram alert
    if [ -n "$TELEGRAM_CHAT_ID" ] && [ -n "$TELEGRAM_BOT_TOKEN" ]; then
        curl -s -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
            -d chat_id="$TELEGRAM_CHAT_ID" \
            -d text="🚨 Trading Bot Alert [$severity]: $message" \
            &>/dev/null
    fi
}

run_health_check() {
    local issues=()
    local warnings=()
    
    echo -e "${BLUE}🏥 Running Comprehensive Health Check...${NC}"
    echo
    
    # Service health
    if ! check_service_health; then
        issues+=("Service not running properly")
    fi
    
    # System resources
    if ! check_system_resources; then
        issues+=("System resource issues detected")
    fi
    
    # Network connectivity
    if ! check_network_connectivity; then
        issues+=("Network connectivity problems")
    fi
    
    # Log health
    if ! check_log_health; then
        warnings+=("Log health issues detected")
    fi
    
    # Configuration
    if ! check_configuration; then
        issues+=("Configuration problems detected")
    fi
    
    # Summary
    echo -e "${CYAN}📊 Health Check Summary:${NC}"
    echo "======================="
    
    if [ ${#issues[@]} -eq 0 ] && [ ${#warnings[@]} -eq 0 ]; then
        echo -e "${GREEN}✅ All systems healthy!${NC}"
        return 0
    elif [ ${#issues[@]} -eq 0 ]; then
        echo -e "${YELLOW}⚠️  System healthy with warnings:${NC}"
        for warning in "${warnings[@]}"; do
            echo "  - $warning"
        done
        return 1
    else
        echo -e "${RED}❌ Critical issues detected:${NC}"
        for issue in "${issues[@]}"; do
            echo "  - $issue"
        done
        
        if [ ${#warnings[@]} -gt 0 ]; then
            echo -e "${YELLOW}⚠️  Additional warnings:${NC}"
            for warning in "${warnings[@]}"; do
                echo "  - $warning"
            done
        fi
        
        # Send alert for critical issues
        local alert_message="Trading Bot Health Check Failed: $(IFS=', '; echo "${issues[*]}")"
        send_alert "$alert_message" "CRITICAL"
        
        return 2
    fi
}

# Main execution
main() {
    case "${1:-check}" in
        check|health)
            print_header
            run_health_check
            ;;
        service)
            check_service_health
            ;;
        system)
            check_system_resources
            ;;
        network)
            check_network_connectivity
            ;;
        logs)
            check_log_health
            ;;
        config)
            check_configuration
            ;;
        report)
            print_header
            generate_health_report
            ;;
        monitor)
            echo -e "${BLUE}🔄 Continuous Health Monitoring (Press Ctrl+C to stop)${NC}"
            echo
            while true; do
                clear
                print_header
                run_health_check
                echo
                echo "Next check in 60 seconds..."
                sleep 60
            done
            ;;
        help|--help|-h)
            echo "Usage: $0 [COMMAND]"
            echo
            echo "Commands:"
            echo "  check      Run complete health check (default)"
            echo "  service    Check service health only"
            echo "  system     Check system resources only"
            echo "  network    Check network connectivity only"
            echo "  logs       Check log health only"
            echo "  config     Check configuration only"
            echo "  report     Generate detailed health report"
            echo "  monitor    Continuous monitoring mode"
            echo "  help       Show this help"
            echo
            ;;
        *)
            echo -e "${RED}❌ Unknown command: $1${NC}"
            echo "Use '$0 help' for available commands"
            exit 1
            ;;
    esac
}

# Trap Ctrl+C for graceful shutdown in monitor mode
trap 'echo -e "\n${YELLOW}🛑 Monitoring stopped${NC}"; exit 0' INT

# Run main function
main "$@"