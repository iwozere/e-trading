#!/bin/bash

# Trading Bot Development Mode Runner
# This script runs the trading bot in development mode with enhanced debugging

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
VENV_DIR="$PROJECT_ROOT/.venv"
LOG_DIR="$PROJECT_ROOT/logs"
CONFIG_DIR="$PROJECT_ROOT/config"
DEFAULT_CONFIG="paper_trading_dev.json"

# Functions
print_header() {
    echo -e "${BLUE}"
    echo "🤖 Trading Bot Development Mode"
    echo "==============================="
    echo -e "${NC}"
}

print_system_info() {
    echo -e "${CYAN}📊 System Information:${NC}"
    echo "Date: $(date)"
    echo "OS: $(uname -s) $(uname -r)"
    echo "Architecture: $(uname -m)"
    echo "Python: $(python3 --version 2>/dev/null || echo 'Not found')"
    echo "Working Directory: $PROJECT_ROOT"
    
    # Check if on Raspberry Pi
    if grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
        echo "Platform: 🥧 Raspberry Pi"
        if [ -f /sys/class/thermal/thermal_zone0/temp ]; then
            temp=$(cat /sys/class/thermal/thermal_zone0/temp)
            temp_c=$((temp/1000))
            echo "CPU Temperature: ${temp_c}°C"
        fi
    else
        echo "Platform: 💻 Desktop/Server"
    fi
    
    echo "Memory: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
    echo
}

check_dependencies() {
    echo -e "${CYAN}🔍 Checking Dependencies:${NC}"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 not found${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Python 3 found${NC}"
    
    # Check virtual environment
    if [ ! -d "$VENV_DIR" ]; then
        echo -e "${YELLOW}⚠️  Virtual environment not found, creating...${NC}"
        python3 -m venv "$VENV_DIR"
        echo -e "${GREEN}✅ Virtual environment created${NC}"
    else
        echo -e "${GREEN}✅ Virtual environment found${NC}"
    fi
    
    # Check requirements
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        echo -e "${YELLOW}📦 Installing/updating requirements...${NC}"
        "$VENV_DIR/bin/pip" install -q --upgrade pip
        "$VENV_DIR/bin/pip" install -q -r "$PROJECT_ROOT/requirements.txt"
        echo -e "${GREEN}✅ Requirements installed${NC}"
    fi
    
    echo
}

setup_environment() {
    echo -e "${CYAN}🔧 Setting up Environment:${NC}"
    
    # Create directories
    mkdir -p "$LOG_DIR"
    mkdir -p "$CONFIG_DIR/trading"
    mkdir -p "$PROJECT_ROOT/data"
    mkdir -p "$PROJECT_ROOT/db"
    
    # Set Python path
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    
    # Load environment variables if .env exists
    if [ -f "$PROJECT_ROOT/.env" ]; then
        echo -e "${GREEN}✅ Loading .env file${NC}"
        set -a
        source "$PROJECT_ROOT/.env"
        set +a
    else
        echo -e "${YELLOW}⚠️  No .env file found${NC}"
    fi
    
    echo -e "${GREEN}✅ Environment setup complete${NC}"
    echo
}

create_dev_config() {
    local config_file="$CONFIG_DIR/trading/$DEFAULT_CONFIG"
    
    if [ ! -f "$config_file" ]; then
        echo -e "${CYAN}📝 Creating development configuration...${NC}"
        
        cat > "$config_file" << EOF
{
  "bot_id": "dev_paper_trading_$(date +%s)",
  "environment": "development",
  "version": "1.0.0-dev",
  "description": "Development paper trading bot with enhanced debugging",
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "risk_per_trade": 0.01,
  "max_open_trades": 2,
  "position_size": 0.05,
  "broker_type": "binance_paper",
  "initial_balance": 1000.0,
  "commission": 0.001,
  "data_source": "binance",
  "lookback_bars": 500,
  "retry_interval": 30,
  "strategy_type": "custom",
  "strategy_params": {
    "entry_logic": {
      "name": "RSIOrBBEntryMixin",
      "params": {
        "e_rsi_period": 14,
        "e_rsi_oversold": 30,
        "e_bb_period": 20,
        "e_bb_dev": 2.0,
        "e_cooldown_bars": 3
      }
    },
    "exit_logic": {
      "name": "ATRExitMixin",
      "params": {
        "x_atr_period": 14,
        "x_sl_multiplier": 1.5
      }
    },
    "use_talib": true
  },
  "paper_trading": true,
  "log_level": "DEBUG",
  "log_file": "$LOG_DIR/trading-bot-dev.log",
  "max_daily_loss": 50.0,
  "max_drawdown_pct": 10.0,
  "notifications_enabled": true,
  "telegram_enabled": false,
  "email_enabled": false,
  "development_mode": true,
  "debug_signals": true,
  "debug_execution": true
}
EOF
        echo -e "${GREEN}✅ Development configuration created: $config_file${NC}"
    else
        echo -e "${GREEN}✅ Development configuration exists: $config_file${NC}"
    fi
    echo
}

validate_config() {
    local config_file="$1"
    
    echo -e "${CYAN}🔍 Validating Configuration:${NC}"
    
    if [ ! -f "$config_file" ]; then
        echo -e "${RED}❌ Configuration file not found: $config_file${NC}"
        exit 1
    fi
    
    # Basic JSON validation
    if ! python3 -c "import json; json.load(open('$config_file'))" 2>/dev/null; then
        echo -e "${RED}❌ Invalid JSON in configuration file${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Configuration file is valid JSON${NC}"
    
    # Check required API keys for paper trading
    if [ -z "$BINANCE_API_KEY" ] || [ -z "$BINANCE_API_SECRET" ]; then
        echo -e "${YELLOW}⚠️  Warning: Binance API keys not set${NC}"
        echo -e "${YELLOW}   Set BINANCE_API_KEY and BINANCE_API_SECRET in .env file${NC}"
        echo -e "${YELLOW}   Or export them as environment variables${NC}"
    else
        echo -e "${GREEN}✅ Binance API keys configured${NC}"
    fi
    
    echo
}

show_menu() {
    echo -e "${PURPLE}🎛️  Development Menu:${NC}"
    echo "1. Run with default config ($DEFAULT_CONFIG)"
    echo "2. Run with custom config"
    echo "3. Validate configuration only"
    echo "4. Show system monitor"
    echo "5. View recent logs"
    echo "6. Clean logs and data"
    echo "7. Run tests"
    echo "8. Exit"
    echo
}

run_bot() {
    local config_file="$1"
    local config_path="$CONFIG_DIR/trading/$config_file"
    
    echo -e "${CYAN}🚀 Starting Trading Bot in Development Mode...${NC}"
    echo "Configuration: $config_path"
    echo "Log Level: DEBUG"
    echo "Paper Trading: Enabled"
    echo
    
    # Validate configuration
    validate_config "$config_path"
    
    echo -e "${GREEN}🎯 Bot Starting...${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    echo "=================================="
    echo
    
    # Run the bot with enhanced error handling
    cd "$PROJECT_ROOT"
    
    # Set development environment variables
    export TRADING_ENV="development"
    export DEBUG_MODE="true"
    export LOG_LEVEL="DEBUG"
    
    # Run with Python unbuffered output for real-time logs
    "$VENV_DIR/bin/python" -u -m src.trading.run_bot "$config_file" 2>&1 | tee -a "$LOG_DIR/dev-session-$(date +%Y%m%d-%H%M%S).log"
}

show_system_monitor() {
    echo -e "${CYAN}📊 System Monitor:${NC}"
    echo "=================="
    
    # System stats
    echo "Uptime: $(uptime -p)"
    echo "Load Average: $(uptime | awk -F'load average:' '{print $2}')"
    echo
    
    # Memory usage
    echo "Memory Usage:"
    free -h
    echo
    
    # Disk usage
    echo "Disk Usage:"
    df -h "$PROJECT_ROOT" | tail -1
    echo
    
    # Process information
    echo "Python Processes:"
    ps aux | grep python | grep -v grep || echo "No Python processes found"
    echo
    
    # Temperature (if Raspberry Pi)
    if [ -f /sys/class/thermal/thermal_zone0/temp ]; then
        temp=$(cat /sys/class/thermal/thermal_zone0/temp)
        temp_c=$((temp/1000))
        echo "CPU Temperature: ${temp_c}°C"
        if [ $temp_c -gt 70 ]; then
            echo -e "${RED}⚠️  High temperature warning!${NC}"
        fi
        echo
    fi
    
    # Network connectivity test
    echo "Network Connectivity:"
    if ping -c 1 8.8.8.8 &> /dev/null; then
        echo -e "${GREEN}✅ Internet connection OK${NC}"
    else
        echo -e "${RED}❌ No internet connection${NC}"
    fi
    
    # Binance API test (if keys are set)
    if [ -n "$BINANCE_API_KEY" ] && [ -n "$BINANCE_API_SECRET" ]; then
        echo "Binance API Test:"
        if curl -s "https://testnet.binance.vision/api/v3/ping" &> /dev/null; then
            echo -e "${GREEN}✅ Binance testnet reachable${NC}"
        else
            echo -e "${RED}❌ Cannot reach Binance testnet${NC}"
        fi
    fi
    echo
}

view_logs() {
    echo -e "${CYAN}📝 Recent Logs:${NC}"
    echo "==============="
    
    if [ -f "$LOG_DIR/trading-bot-dev.log" ]; then
        echo "Last 20 lines from development log:"
        tail -20 "$LOG_DIR/trading-bot-dev.log"
    else
        echo "No development log found"
    fi
    
    echo
    echo "Available log files:"
    ls -la "$LOG_DIR"/*.log 2>/dev/null || echo "No log files found"
    echo
}

clean_data() {
    echo -e "${CYAN}🧹 Cleaning Development Data:${NC}"
    
    read -p "This will delete logs and database files. Continue? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -f "$LOG_DIR"/*.log
        rm -f "$PROJECT_ROOT/db"/*.db
        rm -rf "$PROJECT_ROOT/data/cache"
        echo -e "${GREEN}✅ Development data cleaned${NC}"
    else
        echo "Cancelled"
    fi
    echo
}

run_tests() {
    echo -e "${CYAN}🧪 Running Tests:${NC}"
    
    if [ -d "$PROJECT_ROOT/tests" ]; then
        cd "$PROJECT_ROOT"
        "$VENV_DIR/bin/python" -m pytest tests/ -v
    else
        echo "No tests directory found"
    fi
    echo
}

# Main execution
main() {
    print_header
    print_system_info
    check_dependencies
    setup_environment
    create_dev_config
    
    # Handle command line arguments
    if [ $# -eq 0 ]; then
        # Interactive mode
        while true; do
            show_menu
            read -p "Select option (1-8): " choice
            echo
            
            case $choice in
                1)
                    run_bot "$DEFAULT_CONFIG"
                    ;;
                2)
                    read -p "Enter config file name: " custom_config
                    if [ -n "$custom_config" ]; then
                        run_bot "$custom_config"
                    fi
                    ;;
                3)
                    read -p "Enter config file name: " config_to_validate
                    if [ -n "$config_to_validate" ]; then
                        validate_config "$CONFIG_DIR/trading/$config_to_validate"
                    fi
                    ;;
                4)
                    show_system_monitor
                    ;;
                5)
                    view_logs
                    ;;
                6)
                    clean_data
                    ;;
                7)
                    run_tests
                    ;;
                8)
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
            --config|-c)
                if [ -n "$2" ]; then
                    run_bot "$2"
                else
                    echo -e "${RED}❌ Config file name required${NC}"
                    exit 1
                fi
                ;;
            --validate|-v)
                if [ -n "$2" ]; then
                    validate_config "$CONFIG_DIR/trading/$2"
                else
                    echo -e "${RED}❌ Config file name required${NC}"
                    exit 1
                fi
                ;;
            --monitor|-m)
                show_system_monitor
                ;;
            --logs|-l)
                view_logs
                ;;
            --clean)
                clean_data
                ;;
            --test|-t)
                run_tests
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo
                echo "Options:"
                echo "  -c, --config FILE    Run with specific config file"
                echo "  -v, --validate FILE  Validate config file"
                echo "  -m, --monitor        Show system monitor"
                echo "  -l, --logs           View recent logs"
                echo "  --clean              Clean development data"
                echo "  -t, --test           Run tests"
                echo "  -h, --help           Show this help"
                echo
                echo "Examples:"
                echo "  $0                           # Interactive mode"
                echo "  $0 -c my_config.json        # Run with specific config"
                echo "  $0 -v my_config.json        # Validate config"
                echo "  $0 -m                        # Show system monitor"
                ;;
            *)
                # Assume it's a config file name
                run_bot "$1"
                ;;
        esac
    fi
}

# Trap Ctrl+C for graceful shutdown
trap 'echo -e "\n${YELLOW}🛑 Shutting down gracefully...${NC}"; exit 0' INT

# Run main function
main "$@"