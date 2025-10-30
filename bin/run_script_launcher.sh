#!/bin/bash

# ========================================
# Crypto Trading Platform - Script Launcher
# ========================================

# Get the absolute path to the project root
PROJECT_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"

# Change to project root directory
cd "$PROJECT_ROOT"

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs/log"

# Check if virtual environment exists
if [ ! -f "$PROJECT_ROOT/.venv/bin/python" ]; then
    echo "Error: Python venv not found at $PROJECT_ROOT/.venv/bin/python"
    echo "Please run: python -m venv .venv"
    echo "Then install dependencies: pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$PROJECT_ROOT/.venv/bin/activate"

# Function to check if a process is running
is_running() {
    local script_name=$1
    local pid_file="$PROJECT_ROOT/logs/pids/${script_name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0  # Process is running
        else
            # Clean up stale PID file
            rm -f "$pid_file"
        fi
    fi
    return 1  # Process is not running
}

# Function to start a script in background
start_script() {
    local script_name=$1
    local python_script=$2
    local log_file="$PROJECT_ROOT/logs/log/${script_name}.log"
    local pid_file="$PROJECT_ROOT/logs/pids/${script_name}.pid"
    
    if is_running "$script_name"; then
        echo "❌ $script_name is already running (PID: $(cat "$pid_file"))"
        return 1
    fi
    
    echo "🚀 Starting $script_name..."
    nohup "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/$python_script" > "$log_file" 2>&1 &
    local pid=$!
    echo $pid > "$pid_file"
    echo "✅ $script_name started successfully (PID: $pid)"
    echo "📝 Log file: $log_file"
    return 0
}

# Function to stop a script
stop_script() {
    local script_name=$1
    local pid_file="$PROJECT_ROOT/logs/pids/${script_name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "🛑 Stopping $script_name (PID: $pid)..."
            kill "$pid"
            rm -f "$pid_file"
            echo "✅ $script_name stopped"
        else
            echo "⚠️  $script_name was not running (stale PID file)"
            rm -f "$pid_file"
        fi
    else
        echo "❌ $script_name is not running"
    fi
}

# Function to show logs
show_logs() {
    local script_name=$1
    local log_file="$PROJECT_ROOT/logs/log/${script_name}.log"
    
    if [ -f "$log_file" ]; then
        echo "📋 Recent logs for $script_name:"
        echo "----------------------------------------"
        tail -n 20 "$log_file"
        echo "----------------------------------------"
        echo "Full log file: $log_file"
    else
        echo "❌ No log file found for $script_name"
    fi
}

# Function to show status of all scripts
show_status() {
    echo
    echo "📊 Script Status:"
    echo "=================="
    
    local scripts=(
        "telegram_bot:src/frontend/telegram/bot.py"
        "admin_panel:src/frontend/telegram/screener/admin_panel.py"
        "background_services:src/frontend/telegram/screener/background_services.py"
        "alert_monitor:src/frontend/telegram/screener/alert_monitor.py"
        "schedule_processor:src/frontend/telegram/screener/schedule_processor.py"
        "json2csv:src/backtester/optimizer/run_json2csv.py"
        "optimizer:src/backtester/optimizer/run_optimizer.py"
        "plotter:src/backtester/plotter/run_plotter.py"
        "lstm_optimizer:src/ml/lstm/lstm_optuna_log_return_from_csv.py"
    )
    
    for script_info in "${scripts[@]}"; do
        IFS=':' read -r script_name python_script <<< "$script_info"
        local pid_file="$PROJECT_ROOT/logs/pids/${script_name}.pid"
        
        if is_running "$script_name"; then
            local pid=$(cat "$pid_file")
            echo "🟢 $script_name: Running (PID: $pid)"
        else
            echo "🔴 $script_name: Stopped"
        fi
    done
    echo
}

# Function to display menu
show_menu() {
    clear
    echo
    echo "========================================"
    echo "   Crypto Trading Platform - Scripts"
    echo "========================================"
    echo
    show_status
    echo "Available actions:"
    echo
    echo "[1]  Start Telegram Bot"
    echo "[2]  Start Admin Panel"
    echo "[3]  Start Background Services"
    echo "[4]  Start JSON to CSV Converter"
    echo "[5]  Start Optimizer"
    echo "[6]  Start Plotter"
    echo "[7]  Start LSTM Optimizer"
    echo
    echo "[8]  Stop Telegram Bot"
    echo "[9]  Stop Admin Panel"
    echo "[10] Stop Background Services"
    echo "[11] Stop JSON to CSV Converter"
    echo "[12] Stop Optimizer"
    echo "[13] Stop Plotter"
    echo "[14] Stop LSTM Optimizer"
    echo
    echo "[15] Show Telegram Bot Logs"
    echo "[16] Show Admin Panel Logs"
    echo "[17] Show Background Services Logs"
    echo "[18] Show JSON to CSV Logs"
    echo "[19] Show Optimizer Logs"
    echo "[20] Show Plotter Logs"
    echo "[21] Show LSTM Optimizer Logs"
    echo
    echo "[22] Stop All Scripts"
    echo "[23] Show All Logs"
    echo "[24] Exit"
    echo
}

# Main menu loop
while true; do
    show_menu
    read -p "Select an action (1-24): " choice
    
    case $choice in
        1)
            start_script "telegram_bot" "src/frontend/telegram/bot.py"
            ;;
        2)
            start_script "admin_panel" "src/frontend/telegram/screener/admin_panel.py"
            ;;
        3)
            start_script "background_services" "src/frontend/telegram/screener/background_services.py"
            ;;
        4)
            start_script "json2csv" "src/backtester/optimizer/run_json2csv.py"
            ;;
        5)
            start_script "optimizer" "src/backtester/optimizer/run_optimizer.py"
            ;;
        6)
            start_script "plotter" "src/backtester/plotter/run_plotter.py"
            ;;
        7)
            start_script "lstm_optimizer" "src/ml/lstm/lstm_optuna_log_return_from_csv.py"
            ;;
        8)
            stop_script "telegram_bot"
            ;;
        9)
            stop_script "admin_panel"
            ;;
        10)
            stop_script "background_services"
            ;;
        11)
            stop_script "json2csv"
            ;;
        12)
            stop_script "optimizer"
            ;;
        13)
            stop_script "plotter"
            ;;
        14)
            stop_script "lstm_optimizer"
            ;;
        15)
            show_logs "telegram_bot"
            ;;
        16)
            show_logs "admin_panel"
            ;;
        17)
            show_logs "background_services"
            ;;
        18)
            show_logs "json2csv"
            ;;
        19)
            show_logs "optimizer"
            ;;
        20)
            show_logs "plotter"
            ;;
        21)
            show_logs "lstm_optimizer"
            ;;
        22)
            echo "🛑 Stopping all scripts..."
            stop_script "telegram_bot"
            stop_script "admin_panel"
            stop_script "background_services"
            stop_script "json2csv"
            stop_script "optimizer"
            stop_script "plotter"
            stop_script "lstm_optimizer"
            echo "✅ All scripts stopped"
            ;;
        23)
            echo "📋 Recent logs for all scripts:"
            echo "========================================"
            for script in telegram_bot admin_panel background_services json2csv optimizer plotter lstm_optimizer; do
                show_logs "$script"
                echo
            done
            ;;
        24)
            echo
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid choice. Please try again."
            ;;
    esac
    
    if [ "$choice" -ge 1 ] && [ "$choice" -le 24 ]; then
        echo
        read -p "Press Enter to continue..."
    fi
done
