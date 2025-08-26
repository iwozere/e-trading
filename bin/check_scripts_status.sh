#!/bin/bash

# ========================================
# Crypto Trading Platform - Script Status Checker
# ========================================

# Get the absolute path to the project root
PROJECT_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"

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

# Function to show logs
show_logs() {
    local script_name=$1
    local log_file="$PROJECT_ROOT/logs/log/${script_name}.log"
    
    if [ -f "$log_file" ]; then
        echo "📋 Recent logs for $script_name:"
        echo "----------------------------------------"
        tail -n 10 "$log_file"
        echo "----------------------------------------"
    else
        echo "❌ No log file found for $script_name"
    fi
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

# Show status of all scripts
echo "📊 Crypto Trading Platform - Script Status"
echo "=========================================="
echo

local scripts=(
    "telegram_bot:Telegram Bot"
    "admin_panel:Admin Panel"
    "background_services:Background Services"
    "json2csv:JSON to CSV Converter"
    "optimizer:Optimizer"
    "plotter:Plotter"
    "lstm_optimizer:LSTM Optimizer"
)

running_count=0
total_count=0

for script_info in "${scripts[@]}"; do
    IFS=':' read -r script_name display_name <<< "$script_info"
    local pid_file="$PROJECT_ROOT/logs/pids/${script_name}.pid"
    
    if is_running "$script_name"; then
        local pid=$(cat "$pid_file")
        echo "🟢 $display_name: Running (PID: $pid)"
        ((running_count++))
    else
        echo "🔴 $display_name: Stopped"
    fi
    ((total_count++))
done

echo
echo "Summary: $running_count/$total_count scripts running"
echo

# Interactive menu
if [ "$1" = "--interactive" ] || [ "$1" = "-i" ]; then
    echo "Available actions:"
    echo "[1] Show logs for a script"
    echo "[2] Stop a script"
    echo "[3] Stop all scripts"
    echo "[4] Exit"
    echo
    
    read -p "Select an action (1-4): " choice
    
    case $choice in
        1)
            echo
            echo "Select a script to show logs:"
            echo "[1] Telegram Bot"
            echo "[2] Admin Panel"
            echo "[3] Background Services"
            echo "[4] JSON to CSV Converter"
            echo "[5] Optimizer"
            echo "[6] Plotter"
            echo "[7] LSTM Optimizer"
            echo
            read -p "Select script (1-7): " script_choice
            
            case $script_choice in
                1) show_logs "telegram_bot" ;;
                2) show_logs "admin_panel" ;;
                3) show_logs "background_services" ;;
                4) show_logs "json2csv" ;;
                5) show_logs "optimizer" ;;
                6) show_logs "plotter" ;;
                7) show_logs "lstm_optimizer" ;;
                *) echo "Invalid choice" ;;
            esac
            ;;
        2)
            echo
            echo "Select a script to stop:"
            echo "[1] Telegram Bot"
            echo "[2] Admin Panel"
            echo "[3] Background Services"
            echo "[4] JSON to CSV Converter"
            echo "[5] Optimizer"
            echo "[6] Plotter"
            echo "[7] LSTM Optimizer"
            echo
            read -p "Select script (1-7): " script_choice
            
            case $script_choice in
                1) stop_script "telegram_bot" ;;
                2) stop_script "admin_panel" ;;
                3) stop_script "background_services" ;;
                4) stop_script "json2csv" ;;
                5) stop_script "optimizer" ;;
                6) stop_script "plotter" ;;
                7) stop_script "lstm_optimizer" ;;
                *) echo "Invalid choice" ;;
            esac
            ;;
        3)
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
        4)
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid choice"
            ;;
    esac
fi
