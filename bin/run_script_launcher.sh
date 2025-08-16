#!/bin/bash

# ========================================
# Crypto Trading Platform - Script Launcher
# ========================================

# Get the absolute path to the project root
PROJECT_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"

# Change to project root directory
cd "$PROJECT_ROOT"

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

# Function to display menu
show_menu() {
    clear
    echo
    echo "========================================"
    echo "   Crypto Trading Platform - Scripts"
    echo "========================================"
    echo
    echo "Available scripts:"
    echo
    echo "[1]  Telegram Bot"
    echo "[2]  Telegram Admin Panel"
    echo "[3]  Telegram Background Services"
    echo "[4]  JSON to CSV Converter"
    echo "[5]  Optimizer"
    echo "[6]  Plotter"
    echo "[7]  LSTM Optimizer"
    echo "[8]  Exit"
    echo
}

# Function to run telegram bot
run_telegram_bot() {
    echo
    echo "Starting Telegram Bot..."
    "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/src/frontend/telegram/bot.py"
    echo
    read -p "Press Enter to continue..."
}

# Function to run admin panel
run_admin_panel() {
    echo
    echo "Starting Telegram Admin Panel..."
    echo "The admin panel will start a web server."
    echo "Once it's running, open your web browser and go to:"
    echo "  http://localhost:5000 (or check the console for the correct port)"
    echo
    echo "Login credentials are configured in config/donotshare/.env"
    echo "Press Ctrl+C to stop the admin panel when done."
    echo
    "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/src/frontend/telegram/screener/admin_panel.py"
    echo
    read -p "Press Enter to continue..."
}



# Function to run background services
run_background_services() {
    echo
    echo "Starting Telegram Background Services..."
    "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/src/frontend/telegram/screener/background_services.py"
    echo
    read -p "Press Enter to continue..."
}

# Function to run JSON to CSV converter
run_json2csv() {
    echo
    echo "Starting JSON to CSV Converter..."
    "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/src/backtester/optimizer/run_json2csv.py"
    echo
    read -p "Press Enter to continue..."
}

# Function to run optimizer
run_optimizer() {
    echo
    echo "Starting Optimizer..."
    "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/src/backtester/optimizer/run_optimizer.py"
    echo
    read -p "Press Enter to continue..."
}

# Function to run plotter
run_plotter() {
    echo
    echo "Starting Plotter..."
    "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/src/backtester/plotter/run_plotter.py"
    echo
    read -p "Press Enter to continue..."
}

# Function to run LSTM optimizer
run_lstm_optimizer() {
    echo
    echo "Starting LSTM Optimizer..."
    "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/src/ml/lstm/lstm_optuna_log_return_from_csv.py"
    echo
    read -p "Press Enter to continue..."
}

# Main menu loop
while true; do
    show_menu
    read -p "Select a script to run (1-8): " choice
    
    case $choice in
        1)
            run_telegram_bot
            ;;
        2)
            run_admin_panel
            ;;
        3)
            run_background_services
            ;;
        4)
            run_json2csv
            ;;
        5)
            run_optimizer
            ;;
        6)
            run_plotter
            ;;
        7)
            run_lstm_optimizer
            ;;
        8)
            echo
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid choice. Please try again."
            read -p "Press Enter to continue..."
            ;;
    esac
done
