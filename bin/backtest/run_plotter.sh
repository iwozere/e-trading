#!/bin/bash

# ========================================
# Crypto Trading Platform - Plotter Runner
# ========================================

echo "Starting Crypto Trading Plotter..."

# Get the absolute path to the project root
PROJECT_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"

mkdir -p "$PROJECT_ROOT/logs/log"

if [ ! -f "$PROJECT_ROOT/.venv/bin/python" ]; then
    echo "Error: Python venv not found at $PROJECT_ROOT/.venv/bin/python"
    exit 1
fi

source "$PROJECT_ROOT/.venv/bin/activate"

nohup "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/src/backtester/plotter/run_plotter.py" > "$PROJECT_ROOT/logs/log/plotter.out" 2>&1 &

echo "Plotter started in background. Check $PROJECT_ROOT/logs/log/plotter.log for output."
