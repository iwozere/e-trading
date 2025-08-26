#!/bin/bash

# ========================================
# Crypto Trading Platform - Optimizer Runner
# ========================================

echo "Starting Crypto Trading Optimizer..."

# Get the absolute path to the project root
PROJECT_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"

mkdir -p "$PROJECT_ROOT/logs/log"

if [ ! -f "$PROJECT_ROOT/.venv/bin/python" ]; then
    echo "Error: Python venv not found at $PROJECT_ROOT/.venv/bin/python"
    exit 1
fi

source "$PROJECT_ROOT/.venv/bin/activate"

nohup "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/src/backtester/optimizer/run_optimizer.py" > "$PROJECT_ROOT/logs/log/optimizer.out" 2>&1 &

echo "Optimizer started in background. Check $PROJECT_ROOT/logs/log/optimizer.log for output."
