#!/bin/bash

# ========================================
# Crypto Trading Platform - JSON to CSV Converter
# ========================================

echo "Starting JSON to CSV Converter..."

# Get the absolute path to the project root
PROJECT_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"

mkdir -p "$PROJECT_ROOT/logs/log"

if [ ! -f "$PROJECT_ROOT/.venv/bin/python" ]; then
    echo "Error: Python venv not found at $PROJECT_ROOT/.venv/bin/python"
    exit 1
fi

source "$PROJECT_ROOT/.venv/bin/activate"

nohup "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/src/backtester/optimizer/run_json2csv.py" > "$PROJECT_ROOT/logs/log/json2csv.out" 2>&1 &

echo "JSON to CSV converter started in background. Check $PROJECT_ROOT/logs/log/json2csv.log for output."
