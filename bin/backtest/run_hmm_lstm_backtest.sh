#!/bin/bash
# HMM-LSTM Backtesting Runner for Unix/Linux
# This script runs the HMM-LSTM backtesting optimizer

echo "Starting HMM-LSTM Backtesting..."
echo

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_ROOT/.venv/bin/activate"
elif [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Set Python path and run the optimizer
python3 src/backtester/optimizer/hmm_lstm.py "$@"

echo
echo "HMM-LSTM Backtesting completed."
