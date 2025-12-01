#!/usr/bin/env bash
# Test Scheduler Service Locally
# Run this to test if the scheduler can start without systemd

set -e

# Change to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo "Testing Scheduler Service"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# Check if virtual environment exists
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo "ERROR: Virtual environment not found at $PROJECT_ROOT/.venv"
    echo "Please create it first:"
    echo "  cd $PROJECT_ROOT"
    echo "  python -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT"
export TRADING_ENV=development
export LOG_LEVEL=DEBUG

echo "PYTHONPATH: $PYTHONPATH"
echo "TRADING_ENV: $TRADING_ENV"
echo ""

# Change to project directory
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Starting scheduler in foreground..."
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

# Run scheduler
python -m src.scheduler.main
