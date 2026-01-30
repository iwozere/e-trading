#!/bin/bash

# Navigate to the project root
PROJECT_ROOT="/home/pi/dev/e-trading"
cd $PROJECT_ROOT

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export IBKR_HOST="127.0.0.1"
export IBKR_PORT="7497"
export IBKR_CLIENT_ID="100"
export NOTIFICATION_SERVICE_URL="http://localhost:8000"
export DATA_CACHE_DIR="$PROJECT_ROOT/data/cache"

# Ensure data cache directory exists
mkdir -p $DATA_CACHE_DIR

# Run the screener
python3 src/screeners/main.py
