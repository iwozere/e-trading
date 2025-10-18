#!/bin/bash
# Start Dramatiq Workers
# This script starts the Dramatiq workers for job execution

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Create logs directory if it doesn't exist
mkdir -p logs

# Start Dramatiq workers
echo "Starting Dramatiq workers..."
echo "Workers will process jobs from the 'reports' and 'screeners' queues"

# Start workers with proper configuration
dramatiq src.backend.workers.report_worker src.backend.workers.screener_worker \
    --processes 4 \
    --threads 2 \
    --queues reports,screeners \
    --log-file logs/dramatiq_workers.log \
    --pid-file logs/dramatiq_workers.pid

echo "Dramatiq workers started successfully"


