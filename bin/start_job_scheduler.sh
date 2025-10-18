#!/bin/bash
# Start Job Scheduler Process
# This script starts the APScheduler process for cron-based job triggering

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

# Start the scheduler process
echo "Starting job scheduler process..."
python -m src.backend.scheduler.scheduler_process

echo "Job scheduler process started successfully"


