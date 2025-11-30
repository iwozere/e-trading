#!/usr/bin/env bash
# Reload Scheduler Service Schedules from Database
#
# Usage:
#   ./bin/scheduler/reload.sh

set -e

# Change to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Load environment if .env exists
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "========================================"
echo "Reloading Scheduler Schedules"
echo "========================================"
echo "This will reload all enabled schedules from the database"
echo ""

python -m src.scheduler.cli reload

echo ""
echo "========================================"
echo "Reload complete"
echo "========================================"
