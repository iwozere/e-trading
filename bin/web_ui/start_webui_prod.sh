#!/bin/bash
#
# Trading Web UI Production Startup Script
# ========================================
#
# This script starts the Trading Web UI in production mode.
# It's designed to be used by systemd service or manual production deployment.
#
# Usage:
#   ./start_webui_prod.sh [--host HOST] [--port PORT]
#
# Environment Variables:
#   WEBUI_HOST - Host to bind to (default: 0.0.0.0)
#   WEBUI_PORT - Port to bind to (default: 5003)
#   PROJECT_ROOT - Project root directory (auto-detected)
#

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="5003"

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Parse command line arguments
HOST="${WEBUI_HOST:-$DEFAULT_HOST}"
PORT="${WEBUI_PORT:-$DEFAULT_PORT}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--host HOST] [--port PORT]"
            echo ""
            echo "Options:"
            echo "  --host HOST    Host to bind to (default: $DEFAULT_HOST)"
            echo "  --port PORT    Port to bind to (default: $DEFAULT_PORT)"
            echo "  --help, -h     Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  WEBUI_HOST     Host to bind to"
            echo "  WEBUI_PORT     Port to bind to"
            echo "  PROJECT_ROOT   Project root directory"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Function to log messages
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌${NC} $1"
}

# Print startup banner
echo -e "${BLUE}"
echo "🚀 Trading Web UI Production Startup"
echo "====================================="
echo -e "${NC}"
log "Project Root: $PROJECT_ROOT"
log "Host: $HOST"
log "Port: $PORT"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    log_error "Virtual environment not found at .venv"
    log_error "Please create a virtual environment first:"
    log_error "  python -m venv .venv"
    log_error "  source .venv/bin/activate"
    log_error "  pip install -r requirements.txt"
    exit 1
fi

# Check if Python executable exists in venv
PYTHON_EXEC=".venv/bin/python"
if [ ! -f "$PYTHON_EXEC" ]; then
    log_error "Python executable not found at $PYTHON_EXEC"
    exit 1
fi

log_success "Virtual environment found"

# Check if main script exists
MAIN_SCRIPT="src/web_ui/run_web_ui.py"
if [ ! -f "$MAIN_SCRIPT" ]; then
    log_error "Main script not found at $MAIN_SCRIPT"
    exit 1
fi

log_success "Main script found"

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

# Check if frontend is built for production
FRONTEND_DIST="$PROJECT_ROOT/src/web_ui/frontend/dist"
if [ ! -d "$FRONTEND_DIST" ]; then
    log_warning "Frontend dist directory not found"
    log "Building frontend for production..."
    
    # Check if Node.js is available
    if ! command -v node &> /dev/null; then
        log_error "Node.js is not installed"
        log_error "Please install Node.js >= 18.0.0"
        exit 1
    fi
    
    # Check if npm is available
    if ! command -v npm &> /dev/null; then
        log_error "npm is not installed"
        exit 1
    fi
    
    # Build frontend
    cd "$PROJECT_ROOT/src/web_ui/frontend"
    
    if [ ! -f "package.json" ]; then
        log_error "Frontend package.json not found"
        exit 1
    fi
    
    log "Installing frontend dependencies..."
    npm install
    
    log "Building frontend..."
    npm run build
    
    if [ ! -d "dist" ]; then
        log_error "Frontend build failed - dist directory not created"
        exit 1
    fi
    
    log_success "Frontend built successfully"
    cd "$PROJECT_ROOT"
else
    log_success "Frontend dist directory found"
fi

# Create logs directory if it doesn't exist
mkdir -p logs/web_ui

# Function to handle shutdown
shutdown() {
    log "Received shutdown signal"
    if [ ! -z "$WEBUI_PID" ]; then
        log "Stopping Web UI (PID: $WEBUI_PID)..."
        kill -TERM "$WEBUI_PID" 2>/dev/null || true
        wait "$WEBUI_PID" 2>/dev/null || true
    fi
    log_success "Web UI stopped"
    exit 0
}

# Set up signal handlers
trap shutdown SIGTERM SIGINT

# Start the Web UI
log "Starting Trading Web UI in production mode..."
log "Command: $PYTHON_EXEC $MAIN_SCRIPT --host $HOST --port $PORT"
echo ""

# Start the application
"$PYTHON_EXEC" "$MAIN_SCRIPT" --host "$HOST" --port "$PORT" &
WEBUI_PID=$!

log_success "Web UI started with PID: $WEBUI_PID"
log "Web UI available at: http://$HOST:$PORT"
log "API documentation at: http://$HOST:$PORT/docs"
log ""
log "Press Ctrl+C to stop the service"

# Wait for the process
wait "$WEBUI_PID"