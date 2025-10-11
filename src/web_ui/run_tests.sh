#!/bin/bash
# Web UI Test Runner for Linux/macOS
# ==================================
# 
# This shell script provides an easy way to run all Web UI tests on Unix-like systems.
# It supports both backend Python tests and frontend TypeScript tests.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
VERBOSE=0
NO_COVERAGE=0
BACKEND_ONLY=0
FRONTEND_ONLY=0
SEQUENTIAL=0
SKIP_FRONTEND=0

# Function to print colored output
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to show help
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -v, --verbose      Enable verbose output"
    echo "  --no-coverage      Disable coverage reporting"
    echo "  --backend-only     Run only backend tests"
    echo "  --frontend-only    Run only frontend tests"
    echo "  --sequential       Run tests sequentially (not in parallel)"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Run all tests with coverage"
    echo "  $0 --verbose          # Run with verbose output"
    echo "  $0 --backend-only     # Run only Python backend tests"
    echo "  $0 --frontend-only    # Run only TypeScript frontend tests"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        --no-coverage)
            NO_COVERAGE=1
            shift
            ;;
        --backend-only)
            BACKEND_ONLY=1
            shift
            ;;
        --frontend-only)
            FRONTEND_ONLY=1
            shift
            ;;
        --sequential)
            SEQUENTIAL=1
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_color $RED "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Print header
echo ""
print_color $BLUE "================================================================================"
print_color $BLUE "                           WEB UI TEST RUNNER"
print_color $BLUE "================================================================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    print_color $RED "ERROR: Python is not installed or not in PATH"
    print_color $RED "Please install Python 3.8+ and try again"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
print_color $GREEN "Using Python $PYTHON_VERSION"

# Check if Node.js is available for frontend tests
if ! command -v node &> /dev/null; then
    print_color $YELLOW "WARNING: Node.js is not installed or not in PATH"
    print_color $YELLOW "Frontend tests will be skipped"
    SKIP_FRONTEND=1
else
    NODE_VERSION=$(node --version)
    print_color $GREEN "Using Node.js $NODE_VERSION"
fi

# Build Python command arguments
PYTHON_ARGS=""
if [ $VERBOSE -eq 1 ]; then
    PYTHON_ARGS="$PYTHON_ARGS --verbose"
fi
if [ $NO_COVERAGE -eq 1 ]; then
    PYTHON_ARGS="$PYTHON_ARGS --no-coverage"
fi
if [ $BACKEND_ONLY -eq 1 ]; then
    PYTHON_ARGS="$PYTHON_ARGS --backend-only"
fi
if [ $FRONTEND_ONLY -eq 1 ]; then
    PYTHON_ARGS="$PYTHON_ARGS --frontend-only"
fi
if [ $SEQUENTIAL -eq 1 ]; then
    PYTHON_ARGS="$PYTHON_ARGS --sequential"
fi

# Skip frontend if Node.js not available and not backend-only
if [ $SKIP_FRONTEND -eq 1 ] && [ $BACKEND_ONLY -eq 0 ] && [ $FRONTEND_ONLY -eq 0 ]; then
    print_color $YELLOW "Node.js not available, running backend tests only..."
    PYTHON_ARGS="$PYTHON_ARGS --backend-only"
fi

# Change to script directory
cd "$(dirname "$0")"

print_color $BLUE "Starting Web UI test suite..."
echo ""

# Run the Python test runner
print_color $BLUE "Executing: $PYTHON_CMD run_all_tests.py$PYTHON_ARGS"
echo ""

if $PYTHON_CMD run_all_tests.py$PYTHON_ARGS; then
    TEST_EXIT_CODE=0
    echo ""
    print_color $GREEN "================================================================================"
    print_color $GREEN "                              ALL TESTS PASSED!"
    print_color $GREEN "================================================================================"
else
    TEST_EXIT_CODE=$?
    echo ""
    print_color $RED "================================================================================"
    print_color $RED "                              SOME TESTS FAILED!"
    print_color $RED "================================================================================"
    print_color $RED "Exit code: $TEST_EXIT_CODE"
fi

# Show coverage reports if available
if [ $NO_COVERAGE -eq 0 ]; then
    echo ""
    print_color $BLUE "Coverage reports generated:"
    if [ -f "backend/htmlcov/index.html" ]; then
        print_color $GREEN "  Backend: backend/htmlcov/index.html"
    fi
    if [ -f "frontend/coverage/index.html" ]; then
        print_color $GREEN "  Frontend: frontend/coverage/index.html"
    fi
fi

echo ""
print_color $BLUE "Test run completed."

exit $TEST_EXIT_CODE