#!/bin/bash
# Wrapper script to run the service monitor in the virtual environment

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Path to the virtual environment python
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

# Fallback to system python if venv doesn't exist (though venv is expected)
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Warning: Virtual environment not found at $VENV_PYTHON, falling back to system python"
    VENV_PYTHON="python3"
fi

# Run the monitor
cd "$PROJECT_ROOT"
$VENV_PYTHON src/notification/service_monitor.py "$@"
